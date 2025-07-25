#!/usr/bin/env python
import os
import json
import glob
import orjson
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from collections import defaultdict, Counter

import typer
import pandas as pd
from tqdm import tqdm
from rich import print
from jinja2 import Template

import exifread
import ollama

app = typer.Typer(help="Offline journal backfill (Discord + DayOne + Photos)")

# ----------------------------
# Helpers
# ----------------------------

def load_config(path: Path):
    if not path.exists():
        return dict(
            model="llama3.1:8b",
            ctx=8192,
            temperature=0.2,
            daily_template=DEFAULT_MD_TEMPLATE
        )
    import yaml
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if "daily_template" not in cfg:
        cfg["daily_template"] = DEFAULT_MD_TEMPLATE
    return cfg

def ensure_dirs():
    Path("outputs/days").mkdir(parents=True, exist_ok=True)
    Path("outputs/cache").mkdir(parents=True, exist_ok=True)

def to_date(d):
    if isinstance(d, datetime):
        return d.date()
    return dtparser.parse(d).date()

def read_json_fast(p):
    with open(p, "rb") as f:
        return orjson.loads(f.read())

def call_llm(prompt, model="llama3.1:8b", temperature=0.2, ctx=8192):
    # Ollama respects ctx via env var; still passing is fine
    try:
        res = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_ctx": ctx},
            stream=False
        )
        return res["message"]["content"]
    except Exception as e:
        print(f"[red]LLM call failed: {e}[/red]")
        return f"(LLM processing failed: {str(e)})"

DEFAULT_MD_TEMPLATE = """# {{ date }}

**People I talked to:** {{ people|join(", ") if people else "—" }}
**Places mentioned:** {{ locations|join(" → ") if locations else "—" }}
**How I felt:** {{ mood }}

## My day
{{ summary }}

## What happened when
{% if timeline %}
{% for item in timeline -%}
- {{ item.time }} — {{ item.event }}
{% endfor %}
{% else %}
(didn't note specific times today)
{% endif %}

## Stats
- Messages I saw: {{ stats.discord_messages }}
- Photos taken: {{ stats.photos }}
- Places visited: {{ stats.location_points }}

## Channels I was in
{{ channels|join(", ") if channels else "—" }}
"""

# ----------------------------
# 1) INGEST
# ----------------------------

def ingest_discord(discord_dir: Path) -> pd.DataFrame:
    """
    Read every *.json in `discord_dir` and return one DataFrame with columns:
    source, channel, author, timestamp, text, date
    Combines messages from all Discord channels/chats for each day.
    """

    
    rows = []
    channels_found = []
    
    for fp in glob.glob(str(discord_dir / "*.json")):
        try:
            data = read_json_fast(Path(fp))
        except Exception as e:
            print(f"[red]Could not read {fp}: {e}")
            continue

        messages = data.get("messages") or data  # schema variant
        channel_name = data.get("channel", {}).get("name") or Path(fp).stem
        channels_found.append(channel_name)
        
        print(f"[dim]Processing {channel_name}: ", end="")
        channel_message_count = 0

        for m in messages:
            ts = m.get("timestamp") or m.get("Timestamp") or m.get("Date")
            if not ts:
                continue
            author  = (m.get("author") or {}).get("name") or m.get("Author")
            content = m.get("content") or m.get("Content") or ""

            rows.append(
                dict(
                    source="discord",
                    channel=channel_name,
                    author=author,
                    timestamp=ts,     # raw string for now
                    text=content,
                )
            )
            channel_message_count += 1
        
        print(f"[cyan]{channel_message_count} messages[/cyan]")

    if not rows:
        return pd.DataFrame(
            columns=["source", "channel", "author", "timestamp", "text"]
        )

    print(f"[cyan]Total channels processed: {len(channels_found)} ({', '.join(channels_found)})[/cyan]")
    
    df = pd.DataFrame(rows)

    # 🔧 Robust timestamp parsing; drop rows that fail
    # First try with UTC timezone handling for ISO format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    
    # If that didn't work, try without UTC
    if df["timestamp"].dtype == 'object':
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    
    # Drop rows where timestamp parsing failed
    df = df.dropna(subset=["timestamp"])
    
    # Ensure we have a proper datetime dtype
    if len(df) > 0 and df["timestamp"].dtype == 'object':
        print(f"[yellow]Warning: timestamp column still has object dtype after parsing")
        # Try manual parsing for problematic cases
        df["timestamp"] = df["timestamp"].apply(
            lambda x: dtparser.parse(x) if isinstance(x, str) else x
        )

    df["date"] = df["timestamp"].dt.date

    
    return df


def ingest_dayone(dayone_json: Path) -> pd.DataFrame:
    if not dayone_json.exists():
        return pd.DataFrame(columns=["source", "timestamp", "lat", "lon", "place_name", "text", "weather", "date"])

    data = read_json_fast(dayone_json)
    # Day One export format: {"metadata":..., "entries":[{...}]}
    entries = data.get("entries", [])
    rows = []
    for e in entries:
        # Get timestamp
        tm = e.get("creationDate") or e.get("modifiedDate") or e.get("date")
        if not tm:
            continue
        t = dtparser.parse(tm)
        
        # Get location info
        loc = e.get("location") or {}
        
        # Get journal text content
        text_content = e.get("text", "").strip()
        
        # Get weather info
        weather = e.get("weather") or {}
        weather_desc = ""
        if weather:
            temp = weather.get("temperatureCelsius")
            conditions = weather.get("conditionsDescription", "")
            if temp and conditions:
                temp_f = round(temp * 9/5 + 32)
                weather_desc = f"{conditions}, {temp_f}°F"
            elif conditions:
                weather_desc = conditions
        
        # Create entry for location/journal content
        if loc or text_content:
            rows.append(dict(
                source="dayone",
                timestamp=t,
                lat=loc.get("latitude"),
                lon=loc.get("longitude"),
                place_name=loc.get("placeName") or loc.get("localityName"),
                text=text_content,
                weather=weather_desc
            ))
            
    if not rows:
        return pd.DataFrame(columns=["source", "timestamp", "lat", "lon", "place_name", "text", "weather", "date"])
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.date
    return df

def ingest_photos(photo_dir: Path) -> pd.DataFrame:
    rows = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.heic", "*.JPG", "*.JPEG", "*.PNG", "*.HEIC"):
        for fp in glob.glob(str(photo_dir / "**" / ext), recursive=True):
            timestamp = None
            try:
                with open(fp, "rb") as f:
                    tags = exifread.process_file(f, details=False)
                dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
                if dt:
                    timestamp = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
            except Exception:
                pass
            if not timestamp:
                # fallback: filesystem timestamp
                timestamp = datetime.fromtimestamp(Path(fp).stat().st_mtime)
            rows.append(dict(
                source="photo",
                path=fp,
                timestamp=timestamp
            ))
    if not rows:
        return pd.DataFrame(columns=["source", "path", "timestamp", "date"])
    df = pd.DataFrame(rows)
    df["date"] = df["timestamp"].dt.date
    return df

# ----------------------------
# 2) DAILY FUSION + SUMMARIZATION
# ----------------------------

def chunk_text_by_tokens(texts, max_chars=12000):
    # Larger chunks for more detailed processing - more context per chunk
    # Optimized chunking for detailed summaries while maintaining speed
    chunks, buf = [], []
    size = 0
    for t in texts:
        if size + len(t) > max_chars and buf:
            chunks.append("\n".join(buf))
            buf, size = [t], len(t)
        else:
            buf.append(t)
            size += len(t)
    if buf:
        chunks.append("\n".join(buf))
    
    print(f"[dim]Created {len(chunks)} chunks from {len(texts)} text pieces[/dim]")
    return chunks

def map_reduce_summarize(chunks, model, temperature, ctx):
    # MAP: summarize each chunk
    partial_summaries = []
    print(f"[dim]Processing {len(chunks)} chunks...[/dim]")
    
    for i, c in enumerate(chunks):
        print(f"[dim]Processing chunk {i+1}/{len(chunks)} ({len(c)} chars)...[/dim]")
        prompt = f"""You are Citpyrk writing detailed notes in your personal journal. Analyze this chunk and extract rich details from YOUR perspective:

FOCUS ON:
- Specific conversations you had and what was discussed
- Your reactions, thoughts, and feelings about events
- Interesting quotes or memorable moments
- People you interacted with and the nature of your interactions
- Any locations you visited or mentioned
- Games you played, content you consumed, activities you did
- Your mood changes throughout the time period

Be specific and detailed - capture the nuance and personality of your day.

Chunk #{i+1}:
{c}

Write detailed bullet points from Citpyrk's first-person perspective:"""
        
        try:
            summary = call_llm(prompt, model, temperature, ctx)
            partial_summaries.append(summary)
            print(f"[dim]✓ Chunk {i+1} processed[/dim]")
        except Exception as e:
            print(f"[red]✗ Error processing chunk {i+1}: {e}[/red]")
            partial_summaries.append(f"(Error processing chunk {i+1})")

    print(f"[dim]Merging {len(partial_summaries)} partial summaries...[/dim]")
    # REDUCE: merge bullets
    reduce_prompt = f"""You are Citpyrk writing a detailed, personal journal entry about your day. 
Merge these detailed notes into a rich, comprehensive journal entry from YOUR perspective.

Daily notes to merge:
{chr(10).join(partial_summaries)}

Create a detailed journal entry that:
- Captures specific conversations, events, and interactions
- Shows your personality, thoughts, and reactions
- Includes interesting details and memorable moments
- Mentions specific people and what you discussed with them
- Notes any places you went (look for "LOCATION:" entries)
- Describes your activities, games, content consumed
- Reflects your changing moods and feelings throughout the day

IMPORTANT: Look for any mentions of "LOCATION:" or places you visited!

Return JSON only:
- "summary": 4-6 detailed sentences in first person as Citpyrk, rich with specifics
- "timeline": detailed timeline with specific events and your reactions
- "people": who you interacted with
- "locations": any places mentioned (look specifically for "LOCATION:" entries)
- "mood": one word describing your overall feeling

JSON only, no explanation:"""
    
    try:
        merged = call_llm(reduce_prompt, model, temperature, ctx)
        print(f"[dim]✓ Merge completed[/dim]")
    except Exception as e:
        print(f"[red]✗ Error in merge step: {e}[/red]")
        merged = '{"summary": "Error during summarization", "timeline": [], "people": [], "locations": [], "mood": "unknown"}'
    
    # model may not return perfect JSON; fall back
    try:
        # Clean up the response first
        merged = merged.strip()
        
        # Handle code fences
        if "```json" in merged:
            start = merged.find("```json") + 7
            end = merged.find("```", start)
            if end != -1:
                merged = merged[start:end].strip()
        elif merged.startswith("```"):
            merged = merged.strip("`").strip()
        
        # Find JSON object boundaries
        if not merged.startswith("{"):
            start_idx = merged.find("{")
            if start_idx != -1:
                merged = merged[start_idx:]
        
        if not merged.endswith("}"):
            end_idx = merged.rfind("}")
            if end_idx != -1:
                merged = merged[:end_idx + 1]
        
        # Try to parse
        out = json.loads(merged)
        
        # Validate required keys exist
        if not all(key in out for key in ["summary", "timeline", "people", "locations", "mood"]):
            raise ValueError("Missing required keys")
            
    except Exception as e:
        print(f"[yellow]Warning: JSON parse failed, using fallback: {e}[/yellow]")
        print(f"[dim]Raw response: {merged[:200]}...[/dim]")
        out = {
            "summary": merged.strip(),
            "timeline": [],
            "people": [],
            "locations": [],
            "mood": "unknown"
        }
    return out

def render_markdown(day, fused, stats, channels, template_str):
    template = Template(template_str)
    return template.render(
        date=str(day),
        summary=fused.get("summary", "").strip(),
        timeline=fused.get("timeline", []),
        people=fused.get("people", []),
        locations=fused.get("locations", []),
        mood=fused.get("mood", "unknown"),
        stats=stats,
        channels=sorted(channels)
    )

# ----------------------------
# 3) CLI
# ----------------------------

@app.command()
def run(
    start: datetime = typer.Option(..., help="Start date, e.g. 2023-09-01"),
    end: datetime = typer.Option(None, help="End date, e.g. 2024-11-30 (defaults to start date for single-day processing)"),
    model: str = typer.Option(None, help="Ollama model (overrides config.yaml)"),
    discord_dir: Path = typer.Option(Path("data/discord"), exists=False),
    dayone_json: Path = typer.Option(Path("data/dayone/dayone.json"), exists=False),
    photo_dir: Path = typer.Option(Path("data/photos"), exists=False),
    config_path: Path = typer.Option(Path("config.yaml"), exists=False),
):
    """
    End-to-end daily backfill.
    """
    ensure_dirs()
    cfg = load_config(config_path)
    if model:
        cfg["model"] = model

    # If end date not provided, default to start date (single day processing)
    if end is None:
        end = start

    print("[bold cyan]Ingesting…[/bold cyan]")
    df_discord = ingest_discord(discord_dir)
    df_dayone  = ingest_dayone(dayone_json)
    df_photos  = ingest_photos(photo_dir)

    print(f"Discord messages: {len(df_discord)}")
    print(f"DayOne loc points: {len(df_dayone)}")
    print(f"Photos indexed:   {len(df_photos)}")

    # daily loop
    current = start.date()
    end_date = end.date()
    pbar_total = (end_date - current).days + 1

    for _ in tqdm(range(pbar_total), desc="Days"):
        day = current
        out_path = Path(f"outputs/days/{day}.md")
        cache_path = Path(f"outputs/cache/{day}.json")

        if out_path.exists():
            current += timedelta(days=1)
            continue

        # subset per day
        d_discord = df_discord[df_discord["date"] == day]
        d_dayone  = df_dayone[df_dayone["date"] == day]
        d_photos  = df_photos[df_photos["date"] == day]

        # nothing this day? still produce a skeleton
        if len(d_discord) == 0 and len(d_dayone) == 0 and len(d_photos) == 0:
            md = render_markdown(
                day,
                fused=dict(summary="(Nothing captured)", timeline=[], people=[], locations=[], mood="unknown"),
                stats=dict(discord_messages=0, photos=0, location_points=0),
                channels=[],
                template_str=cfg["daily_template"]
            )
            out_path.write_text(md, encoding="utf-8")
            current += timedelta(days=1)
            continue

        # collect raw day text
        raw_pieces = []

        channels = set()
        if len(d_discord):
            channels |= set(d_discord["channel"].dropna().unique())
            
            # Process all messages but optimize for speed
            discord_sample = d_discord
            if len(d_discord) > 2000:
                print(f"[yellow]Very large day detected ({len(d_discord)} messages), using all messages with optimized processing[/yellow]")
            elif len(d_discord) > 1000:
                print(f"[yellow]Large day detected ({len(d_discord)} messages), using all messages[/yellow]")
            elif len(d_discord) > 500:
                print(f"[yellow]Medium day detected ({len(d_discord)} messages), using all messages[/yellow]")
            
            # Always use all messages, just inform about processing approach
            
            # turn discord messages into lines "[HH:MM] author: text"
            for _, row in discord_sample.iterrows():
                ts = row["timestamp"].strftime("%H:%M")
                raw_pieces.append(f"[{ts}] {row.get('author')}: {row.get('text','')}")

        if len(d_dayone):
            for _, row in d_dayone.sort_values("timestamp").iterrows():
                ts = row["timestamp"].strftime("%H:%M")
                
                # Add location info if available
                if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
                    place = row.get('place_name', 'Unknown location')
                    weather = f" ({row.get('weather', '')})" if row.get('weather') else ""
                    raw_pieces.append(f"[{ts}] LOCATION: {place}{weather}")
                
                # Add journal text if available
                journal_text = row.get('text', '').strip()
                if journal_text:
                    # Truncate very long entries to avoid overwhelming the context
                    if len(journal_text) > 500:
                        journal_text = journal_text[:500] + "..."
                    raw_pieces.append(f"[{ts}] JOURNAL: {journal_text}")

        if len(d_photos):
            for _, row in d_photos.sort_values("timestamp").iterrows():
                ts = row["timestamp"].strftime("%H:%M")
                raw_pieces.append(f"[{ts}] PHOTO: {Path(row['path']).name}")

        # chunk + map-reduce
        chunks = chunk_text_by_tokens(raw_pieces, max_chars=12000)
        fused = map_reduce_summarize(
            chunks,
            model=cfg["model"],
            temperature=cfg.get("temperature", 0.2),
            ctx=cfg.get("ctx", 8192)
        )

        stats = dict(
            discord_messages=int(len(d_discord)),
            photos=int(len(d_photos)),
            location_points=int(len(d_dayone))
        )

        md = render_markdown(day, fused, stats, channels, cfg["daily_template"])
        out_path.write_text(md, encoding="utf-8")
        cache_path.write_text(orjson.dumps(fused).decode("utf-8"), encoding="utf-8")

        current += timedelta(days=1)

    print("[green]Done! Check outputs/days/[/green]")


if __name__ == "__main__":
    app()
