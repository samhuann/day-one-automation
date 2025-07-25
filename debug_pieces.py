#!/usr/bin/env python
import sys
import pandas as pd
sys.path.append('.')
from backfill import ingest_discord, ingest_dayone
from pathlib import Path
from datetime import datetime

day = datetime(2023, 7, 6).date()

# Get data
df_discord = ingest_discord(Path('data/discord'))
df_dayone = ingest_dayone(Path('data/dayone/dayone.json'))

# Filter for the day
d_discord = df_discord[df_discord["date"] == day]
d_dayone = df_dayone[df_dayone["date"] == day]

print(f'Discord messages: {len(d_discord)}')
print(f'DayOne entries: {len(d_dayone)}')

# Simulate the raw_pieces creation
raw_pieces = []

# Discord messages
for _, row in d_discord.sort_values("timestamp").iterrows():
    ts = row["timestamp"].strftime("%H:%M")
    raw_pieces.append(f"[{ts}] {row.get('author')}: {row.get('text','')}")

# DayOne entries
if len(d_dayone):
    for _, row in d_dayone.sort_values("timestamp").iterrows():
        ts = row["timestamp"].strftime("%H:%M")
        
        # Add location info if available
        if pd.notna(row.get('lat')) and pd.notna(row.get('lon')):
            place = row.get('place_name', 'Unknown location')
            weather = f" ({row.get('weather', '')})" if row.get('weather') else ""
            location_entry = f"[{ts}] LOCATION: {place}{weather}"
            raw_pieces.append(location_entry)
            print(f"Added location entry: {location_entry}")
        
        # Add journal text if available
        journal_text = row.get('text', '').strip()
        if journal_text:
            if len(journal_text) > 500:
                journal_text = journal_text[:500] + "..."
            journal_entry = f"[{ts}] JOURNAL: {journal_text}"
            raw_pieces.append(journal_entry)
            print(f"Added journal entry: {journal_entry[:100]}...")

print(f'Total raw pieces: {len(raw_pieces)}')
print(f'Last 5 pieces:')
for piece in raw_pieces[-5:]:
    print(f"  {piece[:100]}...")
