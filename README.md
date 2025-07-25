# day-one-automation

**Fill 400+ missing Day One entries** by automatically summarizing your **Discord chats**, **Day One location history**, and **photos** — entirely **locally** with **Ollama** (no API keys).

- ✅ **No cloud, no API keys** — runs a local LLM via Ollama  
- ✅ **Windows‑friendly** (works on macOS/Linux too)  
- ✅ **Hierarchical summarization** to handle long chat logs  
- ✅ **One Markdown per day** ready to paste into Day One (or import later)

---

## ✨ What it does

1. **Ingests data**
   - Discord JSON exports (DMs/group chats)
   - Day One JSON export (locations included)
   - Photo EXIF timestamps

2. **Fuses by day**: groups everything on the same date.

3. **Summarizes with a local LLM** (Ollama, e.g. `llama3.1:8b`) using a **map‑reduce** pass so long days don’t exceed context limits.

4. **Outputs Markdown** to `outputs/days/YYYY-MM-DD.md` with: narrative, rough timeline, people, locations, and evidence counts.

---

## 🧰 Prerequisites

- **Python 3.11+**
- **Git** (optional but recommended)
- **Ollama** for Windows/macOS/Linux  
  After installing, pull a model (examples):

```powershell
ollama pull llama3.1:8b
# If RAM is tight, try a quantized variant:
# ollama pull llama3.1:8b-instruct-q4_K_M
