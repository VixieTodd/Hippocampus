# Hippocampus — AI Bionic Memory System

🦛 A local three-layer memory system for AI agents.

[中文版](./README_CN.md)

---

## Overview

LLM agents lack persistent memory across sessions. Each session starts with no context from previous interactions.

Hippocampus provides a three-layer memory architecture: agents search relevant history, inject it into context, and store new memories as conversations progress.

```
User input → Agent searches Hippocampus → injects memory → generates response
                                                    ↓
Session ends → Agent writes key information → retrievable next session
```

Three layers:

| Layer | Role | Example |
|-------|------|---------|
| **Short-Term** | Recent turns, sliding window | "User just asked about bubble tea preferences" |
| **Long-Term** | Vector-indexed persistent storage | "User is a VOCALOID fan, prefers osmanthus latte" |
| **Working** | Always-in-context rules & config | "Remind medication at 8am & 8pm daily" |

---

## Quick Start

```bash
# Install (Python 3.10+)
git clone https://github.com/VixieTodd/Hippocampus.git
cd Hippocampus
pip install -e .

# Write memories
hippo write "User prefers osmanthus-flavored milk tea"
hippo write "Rule: remind medication at 8am and 8pm" --layer working

# Search across all layers
hippo search "milk tea"

# Statistics
hippo stats

# Compress short-term → long-term
hippo compress --force

# Trace a single entry
hippo trace <entry_id>

# Export backup
hippo export --format json -o backup.json
```

Built-in TF-IDF vector search (CJK-aware) works offline with no downloads. Optional ChromaDB backend for stronger semantic matching.

---

## Python SDK

```python
from hippocampus import Hippocampus

hippo = Hippocampus("config.yml")

# On session start: load working memory into context
working = hippo.working.get_all()
context = "\n".join([e.content for e in working])

# During conversation: search relevant memories
results = hippo.search("user preferences", top_k=5)
for layer, entries in results.items():
    for e in entries:
        context += f"\n[memory] {e.content}"

# After conversation: save important information
hippo.write("User mentioned new project idea")
hippo.write("Scheduled follow-up for tomorrow afternoon")

# Compression triggers automatically when short-term exceeds threshold
```

---

## CLI Reference

```
hippo install                Guided setup wizard
hippo write <content>        Write a memory entry
hippo search <query>         Three-layer semantic search
hippo stats                  Memory statistics
hippo compress [--force]     Trigger compression (STM → LTM)
hippo trace <id>             Full trace of a single entry
hippo export [--format]      Backup all memories
```

---

## Configuration

```yaml
hippocampus:
  data_dir: "./data"
  short_term:
    window_size: 50
    compression_threshold: 40
  long_term:
    top_k: 5
    embedding_backend: "tfidf"  # or "chroma_default"
  working:
    file: "working.json"
```

---

## Dependencies

- Python 3.10+
- `click` (CLI framework)
- `pyyaml` (configuration)
- `chromadb` (optional — TF-IDF fallback works without it)

---

## License

MIT
