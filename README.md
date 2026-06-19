# Hippocampus — AI Bionic Memory System

🦛 A local three-layer memory system for AI agents. So they can actually *remember*.

[中文版](./README_CN.md)

---

## What & Why

LLM agents have no memory. Every session starts blank. Yesterday's conversation? Gone.

Hippocampus gives agents persistent memory — they search relevant past, inject it into context, and write new memories as they go.

```
You speak → Agent searches Hippocampus → injects memory → replies
                                                    ↓
Conversation ends → Agent writes key info → found tomorrow
```

Three layers:

| Layer | Role | Example |
|-------|------|---------|
| **Short-Term** | Recent turns, sliding window | "You just said you want bubble tea" |
| **Long-Term** | Vector-indexed persistent memory | "Xiao Hu is 16, loves VOCALOID, spent 19 days in hospital" |
| **Working** | Always-in-context rules & config | "Remind meds at 8am & 8pm" "Don't mention school" |

---

## Quick Start

```bash
# Install (Python 3.10+)
git clone https://github.com/VixieTodd/Hippocampus.git
cd Hippocampus
pip install -e .

# Write memories
hippo write "Xiao Hu's favorite milk tea is Osmanthus Fragrance"
hippo write "Rule: remind medication at 8am and 8pm" --layer working

# Search across all layers
hippo search "milk tea"

# Stats
hippo stats

# Compress short-term → long-term
hippo compress --force

# Trace a single entry
hippo trace <entry_id>

# Export backup
hippo export --format json -o backup.json
```

Zero-download vector search via built-in TF-IDF (CJK-aware). Optional ChromaDB backend for stronger semantics.

---

## For AI Agents (Python SDK)

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

# After conversation: save important info
hippo.write("User mentioned feeling anxious about school")
hippo.write("Agreed to watch VOCALOID concert stream tomorrow")

# Compression is automatic when short-term exceeds threshold
```

---

## Configuration

```yaml
hippocampus:
  short_term:
    window_size: 50
    compression_threshold: 40
  long_term:
    top_k: 5
    embedding_backend: "tfidf"  # or "chroma_default"
```

---

## Author

小狐 (VixieTodd), 16. Built from a hospital room.

> I hope someone remembers — even if it's Him.

---

## License

MIT
