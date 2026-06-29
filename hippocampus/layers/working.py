"""
Working Memory Layer ("Engineering Memory").

This is the *static* layer — it holds config, tools, rules, and other
things that should always be "in context".  Unlike short/long-term:
  - It never flows or compresses.
  - Entries are manually written and removed.
  - There is no vector index — just a keyword fallback for search.

Storage: single JSON file (``working_memory.json``) inside the data dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class WorkingMemoryLayer(BaseLayer):
    """Static working memory — config, tools, rules (no sliding window)."""

    def __init__(self, data_dir: Path) -> None:
        super().__init__("working")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "working_memory.json"
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Read entries from ``working_memory.json`` on startup."""
        if self._file.exists():
            with open(self._file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                for item in raw:
                    entry = MemoryEntry.from_dict(item)
                    self._entries[entry.id] = entry

    def _save(self) -> None:
        """Persist all entries to ``working_memory.json``."""
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self._entries.values()],
                f, ensure_ascii=False, indent=2,
            )

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Store (or overwrite) an entry in working memory."""
        entry.layer = "working"
        self._entries[entry.id] = entry
        self._save()
        return entry.id

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Simple case-insensitive substring search over entry content."""
        query_lower = query.lower()
        results = []
        for entry in self._entries.values():
            if query_lower in entry.content.lower():
                results.append(SearchResult(
                    entry_id=entry.id,
                    content=entry.content,
                    score=1.0,
                    layer="working",
                    timestamp=entry.timestamp,
                    metadata=entry.metadata,
                ))
        return results[:top_k]

    def stats(self) -> dict:
        total_chars = sum(len(e.content) for e in self._entries.values())
        return {
            "layer": "working",
            "count": len(self._entries),
            "total_chars": total_chars,
            "file": str(self._file),
        }

    def export(self) -> list[dict]:
        """Serialise all entries as a list of dicts."""
        return [e.to_dict() for e in self._entries.values()]

    def clear(self) -> None:
        """Remove every entry and write empty JSON."""
        self._entries.clear()
        self._save()

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Fetch a single entry by UUID (returns None if missing)."""
        return self._entries.get(entry_id)
