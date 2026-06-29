"""
Short-Term Memory Layer.

Behaviour:
  - Sliding window of the *N* most recent entries.
  - Stored as a JSON file (``short_term_memory.json``).
  - When the window reaches the fill threshold, the MemoryStore caller
    triggers compression → migration to long-term.
  - Search is keyword-based (newest-first); no vector index needed since
    this layer holds only recent, small data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class ShortTermMemoryLayer(BaseLayer):
    """Recent entries with sliding window."""

    def __init__(self, data_dir: Path, window_size: int = 100) -> None:
        super().__init__("short_term")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "short_term_memory.json"
        self._window_size = window_size
        self._entries: list[MemoryEntry] = []
        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load entries from disk (on init)."""
        if self._file.exists():
            with open(self._file, "r", encoding="utf-8") as f:
                raw = json.load(f)
                self._entries = [MemoryEntry.from_dict(item) for item in raw]

    def _save(self) -> None:
        """Persist entries to disk (on every write or removal)."""
        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(
                [e.to_dict() for e in self._entries],
                f, ensure_ascii=False, indent=2,
            )

    def _enforce_window(self) -> int:
        """Drop the oldest entries if we exceed window_size.

        Returns the number of entries trimmed.
        """
        if len(self._entries) <= self._window_size:
            return 0
        trimmed = len(self._entries) - self._window_size
        self._entries = self._entries[trimmed:]
        return trimmed

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Append an entry to the window.  Trims oldest if full."""
        entry.layer = "short_term"
        self._entries.append(entry)
        self._enforce_window()
        self._save()
        return entry.id

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Case-insensitive keyword search (newest first).

        Scores are based on frequency of query term occurrences in content.
        This is deliberately approximate — short-term search is more about
        recency than semantic relevance.
        """
        query_lower = query.lower()
        # Split into individual keywords for multi-word queries.
        keywords = [w for w in query_lower.split() if w]
        if not keywords:
            return []

        results: list[SearchResult] = []
        for entry in reversed(self._entries):  # newest first
            content_lower = entry.content.lower()
            # Count how many keywords appear in the content.
            hits = sum(1 for kw in keywords if kw in content_lower)
            if hits == 0:
                continue
            score = min(1.0, hits / len(keywords) * 0.8 + 0.2)
            results.append(SearchResult(
                entry_id=entry.id,
                content=entry.content,
                score=round(score, 4),
                layer="short_term",
                timestamp=entry.timestamp,
                metadata=entry.metadata,
            ))

        return results[:top_k]

    def stats(self) -> dict:
        total_chars = sum(len(e.content) for e in self._entries)
        fill_ratio = len(self._entries) / max(self._window_size, 1)
        return {
            "layer": "short_term",
            "count": len(self._entries),
            "window_size": self._window_size,
            "fill_ratio": round(fill_ratio, 2),
            "total_chars": total_chars,
            "needs_compression": fill_ratio >= 0.8,
        }

    def export(self) -> list[dict]:
        return [e.to_dict() for e in self._entries]

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    # ── Batch operations (used by Compressor) ─────────────────────────

    def get_entries_batch(self, count: int) -> list[MemoryEntry]:
        """Return the oldest *count* entries (for compression migration)."""
        return self._entries[:count]

    def remove_batch(self, entry_ids: list[str]) -> None:
        """Remove entries by ID (called after successful migration)."""
        ids_set = set(entry_ids)
        self._entries = [e for e in self._entries if e.id not in ids_set]
        self._save()

    def all_entries(self) -> list[MemoryEntry]:
        return list(self._entries)

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        for e in self._entries:
            if e.id == entry_id:
                return e
        return None
