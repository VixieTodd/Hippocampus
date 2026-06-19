"""
Working Memory — static config, rules, and tools.
Always loaded, never flows into compression.
"""

import json
import os
from typing import List, Optional

from .base import MemoryEntry


class WorkingMemory:
    """Static working memory layer.

    Holds persistent rules, config, tool definitions — things
    that should always be in context but don't participate in
    the short→long compression lifecycle.
    """

    def __init__(self, data_dir: str, filename: str = "working.json"):
        self.file_path = os.path.join(data_dir, filename)
        self._entries: List[MemoryEntry] = []
        self._load()

    def _load(self):
        """Load working memory from JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [MemoryEntry.from_dict(d) for d in data]
        else:
            self._entries = []

    def _save(self):
        """Persist to JSON file atomically."""
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        tmp_path = self.file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.file_path)

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add a rule/entry to working memory."""
        entry.layer = "working"
        self._entries.append(entry)
        self._save()
        return entry

    def get_all(self) -> List[MemoryEntry]:
        """Return all working memory entries."""
        return list(self._entries)

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Simple text search over working memory."""
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            if query_lower in entry.content.lower():
                scored.append((1.0, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def find_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Find an entry by ID."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def count(self) -> int:
        """Number of entries."""
        return len(self._entries)

    def clear(self):
        """Clear all working memory entries."""
        self._entries = []
        self._save()

    def stats(self) -> dict:
        """Return statistics for this layer."""
        return {
            "layer": "working",
            "count": len(self._entries),
            "file": self.file_path,
        }
