"""
Short-Term Memory — sliding window of recent entries.
Stored as JSON file. Triggers compression when exceeding threshold.
"""

import json
import os
from typing import List, Optional

from .base import MemoryEntry
from ..utils import truncate


class ShortTermMemory:
    """In-memory + JSON-backed short-term memory layer.

    Holds the most recent N entries. When entry count exceeds
    compression_threshold, older entries should be compressed
    into long-term memory.
    """

    def __init__(self, data_dir: str, window_size: int = 50,
                 compression_threshold: int = 40):
        self.file_path = os.path.join(data_dir, "short_term.json")
        self.window_size = window_size
        self.compression_threshold = compression_threshold
        self._entries: List[MemoryEntry] = []
        self._load()

    def _load(self):
        """Load entries from JSON file."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._entries = [MemoryEntry.from_dict(d) for d in data]
        else:
            self._entries = []

    def _save(self):
        """Persist entries to JSON file atomically.
        
        Writes to temp file first, then atomically replaces the target.
        This prevents corruption from concurrent writes or crashes mid-write.
        """
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        tmp_path = self.file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self._entries], f,
                      ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.file_path)

    def add(self, entry: MemoryEntry) -> MemoryEntry:
        """Add an entry to short-term memory."""
        entry.layer = "short_term"
        self._entries.append(entry)
        self._save()
        return entry

    def get_all(self) -> List[MemoryEntry]:
        """Return all entries (newest last)."""
        return list(self._entries)

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """Return the most recent n entries."""
        return self._entries[-n:]

    def search(self, query: str, top_k: int = 5) -> List[MemoryEntry]:
        """Simple substring + keyword search over short-term memory.
        
        For MVP, uses case-insensitive substring matching.
        Returns entries sorted by relevance (match quality) then recency.
        """
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            content_lower = entry.content.lower()
            # Score: exact phrase match > all words present > partial match
            score = 0
            if query_lower in content_lower:
                score = 10 + (len(query_lower) / max(len(content_lower), 1)) * 10
            else:
                # Check individual words
                words = query_lower.split()
                matches = sum(1 for w in words if w in content_lower)
                if matches > 0:
                    score = (matches / len(words)) * 5
            if score > 0:
                scored.append((score, entry))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def count(self) -> int:
        """Number of entries."""
        return len(self._entries)

    def needs_compression(self) -> bool:
        """Check if compression should be triggered."""
        return self.count() > self.compression_threshold

    def pop_oldest(self, n: int = 1) -> List[MemoryEntry]:
        """Remove and return the oldest n entries."""
        removed = self._entries[:n]
        self._entries = self._entries[n:]
        self._save()
        return removed

    def remove_by_id(self, entry_id: str) -> bool:
        """Remove an entry by ID. Returns True if found and removed."""
        for i, entry in enumerate(self._entries):
            if entry.id == entry_id:
                self._entries.pop(i)
                self._save()
                return True
        return False

    def find_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Find an entry by ID."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry
        return None

    def stats(self) -> dict:
        """Return statistics for this layer."""
        return {
            "layer": "short_term",
            "count": len(self._entries),
            "window_size": self.window_size,
            "compression_threshold": self.compression_threshold,
            "needs_compression": self.needs_compression(),
            "file": self.file_path,
        }
