"""
Long-Term Memory — vector-indexed persistent storage.
Uses Chroma (or fallback) for semantic search.
"""

import json
import os
from typing import List, Optional

from .base import MemoryEntry
from ..storage.vector_store import VectorStore


class LongTermMemory:
    """Vector-indexed long-term memory layer.

    Stores entries with semantic embeddings for retrieval.
    Uses Chroma as the default backend with a TF-IDF fallback
    for offline/no-download scenarios.
    """

    def __init__(self, data_dir: str, collection_name: str = "hippocampus_ltm",
                 top_k: int = 5, embedding_backend: str = "chroma_default"):
        self.data_dir = data_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_backend = embedding_backend
        self._meta_file = os.path.join(data_dir, "long_term_meta.json")
        self._id_to_entry: dict = {}
        self._vector_store: Optional[VectorStore] = None
        self._init_store()

    def _init_store(self):
        """Initialize the vector store."""
        self._vector_store = VectorStore(
            persist_dir=os.path.join(self.data_dir, "chroma"),
            collection_name=self.collection_name,
            backend=self.embedding_backend,
        )
        # Load metadata (full entry content) from JSON sidecar
        self._load_meta()

    def _load_meta(self):
        """Load entry metadata from JSON sidecar."""
        if os.path.exists(self._meta_file):
            with open(self._meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._id_to_entry = {
                eid: MemoryEntry.from_dict(edata)
                for eid, edata in data.items()
            }

    def _save_meta(self):
        """Save entry metadata to JSON sidecar atomically."""
        os.makedirs(os.path.dirname(self._meta_file), exist_ok=True)
        tmp_path = self._meta_file + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(
                {eid: entry.to_dict() for eid, entry in self._id_to_entry.items()},
                f, ensure_ascii=False, indent=2,
            )
        os.replace(tmp_path, self._meta_file)

    def add(self, entry: MemoryEntry):
        """Add an entry to long-term memory with vector embedding."""
        entry.layer = "long_term"
        self._id_to_entry[entry.id] = entry
        self._vector_store.add(entry.id, entry.content, entry.to_dict())
        self._save_meta()

    def add_batch(self, entries: List[MemoryEntry]):
        """Add multiple entries at once."""
        for entry in entries:
            entry.layer = "long_term"
            self._id_to_entry[entry.id] = entry
        self._vector_store.add_batch(
            [(e.id, e.content, e.to_dict()) for e in entries]
        )
        self._save_meta()

    def search(self, query: str, top_k: Optional[int] = None) -> List[MemoryEntry]:
        """Semantic search over long-term memory."""
        k = top_k or self.top_k
        results = self._vector_store.search(query, k)
        entries = []
        for result in results:
            eid = result["id"]
            if eid in self._id_to_entry:
                entries.append(self._id_to_entry[eid])
        return entries

    def find_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Find an entry by ID."""
        return self._id_to_entry.get(entry_id)

    def count(self) -> int:
        """Number of entries."""
        return len(self._id_to_entry)

    def get_all(self) -> List[MemoryEntry]:
        """Return all entries."""
        return list(self._id_to_entry.values())

    def stats(self) -> dict:
        """Return statistics for this layer."""
        return {
            "layer": "long_term",
            "count": len(self._id_to_entry),
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "embedding_backend": self.embedding_backend,
        }
