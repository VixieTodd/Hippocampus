"""
Long-Term Memory Layer.

This is the *persistent*, *vector-indexed* memory:
  - Uses ChromaDB (local, persistent) as the vector store.
  - Sentence-transformers (``all-MiniLM-L6-v2``) for embeddings.
  - Full metadata is kept in a sidecar JSON file (Chroma's metadata fields
    are limited — flat strings only — so we store the richer dict there).

The layer is meant for memories that have "settled" — compressed or
explicit long-term writes.  Search is semantic (cosine similarity).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None  # type: ignore[assignment]

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class LongTermMemoryLayer(BaseLayer):
    """Persistent vector-indexed long-term memory (ChromaDB backend)."""

    def __init__(
        self,
        data_dir: Path,
        collection_name: str = "hippocampus_long_term",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        super().__init__("long_term")

        # ══ Dependency guards ════════════════════════════════════════
        if chromadb is None:
            raise ImportError(
                "chromadb is required for LongTermMemoryLayer. "
                "Install it with: pip install chromadb"
            )
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install it with: pip install sentence-transformers"
            )

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        self._min_score = min_score
        self._collection_name = collection_name

        # Sidecar metadata file — Chroma's own metadata is flat
        # (string-only values), so we keep the full MemoryEntry dict here.
        self._meta_file = self._data_dir / "long_term_metadata.json"
        self._metadata: dict[str, dict] = {}
        self._load_metadata()

        # ── ChromaDB ───────────────────────────────────────────────
        chroma_dir = str(self._data_dir / "chroma")
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # ── Embedding model ─────────────────────────────────────────
        self._embedder = SentenceTransformer(embedding_model_name)

    # ── Sidecar metadata helpers ───────────────────────────────────────

    def _load_metadata(self) -> None:
        if self._meta_file.exists():
            with open(self._meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _save_metadata(self) -> None:
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Index a single entry into ChromaDB + metadata store."""
        entry.layer = "long_term"

        embedding = self._embedder.encode(
            entry.content, normalize_embeddings=True,
        )

        self._collection.add(
            ids=[entry.id],
            embeddings=[embedding.tolist()],
            documents=[entry.content],
            metadatas=[{
                "timestamp": entry.timestamp,
                "source": entry.source,
                "parent_id": entry.parent_id or "",
            }],
        )

        self._metadata[entry.id] = entry.to_dict()
        self._save_metadata()

        return entry.id

    def write_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Index multiple entries in a single ChromaDB call (faster)."""
        if not entries:
            return []

        ids: list[str] = []
        embeddings_list: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for entry in entries:
            entry.layer = "long_term"
            ids.append(entry.id)
            emb = self._embedder.encode(
                entry.content, normalize_embeddings=True,
            )
            embeddings_list.append(emb.tolist())
            documents.append(entry.content)
            metadatas.append({
                "timestamp": entry.timestamp,
                "source": entry.source,
                "parent_id": entry.parent_id or "",
            })
            self._metadata[entry.id] = entry.to_dict()

        self._collection.add(
            ids=ids,
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
        )
        self._save_metadata()

        return ids

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic (cosine-similarity) search over long-term memory."""
        k = top_k or self._top_k

        query_embedding = self._embedder.encode(
            query, normalize_embeddings=True,
        )

        raw = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[SearchResult] = []
        for i, entry_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, min(1.0, 1.0 - distance))  # cos-dist → similarity

            if score < self._min_score:
                continue

            content = documents[i] if i < len(documents) else ""
            meta = metadatas[i] if i < len(metadatas) else {}
            timestamp = meta.get("timestamp", "")

            results.append(SearchResult(
                entry_id=entry_id,
                content=content,
                score=round(score, 4),
                layer="long_term",
                timestamp=timestamp,
                metadata=meta,
            ))

        return results

    def stats(self) -> dict:
        count = self._collection.count()
        # Use Chroma's stored documents (not sidecar metadata) for char count.
        total_chars = 0
        if count > 0:
            try:
                peek = self._collection.get(limit=count)
                total_chars = sum(len(d or "") for d in (peek.get("documents") or []))
            except Exception:
                total_chars = 0

        return {
            "layer": "long_term",
            "count": count,
            "total_chars": total_chars,
            "collection": self._collection_name,
            "backend": "chromadb",
            "embedding_model": str(self._embedder),
        }

    def export(self) -> list[dict]:
        return list(self._metadata.values())

    def clear(self) -> None:
        try:
            self._client.delete_collection(name=self._collection_name)
        except Exception:
            pass
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._metadata.clear()
        self._save_metadata()

    def get_entry(self, entry_id: str) -> dict | None:
        return self._metadata.get(entry_id)

    @property
    def count(self) -> int:
        return self._collection.count()
