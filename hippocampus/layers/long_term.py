"""
Long-Term Memory Layer.

Two backends, selected via config (``long_term.backend``):

  - **tfidf** (default, Lite mode): zero extra dependencies, CJK-aware
    TF-IDF + cosine similarity.  Suitable for small-to-medium collections.
  - **chroma** (Full mode): ChromaDB + sentence-transformers for dense
    semantic embeddings.  Requires ``pip install chromadb sentence-transformers``.

The layer interface is identical regardless of backend — MemoryStore
doesn't need to know which one is active.
"""

from __future__ import annotations

import json
from pathlib import Path

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class LongTermMemoryLayer(BaseLayer):
    """Persistent long-term memory with selectable backend."""

    def __init__(
        self,
        data_dir: Path,
        backend: str = "tfidf",
        collection_name: str = "hippocampus_long_term",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> None:
        super().__init__("long_term")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        self._min_score = min_score
        self._backend_name = backend

        # Sidecar metadata — full MemoryEntry dicts live here regardless
        # of backend, so we always have the complete record for trace/export.
        self._meta_file = self._data_dir / "long_term_metadata.json"
        self._metadata: dict[str, dict] = {}
        self._load_metadata()

        # ── Select backend ──────────────────────────────────────────
        if backend == "chroma":
            self._init_chroma(collection_name, embedding_model_name)
        else:
            self._init_tfidf()

    # ── TF-IDF backend (Lite, default) ──────────────────────────────

    def _init_tfidf(self) -> None:
        """Initialise the zero-dependency TF-IDF backend.

        This is the default path — no ChromaDB or sentence-transformers needed.
        The backend stores documents inline in ``tfidf_store.json``.
        """
        from hippocampus.layers.tfidf_backend import TFIDFBackend

        self._be_tfidf: "TFIDFBackend" = TFIDFBackend(self._data_dir)
        self._be_chroma = None

    # ── ChromaDB backend (Full) ─────────────────────────────────────

    def _init_chroma(
        self, collection_name: str, embedding_model_name: str
    ) -> None:
        try:
            import chromadb  # noqa: F811
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            raise ImportError(
                "chromadb is required for Chroma backend. "
                "Install with: pip install chromadb\n"
                "Or switch to tfidf backend in config.yml."
            )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for Chroma backend. "
                "Install with: pip install sentence-transformers\n"
                "Or switch to tfidf backend in config.yml."
            )

        self._embedder = SentenceTransformer(embedding_model_name)

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

        self._collection_name = collection_name
        self._be_tfidf = None

    # ── Sidecar metadata helpers ───────────────────────────────────────

    def _load_metadata(self) -> None:
        if self._meta_file.exists():
            with open(self._meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _save_metadata(self) -> None:
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    @property
    def _is_chroma(self) -> bool:
        return self._backend_name == "chroma"

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Index a single entry."""
        entry.layer = "long_term"

        if self._is_chroma:
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
        else:
            self._be_tfidf.add(
                doc_id=entry.id,
                content=entry.content,
                metadata={
                    "timestamp": entry.timestamp,
                    "source": entry.source,
                    "parent_id": entry.parent_id or "",
                },
            )

        self._metadata[entry.id] = entry.to_dict()
        self._save_metadata()
        return entry.id

    def write_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Index multiple entries in one call."""
        if not entries:
            return []

        if self._is_chroma:
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
        else:
            batch = []
            for entry in entries:
                entry.layer = "long_term"
                batch.append((
                    entry.id,
                    entry.content,
                    {
                        "timestamp": entry.timestamp,
                        "source": entry.source,
                        "parent_id": entry.parent_id or "",
                    },
                ))
                self._metadata[entry.id] = entry.to_dict()
            self._be_tfidf.add_batch(batch)

        self._save_metadata()
        return [e.id for e in entries]

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search long-term memory."""
        k = top_k or self._top_k
        results: list[SearchResult] = []

        if self._is_chroma:
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

            for i, entry_id in enumerate(ids):
                distance = distances[i] if i < len(distances) else 1.0
                score = max(0.0, min(1.0, 1.0 - distance))
                if score < self._min_score:
                    continue
                content = documents[i] if i < len(documents) else ""
                meta = metadatas[i] if i < len(metadatas) else {}
                results.append(SearchResult(
                    entry_id=entry_id,
                    content=content,
                    score=round(score, 4),
                    layer="long_term",
                    timestamp=meta.get("timestamp", ""),
                    metadata=meta,
                ))
        else:
            hits = self._be_tfidf.search(query, top_k=k, min_score=self._min_score)
            for hit in hits:
                results.append(SearchResult(
                    entry_id=hit["id"],
                    content=hit["content"],
                    score=hit["score"],
                    layer="long_term",
                    timestamp=hit.get("timestamp", ""),
                    metadata={
                        "source": hit.get("source", ""),
                        "parent_id": hit.get("parent_id", ""),
                    },
                ))

        return results

    def stats(self) -> dict:
        if self._is_chroma:
            count = self._collection.count()
            total_chars = 0
            if count > 0:
                try:
                    peek = self._collection.get(limit=count)
                    total_chars = sum(len(d or "") for d in (peek.get("documents") or []))
                except Exception:
                    total_chars = 0
        else:
            count = self._be_tfidf.count()
            docs = self._be_tfidf.all_docs()
            total_chars = sum(len(d.get("content", "")) for d in docs)

        return {
            "layer": "long_term",
            "count": count,
            "total_chars": total_chars,
            "collection": getattr(self, "_collection_name", "tfidf"),
            "backend": self._backend_name,
        }

    def export(self) -> list[dict]:
        return list(self._metadata.values())

    def clear(self) -> None:
        if self._is_chroma:
            try:
                self._client.delete_collection(name=self._collection_name)
            except Exception:
                pass
            self._collection = self._client.create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._be_tfidf.clear()
        self._metadata.clear()
        self._save_metadata()

    def get_entry(self, entry_id: str) -> dict | None:
        return self._metadata.get(entry_id)

    # ── Backend introspection ───────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active backend name: ``"tfidf"`` or ``"chroma"``."""
        return self._backend_name

    @property
    def count(self) -> int:
        if self._is_chroma:
            return self._collection.count()
        return self._be_tfidf.count()
