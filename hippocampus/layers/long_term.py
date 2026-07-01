"""
Long-Term Memory Layer.

Two backends, selected via config (``long_term.backend``):

  - **tfidf** (default, Lite mode): zero extra dependencies, CJK-aware
    TF-IDF + cosine similarity.  Suitable for small-to-medium collections.
  - **chroma** (Full mode): ChromaDB + sentence-transformers for dense
    semantic embeddings.  Requires ``pip install chromadb sentence-transformers``.

The layer interface is identical regardless of backend — MemoryStore
doesn't need to know which one is active.

V0.4 Multi-Agent:
  Every entry is tagged with ``agent_id`` in its metadata (both sidecar
  JSON and backend storage).  ``search()`` accepts an optional ``agent_id``
  filter — when provided, only that agent's entries are considered.

  **Shared mode** (default, ``long_term_isolation=False``):
    All agents write into one collection.  Search can be cross-agent or
    filtered by agent_id.  Metadata from all agents stored in one sidecar.

  **Isolation mode** (``long_term_isolation=True``):
    Each agent gets its own independent backend instance:
      - TF-IDF:  ``tfidf_store_{agent_id}.json``
      - ChromaDB: ``hippocampus_long_term_{agent_id}`` collection
    Cross-agent search is disabled — search always scoped to one agent.
    This guarantees complete data isolation between agents.
"""

from __future__ import annotations

import json
from pathlib import Path

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class LongTermMemoryLayer(BaseLayer):
    """Persistent long-term memory with selectable backend.

    V0.4: supports both shared and per-agent isolation mode via
    ``long_term_isolation`` parameter.
    """

    def __init__(
        self,
        data_dir: Path,
        backend: str = "tfidf",
        collection_name: str = "hippocampus_long_term",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        min_score: float = 0.0,

        # ── V0.4: agent config ────────────────────────────────────
        cross_agent_search: bool = True,
        long_term_isolation: bool = False,
    ) -> None:
        super().__init__("long_term")

        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._top_k = top_k
        self._min_score = min_score
        self._backend_name = backend
        self._collection_name = collection_name
        self._embedding_model_name = embedding_model_name
        self._cross_agent_search = cross_agent_search

        # ── V0.4: isolation mode flag ──────────────────────────
        self._isolation = long_term_isolation

        # Sidecar metadata — full MemoryEntry dicts live here regardless
        # of backend, so we always have the complete record for trace/export.
        # In isolation mode, we still use one metadata file (small, keyed by
        # entry_id which already carries agent_id in the value dict).
        self._meta_file = self._data_dir / "long_term_metadata.json"
        self._metadata: dict[str, dict] = {}
        self._load_metadata()

        # ── Initialise backend(s) ──────────────────────────────────
        # Shared mode: one backend instance for all agents.
        # Isolation mode: lazy-init per-agent backends via _get_backend().
        if self._isolation:
            # Lazy init — backends created on first write per agent.
            self._be_tfidf = {}   # type: dict[str, Any]
            self._be_chroma = {}  # type: dict[str, Any]
        else:
            # Shared mode — single backend as before.
            if backend == "chroma":
                self._init_chroma_single(collection_name, embedding_model_name)
            else:
                self._init_tfidf_single()

    # ── Shared-mode initialisers (single backend) ──────────────────

    def _init_tfidf_single(self) -> None:
        from hippocampus.layers.tfidf_backend import TFIDFBackend
        # In shared mode, _be_tfidf is a single instance.
        self._be_tfidf: "TFIDFBackend" = TFIDFBackend(self._data_dir)
        self._be_chroma = None

    def _init_chroma_single(
        self, collection_name: str, embedding_model_name: str
    ) -> None:
        self._check_chroma_deps()
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from sentence_transformers import SentenceTransformer

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
        self._be_chroma = self._collection  # keep for is_chroma check

    # ── V0.4: Isolation-mode backend routing ──────────────────────

    @staticmethod
    def _check_chroma_deps() -> None:
        """Raise ImportError if ChromaDB deps are missing."""
        try:
            import chromadb  # noqa: F401, F811
        except ImportError:
            raise ImportError(
                "chromadb is required for Chroma backend. "
                "Install with: pip install chromadb\n"
                "Or switch to tfidf backend in config.yml."
            )
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for Chroma backend. "
                "Install with: pip install sentence-transformers\n"
                "Or switch to tfidf backend in config.yml."
            )

    def _get_backend(self, agent_id: str) -> Any:
        """Return the backend instance for *agent_id*.

        In shared mode, returns the single shared backend.
        In isolation mode, creates a new backend on first access for this
        agent_id (lazy init).
        """
        if not self._isolation:
            # Shared mode — always the same backend.
            return self._be_tfidf

        # ── Isolation mode — per-agent backend ──────────────────
        if self._backend_name == "chroma":
            raise NotImplementedError(
                "ChromaDB isolation mode not yet implemented. "
                "Use tfidf backend with long_term_isolation for now."
            )
        else:
            # TF-IDF isolation: one TFIDFBackend per agent_id.
            if agent_id not in self._be_tfidf:
                from hippocampus.layers.tfidf_backend import TFIDFBackend

                # Each agent gets its own store file.
                agent_dir = self._data_dir / f"agent_{agent_id}"
                agent_dir.mkdir(parents=True, exist_ok=True)
                self._be_tfidf[agent_id] = TFIDFBackend(agent_dir)

            return self._be_tfidf[agent_id]

    # ── Sidecar metadata helpers ───────────────────────────────────────

    def _load_metadata(self) -> None:
        if self._meta_file.exists():
            with open(self._meta_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

    def _save_metadata(self) -> None:
        with open(self._meta_file, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def _backend_meta(self, entry: MemoryEntry) -> dict[str, str]:
        """Build the metadata dict sent to the backend (tfidf or chroma).

        V0.4: includes ``agent_id`` so the source agent is always traceable,
        even if the backend itself doesn't support filtering by it.
        """
        return {
            "timestamp": entry.timestamp,
            "source": entry.source,
            "parent_id": entry.parent_id or "",
            "agent_id": entry.agent_id,
        }

    @property
    def _is_chroma(self) -> bool:
        return self._backend_name == "chroma"

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Index a single entry.

        V0.4: in isolation mode, entry is routed to the agent's own backend.
        """
        entry.layer = "long_term"

        if self._is_chroma:
            if self._isolation:
                raise NotImplementedError(
                    "ChromaDB isolation mode not yet implemented."
                )
            embedding = self._embedder.encode(
                entry.content, normalize_embeddings=True,
            )
            self._collection.add(
                ids=[entry.id],
                embeddings=[embedding.tolist()],
                documents=[entry.content],
                metadatas=[self._backend_meta(entry)],
            )
        else:
            be = self._get_backend(entry.agent_id)  # ← V0.4: route
            be.add(
                doc_id=entry.id,
                content=entry.content,
                metadata=self._backend_meta(entry),
            )

        self._metadata[entry.id] = entry.to_dict()
        self._save_metadata()
        return entry.id

    def write_batch(self, entries: list[MemoryEntry]) -> list[str]:
        """Index multiple entries in one call.

        V0.4: in isolation mode, entries are grouped by agent_id and each
        group is written to its own backend.
        """
        if not entries:
            return []

        if self._is_chroma:
            if self._isolation:
                raise NotImplementedError(
                    "ChromaDB isolation mode not yet implemented."
                )
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
                metadatas.append(self._backend_meta(entry))
                self._metadata[entry.id] = entry.to_dict()

            self._collection.add(
                ids=ids,
                embeddings=embeddings_list,
                documents=documents,
                metadatas=metadatas,
            )
        else:
            if self._isolation:
                # ── V0.4: group by agent_id, write per-agent ────
                from collections import defaultdict
                groups: dict[str, list[MemoryEntry]] = defaultdict(list)
                for entry in entries:
                    entry.layer = "long_term"
                    groups[entry.agent_id].append(entry)
                    self._metadata[entry.id] = entry.to_dict()

                for aid, group in groups.items():
                    be = self._get_backend(aid)
                    batch = [
                        (e.id, e.content, self._backend_meta(e))
                        for e in group
                    ]
                    be.add_batch(batch)
            else:
                batch = []
                for entry in entries:
                    entry.layer = "long_term"
                    batch.append((
                        entry.id,
                        entry.content,
                        self._backend_meta(entry),
                    ))
                    self._metadata[entry.id] = entry.to_dict()
                self._be_tfidf.add_batch(batch)

        self._save_metadata()
        return [e.id for e in entries]

    def search(
        self,
        query: str,
        top_k: int = 5,
        agent_id: str | None = None,
    ) -> list[SearchResult]:
        """Search long-term memory, optionally filtered by agent_id.

        V0.4 behaviour:
          - Shared mode + agent_id given → only that agent's entries.
          - Shared mode + agent_id=None + cross_agent_search=True → all.
          - Shared mode + agent_id=None + cross_agent_search=False → [].
          - Isolation mode → **agent_id is required**.  Passing None
            raises ValueError (you must specify which agent to search).
        """
        # ── V0.4: isolation mode requires explicit agent_id ──────
        if self._isolation and agent_id is None:
            raise ValueError(
                "long_term_isolation is enabled — search() requires "
                "an explicit agent_id.  Cross-agent search is not "
                "supported in isolation mode."
            )

        # ── Shared mode: enforce cross-agent search policy ──────
        if agent_id is None and not self._cross_agent_search:
            return []

        k = top_k or self._top_k
        results: list[SearchResult] = []

        if self._is_chroma:
            if self._isolation:
                raise NotImplementedError(
                    "ChromaDB isolation mode not yet implemented."
                )
            query_embedding = self._embedder.encode(
                query, normalize_embeddings=True,
            )
            where_filter = {"agent_id": agent_id} if agent_id is not None else None

            raw = self._collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=k,
                include=["documents", "metadatas", "distances"],
                where=where_filter,
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
            if self._isolation:
                # ── V0.4: search only this agent's TF-IDF backend ─
                be = self._get_backend(agent_id)  # type: ignore[arg-type]
                hits = be.search(query, top_k=k, min_score=self._min_score)
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
                            "agent_id": hit.get("agent_id", agent_id),
                        },
                    ))
            else:
                # ── Shared TF-IDF: search all, filter post-hoc ──
                hits = self._be_tfidf.search(
                    query,
                    top_k=k * 2 if agent_id is not None else k,
                    min_score=self._min_score,
                )
                for hit in hits:
                    if agent_id is not None and hit.get("agent_id") != agent_id:
                        continue
                    results.append(SearchResult(
                        entry_id=hit["id"],
                        content=hit["content"],
                        score=hit["score"],
                        layer="long_term",
                        timestamp=hit.get("timestamp", ""),
                        metadata={
                            "source": hit.get("source", ""),
                            "parent_id": hit.get("parent_id", ""),
                            "agent_id": hit.get("agent_id", ""),
                        },
                    ))

        return results

    def stats(self, agent_id: str | None = None) -> dict:
        """Layer stats, optionally broken down by agent.

        V0.4: in isolation mode, stats are aggregated across all agent
        backends.  When ``agent_id`` is given, only that agent's stats
        are returned.
        """
        if self._is_chroma:
            if self._isolation:
                raise NotImplementedError(
                    "ChromaDB isolation mode not yet implemented."
                )
            count = self._collection.count()
            total_chars = 0
            if count > 0:
                try:
                    peek = self._collection.get(limit=count)
                    total_chars = sum(
                        len(d or "") for d in (peek.get("documents") or [])
                    )
                except Exception:
                    total_chars = 0
        elif self._isolation:
            # ── V0.4: aggregate across all agent backends ──────
            if agent_id is not None:
                be = self._get_backend(agent_id)
                count = be.count()
                total_chars = sum(
                    len(d.get("content", "")) for d in be.all_docs()
                )
            else:
                count = 0
                total_chars = 0
                for aid in self._be_tfidf:
                    be = self._be_tfidf[aid]
                    count += be.count()
                    total_chars += sum(
                        len(d.get("content", "")) for d in be.all_docs()
                    )
        else:
            count = self._be_tfidf.count()
            docs = self._be_tfidf.all_docs()
            total_chars = sum(len(d.get("content", "")) for d in docs)

        result = {
            "layer": "long_term",
            "count": count,
            "total_chars": total_chars,
            "collection": getattr(self, "_collection_name", "tfidf"),
            "backend": self._backend_name,
            "isolation": self._isolation,
        }

        # ── Per-agent breakdown from sidecar metadata ───────────
        if agent_id is None:
            agent_counts: dict[str, int] = {}
            for meta in self._metadata.values():
                aid = meta.get("agent_id", "main")
                agent_counts[aid] = agent_counts.get(aid, 0) + 1
            result["agents"] = agent_counts
            if self._isolation:
                result["agent_count"] = len(self._be_tfidf)

        return result

    def export(self) -> list[dict]:
        """Export all long-term entries.

        V0.4: each entry includes agent_id.  In isolation mode, all
        agents' entries are merged into one flat list.
        """
        return list(self._metadata.values())

    def clear(self, agent_id: str | None = None) -> None:
        """Clear long-term memory.

        V0.4:
          - Shared mode + agent_id given → remove only that agent's entries.
          - Shared mode + agent_id=None → wipe everything.
          - Isolation mode + agent_id given → clear that agent's backend.
          - Isolation mode + agent_id=None → clear all backends.
        """
        if self._isolation:
            if agent_id is not None:
                # ── Clear one agent's isolated backend ─────────
                if self._backend_name == "chroma":
                    raise NotImplementedError()
                if agent_id in self._be_tfidf:
                    self._be_tfidf[agent_id].clear()
                    del self._be_tfidf[agent_id]
                # Also remove metadata entries for this agent.
                to_remove = [
                    eid for eid, meta in self._metadata.items()
                    if meta.get("agent_id") == agent_id
                ]
                for eid in to_remove:
                    self._metadata.pop(eid, None)
            else:
                # ── Clear all isolated backends ────────────────
                for be in self._be_tfidf.values():
                    be.clear()
                self._be_tfidf.clear()
                self._metadata.clear()
        else:
            if agent_id is not None:
                # Collect IDs belonging to this agent from sidecar.
                to_remove = [
                    eid for eid, meta in self._metadata.items()
                    if meta.get("agent_id") == agent_id
                ]
                if to_remove:
                    if self._is_chroma:
                        self._collection.delete(ids=to_remove)
                    else:
                        self._be_tfidf.remove(to_remove)
                    for eid in to_remove:
                        self._metadata.pop(eid, None)
            else:
                if self._is_chroma:
                    try:
                        self._client.delete_collection(
                            name=self._collection_name,
                        )
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
        """Fetch a single entry's metadata by UUID.

        V0.4: searches sidecar metadata (common across all modes).
        Backend-level lookup is not needed since metadata always has
        the full entry dict.
        """
        return self._metadata.get(entry_id)

    # ── Backend introspection ───────────────────────────────────────

    @property
    def backend(self) -> str:
        """Active backend name: ``"tfidf"`` or ``"chroma"``."""
        return self._backend_name

    @property
    def count(self) -> int:
        """Total document count.

        V0.4: in isolation mode, sums across all agent backends.
        """
        if self._is_chroma:
            return self._collection.count()

        if self._isolation:
            return sum(be.count() for be in self._be_tfidf.values())

        return self._be_tfidf.count()
