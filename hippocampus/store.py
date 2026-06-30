"""
MemoryStore — unified orchestrator for all three layers.

This is the single entry point for every memory operation.  It:

1. Initialises the three layers (working, short-term, long-term).
2. Writes entries to the correct layer and triggers auto-compression when
   short-term memory fills up.
3. Searches across all layers, merging results sorted by relevance score.
4. Exports, traces, and compresses on demand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from hippocampus.config import Config
from hippocampus.memory import MemoryEntry
from hippocampus.tracer import Tracer
from hippocampus.layers import SearchResult
from hippocampus.layers.working import WorkingMemoryLayer
from hippocampus.layers.short_term import ShortTermMemoryLayer
from hippocampus.layers.long_term import LongTermMemoryLayer
from hippocampus.compressor import Compressor


class MemoryStore:
    """Unified memory store managing all three layers."""

    def __init__(self, config: Config) -> None:
        self._config = config
        data_dir = config.data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

        # ── Initialise each layer ──────────────────────────────────────
        self.working = WorkingMemoryLayer(data_dir)
        self.short_term = ShortTermMemoryLayer(
            data_dir,
            window_size=config.short_term.window_size,
        )
        self.long_term = LongTermMemoryLayer(
            data_dir,
            backend=config.long_term.backend,
            collection_name=config.long_term.collection_name,
            embedding_model_name=config.long_term.embedding_model,
            top_k=config.long_term.top_k,
            min_score=config.long_term.min_score,
        )

        # ── Compression bridge ─────────────────────────────────────────
        self.compressor = Compressor(
            self.short_term,
            self.long_term,
            strategy=config.compression.strategy,
            batch_size=config.compression.batch_size,
            max_chars=config.compression.max_chars,
        )

        # ── Traceability ───────────────────────────────────────────────
        trace_path: Path | None = None
        if config.trace.enabled:
            trace_path = data_dir / config.trace.log_file
        self.tracer = Tracer(trace_path)

    # ── Write ─────────────────────────────────────────────────────────

    def write(
        self,
        content: str,
        source: str = "user",
        layer: str = "short_term",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Write a new memory entry.

        Args:
            content:  The text to store.
            source:   Who said it — ``"user"``, ``"agent"``, or ``"system"``.
            layer:    Target layer — ``"short_term"`` (default),
                      ``"working"``, or ``"long_term"``.
            metadata: Optional extra key-value pairs attached to the entry.

        Returns:
            The newly-created entry's UUID.
        """
        entry = MemoryEntry(
            content=content,
            layer=layer,
            source=source,
            metadata=metadata or {},
        )

        if layer == "working":
            eid = self.working.write(entry)
        elif layer == "long_term":
            eid = self.long_term.write(entry)
        else:
            eid = self.short_term.write(entry)
            # Auto-trigger compression when short-term fills up.
            st = self.short_term.stats()
            if st.get("needs_compression"):
                self.compressor.compress(force=False)

        self.tracer.log("write", entry_id=eid, layer=layer, detail={
            "content_len": len(content),
            "source": source,
        })

        return eid

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        layers: list[str] | None = None,
    ) -> list[SearchResult]:
        """Search across one or more layers.

        Args:
            query:  Free-text search string.
            top_k:  Max results *per layer* (final list may be larger).
            layers: Which layers to search; defaults to all three.

        Returns:
            Combined results sorted by score descending.
            *Note*: scores across layers are not directly comparable
            (keyword match vs. semantic similarity).  Use the ``layer``
            field to distinguish.
        """
        if layers is None:
            layers = ["short_term", "long_term", "working"]

        all_results: list[SearchResult] = []
        for name in layers:
            if name == "short_term":
                all_results.extend(self.short_term.search(query, top_k))
            elif name == "long_term":
                all_results.extend(self.long_term.search(query, top_k))
            elif name == "working":
                all_results.extend(self.working.search(query, top_k))

        all_results.sort(key=lambda r: r.score, reverse=True)

        self.tracer.log("search", detail={
            "query": query,
            "top_k": top_k,
            "results_count": len(all_results),
        })

        return all_results

    # ── Compress ───────────────────────────────────────────────────────

    def compress(self, force: bool = False) -> dict:
        """Migrate short-term entries → long-term.

        Args:
            force:  Skip the fill-ratio check and compress everything.

        Returns:
            Stats dict with compressed/migrated counts.
        """
        result = self.compressor.compress(force=force)
        self.tracer.log("compress", detail=result)
        return result

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Layer-wise statistics: counts, sizes, config values."""
        return {
            "short_term": self.short_term.stats(),
            "long_term": self.long_term.stats(),
            "working": self.working.stats(),
        }

    # ── Trace ──────────────────────────────────────────────────────────

    def trace(self, entry_id: str) -> dict:
        """Full audit trail for one entry across all layers.

        Searches every layer for the entry body, then pulls its trace
        records from the operation log.  Useful for debugging where a
        piece of memory ended up and how it got there.
        """
        entry_data: dict[str, Any] | None = None
        layer_name: str | None = None

        for name, layer in (
            ("short_term", self.short_term),
            ("long_term", self.long_term),
            ("working", self.working),
        ):
            # Long-term stores metadata dicts; short/working store MemoryEntry.
            # Both support get_entry() but return type differs — we normalise here.
            data: Any = layer.get_entry(entry_id)
            if data is not None:
                entry_data = (
                    data if isinstance(data, dict)
                    else data.to_dict()
                )
                layer_name = name
                break

        traces = self.tracer.read(entry_id)

        return {
            "entry": entry_data,
            "layer": layer_name,
            "traces": traces,
            "trace_count": len(traces),
        }

    # ── Export ─────────────────────────────────────────────────────────

    def export(self, fmt: str = "json") -> dict:
        """Dump all memories as a single JSON-serialisable structure."""
        from hippocampus import __version__
        return {
            "version": __version__,
            "format": fmt,
            "short_term": self.short_term.export(),
            "long_term": self.long_term.export(),
            "working": self.working.export(),
        }
