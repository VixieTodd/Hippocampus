"""
MemoryStore — unified orchestrator for all three layers.

This is the single entry point for every memory operation.  It:

1. Initialises the three layers (working, short-term, long-term).
2. Writes entries to the correct layer and triggers auto-compression when
   short-term memory fills up.
3. Searches across all layers, merging results sorted by relevance score.
4. Exports, traces, and compresses on demand.

V0.4 Multi-Agent:
  Every public method accepts an ``agent_id`` parameter.  When isolation
  is enabled (config), short-term memory is automatically partitioned.
  Long-term search can be cross-agent or scoped.  The default agent_id
  comes from config.agent.default_agent_id (usually "main").
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
    """Unified memory store managing all three layers.

    V0.4: all write/search/compress methods accept agent_id.
    """

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

            # ── V0.4: multi-agent config ─────────────────────────
            cross_agent_search=config.agent.cross_agent_search,
            long_term_isolation=config.agent.long_term_isolation,
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

    # ── V0.4: helpers ──────────────────────────────────────────────

    @property
    def _default_agent_id(self) -> str:
        """The default agent_id when none is explicitly given.

        Comes from config (agent.default_agent_id), typically "main".
        """
        return self._config.agent.default_agent_id

    def _resolve_agent_id(self, agent_id: str | None) -> str:
        """Resolve an optional agent_id to a concrete value.

        If agent_id is None, uses the config default ("main").
        """
        return agent_id if agent_id is not None else self._default_agent_id

    # ── Write ─────────────────────────────────────────────────────────

    def write(
        self,
        content: str,
        source: str = "user",
        layer: str = "short_term",
        metadata: dict[str, Any] | None = None,

        # ── V0.4: agent_id ──────────────────────────────────────────
        # When None, uses config.agent.default_agent_id (usually "main").
        # Explicit agent_id is required for sub-agent isolation.
        agent_id: str | None = None,
    ) -> str:
        """Write a new memory entry.

        Args:
            content:  The text to store.
            source:   Who said it — ``"user"``, ``"agent"``, or ``"system"``.
            layer:    Target layer — ``"short_term"`` (default),
                      ``"working"``, or ``"long_term"``.
            metadata: Optional extra key-value pairs attached to the entry.
            agent_id: Which agent owns this entry.  Defaults to config
                      default (``"main"``).  For sub-agents, pass the
                      agent's ID explicitly.

        Returns:
            The newly-created entry's UUID.

        V0.4: agent_id is stored on the entry and used for partitioning
        in short-term and filtering in long-term.
        """
        resolved_agent = self._resolve_agent_id(agent_id)

        entry = MemoryEntry(
            content=content,
            layer=layer,
            source=source,
            metadata=metadata or {},
            agent_id=resolved_agent,
        )

        if layer == "working":
            eid = self.working.write(entry)
        elif layer == "long_term":
            eid = self.long_term.write(entry)
        else:
            eid = self.short_term.write(entry)

            # ── V0.4: per-agent compression trigger ───────────────
            # Check if THIS agent's short-term window needs compression,
            # not the global pool.
            st = self.short_term.stats(agent_id=resolved_agent)
            if st.get("needs_compression"):
                self.compressor.compress(
                    force=False, agent_id=resolved_agent,
                )

        self.tracer.log("write", entry_id=eid, layer=layer, detail={
            "content_len": len(content),
            "source": source,
            "agent_id": resolved_agent,
        })

        return eid

    # ── Search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        layers: list[str] | None = None,

        # ── V0.4: agent scope ─────────────────────────────────────
        # When provided, scopes short-term and long-term search to this
        # agent (working memory always includes shared entries).
        # When None (default), uses cross-agent search (all agents).
        agent_id: str | None = None,
    ) -> list[SearchResult]:
        """Search across one or more layers.

        Args:
            query:    Free-text search string.
            top_k:    Max results *per layer* (final list may be larger).
            layers:   Which layers to search; defaults to all three.
            agent_id: Scope to one agent (applies to short-term and
                      long-term).  Working memory always includes shared
                      entries.  None = cross-agent.

        Returns:
            Combined results sorted by score descending.
            *Note*: scores across layers are not directly comparable
            (keyword match vs. semantic similarity).  Use the ``layer``
            field to distinguish.

        V0.4: short-term and long-term search respect agent_id filtering.
        Working memory returns agent-specific + shared entries.
        """
        if layers is None:
            layers = ["short_term", "long_term", "working"]

        all_results: list[SearchResult] = []
        for name in layers:
            if name == "short_term":
                all_results.extend(
                    self.short_term.search(query, top_k, agent_id=agent_id)
                )
            elif name == "long_term":
                all_results.extend(
                    self.long_term.search(query, top_k, agent_id=agent_id)
                )
            elif name == "working":
                # ── V0.4: working memory always includes shared ─
                # Pass the caller's agent_id so they see their own
                # entries + shared entries, even when agent_id is None.
                all_results.extend(
                    self.working.search(
                        query, top_k,
                        agent_id=agent_id,
                        include_shared=True,
                    )
                )

        all_results.sort(key=lambda r: r.score, reverse=True)

        self.tracer.log("search", detail={
            "query": query,
            "top_k": top_k,
            "results_count": len(all_results),
            "agent_id": agent_id,
        })

        return all_results

    # ── Compress ───────────────────────────────────────────────────────

    def compress(
        self,
        force: bool = False,

        # ── V0.4: per-agent compression ───────────────────────────
        # When given, only compresses that agent's short-term entries.
        # When None, compresses ALL agents' short-term entries in sequence
        # (useful for maintenance / forced cleanup).
        agent_id: str | None = None,
    ) -> dict:
        """Migrate short-term entries → long-term.

        V0.4: compression is per-agent by default.  When agent_id is None
        and force=True, all agents are compressed in sequence.

        Args:
            force:    Skip the fill-ratio check and compress everything.
            agent_id: Which agent to compress (None = all).

        Returns:
            Stats dict with compressed/migrated counts.
        """
        if agent_id is not None:
            # ── Single-agent compression ──────────────────────────
            result = self.compressor.compress(
                force=force, agent_id=agent_id,
            )
            self.tracer.log("compress", detail=result)
            return result

        # ── V0.4: compress all agents ─────────────────────────────
        # When no specific agent_id is given, iterate over every agent
        # that has entries in short-term memory.
        all_results: dict[str, dict] = {}
        total_compressed = 0
        total_migrated = 0

        for aid in self.short_term.list_agents():
            result = self.compressor.compress(
                force=force, agent_id=aid,
            )
            all_results[aid] = result
            total_compressed += result.get("compressed", 0)
            total_migrated += result.get("migrated", 0)

        summary = {
            "agents": all_results,
            "total_compressed": total_compressed,
            "total_migrated": total_migrated,
        }
        self.tracer.log("compress", detail=summary)
        return summary

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self, agent_id: str | None = None) -> dict:
        """Layer-wise statistics, optionally scoped to one agent.

        V0.4: when ``agent_id`` is given, each layer's stats are scoped.
        When None, returns per-agent breakdowns from each layer.
        """
        return {
            "short_term": self.short_term.stats(agent_id=agent_id),
            "long_term": self.long_term.stats(agent_id=agent_id),
            "working": self.working.stats(agent_id=agent_id),
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
        """Dump all memories as a single JSON-serialisable structure.

        V0.4: exported entries include agent_id so exports can be
        inspected per-agent.
        """
        from hippocampus import __version__
        return {
            "version": __version__,
            "format": fmt,
            "short_term": self.short_term.export(),
            "long_term": self.long_term.export(),
            "working": self.working.export(),
        }
