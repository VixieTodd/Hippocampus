"""
Memory compression — short-term → long-term migration.

The flow:
  Short-Term Memory (many small entries, recent window)
     ↓  (compression / migration)
  Long-Term Memory  (fewer, denser entries, vector-indexed)

Current strategies:
  - simple_concat:  Groups entries and concatenates them block-by-block.
    Reduces entry count while preserving raw content.
  - (future) llm_summary:  Uses an LLM to produce a concise summary.

The Compressor is triggered automatically when short-term fill_ratio
exceeds the configured threshold, or manually via ``--force``.

V0.4 Multi-Agent:
  Compression is always per-agent.  Each agent's short-term window is
  compressed independently, so Agent A's compression never touches
  Agent B's entries.  Compressed entries inherit the source agent's
  agent_id so they remain filterable in long-term.
"""

from __future__ import annotations

from hippocampus.memory import MemoryEntry
from hippocampus.layers.short_term import ShortTermMemoryLayer
from hippocampus.layers.long_term import LongTermMemoryLayer


class Compressor:
    """Migrates short-term entries into long-term memory.

    V0.4: ``compress()`` requires an ``agent_id``.  It pulls entries
    from that agent's short-term partition, compresses them, and writes
    them to long-term with the same agent_id.
    """

    def __init__(
        self,
        short_term: ShortTermMemoryLayer,
        long_term: LongTermMemoryLayer,
        strategy: str = "simple_concat",
        batch_size: int = 20,
        max_chars: int = 2000,
        threshold: float = 0.8,
    ) -> None:
        self._st = short_term
        self._lt = long_term
        self._strategy = strategy
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._threshold = threshold

    def compress(
        self,
        force: bool = False,

        # ── V0.4: agent_id is now required ─────────────────────────
        # Compression always targets a single agent's short-term entries.
        # This prevents cross-agent memory mixing during compression.
        agent_id: str = "main",
    ) -> dict:
        """Migrate a batch of short-term entries into long-term.

        V0.4: targets a single agent's partition.  Each agent's window
        is compressed independently.

        Args:
            force:     Skip the threshold check; compress regardless.
            agent_id:  Which agent's short-term memory to compress.

        Returns:
            Stats dict::

                {
                    "agent_id":   "...",
                    "compressed":  N,       # entries read from ST
                    "migrated":    M,       # entries written to LT (≤ N)
                    "strategy":    "...",
                    "short_term_remaining": ...,
                    "long_term_total": ...,
                }
        """
        # ── V0.4: check this agent's stats, not global ────────────
        st = self._st.stats(agent_id=agent_id)
        fill_ratio = st.get("fill_ratio", 0.0)
        entry_count = st.get("count", 0)

        # Bail early if there's nothing to do.
        if entry_count == 0:
            return {
                "agent_id": agent_id,
                "compressed": 0,
                "migrated": 0,
                "reason": "no entries",
            }

        if not force and fill_ratio < self._threshold:
            return {
                "agent_id": agent_id,
                "compressed": 0,
                "migrated": 0,
                "reason": (
                    f"fill_ratio {fill_ratio:.2f} "
                    f"< threshold {self._threshold}"
                ),
                "short_term_remaining": entry_count,
            }

        # ── V0.4: pull from this agent's partition ────────────────
        batch_count = min(self._batch_size, entry_count)
        entries = self._st.get_entries_batch(batch_count, agent_id=agent_id)

        # Compress.
        if self._strategy == "simple_concat":
            compressed = self._concat_block(entries, agent_id=agent_id)
        else:
            # Unknown strategy → fall back to simple concat.
            compressed = self._concat_block(entries, agent_id=agent_id)

        # ── V0.4: migrate to long-term with agent_id preserved ───
        if compressed:
            self._lt.write_batch(compressed)
            ids_removed = [e.id for e in entries]
            self._st.remove_batch(ids_removed, agent_id=agent_id)

        # ── V0.4: get updated count for THIS agent post-compression ─
        remaining = self._st.stats(agent_id=agent_id)["count"]

        return {
            "agent_id": agent_id,
            "compressed": len(entries),
            "migrated": len(compressed),
            "strategy": self._strategy,
            "short_term_remaining": remaining,
            "long_term_total": self._lt.count,
        }

    # ── Strategies ─────────────────────────────────────────────────────

    def _concat_block(
        self,
        entries: list[MemoryEntry],

        # ── V0.4: agent_id for compressed entries ────────────────
        # Compressed entries inherit the agent_id of their sources
        # so they remain filterable in long-term searches.
        agent_id: str = "main",
    ) -> list[MemoryEntry]:
        """Concatenate a batch of short-term entries into fewer long-term
        chunks.

        Merges entry content with "\n---\n" separators and splits into
        chunks of at most ``max_chars`` characters each.  Each chunk
        becomes one long-term MemoryEntry with traceability metadata
        pointing back to the original entries.

        V0.4: compressed entries carry the same agent_id as their source
        entries, keeping them in the right partition.

        Example: 20 short entries × ~50 chars each → 1 chunk (1000 chars)
        """
        if not entries:
            return []

        # Build a combined text with separators.
        parts: list[str] = []
        for entry in entries:
            content = entry.content.strip()
            if content:
                parts.append(content)

        if not parts:
            return []

        full_text = "\n---\n".join(parts)

        # Split into max_chars chunks.
        compressed: list[MemoryEntry] = []
        offset = 0
        while offset < len(full_text):
            chunk = full_text[offset:offset + self._max_chars]
            if offset + self._max_chars < len(full_text):
                chunk += "\u2026"  # … (ellipsis)

            # Collect parent IDs for traceability.
            parent_ids = [e.id for e in entries if e.content.strip()]

            compressed.append(MemoryEntry(
                content=chunk,
                layer="long_term",
                source="compressed",
                agent_id=agent_id,         # ← V0.4: preserve agent identity
                parent_id=parent_ids[0] if parent_ids else None,
                metadata={
                    "original_sources": list(
                        set(e.source for e in entries)
                    ),
                    "original_count": len(entries),
                    "original_agent_ids": list(
                        set(e.agent_id for e in entries)
                    ),
                    "compression_strategy": self._strategy,
                    "original_total_chars": len(full_text),
                },
            ))
            offset += self._max_chars

        return compressed
