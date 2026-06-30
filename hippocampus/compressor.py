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
"""

from __future__ import annotations

from hippocampus.memory import MemoryEntry
from hippocampus.layers.short_term import ShortTermMemoryLayer
from hippocampus.layers.long_term import LongTermMemoryLayer


class Compressor:
    """Migrates short-term entries into long-term memory."""

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

    def compress(self, force: bool = False) -> dict:
        """Migrate a batch of short-term entries into long-term.

        Args:
            force:  Skip the threshold check; compress regardless.

        Returns:
            Stats dict::

                {
                    "compressed":  N,       # entries read from ST
                    "migrated":    M,       # entries written to LT (≤ N)
                    "strategy":    "...",
                    "short_term_remaining": ...,
                    "long_term_total": ...,
                }
        """
        st = self._st.stats()
        fill_ratio = st.get("fill_ratio", 0.0)
        entry_count = st.get("count", 0)

        # Bail early if there's nothing to do.
        if entry_count == 0:
            return {"compressed": 0, "migrated": 0, "reason": "no entries"}

        if not force and fill_ratio < self._threshold:
            return {
                "compressed": 0,
                "migrated": 0,
                "reason": f"fill_ratio {fill_ratio:.2f} < threshold {self._threshold}",
            }

        # Grab the oldest batch.
        batch_count = min(self._batch_size, entry_count)
        entries = self._st.get_entries_batch(batch_count)

        # Compress.
        if self._strategy == "simple_concat":
            compressed = self._concat_block(entries)
        else:
            # Unknown strategy → fall back to simple concat.
            compressed = self._concat_block(entries)

        # Migrate.
        if compressed:
            self._lt.write_batch(compressed)
            ids_removed = [e.id for e in entries]
            self._st.remove_batch(ids_removed)

        return {
            "compressed": len(entries),
            "migrated": len(compressed),
            "strategy": self._strategy,
            "short_term_remaining": self._st.stats()["count"],
            "long_term_total": self._lt.count,
        }

    # ── Strategies ─────────────────────────────────────────────────────

    def _concat_block(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Concatenate a batch of short-term entries into fewer long-term chunks.

        Merges entry content with "\n---\n" separators and splits into chunks
        of at most ``max_chars`` characters each.  Each chunk becomes one long-term
        MemoryEntry with traceability metadata pointing back to the original entries.

        Example: 20 short entries × ~50 chars each → 1 chunk (1000 chars ≤ max_chars)
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
                chunk += "…"

            # Collect parent IDs for traceability.
            parent_ids = [e.id for e in entries if e.content.strip()]
            compressed.append(MemoryEntry(
                content=chunk,
                layer="long_term",
                source="compressed",
                parent_id=parent_ids[0] if parent_ids else None,
                metadata={
                    "original_sources": list(set(e.source for e in entries)),
                    "original_count": len(entries),
                    "compression_strategy": self._strategy,
                    "original_total_chars": len(full_text),
                },
            ))
            offset += self._max_chars

        return compressed
