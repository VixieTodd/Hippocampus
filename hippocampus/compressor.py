"""
Compressor — migrates entries from short-term to long-term memory.
"""

from typing import List

from .memory.base import MemoryEntry
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .utils import generate_id, utc_now_iso


class Compressor:
    """Handles short-term → long-term memory compression."""

    def __init__(self, short_term: ShortTermMemory, long_term: LongTermMemory,
                 id_prefix: str = "hippo"):
        self.short_term = short_term
        self.long_term = long_term
        self.id_prefix = id_prefix

    def compress(self, force: bool = False) -> int:
        """Run compression cycle.
        
        If force=True, compress ALL short-term entries.
        Otherwise, only compress entries exceeding window_size.
        
        Returns the number of entries compressed.
        """
        if not force and not self.short_term.needs_compression():
            return 0

        if force:
            # Compress all short-term entries
            entries = self.short_term.pop_oldest(self.short_term.count())
        else:
            # Compress oldest entries that exceed window_size
            excess = self.short_term.count() - self.short_term.window_size
            if excess <= 0:
                return 0
            entries = self.short_term.pop_oldest(excess)

        if not entries:
            return 0

        # Generate summary entries and migrate to long-term
        summarized = self._summarize_batch(entries)
        self.long_term.add_batch(summarized)

        return len(entries)

    def _summarize_batch(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """Summarize a batch of short-term entries into long-term entries.
        
        V0.1 strategy: group by temporal proximity, create one summary
        entry per batch. Without an LLM, we concatenate with markers.
        """
        if len(entries) <= 1:
            # Single entry: migrate directly with a "[Archived]" prefix
            for e in entries:
                e.content = f"[Archived from STM] {e.content}"
            return entries

        # Multiple entries: create a merged summary
        timestamps = [e.timestamp for e in entries]
        contents = [f"- {e.content}" for e in entries]
        
        merged_content = (
            f"[Compressed batch: {len(entries)} entries, "
            f"{timestamps[0][:10]} to {timestamps[-1][:10]}]\n"
            + "\n".join(contents)
        )

        merged_entry = MemoryEntry.create(
            content=merged_content,
            source="compressor",
            layer="long_term",
            id_prefix=self.id_prefix,
            metadata={
                "compressed_from": [e.id for e in entries],
                "original_count": len(entries),
                "original_timestamps": timestamps,
            },
        )

        return [merged_entry]
