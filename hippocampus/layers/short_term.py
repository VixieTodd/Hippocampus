"""
Short-Term Memory Layer.

Behaviour:
  - Sliding window of the *N* most recent entries per agent.
  - Stored as a JSON file (``short_term_memory.json``).
  - When the window reaches the fill threshold, the MemoryStore caller
    triggers compression → migration to long-term.
  - Search is keyword-based (newest-first); no vector index needed since
    this layer holds only recent, small data.

V0.4 Multi-Agent:
  Entries are partitioned by ``agent_id``.  Each agent gets its own
  independent sliding window (same window_size for all).  Search,
  write, stats all accept an agent_id parameter.

  When ``enable_isolation`` is False (config), all agents share one
  global window (legacy single-agent behaviour).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult


class ShortTermMemoryLayer(BaseLayer):
    """Recent entries with sliding window, partitioned by agent_id.

    V0.4: internal storage changed from a flat list to
    ``dict[agent_id, list[MemoryEntry]]`` so each agent has its own window.
    The on-disk format remains a single JSON array — partitioning happens
    in memory on load and is transparent to the serialisation format.
    """

    def __init__(self, data_dir: Path, window_size: int = 100) -> None:
        super().__init__("short_term")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "short_term_memory.json"
        self._window_size = window_size

        # ── V0.4: partitioned storage ──────────────────────────────
        # key = agent_id, value = list of MemoryEntry (newest at end).
        # Default dict so we don't need to pre-register agent IDs.
        self._entries: dict[str, list[MemoryEntry]] = defaultdict(list)

        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load entries from disk and re-partition by agent_id.

        V0.4: entries are read as a flat list, then grouped by agent_id
        in memory.  Entries serialised before the agent_id field existed
        will default to "main" (via MemoryEntry.from_dict).
        """
        if not self._file.exists():
            return

        with open(self._file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Rebuild partitioned storage from flat list.
        # Using defaultdict to avoid key errors for new agent_ids.
        partitioned: dict[str, list[MemoryEntry]] = defaultdict(list)
        for item in raw:
            entry = MemoryEntry.from_dict(item)
            partitioned[entry.agent_id].append(entry)

        self._entries = partitioned

    def _save(self) -> None:
        """Persist all partitions as a single flat JSON array.

        V0.4: all agents' entries are serialised together into one file.
        The agent_id field on each entry preserves the partition on reload.
        """
        # Flatten all partitions into one list.
        all_entries: list[dict] = []
        for agent_entries in self._entries.values():
            for entry in agent_entries:
                all_entries.append(entry.to_dict())

        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

    def _enforce_window(self, agent_id: str) -> int:
        """Drop the oldest entries for *agent_id* if its window exceeds the limit.

        Each agent's window is independent — Agent A writing doesn't
        evict Agent B's recent entries.

        Returns the number of entries trimmed for this agent.
        """
        agent_entries = self._entries[agent_id]
        if len(agent_entries) <= self._window_size:
            return 0
        trimmed = len(agent_entries) - self._window_size
        # Keep only the newest window_size entries.
        self._entries[agent_id] = agent_entries[trimmed:]
        return trimmed

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Append an entry to the appropriate agent's window.

        V0.4: entry.agent_id determines which partition gets the entry.
        The agent's own window is enforced independently.
        """
        entry.layer = "short_term"
        agent_id = entry.agent_id

        # Append to this agent's partition.
        self._entries[agent_id].append(entry)

        # Enforce this agent's sliding window.
        self._enforce_window(agent_id)

        self._save()
        return entry.id

    def search(
        self,
        query: str,
        top_k: int = 5,
        agent_id: str | None = None,
    ) -> list[SearchResult]:
        """Case-insensitive keyword search, optionally scoped to one agent.

        V0.4: when ``agent_id`` is provided, only that agent's partition
        is searched.  When ``None`` (default), all agents' entries are
        searched together.  Useful for cross-agent introspection.

        Args:
            query:    Free-text search string.
            top_k:    Max results to return.
            agent_id: Scope to one agent, or None for all.

        Returns:
            SearchResult list sorted by score descending (then newest first).
        """
        query_lower = query.lower()
        keywords = [w for w in query_lower.split() if w]
        if not keywords:
            return []

        # ── V0.4: select which partitions to search ────────────────
        if agent_id is not None:
            # Scoped to a single agent.
            partitions = {agent_id: self._entries.get(agent_id, [])}
        else:
            # Search all agents together.
            partitions = self._entries

        results: list[SearchResult] = []
        for aid, agent_entries in partitions.items():
            for entry in reversed(agent_entries):  # newest first
                content_lower = entry.content.lower()
                hits = sum(1 for kw in keywords if kw in content_lower)
                if hits == 0:
                    continue

                # Score: 0.2 base bias for any match, + 0.8 × hit_ratio.
                score = min(1.0, hits / len(keywords) * 0.8 + 0.2)

                results.append(SearchResult(
                    entry_id=entry.id,
                    content=entry.content,
                    score=round(score, 4),
                    layer="short_term",
                    timestamp=entry.timestamp,
                    metadata=entry.metadata,
                ))

        # Sort by score desc; ties broken by newest first (implicit because
        # we iterated newest-first within each agent, but with cross-agent
        # results we need explicit sorting).
        results.sort(key=lambda r: (r.score, r.timestamp), reverse=True)
        return results[:top_k]

    def stats(self, agent_id: str | None = None) -> dict:
        """Layer stats, optionally scoped to one agent.

        V0.4: when ``agent_id`` is given, stats reflect only that agent's
        partition.  When None, returns per-agent breakdown + global totals.
        """
        if agent_id is not None:
            entries = self._entries.get(agent_id, [])
            total_chars = sum(len(e.content) for e in entries)
            fill_ratio = len(entries) / max(self._window_size, 1)
            return {
                "layer": "short_term",
                "agent_id": agent_id,
                "count": len(entries),
                "window_size": self._window_size,
                "fill_ratio": round(fill_ratio, 2),
                "total_chars": total_chars,
                "needs_compression": fill_ratio >= 0.8,
            }

        # ── Global stats: per-agent breakdown ──────────────────────
        agent_stats: dict[str, dict] = {}
        total_count = 0
        total_chars = 0
        any_needs_compression = False

        for aid, entries in sorted(self._entries.items()):
            chars = sum(len(e.content) for e in entries)
            ratio = len(entries) / max(self._window_size, 1)
            needs = ratio >= 0.8
            agent_stats[aid] = {
                "count": len(entries),
                "fill_ratio": round(ratio, 2),
                "total_chars": chars,
                "needs_compression": needs,
            }
            total_count += len(entries)
            total_chars += chars
            if needs:
                any_needs_compression = True

        return {
            "layer": "short_term",
            "agent_count": len(self._entries),
            "total_count": total_count,
            "total_chars": total_chars,
            "window_size": self._window_size,
            "any_needs_compression": any_needs_compression,
            "agents": agent_stats,
        }

    def export(self) -> list[dict]:
        """Export all entries across all agents as a flat list."""
        all_entries: list[dict] = []
        for agent_entries in self._entries.values():
            for entry in agent_entries:
                all_entries.append(entry.to_dict())
        return all_entries

    def clear(self, agent_id: str | None = None) -> None:
        """Clear entries.

        V0.4: when ``agent_id`` is given, only that agent's partition is
        cleared.  When None, all partitions are wiped.
        """
        if agent_id is not None:
            self._entries.pop(agent_id, None)
        else:
            self._entries.clear()
        self._save()

    # ── Batch operations (used by Compressor, V0.4: agent-scoped) ────

    def get_entries_batch(
        self, count: int, agent_id: str
    ) -> list[MemoryEntry]:
        """Return the oldest *count* entries for a specific agent.

        V0.4: requires ``agent_id`` — compression is always per-agent,
        so this method only pulls from one partition at a time.
        """
        agent_entries = self._entries.get(agent_id, [])
        return agent_entries[:count]

    def remove_batch(
        self, entry_ids: list[str], agent_id: str
    ) -> None:
        """Remove entries by ID for a specific agent.

        V0.4: requires ``agent_id`` — we only scan one partition,
        which is both faster and prevents accidentally deleting
        another agent's entries with colliding UUIDs.
        """
        ids_set = set(entry_ids)
        if agent_id in self._entries:
            self._entries[agent_id] = [
                e for e in self._entries[agent_id]
                if e.id not in ids_set
            ]
        self._save()

    def all_entries(self, agent_id: str | None = None) -> list[MemoryEntry]:
        """Get all entries.

        V0.4: when ``agent_id`` is given, returns only that agent's entries.
        When None, returns entries from ALL agents (flattened).
        """
        if agent_id is not None:
            return list(self._entries.get(agent_id, []))
        # Flatten all partitions.
        result: list[MemoryEntry] = []
        for agent_entries in self._entries.values():
            result.extend(agent_entries)
        return result

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Find an entry by ID across all agent partitions."""
        for agent_entries in self._entries.values():
            for e in agent_entries:
                if e.id == entry_id:
                    return e
        return None

    # ── V0.4: Agent management helpers ──────────────────────────────

    def list_agents(self) -> list[str]:
        """Return all agent_ids currently with entries in short-term."""
        return sorted(self._entries.keys())

    def agent_count(self) -> int:
        """Number of agents with entries in short-term."""
        return len(self._entries)
