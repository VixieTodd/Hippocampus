"""
Working Memory Layer ("Engineering Memory").

This is the *static* layer — it holds config, tools, rules, and other
things that should always be "in context".  Unlike short/long-term:
  - It never flows or compresses.
  - Entries are manually written and removed.
  - There is no vector index — just a keyword fallback for search.

Storage: single JSON file (``working_memory.json``) inside the data dir.

V0.4 Multi-Agent:
  Working memory entries are tagged with ``agent_id``.  By default,
  entries are "shared" (agent_id="shared") so all agents see common
  config and rules.  Agent-specific entries use the agent's own ID.

  Search can be scoped to a specific agent, or include shared entries
  alongside agent-specific ones (the default).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from hippocampus.memory import MemoryEntry
from hippocampus.layers import BaseLayer, SearchResult

# ── V0.4: reserved agent_id for shared entries ──────────────────────
# Entries written with this agent_id are visible to ALL agents during
# search.  Use it for global config, shared tools, or common rules.
SHARED_AGENT_ID = "shared"


class WorkingMemoryLayer(BaseLayer):
    """Static working memory — config, tools, rules.

    V0.4: entries are partitioned by agent_id in memory, stored in a
    single JSON file.  "shared" is a reserved agent_id for entries
    visible to all agents.
    """

    def __init__(self, data_dir: Path) -> None:
        super().__init__("working")
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._file = self._data_dir / "working_memory.json"

        # ── V0.4: partitioned by agent_id ─────────────────────────
        # key = agent_id ("shared", "main", "coder", …)
        # value = dict[entry_id, MemoryEntry]
        self._entries: dict[str, dict[str, MemoryEntry]] = defaultdict(dict)

        self._load()

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Read entries from ``working_memory.json`` and partition by agent_id.

        V0.4: entries are loaded as a flat list, then grouped by agent_id.
        Entries from before V0.4 will default to agent_id="shared".
        """
        if not self._file.exists():
            return

        with open(self._file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        partitioned: dict[str, dict[str, MemoryEntry]] = defaultdict(dict)
        for item in raw:
            entry = MemoryEntry.from_dict(item)
            partitioned[entry.agent_id][entry.id] = entry

        self._entries = partitioned

    def _save(self) -> None:
        """Persist all entries as a flat JSON array.

        V0.4: all partitions are flattened into one file.  The agent_id
        field on each entry preserves the partition on reload.
        """
        all_entries: list[dict] = []
        for agent_entries in self._entries.values():
            for entry in agent_entries.values():
                all_entries.append(entry.to_dict())

        with open(self._file, "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)

    # ── Layer interface ───────────────────────────────────────────────

    def write(self, entry: MemoryEntry) -> str:
        """Store (or overwrite) an entry in working memory.

        V0.4: entry goes into the partition matching entry.agent_id.
        If agent_id is not set, it defaults to ``"shared"`` so all
        agents can see it.

        Overwrite behaviour: if an entry with the same UUID already
        exists in this agent's partition, it is replaced.
        """
        entry.layer = "working"
        agent_id = entry.agent_id

        self._entries[agent_id][entry.id] = entry
        self._save()
        return entry.id

    def search(
        self,
        query: str,
        top_k: int = 5,

        # ── V0.4: agent scope ─────────────────────────────────────
        # None (default): search this agent's own entries + shared.
        # Specific agent_id: search only that agent's + shared.
        # To search ONLY the calling agent without shared entries,
        # pass include_shared=False.
        agent_id: str | None = None,
        include_shared: bool = True,
    ) -> list[SearchResult]:
        """Simple case-insensitive substring search over entry content.

        V0.4: by default, returns matches from both the requesting agent's
        partition AND shared entries.  This means every agent automatically
        sees global config/rules without extra code.
        """
        query_lower = query.lower()
        results: list[SearchResult] = []

        # ── V0.4: decide which partitions to search ──────────────
        partitions_to_search: list[list[MemoryEntry]] = []

        if agent_id is not None:
            # Agent-specific entries for the requested agent.
            partitions_to_search.append(
                list(self._entries.get(agent_id, {}).values())
            )
        else:
            # Search all agent-specific partitions (not shared yet).
            for aid, entries_dict in self._entries.items():
                if aid != SHARED_AGENT_ID:
                    partitions_to_search.append(list(entries_dict.values()))

        if include_shared:
            partitions_to_search.append(
                list(self._entries.get(SHARED_AGENT_ID, {}).values())
            )

        for entries in partitions_to_search:
            for entry in entries:
                if query_lower in entry.content.lower():
                    results.append(SearchResult(
                        entry_id=entry.id,
                        content=entry.content,
                        score=1.0,  # Working memory: exact substring = perfect match.
                        layer="working",
                        timestamp=entry.timestamp,
                        metadata=entry.metadata,
                    ))

        return results[:top_k]

    def stats(self, agent_id: str | None = None) -> dict:
        """Layer stats.

        V0.4: when ``agent_id`` is given, returns only that agent's
        partition.  When None, returns per-agent breakdown + global totals.
        """
        if agent_id is not None:
            entries = self._entries.get(agent_id, {})
            total_chars = sum(len(e.content) for e in entries.values())
            return {
                "layer": "working",
                "agent_id": agent_id,
                "count": len(entries),
                "total_chars": total_chars,
                "file": str(self._file),
            }

        # ── Per-agent breakdown ──────────────────────────────────
        agent_stats: dict[str, dict] = {}
        total_count = 0
        total_chars = 0

        for aid, entries_dict in sorted(self._entries.items()):
            chars = sum(len(e.content) for e in entries_dict.values())
            agent_stats[aid] = {
                "count": len(entries_dict),
                "total_chars": chars,
            }
            total_count += len(entries_dict)
            total_chars += chars

        return {
            "layer": "working",
            "agent_count": len(self._entries),
            "total_count": total_count,
            "total_chars": total_chars,
            "file": str(self._file),
            "agents": agent_stats,
        }

    def export(self) -> list[dict]:
        """Serialise all entries as a flat list."""
        all_entries: list[dict] = []
        for entries_dict in self._entries.values():
            for entry in entries_dict.values():
                all_entries.append(entry.to_dict())
        return all_entries

    def clear(self, agent_id: str | None = None) -> None:
        """Remove entries.

        V0.4: when ``agent_id`` is given, only that partition is cleared.
        When None, all partitions (including shared) are wiped.
        """
        if agent_id is not None:
            self._entries.pop(agent_id, None)
        else:
            self._entries.clear()
        self._save()

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        """Fetch a single entry by UUID across all partitions."""
        for entries_dict in self._entries.values():
            if entry_id in entries_dict:
                return entries_dict[entry_id]
        return None

    # ── V0.4: Agent management helpers ──────────────────────────────

    def list_agents(self) -> list[str]:
        """Return all agent_ids with entries in working memory."""
        return sorted(self._entries.keys())

    def agent_entries(self, agent_id: str) -> list[MemoryEntry]:
        """Get all entries for a specific agent (not including shared)."""
        return list(self._entries.get(agent_id, {}).values())

    def shared_entries(self) -> list[MemoryEntry]:
        """Get all shared entries (visible to all agents)."""
        return list(self._entries.get(SHARED_AGENT_ID, {}).values())
