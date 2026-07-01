"""
Memory entry data model.

Every piece of information stored in Hippocampus is a MemoryEntry.
An entry always carries:
  - A unique ID  (UUID v4, generated on creation)
  - An ISO-8601 timestamp (UTC)
  - Source attribution (user / agent / system / compressed)
  - An agent_id that identifies which Agent owns this memory
  - A layer label (short_term / long_term / working)
  - Optional parent_id for traceability (e.g. compression lineage)

Multi-Agent support (V0.4):
  agent_id is the key field for multi-tenant isolation.  Each sub-agent
  (or the main agent) receives its own partition in short-term and can
  optionally filter long-term searches to its own records.  The default
  value "main" keeps backward compatibility with single-agent setups.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class MemoryEntry:
    """A single memory entry in any layer.

    Fields are all plain types (str, dict) so the object can be serialized
    to JSON via to_dict() / from_dict() without a custom encoder.

    V0.4: added ``agent_id`` for multi-agent isolation.  When an agent_id
    is explicitly given on write(), it is stored here; otherwise defaults
    to ``"main"`` (single-agent backward-compatible).
    """

    content: str
    layer: str = "short_term"         # short_term | long_term | working
    source: str = "user"              # user | agent | system | compressed
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    )
    parent_id: str | None = None       # Links to source entry (e.g. after compression)

    # ── V0.4: Multi-Agent ───────────────────────────────────────────
    # Identifies which agent (or "main") owns this memory entry.
    # Used by short-term partitioning and long-term search filtering.
    agent_id: str = "main"

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable dict.

        V0.4: agent_id is included automatically via asdict(),
        no special handling needed.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Rebuild from a dict (as returned by to_dict()).

        Silently ignores unknown keys so older serialised entries survive
        schema additions without migration — forward compatible by default.

        V0.4 note: entries serialised before the agent_id field was added
        will have agent_id default to "main", which is exactly what we want
        for backward compatibility.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:
        return (
            f"<MemoryEntry id={self.id[:8]}... "
            f"agent={self.agent_id} "
            f"layer={self.layer} ts={self.timestamp[:19]}>"
        )
