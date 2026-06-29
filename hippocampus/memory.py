"""
Memory entry data model.

Every piece of information stored in Hippocampus is a MemoryEntry.
An entry always carries:
  - A unique ID  (UUID v4, generated on creation)
  - An ISO-8601 timestamp (UTC)
  - Source attribution (user / agent / system / compressed)
  - A layer label (short_term / long_term / working)
  - Optional parent_id for traceability (e.g. compression lineage)
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

    def to_dict(self) -> dict[str, Any]:
        """Flatten to a JSON-serialisable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Rebuild from a dict (as returned by to_dict()).

        Silently ignores unknown keys so older serialised entries survive
        schema additions without migration — forward compatible by default.
        """
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    def __repr__(self) -> str:
        return (
            f"<MemoryEntry id={self.id[:8]}... "
            f"layer={self.layer} ts={self.timestamp[:19]}>"
        )
