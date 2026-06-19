"""
Base memory entry and layer interface.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional

from ..utils import generate_id, utc_now_iso


@dataclass
class MemoryEntry:
    """A single memory entry with full traceability."""
    id: str
    content: str
    timestamp: str
    source: str = "cli"
    layer: str = "short_term"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        content: str,
        source: str = "cli",
        layer: str = "short_term",
        id_prefix: str = "hippo",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryEntry":
        """Factory to create a new entry with auto-generated ID and timestamp."""
        return cls(
            id=generate_id(prefix=id_prefix),
            content=content,
            timestamp=utc_now_iso(),
            source=source,
            layer=layer,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        """Deserialize from dictionary."""
        return cls(**data)

    def summary(self, max_len: int = 80) -> str:
        """Short preview of this entry."""
        text = self.content[:max_len]
        if len(self.content) > max_len:
            text += "..."
        return f"[{self.id}] {text}"
