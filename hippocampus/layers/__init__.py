"""
Abstract base layer for all memory layers.

All three memory layers (working / short-term / long-term) implement this
interface. This guarantees the MemoryStore can treat them uniformly.

Methods each layer must provide:
  - write(entry)     → str (entry ID)
  - search(query, k) → list[SearchResult]
  - stats()          → dict
  - export()         → list[dict]
  - clear()          → None
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SearchResult:
    """
    A single search hit from any memory layer.

    Attributes:
        entry_id:  UUID of the matched entry.
        content:   Text content (may be truncated for display).
        score:     Relevance score in [0.0, 1.0], higher = better match.
        layer:     Which layer this result came from (e.g. "short_term").
        timestamp: ISO-8601 timestamp of the original entry.
        metadata:  Arbitrary key-value pairs attached at write time.
    """

    entry_id: str
    content: str
    score: float
    layer: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLayer(ABC):
    """
    Abstract base class that every memory layer must subclass.

    Layers are interchangeable: MemoryStore calls these same methods
    regardless of whether the backend is JSON, ChromaDB, or in-memory.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def write(self, entry: Any) -> str: ...

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]: ...

    @abstractmethod
    def stats(self) -> dict: ...

    @abstractmethod
    def export(self) -> list[dict]: ...

    @abstractmethod
    def clear(self) -> None: ...
