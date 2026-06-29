"""
Traceability — append-only operation log for Hippocampus.

Every write, search, compress, and export operation is logged here with
a timestamp and entry ID.  This provides a full audit trail for debugging
and understanding how memory flowed through the system.

**Design note:** The log is a simple JSON-lines file.  ``trace(entry_id)``
scans sequentially (O(n) per lookup).  For small-to-medium scale this is
fine; if trace files grow beyond ~10 K lines, consider switching to a
line-oriented database or building an in-memory index.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Tracer:
    """Simple append-only trace log backed by a JSON-lines file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self._path = log_path
        self._enabled = log_path is not None

    def log(
        self,
        action: str,
        entry_id: str | None = None,
        layer: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Append one trace record to the log file.

        Silently discards if tracing is disabled (no log_path was given)
        or if the file can't be written (permission, disk full, …).
        """
        if not self._enabled or not self._path:
            return

        record = {
            "action": action,
            "entry_id": entry_id,
            "layer": layer,
            "timestamp": time.time(),
            "detail": detail or {},
        }

        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass  # Don't let a trace write failure crash the caller.

    def read(self, entry_id: str) -> list[dict[str, Any]]:
        """Return all trace records that reference *entry_id*.

        Performs a linear scan of the entire trace file.  The result is
        empty when the file doesn't exist or is empty.
        """
        if not self._path or not self._path.exists():
            return []

        traces: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("entry_id") == entry_id:
                        traces.append(record)
                except json.JSONDecodeError:
                    continue
        return traces

    def all(self) -> list[dict[str, Any]]:
        """Return every trace record in the log (unsorted)."""
        if not self._path or not self._path.exists():
            return []

        traces: list[dict[str, Any]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return traces
