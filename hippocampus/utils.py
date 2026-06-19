"""
Utilities: ID generation, timestamps, text helpers.
"""

import uuid
import time
from datetime import datetime, timezone


def generate_id(prefix: str = "hippo") -> str:
    """Generate a time-sortable unique ID (UUID7-like).
    
    Uses UUID4 + epoch timestamp prefix for rough time-sorting.
    Format: {prefix}_{timestamp}_{uuid_short}
    """
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:12]
    return f"{prefix}_{ts}_{uid}"


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def local_now_iso() -> str:
    """Return current local time as ISO 8601 string."""
    return datetime.now().astimezone().isoformat()


def truncate(text: str, max_len: int = 200) -> str:
    """Truncate text to max_len with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
