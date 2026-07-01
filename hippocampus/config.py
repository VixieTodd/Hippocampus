"""
Configuration — typed YAML loader for Hippocampus.

The config.yml file lives next to pyproject.toml (or in the current working
directory) and controls every tunable parameter: storage paths, window sizes,
embedding models, compression thresholds, etc.

If no config.yml is found, from_file() creates one with sensible defaults.

V0.4: added ``agent`` section for multi-agent support.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ── Default YAML ──────────────────────────────────────────────────────────
# This is written out verbatim when no config.yml exists.  Every dataclass
# below mirrors its structure so the code is self-documenting.
#
# V0.4: added ``agent`` section with multi-agent isolation settings.
DEFAULT_CONFIG_YAML = """\
# Hippocampus Configuration
#
# Backend: "tfidf" (default, zero extra deps) or "chroma" (semantic, needs pip install)
storage:
  data_dir: "./data"
short_term:
  window_size: 100
  compression_threshold: 0.8
  format: "json"
long_term:
  backend: "tfidf"
  embedding_model: "all-MiniLM-L6-v2"
  top_k: 5
  min_score: 0.0
  collection_name: "hippocampus_long_term"
compression:
  strategy: "simple_concat"
  max_chars: 2000
  batch_size: 20
working:
  entries_file: ""
trace:
  enabled: true
  log_file: "trace.log"
cli:
  default_top_k: 5
agent:
  default_agent_id: "main"
  enable_isolation: true
  cross_agent_search: true
  long_term_isolation: false
"""

# ── Sub-config dataclasses ────────────────────────────────────────────────


@dataclass
class StorageConfig:
    data_dir: str = "./data"


@dataclass
class ShortTermConfig:
    window_size: int = 100
    compression_threshold: float = 0.8
    format: str = "json"


@dataclass
class LongTermConfig:
    backend: str = "tfidf"  # "tfidf" (Lite) or "chroma" (Full)
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.0
    collection_name: str = "hippocampus_long_term"


@dataclass
class CompressionConfig:
    strategy: str = "simple_concat"
    max_chars: int = 2000
    batch_size: int = 20


@dataclass
class WorkingConfig:
    entries_file: str = ""


@dataclass
class TraceConfig:
    enabled: bool = True
    log_file: str = "trace.log"


@dataclass
class CLIConfig:
    default_top_k: int = 5


# ── V0.4: Agent config ─────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Multi-agent isolation settings.

    Attributes:
        default_agent_id:  Fallback agent_id when none is supplied to
                           write()/search().  Typically ``"main"`` for the
                           primary session.
        enable_isolation:  When True, short-term memory is partitioned
                           per agent_id (each agent gets its own sliding
                           window).  When False, all agents share one
                           pool (legacy single-agent behaviour).
        cross_agent_search: When True, long-term search without an
                           agent_id filter returns results from ALL agents.
                           When False, it returns only the requesting
                           agent's records.  Ignored when long_term_isolation
                           is True (search always scoped to one agent).
        long_term_isolation: When True, each agent gets its own independent
                           long-term collection (separate TF-IDF file or
                           ChromaDB collection).  When False (default), all
                           agents share one long-term pool with agent_id
                           tags for optional filtering.
    """
    default_agent_id: str = "main"
    enable_isolation: bool = True
    cross_agent_search: bool = True
    long_term_isolation: bool = False


# ── Top-level config ──────────────────────────────────────────────────────


@dataclass
class Config:
    """Hippocampus configuration with typed sub-configs."""

    storage: StorageConfig = field(default_factory=StorageConfig)
    short_term: ShortTermConfig = field(default_factory=ShortTermConfig)
    long_term: LongTermConfig = field(default_factory=LongTermConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    working: WorkingConfig = field(default_factory=WorkingConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)
    cli: CLIConfig = field(default_factory=CLIConfig)

    # V0.4: Multi-agent configuration.
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Internally set after from_file() to resolve relative paths.
    _config_path: Optional[Path] = field(default=None, repr=False)

    @property
    def data_dir(self) -> Path:
        """Absolute path to the data directory.

        Resolved relative to the config file's parent directory (or CWD if
        no config path is set yet — though in practice from_file always sets
        it before data_dir is accessed).
        """
        base = self._config_path.parent if self._config_path else Path.cwd()
        return (base / self.storage.data_dir).resolve()

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config from a YAML file.  Creates default if missing."""
        path = Path(path).resolve()
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        config = cls()
        config._config_path = path

        # Walk through each section and overlay values from YAML.
        # V0.4: agent section added.
        for section in (
            "storage", "short_term", "long_term",
            "compression", "working", "trace", "cli", "agent"
        ):
            if section in raw and raw[section]:
                sub = getattr(config, section)
                for key, value in raw[section].items():
                    if hasattr(sub, key):
                        setattr(sub, key, value)

        return config

    def to_dict(self) -> dict:
        """Rebuild the nested dict (symmetric with the YAML file)."""
        return {
            "storage": {k: v for k, v in self.storage.__dict__.items() if not k.startswith("_")},
            "short_term": {k: v for k, v in self.short_term.__dict__.items() if not k.startswith("_")},
            "long_term": {k: v for k, v in self.long_term.__dict__.items() if not k.startswith("_")},
            "compression": {k: v for k, v in self.compression.__dict__.items() if not k.startswith("_")},
            "working": {k: v for k, v in self.working.__dict__.items() if not k.startswith("_")},
            "trace": {k: v for k, v in self.trace.__dict__.items() if not k.startswith("_")},
            "cli": {k: v for k, v in self.cli.__dict__.items() if not k.startswith("_")},
            "agent": {k: v for k, v in self.agent.__dict__.items() if not k.startswith("_")},
        }
