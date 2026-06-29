"""
Configuration — typed YAML loader for Hippocampus.

The config.yml file lives next to pyproject.toml (or in the current working
directory) and controls every tunable parameter: storage paths, window sizes,
embedding models, compression thresholds, etc.

If no config.yml is found, from_file() creates one with sensible defaults.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml


# ── Default YAML ──────────────────────────────────────────────────────────
# This is written out verbatim when no config.yml exists.  Every dataclass
# below mirrors its structure so the code is self-documenting.
DEFAULT_CONFIG_YAML = """\
# Hippocampus Configuration
storage:
  data_dir: "./data"
short_term:
  window_size: 100
  compression_threshold: 0.8
  format: "json"
long_term:
  backend: "chroma"
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
    backend: str = "chroma"
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
        for section in (
            "storage", "short_term", "long_term",
            "compression", "working", "trace", "cli"
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
            "storage": self.storage.__dict__,
            "short_term": self.short_term.__dict__,
            "long_term": self.long_term.__dict__,
            "compression": self.compression.__dict__,
            "working": self.working.__dict__,
            "trace": self.trace.__dict__,
            "cli": self.cli.__dict__,
        }
