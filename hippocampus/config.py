"""
Configuration loader — reads config.yml, merges with defaults.
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "hippocampus": {
        "data_dir": "./data",
        "short_term": {
            "window_size": 50,
            "compression_threshold": 40,
        },
        "long_term": {
            "top_k": 5,
            "collection_name": "hippocampus_ltm",
            "embedding_backend": "chroma_default",
        },
        "working": {
            "file": "working.json",
        },
        "id_prefix": "hippo",
    }
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str = "config.yml") -> Dict[str, Any]:
    """Load config from YAML file, falling back to defaults.
    
    Resolves data_dir relative to config file location.
    """
    config = DEFAULT_CONFIG.copy()

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = deep_merge(config, user_config)

    # Resolve data_dir relative to config file's directory
    data_dir = config["hippocampus"]["data_dir"]
    if not os.path.isabs(data_dir):
        config["hippocampus"]["data_dir"] = str(path.parent / data_dir)

    # Ensure data_dir exists
    os.makedirs(config["hippocampus"]["data_dir"], exist_ok=True)

    return config


def get_layer_config(config: dict, layer: str) -> dict:
    """Get config for a specific memory layer."""
    hc = config["hippocampus"]
    if layer == "short_term":
        return hc["short_term"]
    elif layer == "long_term":
        return hc["long_term"]
    elif layer == "working":
        return hc["working"]
    else:
        raise ValueError(f"Unknown layer: {layer}")
