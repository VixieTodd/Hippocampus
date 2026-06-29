"""
Dependency checker — verify and auto-install required packages.

Hippocampus relies on several PyPI packages that may not be installed yet.
This module provides a ``check()`` function that reports status and an
``install_missing()`` function that offers to pip-install whatever's
missing, without requiring any external shell scripting.
"""

from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from typing import NamedTuple


# ── Manifest ─────────────────────────────────────────────────────────────
# (import_name, pip_package, minimum_version, human_label)
REQUIRED: list[tuple[str, str, str, str]] = [
    ("click",       "click",       "8.0", "CLI framework"),
    ("yaml",        "pyyaml",      "6.0", "YAML config parser"),
    ("chromadb",    "chromadb",    "0.4.0", "Vector database (long-term)"),
    ("sentence_transformers", "sentence-transformers",
     "2.2.0", "Embedding model backend"),
]


class DepStatus(NamedTuple):
    """Result of checking a single dependency."""

    name: str           # import name
    label: str          # human-readable description
    installed: bool
    version: str | None
    min_version: str
    ok: bool            # installed AND ≥ min_version


def check_python() -> tuple[bool, str]:
    """Verify Python ≥ 3.10."""
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 10
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    return ok, version_str


def check_one(import_name: str, pip_name: str, min_ver: str, label: str) -> DepStatus:
    """Check if *import_name* is importable and meets *min_ver*."""
    installed = False
    version: str | None = None
    try:
        mod = importlib.import_module(import_name)
        installed = True
        version = getattr(mod, "__version__", None) or "?"
    except ImportError:
        pass

    ok = installed

    # Version comparison (best-effort: fail open if we can't parse either)
    if installed and version and version != "?" and min_ver:
        try:
            # Split into int tuples for very rough comparison.
            def _int_tuple(v: str) -> tuple[int, ...]:
                parts = v.split(".")
                out = []
                for p in parts:
                    try:
                        out.append(int(p))
                    except ValueError:
                        out.append(0)
                return tuple(out)

            ok = _int_tuple(version) >= _int_tuple(min_ver)
        except Exception:
            ok = True  # fail open

    return DepStatus(
        name=import_name,
        label=label,
        installed=installed,
        version=version,
        min_version=min_ver,
        ok=ok,
    )


def check_all() -> list[DepStatus]:
    """Return status for every required dependency."""
    return [check_one(*d) for d in REQUIRED]


def install_missing(dry_run: bool = False) -> bool:
    """Install every missing/outdated dependency via pip.

    Returns True if all dependencies are now satisfied (or dry-run).
    """
    statuses = check_all()
    missing = [s for s in statuses if not s.ok]

    if not missing:
        return True

    print("\n── 检测到缺失的依赖 ──")
    for s in missing:
        tag = "未安装" if not s.installed else f"版本过低 ({s.version} < {s.min_version})"
        print(f"  · {s.label}  ({s.name})  —  [{tag}]")

    if dry_run:
        print("\n执行安装需要运行：pip install <包名>")
        return False

    # Collect pip package names.
    pkgs = []
    for s in missing:
        # Find it in the manifest.
        for imp, pip_name, *_ in REQUIRED:
            if imp == s.name:
                pkgs.append(pip_name)
                break

    if not pkgs:
        return True

    print(f"\n即将安装：{' '.join(pkgs)}")
    print("是否继续？[Y/n] ", end="", flush=True)

    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False

    if answer not in ("", "y", "yes"):
        print("已取消。")
        return False

    print("\n正在安装，请稍候…")
    for pkg in pkgs:
        print(f"  pip install {pkg}…")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", pkg],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            fail_icon = chr(0x274C)
            print(f"  {fail_icon} {pkg} 安装失败：{result.stderr.strip()[-200:]}")
            return False
        ok_icon = chr(0x2705)
        print(f"  {ok_icon} {pkg} 安装成功")

    # Final verification.
    final = check_all()
    all_ok = all(s.ok for s in final)
    ok_icon = chr(0x2705)
    if all_ok:
        print(f"\n{ok_icon} 全部依赖就绪。")
    else:
        still_bad = [s for s in final if not s.ok]
        warn = chr(0x26A0) + chr(0xFE0F)
        print(f"\n{warn} 仍有 {len(still_bad)} 项未满足：")
        for s in still_bad:
            print(f"  · {s.label}")
    return all_ok
