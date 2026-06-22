"""
Installer — guided setup wizard for Hippocampus.

Handles:
  1. Environment detection (OpenClaw workspace, memory files, skills)
  2. Memory migration (MEMORY.md + memory/*.md → LongTermMemory)
  3. Skill conflict resolution (disable/quarantine conflicting skills)

Usage:  hippo install
"""

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml


# ── Constants ──────────────────────────────────────────────

WORKSPACE_MARKERS = ["MEMORY.md", "AGENTS.md", "SOUL.md"]
CONFLICT_SKILLS = [
    "memory-setup",
    "proactive-agent",
    "self-improving-agent-skill",
]

# Markdown heading patterns
H2_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
H3_PATTERN = re.compile(r"^###\s+(.+)$", re.MULTILINE)


# ── Detection ──────────────────────────────────────────────

def _find_workspace(start_dir: Optional[str] = None) -> Optional[str]:
    """Find OpenClaw workspace by looking for markers upward from start_dir."""
    if start_dir is None:
        start_dir = os.getcwd()

    candidates = [
        start_dir,
        os.path.expanduser("~/.openclaw/workspace"),
        os.path.expanduser("~/openclaw-workspace"),
    ]

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            for marker in WORKSPACE_MARKERS:
                if os.path.isfile(os.path.join(candidate, marker)):
                    return os.path.abspath(candidate)

    # Walk up from current dir
    path = Path(start_dir).resolve()
    for parent in [path] + list(path.parents):
        for marker in WORKSPACE_MARKERS:
            if (parent / marker).is_file():
                return str(parent)

    return None


def detect_environment(config_dir: Optional[str] = None) -> dict:
    """Scan environment and return structured report.

    Returns:
        {
            "workspace": str | None,
            "memory_files": [
                {"path": "...", "name": "...", "size": 1234, "lines": 50}
            ],
            "memory_total_size": 12345,
            "skills": [
                {"name": "memory-setup", "path": "...", "conflicts": true}
            ],
            "installed_skills": 6,
            "conflict_count": 3,
        }
    """
    report: Dict[str, Any] = {
        "workspace": None,
        "memory_files": [],
        "memory_total_size": 0,
        "skills": [],
        "installed_skills": 0,
        "conflict_count": 0,
    }

    # Find workspace
    ws = _find_workspace(config_dir)
    report["workspace"] = ws

    if ws:
        # Scan memory files
        mem_path = os.path.join(ws, "MEMORY.md")
        if os.path.isfile(mem_path):
            size = os.path.getsize(mem_path)
            with open(mem_path, "r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
            report["memory_files"].append({
                "path": mem_path,
                "name": "MEMORY.md",
                "size": size,
                "lines": lines,
            })
            report["memory_total_size"] += size

        mem_dir = os.path.join(ws, "memory")
        if os.path.isdir(mem_dir):
            for fname in sorted(os.listdir(mem_dir)):
                if fname.endswith(".md"):
                    fpath = os.path.join(mem_dir, fname)
                    size = os.path.getsize(fpath)
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = sum(1 for _ in f)
                    report["memory_files"].append({
                        "path": fpath,
                        "name": f"memory/{fname}",
                        "size": size,
                        "lines": lines,
                    })
                    report["memory_total_size"] += size

        # Scan skills
        skills_dir = os.path.join(ws, "skills")
        if os.path.isdir(skills_dir):
            for skill_name in sorted(os.listdir(skills_dir)):
                skill_path = os.path.join(skills_dir, skill_name)
                if os.path.isdir(skill_path):
                    conflicts = skill_name in CONFLICT_SKILLS
                    report["skills"].append({
                        "name": skill_name,
                        "path": skill_path,
                        "conflicts": conflicts,
                    })
                    report["installed_skills"] += 1
                    if conflicts:
                        report["conflict_count"] += 1

    return report


# ── Pretty Print ───────────────────────────────────────────

def _format_size(size: int) -> str:
    """Format bytes to human-readable string."""
    if size < 1024:
        return f"{size}B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    else:
        return f"{size / (1024 * 1024):.1f}MB"


def print_environment_report(report: dict):
    """Print a formatted environment detection report."""
    click.echo()
    if report["workspace"]:
        click.echo(f"  {click.style('✓', fg='green')} 找到 OpenClaw 工作区: "
                   f"{click.style(report['workspace'], fg='cyan')}")
    else:
        click.echo(f"  {click.style('○', fg='yellow')} 未检测到 OpenClaw 工作区")
        return

    if report["memory_files"]:
        click.echo(f"\n  {click.style('记忆文件', bold=True)} "
                   f"({len(report['memory_files'])} 个文件, "
                   f"共 {_format_size(report['memory_total_size'])})")
        for f in report["memory_files"]:
            click.echo(f"    • {click.style(f['name'], fg='yellow')} "
                       f"— {f['lines']} 行, {_format_size(f['size'])}")
    else:
        click.echo(f"\n  {click.style('○', fg='yellow')} 未找到记忆文件")

    if report["skills"]:
        all_skills = [s for s in report["skills"] if s["conflicts"]]
        non_conflict = [s for s in report["skills"] if not s["conflicts"]]
        click.echo(f"\n  {click.style('Skill', bold=True)} "
                   f"({report['installed_skills']} 个已安装)")
        for s in non_conflict:
            click.echo(f"    • {click.style(s['name'], fg='green')}")
        for s in all_skills:
            click.echo(f"    • {click.style(s['name'], fg='red')} "
                       f"{click.style('[冲突风险]', fg='yellow')}")
    else:
        click.echo(f"\n  {click.style('○', fg='yellow')} 未找到已安装的 Skill")

    click.echo()


# ── Memory Migration ───────────────────────────────────────

def _parse_markdown_sections(content: str, filename: str) -> List[dict]:
    """Split markdown into sections by ## headings.

    Each section becomes a dict with:
        {title, content, level, filename}
    """
    sections = []

    # Split by H2
    parts = H2_PATTERN.split(content)
    # parts[0] = text before first ##
    # parts[1] = first title, parts[2] = first body, ...

    # Handle preamble (text before first ##)
    preamble = parts[0].strip()
    if preamble and not preamble.startswith("# "):
        # Skip the top-level # title line
        lines = preamble.split("\n")
        filtered = [l for l in lines if not l.startswith("# ")]
        if filtered:
            sections.append({
                "title": f"{filename} (preamble)",
                "content": "\n".join(filtered).strip(),
                "level": 0,
                "filename": filename,
            })

    # Process H2 sections
    for i in range(1, len(parts), 2):
        title = parts[i].strip() if i < len(parts) else "Untitled"
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Split H2 sections further by H3 if they're large
        h3_parts = H3_PATTERN.split(body)
        if len(h3_parts) >= 3:
            # Has H3 subsections
            h3_preamble = h3_parts[0].strip()
            if h3_preamble:
                sections.append({
                    "title": f"{filename} → {title}",
                    "content": h3_preamble,
                    "level": 2,
                    "filename": filename,
                })
            for j in range(1, len(h3_parts), 2):
                h3_title = h3_parts[j].strip() if j < len(h3_parts) else "Untitled"
                h3_body = h3_parts[j + 1].strip() if j + 1 < len(h3_parts) else ""
                if h3_body:
                    sections.append({
                        "title": f"{filename} → {title} → {h3_title}",
                        "content": h3_body,
                        "level": 3,
                        "filename": filename,
                    })
        else:
            # No H3 subsections, use whole body
            if body:
                sections.append({
                    "title": f"{filename} → {title}",
                    "content": body,
                    "level": 2,
                    "filename": filename,
                })

    return sections


def migrate_memories(
    report: dict,
    long_term_memory,
    id_prefix: str = "hippo",
    archive: bool = True,
) -> Tuple[int, List[str]]:
    """Migrate detected memory files into Hippocampus long-term memory.

    Args:
        report: detect_environment() output
        long_term_memory: LongTermMemory instance
        id_prefix: ID prefix for new entries
        archive: If True, rename originals to .bak

    Returns:
        (total_entries_migrated, list_of_warnings)
    """
    from .memory.base import MemoryEntry

    total = 0
    warnings = []
    archive_dir = None

    if archive and report["workspace"]:
        archive_dir = os.path.join(report["workspace"], "memory", ".archive")
        os.makedirs(archive_dir, exist_ok=True)

    for f in report["memory_files"]:
        try:
            with open(f["path"], "r", encoding="utf-8") as fh:
                content = fh.read()
        except Exception as e:
            warnings.append(f"读取失败: {f['name']} — {e}")
            continue

        sections = _parse_markdown_sections(content, f["name"])
        if not sections:
            warnings.append(f"跳过: {f['name']} — 无可解析的记忆条目")
            continue

        entries = []
        for sec in sections:
            # Skip sections that are too short or noise
            if len(sec["content"]) < 10:
                continue

            entry = MemoryEntry.create(
                content=sec["content"],
                source="migration",
                layer="long_term",
                id_prefix=id_prefix,
                metadata={
                    "migrated_from": f["name"],
                    "section": sec["title"],
                    "original_file": f["path"],
                    "migrated_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            entries.append(entry)

        if entries:
            long_term_memory.add_batch(entries)
            total += len(entries)

            # Archive original
            if archive and archive_dir:
                dest = os.path.join(archive_dir, os.path.basename(f["path"]))
                try:
                    shutil.move(f["path"], dest)
                except OSError as e:
                    warnings.append(f"归档失败: {f['name']} — {e}")

    return total, warnings


# ── Skill Conflict Handling ────────────────────────────────

def disable_skills(report: dict, selected_skills: Optional[List[str]] = None) -> dict:
    """Disable conflicting skills by writing .disabled markers.

    Args:
        report: detect_environment() output
        selected_skills: Specific skill names to disable.
                         If None, disable all conflicting skills.

    Returns:
        {skill_name: "disabled" | "skipped" | "error: ..."}
    """
    results = {}

    for s in report["skills"]:
        if not s["conflicts"]:
            continue
        if selected_skills is not None and s["name"] not in selected_skills:
            results[s["name"]] = "skipped"
            continue

        disabled_file = os.path.join(s["path"], ".disabled")
        try:
            with open(disabled_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "disabled_by": "hippocampus",
                    "disabled_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "Conflicts with Hippocampus memory system",
                }, ensure_ascii=False, indent=2))
            results[s["name"]] = "disabled"
        except OSError as e:
            results[s["name"]] = f"error: {e}"

    return results


# ── Interactive Wizard ─────────────────────────────────────

def run_install_wizard(config_dir: Optional[str] = None):
    """Run the full interactive installation wizard.

    Steps:
        1. Detect environment
        2. Show report
        3. Ask: migrate existing memories?
        4. Ask: disable conflicting skills?
        5. Execute choices
        6. Show summary
    """
    click.echo()
    click.echo(click.style("🦛 Hippocampus 安装向导", bold=True, fg="magenta"))
    click.echo(click.style("━" * 40, fg="bright_black"))

    # ── Step 1: Detect ──
    click.echo(f"\n{click.style('[1/3]', fg='cyan', bold=True)} 检测环境...")
    report = detect_environment(config_dir)

    if not report["workspace"]:
        click.echo()
        click.echo("  未检测到 OpenClaw 工作区。")
        click.echo("  Hippocampus 将以独立模式运行，不进行迁移或冲突处理。")
        click.echo()
        return

    print_environment_report(report)

    # ── Step 2: Memory Migration ──
    do_migrate = False
    if report["memory_files"]:
        click.echo(f"{click.style('[2/3]', fg='cyan', bold=True)} 记忆迁移")
        click.echo()
        click.echo("  检测到已有记忆文件。是否将现有记忆重组导入 Hippocampus？")
        click.echo()
        click.echo(f"    {click.style('[Y]', fg='green')} 是 — 解析并导入所有记忆，原文件归档为 .archive/")
        click.echo(f"    {click.style('[N]', fg='yellow')} 否 — 保留原文件不动，Hippocampus 从零开始")
        click.echo()

        choice = click.prompt(
            "  选择",
            type=click.Choice(["Y", "y", "N", "n", "yes", "no"]),
            default="Y",
            show_default=False,
        )
        do_migrate = choice.lower() in ("y", "yes")
    else:
        click.echo(f"\n{click.style('[2/3]', fg='cyan', bold=True)} 记忆迁移 — 无记忆文件，跳过")

    # ── Step 3: Skill Conflicts ──
    do_disable = False
    all_conflicting = [s for s in report["skills"] if s["conflicts"]]
    if all_conflicting:
        click.echo(f"\n{click.style('[3/3]', fg='cyan', bold=True)} Skill 冲突处理")
        click.echo()
        click.echo("  以下 Skill 可能与 Hippocampus 冲突：")
        click.echo()
        for s in all_conflicting:
            click.echo(f"    {click.style('⚠', fg='yellow')} {s['name']}")
        click.echo()
        click.echo("  是否禁用这些 Skill？")
        click.echo()
        click.echo(f"    {click.style('[Y]', fg='green')} 是 — 全部禁用（写入 .disabled 标记，可恢复）")
        click.echo(f"    {click.style('[N]', fg='yellow')} 否 — 保留它们")
        click.echo()

        choice = click.prompt(
            "  选择",
            type=click.Choice(["Y", "y", "N", "n", "yes", "no"]),
            default="Y",
            show_default=False,
        )
        do_disable = choice.lower() in ("y", "yes")
    else:
        click.echo(f"\n{click.style('[3/3]', fg='cyan', bold=True)} Skill 冲突处理 — 无冲突 Skill，跳过")

    # ── Execute ──
    click.echo(f"\n{click.style('━' * 40, fg='bright_black')}")
    click.echo(f"\n{click.style('执行中...', bold=True)}")

    total_migrated = 0
    migrate_warnings = []

    if do_migrate and report["memory_files"]:
        # Need a Hippocampus instance for long_term memory
        from .config import load_config
        from .memory.long_term import LongTermMemory

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yml"
        )
        config = load_config(config_path)
        hc = config["hippocampus"]
        data_dir = hc["data_dir"]

        ltm = LongTermMemory(
            data_dir=data_dir,
            collection_name=hc["long_term"]["collection_name"],
            top_k=hc["long_term"]["top_k"],
            embedding_backend=hc["long_term"]["embedding_backend"],
        )

        total_migrated, migrate_warnings = migrate_memories(
            report, ltm, id_prefix=hc["id_prefix"], archive=True,
        )
        click.echo(f"  → 迁移完成: {click.style(str(total_migrated), fg='green', bold=True)} 条记忆已导入长期记忆层")
        for w in migrate_warnings:
            click.echo(f"    {click.style('⚠', fg='yellow')} {w}")

    if do_disable and all_conflicting:
        results = disable_skills(report)
        for name, status in results.items():
            if status == "disabled":
                click.echo(f"  → {click.style('✓', fg='green')} {name} 已禁用")
            elif status.startswith("error"):
                click.echo(f"  → {click.style('✗', fg='red')} {name}: {status}")
            else:
                click.echo(f"  → {click.style('○', fg='yellow')} {name}: 已跳过")

    # ── Summary ──
    click.echo(f"\n{click.style('━' * 40, fg='bright_black')}")
    click.echo(f"\n{click.style('✓ 安装完成', fg='green', bold=True)}")

    summary_parts = []
    if total_migrated > 0:
        summary_parts.append(f"导入了 {total_migrated} 条记忆")
    if do_disable:
        disabled_count = sum(
            1 for s in all_conflicting
            if s["name"] in disable_skills.__defaults__ or True
        )
        summary_parts.append(f"禁用了 {len(all_conflicting)} 个冲突 Skill")
    if not summary_parts:
        summary_parts.append("无操作需要执行")

    click.echo(f"  {' | '.join(summary_parts)}")
    click.echo(f"  Hippocampus 现在是你的主记忆引擎。🦛")
    click.echo()


# ── Non-interactive API ────────────────────────────────────

def auto_install(
    migrate: bool = True,
    disable_skills_flag: bool = True,
    config_dir: Optional[str] = None,
) -> dict:
    """Non-interactive installation for scripting.

    Returns:
        {migrated, disabled, warnings}
    """
    result = {
        "migrated": 0,
        "disabled": [],
        "warnings": [],
    }

    report = detect_environment(config_dir)

    if migrate and report["workspace"] and report["memory_files"]:
        from .config import load_config
        from .memory.long_term import LongTermMemory

        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yml"
        )
        config = load_config(config_path)
        hc = config["hippocampus"]
        ltm = LongTermMemory(
            data_dir=hc["data_dir"],
            collection_name=hc["long_term"]["collection_name"],
            top_k=hc["long_term"]["top_k"],
            embedding_backend=hc["long_term"]["embedding_backend"],
        )
        n, warnings = migrate_memories(report, ltm, hc["id_prefix"], archive=True)
        result["migrated"] = n
        result["warnings"].extend(warnings)

    if disable_skills_flag and report["workspace"]:
        results = disable_skills(report)
        for name, status in results.items():
            if status == "disabled":
                result["disabled"].append(name)
            elif status.startswith("error"):
                result["warnings"].append(f"{name}: {status}")

    return result
