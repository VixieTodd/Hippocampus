"""
Hippocampus CLI — terminal commands for the 3-layer memory system.

Usage:
  hippo write "content" [--source user|agent|system] [--layer st|lt|wk]
  hippo search "query" [--top 5] [--layers st,lt,wk]
  hippo stats
  hippo compress [--force]
  hippo trace <id>
  hippo export [--format json] [--output file.json]
"""

from __future__ import annotations

import json
from pathlib import Path

import click

DEFAULT_CONFIG_PATH = Path.cwd() / "config.yml"


def _resolve_config_path() -> Path:
    """Find config.yml: check cwd first, then the package root."""
    cwd_config = Path.cwd() / "config.yml"
    if cwd_config.exists():
        return cwd_config
    package_config = Path(__file__).parent.parent / "config.yml"
    if package_config.exists():
        return package_config
    return cwd_config  # Let the caller handle creation.


@click.group()
@click.option(
    "-c", "--config", "config_path",
    default=None,
    help="Path to config.yml",
    type=click.Path(exists=False),
)
@click.pass_context
def cli(ctx, config_path) -> None:
    """Hippocampus — A bionic memory system for AI."""
    ctx.ensure_object(dict)

    # Lazy import so --help is fast even without deps.
    from hippocampus.config import Config
    from hippocampus.store import MemoryStore

    path = Path(config_path) if config_path else _resolve_config_path()
    config = Config.from_file(path)
    store = MemoryStore(config)

    ctx.obj["config"] = config
    ctx.obj["store"] = store


# ── Bilingual strings ────────────────────────────────────────────────────

_L = {
    "zh": {
        "banner_title":         "Hippocampus 安装向导",
        "banner_sub":           "三层仿生记忆系统 — 为 AI Agent 提供持久记忆",
        "step_env":             "[1/3] 环境检查",
        "step_skill":           "[2/3] Skill 冲突检查",
        "step_migrate":         "[3/3] 导入数据",
        "py_ok":                "Python {ver} 符合要求",
        "py_too_old":           "Python {ver} 版本过低（需要 >=3.10）",
        "py_ask":               "Python 3.10+ 是运行 Hippocampus 的必要条件。",
        "py_option_open":       "[1] 打开下载页面（手动安装后重试）",
        "py_option_exit":       "[2] 退出安装程序",
        "py_opening":           "正在打开 Python 下载页面...",
        "deps_checking":        "正在检查依赖...",
        "deps_ok":              "所有依赖就绪",
        "deps_missing":         "检测到缺失依赖",
        "deps_ask":             "检测到不支持的依赖",
        "deps_auto":            "[1] 由程序自动安装并继续",
        "deps_manual":          "[2] 手动安装并退出安装程序",
        "deps_installing":      "正在自动安装...",
        "deps_failed":          "依赖安装失败",
        "found_workspace":      "找到 OpenClaw 工作区",
        "not_found_workspace":  "未检测到 OpenClaw 工作区（暂仅支持 OpenClaw），请确认已安装并已设置工作目录。",
        "found_memory":         "检测到 {n} 个已有记忆文件（约 {lines} 行有效内容）",
        "no_memory":            "未找到现有记忆文件，已跳过",
        "ask_migrate":          "是否将现有记忆导入 Hippocampus？",
        "imported":             "导入了 {n} 条记忆",
        "import_failed":        "导入失败",
        "skipped":              "已跳过",
        "no_conflict":          "未发现冲突 Skill",
        "ask_disable":          "是否禁用这些 Skill？",
        "disabled":             "已禁用",
        "done":                 "安装完成",
        "summary":              "导入了 {n} 条记忆 | 禁用了 {m} 个冲突 Skill",
        "press_enter":          "按 Enter 退出...",
    },
    "en": {
        "banner_title":         "Hippocampus Setup Wizard",
        "banner_sub":           "A bionic memory system for AI — layered architecture with vector retrieval",
        "step_env":             "[1/3] Environment check",
        "step_skill":           "[2/3] Skill conflict check",
        "step_migrate":         "[3/3] Import data",
        "py_ok":                "Python {ver} meets requirements",
        "py_too_old":           "Python {ver} is too old (>=3.10 required)",
        "py_ask":               "Python 3.10+ is required to run Hippocampus.",
        "py_option_open":       "[1] Open download page (install manually, then re-run)",
        "py_option_exit":       "[2] Exit installer",
        "py_opening":           "Opening Python download page...",
        "deps_checking":        "Checking dependencies...",
        "deps_ok":              "All dependencies ready",
        "deps_missing":         "Missing dependencies detected",
        "deps_ask":             "Unsupported dependencies detected",
        "deps_auto":            "[1] Auto-install and continue",
        "deps_manual":          "[2] Install manually and exit installer",
        "deps_installing":      "Auto-installing...",
        "deps_failed":          "Dependency installation failed",
        "found_workspace":      "Found OpenClaw workspace",
        "not_found_workspace":  "No OpenClaw workspace detected (currently only OpenClaw is supported). Please confirm OpenClaw is installed and a workspace is configured.",
        "found_memory":         "Found {n} existing memory file(s) (~{lines} meaningful lines)",
        "no_memory":            "No existing memory files found, skipped",
        "ask_migrate":          "Import existing memories into Hippocampus?",
        "imported":             "Imported {n} memories",
        "import_failed":        "Import failed",
        "skipped":              "Skipped",
        "no_conflict":          "No conflicting skills found",
        "ask_disable":          "Disable these skills?",
        "disabled":             "Disabled",
        "done":                 "Setup complete",
        "summary":              "Imported {n} memories | Disabled {m} conflicting skill(s)",
        "press_enter":          "Press Enter to exit...",
    },
}


# ── install (setup wizard) ──────────────────────────────────────────────


def _ask_lang() -> tuple[str, dict]:
    """Ask user to pick language."""
    print()
    print("Hippocampus " + chr(0x1F9E0) + " Installer")
    print()
    print("Language Selection:")
    print()
    print("  [1] Simplified Chinese " + chr(0x7B80) + chr(0x4F53) + chr(0x4E2D) + chr(0x6587))
    print("  [2] English")
    print()
    while True:
        try:
            c = input("  Select [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ("zh", _L["zh"])
        if c in ("", "1"):
            return ("zh", _L["zh"])
        if c == "2":
            return ("en", _L["en"])


def _ask_yn(L, prompt_text: str, default_yes: bool = True) -> bool:
    prompt = prompt_text + " [Y/n] " if default_yes else " [y/N] "
    try:
        answer = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default_yes
    if default_yes:
        return answer in ("", "y", "yes", "是")
    return answer in ("y", "yes", "是")


def _find_workspace() -> Path | None:
    """Detect a project workspace root (has MEMORY.md or AGENTS.md)."""
    candidates = [
        Path.home() / ".openclaw" / "workspace",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "MEMORY.md").exists() or (p / "AGENTS.md").exists() or (p / ".git").is_dir():
            return p.resolve()
    return None


def _find_memory_files(workspace: Path) -> list[Path]:
    """Find memory-related markdown files in the workspace."""
    files = []
    for name in ("MEMORY.md", "notes.md"):
        f = workspace / name
        if f.exists():
            files.append(f)
    memory_dir = workspace / "memory"
    if memory_dir.is_dir():
        files.extend(sorted(memory_dir.glob("*.md")))
    return files


def _count_lines(file: Path) -> int:
    count = 0
    try:
        for line in file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("<!--"):
                count += 1
    except Exception:
        pass
    return count


def _find_all_skills(workspace: Path) -> list[tuple[str, str | None]]:
    """Scan workspace/skills/ for SKILL.md files and read their names."""
    skill_dir = workspace / "skills"
    if not skill_dir.is_dir():
        return []
    found = []
    for child in sorted(skill_dir.iterdir()):
        skill_file = child / "SKILL.md"
        if skill_file.exists():
            name = None
            try:
                for line in skill_file.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"').strip("'")
                        break
            except Exception:
                pass
            found.append((child.name, name))
    return found


@cli.command()
def install() -> None:
    """Setup wizard: detect environment, migrate memories, handle skill conflicts."""
    import sys
    import webbrowser
    from hippocampus.deps import check_python, check_all, install_missing

    lang, L = _ask_lang()

    check_mark = chr(0x2713)
    cross_mark = "x"
    divider = chr(0x2501) * 44

    print()
    print(divider)
    print(" " + chr(0x1F9E0) + "  " + L["banner_title"])
    print(divider)
    print()

    # ── [1/3] Environment check ─────────────────────────────────────────┐
    print(L["step_env"] + "...")

    # Python version
    py_ok, py_ver = check_python()
    if py_ok:
        print(f" {check_mark} {L['py_ok'].format(ver=py_ver)}")
    else:
        print(f" {cross_mark} {L['py_too_old'].format(ver=py_ver)}")
        print()
        print(" " + L["py_ask"])
        print()
        print(" " + L["py_option_open"])
        print(" " + L["py_option_exit"])
        print()
        while True:
            try:
                c = input("  Select [1/2]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                c = "2"
            if c == "1":
                print(" " + L["py_opening"])
                webbrowser.open("https://www.python.org/downloads/")
                input("  " + L["press_enter"])
                sys.exit(0)
            elif c in ("", "2"):
                input("  " + L["press_enter"])
                sys.exit(1)

    # Dependencies
    statuses = check_all()
    missing_deps = [s for s in statuses if not s.ok]
    for s in statuses:
        if s.ok:
            print(f" {check_mark} {s.label:30s}  v{s.version}")
        else:
            tag = "missing" if not s.installed else f"too old ({s.version})"
            print(f" {cross_mark} {s.label:30s}  [{tag}]")
    if missing_deps:
        print()
        print(" " + L["deps_ask"])
        print()
        print(" " + L["deps_auto"])
        print(" " + L["deps_manual"])
        print()
        while True:
            try:
                c = input("  Select [1/2]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                c = "2"
            if c == "1":
                print(" " + L["deps_installing"])
                print()
                ok_all = install_missing(dry_run=False)
                if not ok_all:
                    print(f" {cross_mark} {L['deps_failed']}")
                    input("  " + L["press_enter"])
                    sys.exit(1)
                break
            elif c in ("", "2"):
                print()
                print("  " + ("请手动安装后重试：" if lang == "zh" else "Install manually and re-run:"))
                for s in missing_deps:
                    pip_name = ""
                    for imp, pn, _, _ in [
                        ("click", "click", "8.0", ""),
                        ("yaml", "pyyaml", "6.0", ""),
                        ("chromadb", "chromadb", "0.4.0", ""),
                        ("sentence_transformers", "sentence-transformers", "2.2.0", ""),
                    ]:
                        if imp == s.name:
                            pip_name = pn
                            break
                    print(f"    pip install {pip_name or s.name}")
                print()
                input("  " + L["press_enter"])
                sys.exit(1)
    else:
        print(f" {check_mark} {L['deps_ok']}")

    # Workspace
    workspace = _find_workspace()
    if workspace:
        print(f" {check_mark} {L['found_workspace']}: {workspace}")
        config_path = workspace / "hippocampus" / "config.yml"
        if not config_path.parent.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(f" {cross_mark} {L['not_found_workspace']}")
        print()
        input("  " + L["press_enter"])
        sys.exit(1)

    print()

    # ── [2/3] Skill conflict check ──────────────────────────────────
    print(L["step_skill"])
    all_skills = _find_all_skills(workspace)

    if all_skills:
        print(f"  Found {len(all_skills)} installed skill(s) that may overlap.")
        for dir_name, display_name in all_skills:
            label = display_name or dir_name
            print(f" {chr(0x26A0)} {label}")
        print(" " + L["ask_disable"])
        if _ask_yn(L, "ask_disable", default_yes=True):
            disabled = []
            for dir_name, _ in all_skills:
                skill_file = workspace / "skills" / dir_name / "SKILL.md"
                disabled_file = skill_file.with_suffix(".md.disabled")
                try:
                    if skill_file.exists() and not disabled_file.exists():
                        skill_file.rename(disabled_file)
                        disabled.append(dir_name)
                    elif disabled_file.exists():
                        disabled.append(dir_name)
                except Exception:
                    pass
            if disabled:
                for name in disabled:
                    print(f" {check_mark} {L['disabled']}: {name}")
            else:
                print(f" {L['skipped']}.")
        else:
            print(f" {L['skipped']}.")
    else:
        print(f" {L['no_conflict']}.")

    print()

    # ── [3/3] Import data ───────────────────────────────────────────
    print(L["step_migrate"])

    memory_files = _find_memory_files(workspace)

    if memory_files:
        total_lines = sum(_count_lines(f) for f in memory_files)
        print(" " + L["found_memory"].format(n=len(memory_files), lines=total_lines))
        print(" " + L["ask_migrate"])
        if _ask_yn(L, "ask_migrate", default_yes=True):
            from hippocampus.config import Config, DEFAULT_CONFIG_YAML
            from hippocampus.store import MemoryStore

            config_path.parent.mkdir(parents=True, exist_ok=True)
            if not config_path.exists():
                config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")

            import_count = 0
            try:
                _cfg = Config.from_file(config_path)
                _store = MemoryStore(_cfg)
                for mf in memory_files:
                    content = mf.read_text(encoding="utf-8")
                    for line in content.splitlines():
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#") and not stripped.startswith("<!--"):
                            _store.write(
                                stripped,
                                source="system",
                                layer="short_term",
                                metadata={"source_file": str(mf)},
                            )
                            import_count += 1
                print(f" {check_mark} {L['imported'].format(n=import_count)}")
            except Exception as e:
                print(f" {cross_mark} {L['import_failed']}: {e}")
        else:
            print(f" {L['skipped']}.")
    else:
        print(f" {L['no_memory']}.")

    print()

    # ── Done ────────────────────────────────────────────────────────
    print(divider)
    print(f"{chr(0x2713)} {L['done']}")
    imported_n = locals().get("import_count", 0)
    disabled_n = len(locals().get("disabled", []))
    print(f" {L['summary'].format(n=imported_n, m=disabled_n)}")
    print()


# ── write ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("content")
@click.option("--source", default="user", help="Source: user, agent, system")
@click.option(
    "--layer", default="short_term",
    help="Target layer: short_term (default), working, long_term",
)
@click.pass_context
def write(ctx, content, source, layer) -> None:
    """Write a memory entry."""
    store = ctx.obj["store"]
    entry_id = store.write(content, source=source, layer=layer)
    click.echo(json.dumps({"status": "ok", "id": entry_id, "layer": layer}))


# ── search ───────────────────────────────────────────────────────────────


@cli.command()
@click.argument("query")
@click.option("--top", default=5, help="Max results per layer")
@click.option(
    "--layers", default="short_term,long_term,working",
    help="Comma-separated layers to search",
)
@click.pass_context
def search(ctx, query, top, layers) -> None:
    """Search memories across layers."""
    store = ctx.obj["store"]
    layer_list = [l.strip() for l in layers.split(",") if l.strip()]
    results = store.search(query, top_k=top, layers=layer_list)

    if not results:
        click.echo(json.dumps({"status": "ok", "results": []}))
        return

    output = []
    for r in results:
        output.append({
            "id": r.entry_id,
            "content": (
                r.content[:300] + ("..." if len(r.content) > 300 else "")
            ),
            "score": r.score,
            "layer": r.layer,
            "timestamp": r.timestamp,
        })

    click.echo(
        json.dumps({"status": "ok", "results": output},
                   ensure_ascii=False, indent=2)
    )


# ── stats ────────────────────────────────────────────────────────────────


@cli.command()
@click.pass_context
def stats(ctx) -> None:
    """Show memory statistics for all layers."""
    store = ctx.obj["store"]
    s = store.stats()
    click.echo(json.dumps(s, ensure_ascii=False, indent=2))


# ── compress ─────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--force", is_flag=True, default=False,
    help="Force compression regardless of threshold",
)
@click.pass_context
def compress(ctx, force) -> None:
    """Trigger memory compression (short-term -> long-term)."""
    store = ctx.obj["store"]
    result = store.compress(force=force)
    click.echo(json.dumps(result, ensure_ascii=False, indent=2))


# ── trace ────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("entry_id")
@click.pass_context
def trace(ctx, entry_id) -> None:
    """Show full trace of a single memory entry."""
    store = ctx.obj["store"]
    result = store.trace(entry_id)

    if result["entry"] is None:
        click.echo(json.dumps({"status": "not_found", "id": entry_id}))
        return

    output = {
        "status": "ok",
        "id": entry_id,
        "layer": result["layer"],
        "entry": result["entry"],
        "trace_count": result["trace_count"],
        "traces": result["traces"],
    }
    click.echo(json.dumps(output, ensure_ascii=False, indent=2))


# ── export ────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--format", "fmt", default="json", help="Export format (json)")
@click.option("--output", "-o", default=None, help="Output file path (stdout if omitted)")
@click.pass_context
def export(ctx, fmt, output) -> None:
    """Export all memories."""
    store = ctx.obj["store"]
    data = store.export(format=fmt)

    text = json.dumps(data, ensure_ascii=False, indent=2)

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        click.echo(
            json.dumps({
                "status": "ok",
                "file": str(path.resolve()),
                "bytes": len(text),
            })
        )
    else:
        click.echo(text)


# ── doctor ──────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "--install", is_flag=True, default=False,
    help="Auto-install missing dependencies (skip confirmation)",
)
@click.option(
    "--dry-run", is_flag=True, default=False,
    help="Check only, do not install",
)
def doctor(install_flag, dry_run) -> None:
    """Check runtime dependencies and optionally install missing ones."""
    import sys
    from hippocampus.deps import check_all, check_python, install_missing

    pass_icon = chr(0x2705)
    fail_icon = chr(0x274C)
    warn_icon = chr(0x26A0) + chr(0xFE0F)

    py_ok, py_ver = check_python()
    icon = pass_icon if py_ok else fail_icon
    print(f"Python {py_ver}  {icon}  (>=3.10 required)")
    if not py_ok:
        print(f"{warn_icon} Python version too old, upgrade to 3.10+.")
        return

    statuses = check_all()
    ok_count = sum(1 for s in statuses if s.ok)
    total = len(statuses)

    print(f"\nDependency check ({ok_count}/{total} ok):")
    for s in statuses:
        if s.ok:
            print(f"  {pass_icon} {s.label:30s}  {s.name:22s}  v{s.version}")
        else:
            tag = "missing" if not s.installed else f"too old ({s.version})"
            print(f"  {fail_icon} {s.label:30s}  {s.name:22s}  [{tag}]")

    if ok_count < total:
        if install_flag:
            print()
            install_missing(dry_run=False)
        elif not dry_run:
            print("\nRun `hippo doctor --install` to auto-install missing dependencies.")
        else:
            print("\nInstall the missing packages above.")
    else:
        print(f"\n{pass_icon} All dependencies ready.")


# ── Entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Entry point for `hippo` console_scripts."""
    cli(obj={})


if __name__ == "__main__":
    main()
