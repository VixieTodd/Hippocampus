"""
Hippocampus CLI — command-line interface for the bionic memory system.

Commands:
  hippo install                Guided setup wizard
  hippo write <content>        Write a memory entry
  hippo search <query>         Three-layer semantic search
  hippo stats                  Memory statistics
  hippo compress [--force]     Trigger compression (STM → LTM)
  hippo trace <id>             Full trace of a single entry
  hippo export [--format]      Backup all memories
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import click
import yaml

from .config import load_config
from .memory.base import MemoryEntry
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .memory.working import WorkingMemory
from .compressor import Compressor
from .installer import run_install_wizard, auto_install


class Hippocampus:
    """Main Hippocampus application — holds all three layers."""

    def __init__(self, config_path: str = "config.yml"):
        self.config = load_config(config_path)
        hc = self.config["hippocampus"]
        self.data_dir = hc["data_dir"]
        self.id_prefix = hc["id_prefix"]

        # Initialize layers
        self.short_term = ShortTermMemory(
            data_dir=self.data_dir,
            window_size=hc["short_term"]["window_size"],
            compression_threshold=hc["short_term"]["compression_threshold"],
        )
        self.long_term = LongTermMemory(
            data_dir=self.data_dir,
            collection_name=hc["long_term"]["collection_name"],
            top_k=hc["long_term"]["top_k"],
            embedding_backend=hc["long_term"]["embedding_backend"],
        )
        self.working = WorkingMemory(
            data_dir=self.data_dir,
            filename=hc["working"]["file"],
        )
        self.compressor = Compressor(
            short_term=self.short_term,
            long_term=self.long_term,
            id_prefix=self.id_prefix,
        )

    def write(self, content: str, source: str = "cli",
              layer: str = "short_term") -> MemoryEntry:
        """Write a memory entry. Defaults to short-term."""
        entry = MemoryEntry.create(
            content=content,
            source=source,
            layer=layer,
            id_prefix=self.id_prefix,
        )

        if layer == "working":
            self.working.add(entry)
        elif layer == "long_term":
            self.long_term.add(entry)
        else:
            self.short_term.add(entry)
            # Auto-compress if needed
            if self.short_term.needs_compression():
                n = self.compressor.compress(force=False)
                if n > 0:
                    click.echo(f"  ⚡ Auto-compressed {n} entries to long-term memory")

        return entry

    def search(self, query: str, top_k: int = 5) -> dict:
        """Search across all three layers."""
        return {
            "short_term": self.short_term.search(query, top_k),
            "long_term": self.long_term.search(query, top_k),
            "working": self.working.search(query, top_k),
        }

    def stats(self) -> dict:
        """Get statistics for all layers."""
        return {
            "short_term": self.short_term.stats(),
            "long_term": self.long_term.stats(),
            "working": self.working.stats(),
            "total_entries": (
                self.short_term.count()
                + self.long_term.count()
                + self.working.count()
            ),
        }

    def trace(self, entry_id: str) -> Optional[dict]:
        """Trace a single entry across all layers."""
        for layer_name, layer in [
            ("short_term", self.short_term),
            ("long_term", self.long_term),
            ("working", self.working),
        ]:
            entry = layer.find_by_id(entry_id)
            if entry:
                return {
                    "found_in": layer_name,
                    "entry": entry.to_dict(),
                }
        return None

    def export(self, format: str = "json") -> str:
        """Export all memories."""
        all_entries = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "version": "0.1.0",
            "short_term": [e.to_dict() for e in self.short_term.get_all()],
            "long_term": [e.to_dict() for e in self.long_term.get_all()],
            "working": [e.to_dict() for e in self.working.get_all()],
        }
        if format == "json":
            return json.dumps(all_entries, ensure_ascii=False, indent=2)
        elif format == "jsonl":
            lines = []
            for layer in ["short_term", "long_term", "working"]:
                for entry in all_entries[layer]:
                    lines.append(json.dumps(entry, ensure_ascii=False))
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported format: {format}")


# ── CLI Definition ──────────────────────────────────────────

def _find_config():
    """Find config.yml in current directory or default locations."""
    candidates = [
        "config.yml",
        "hippocampus.yml",
        os.path.expanduser("~/.hippocampus/config.yml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    # If not found, create default in current dir
    return "config.yml"


def _format_entry_summary(entry: MemoryEntry) -> str:
    """Format a single entry for display."""
    ts = entry.timestamp[:19].replace("T", " ")
    return (
        f"  [{click.style(entry.id[:20] + '…', fg='cyan')}] "
        f"{click.style(ts, fg='bright_black')} "
        f"{entry.content[:100]}"
    )


@click.group()
@click.option("--config", "-c", "config_path", default=None,
              help="Path to config.yml")
@click.pass_context
def cli(ctx, config_path):
    """🦛 Hippocampus — AI Bionic Memory System V0.1
    
    Three-layer memory: Short-Term → Long-Term → Working.
    """
    ctx.ensure_object(dict)
    path = config_path or _find_config()
    ctx.obj["hippo"] = Hippocampus(path)


@cli.command()
@click.argument("content")
@click.option("--layer", "-l", default="short_term",
              type=click.Choice(["short_term", "long_term", "working"]),
              help="Target memory layer")
@click.option("--source", "-s", default="cli", help="Source tag")
@click.pass_context
def write(ctx, content, layer, source):
    """Write a new memory entry."""
    hippo: Hippocampus = ctx.obj["hippo"]
    entry = hippo.write(content, source=source, layer=layer)
    click.echo(f"[OK] Written to {click.style(layer, fg='green')}")
    click.echo(f"  ID: {click.style(entry.id, fg='cyan')}")
    click.echo(f"  Content: {entry.content[:120]}")


@cli.command()
@click.argument("query")
@click.option("--top", "-k", "top_k", default=5, help="Results per layer")
@click.pass_context
def search(ctx, query, top_k):
    """Search across all three memory layers."""
    hippo: Hippocampus = ctx.obj["hippo"]
    results = hippo.search(query, top_k)

    for layer_name, label, color in [
        ("short_term", "Short-Term Memory", "yellow"),
        ("long_term", "Long-Term Memory", "blue"),
        ("working", "Working Memory", "magenta"),
    ]:
        entries = results[layer_name]
        header = f"── {label} ({len(entries)} hits) "
        click.echo(click.style(header.ljust(50, "─"), fg=color))
        if entries:
            for entry in entries:
                click.echo(_format_entry_summary(entry))
        else:
            click.echo("  (no results)")
        click.echo()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show memory statistics for all layers."""
    hippo: Hippocampus = ctx.obj["hippo"]
    s = hippo.stats()

    click.echo(click.style("🦛 Hippocampus Memory Stats", bold=True))
    click.echo(f"  Total entries: {click.style(str(s['total_entries']), fg='green', bold=True)}")
    click.echo()

    for layer_key, label, color in [
        ("short_term", "📝 Short-Term Memory", "yellow"),
        ("long_term", "🧠 Long-Term Memory", "blue"),
        ("working", "⚙️  Working Memory", "magenta"),
    ]:
        ls = s[layer_key]
        click.echo(click.style(f"  {label}", fg=color, bold=True))
        click.echo(f"    Entries: {ls['count']}")
        if layer_key == "short_term":
            click.echo(f"    Window: {ls['window_size']} | Threshold: {ls['compression_threshold']}")
            flag = click.style("YES", fg="red") if ls["needs_compression"] else click.style("no", fg="green")
            click.echo(f"    Needs compression: {flag}")
        elif layer_key == "long_term":
            click.echo(f"    Top-K: {ls['top_k']} | Backend: {ls['embedding_backend']}")
        click.echo()


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Compress ALL short-term entries")
@click.pass_context
def compress(ctx, force):
    """Manually trigger compression (Short-Term → Long-Term)."""
    hippo: Hippocampus = ctx.obj["hippo"]
    before = hippo.short_term.count()
    n = hippo.compressor.compress(force=force)
    after = hippo.short_term.count()

    if n == 0:
        click.echo("Nothing to compress.")
    else:
        click.echo(f"[OK] Compressed {click.style(str(n), fg='green', bold=True)} entries")
        click.echo(f"  Short-term: {before} → {after}")
        click.echo(f"  Long-term: +{n} entries")


@cli.command()
@click.argument("entry_id")
@click.pass_context
def trace(ctx, entry_id):
    """Trace a single memory entry by ID."""
    hippo: Hippocampus = ctx.obj["hippo"]
    result = hippo.trace(entry_id)

    if result is None:
        click.echo(click.style(f"[X] Entry not found: {entry_id}", fg="red"))
        return

    entry = result["entry"]
    click.echo(click.style("🔍 Memory Trace", bold=True))
    click.echo(f"  ID:        {click.style(entry['id'], fg='cyan')}")
    click.echo(f"  Layer:     {click.style(result['found_in'], fg='green')}")
    click.echo(f"  Timestamp: {entry['timestamp']}")
    click.echo(f"  Source:    {entry['source']}")
    if entry.get("metadata"):
        click.echo(f"  Metadata:  {json.dumps(entry['metadata'], ensure_ascii=False)}")
    click.echo(f"  Content:")
    click.echo(f"    {entry['content']}")


@cli.command()
@click.option("--yes", "-y", "non_interactive", is_flag=True,
              help="Non-interactive: migrate all and disable conflicts")
@click.pass_context
def install(ctx, non_interactive):
    """Guided installation wizard — migrate memories & handle skill conflicts."""
    if non_interactive:
        result = auto_install(migrate=True, disable_skills_flag=True)
        click.echo(f"[OK] Migrated {result['migrated']} entries, "
                   f"disabled {len(result['disabled'])} skills")
        for w in result.get("warnings", []):
            click.echo(f"  ⚠ {w}")
    else:
        run_install_wizard()


@cli.command()
@click.option("--format", "-f", "fmt", default="json",
              type=click.Choice(["json", "jsonl"]),
              help="Export format")
@click.option("--output", "-o", default=None, help="Output file path")
@click.pass_context
def export(ctx, fmt, output):
    """Export all memories (full backup)."""
    hippo: Hippocampus = ctx.obj["hippo"]
    data = hippo.export(format=fmt)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(data)
        size = len(data.encode("utf-8"))
        click.echo(f"[OK] Exported to {click.style(output, fg='green')} ({size:,} bytes)")
    else:
        click.echo(data)


def main():
    """Entry point for console script."""
    # Force UTF-8 on Windows to avoid GBK encoding issues
    import sys
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cli(auto_envvar_prefix="HIPPO")


if __name__ == "__main__":
    main()
