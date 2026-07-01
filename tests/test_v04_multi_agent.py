"""
V0.4 Multi-Agent integration test.

Tests:
  1. MemoryEntry carries agent_id by default ("main")
  2. Short-term partitions per agent (independent windows)
  3. Short-term search scoped to agent
  4. Long-term agent_id in metadata
  5. Long-term search filtered by agent
  6. Working memory shared + agent-specific
  7. Compression is per-agent
  8. Store-level write/search/compress with agent_id
  9. Stats breakdown by agent
"""

import json
import tempfile
import shutil
from pathlib import Path

from hippocampus.config import Config, DEFAULT_CONFIG_YAML
from hippocampus.store import MemoryStore
from hippocampus.memory import MemoryEntry


def test_agent_id_default():
    """MemoryEntry defaults to agent_id='main'."""
    entry = MemoryEntry(content="hello")
    assert entry.agent_id == "main", f"Expected 'main', got {entry.agent_id}"
    d = entry.to_dict()
    assert d["agent_id"] == "main"
    # Round-trip
    e2 = MemoryEntry.from_dict(d)
    assert e2.agent_id == "main"


def test_short_term_partition():
    """Each agent has its own sliding window."""
    import tempfile
    from hippocampus.layers.short_term import ShortTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = ShortTermMemoryLayer(Path(tmpdir), window_size=3)

        # Write 5 entries for "main" — only last 3 remain.
        for i in range(5):
            layer.write(MemoryEntry(
                content=f"main-{i}",
                agent_id="main",
            ))

        assert layer.stats(agent_id="main")["count"] == 3

        # Write 2 entries for "coder" — both remain (below window).
        for i in range(2):
            layer.write(MemoryEntry(
                content=f"coder-{i}",
                agent_id="coder",
            ))

        assert layer.stats(agent_id="coder")["count"] == 2
        # Main still has 3.
        assert layer.stats(agent_id="main")["count"] == 3

        # Global stats.
        g = layer.stats()
        assert g["total_count"] == 5
        assert g["agent_count"] == 2


def test_short_term_search_scoped():
    """Search returns only the requested agent's entries."""
    import tempfile
    from hippocampus.layers.short_term import ShortTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = ShortTermMemoryLayer(Path(tmpdir), window_size=10)
        layer.write(MemoryEntry(content="main knows Python", agent_id="main"))
        layer.write(MemoryEntry(content="coder knows Rust", agent_id="coder"))

        # Scoped to main.
        r = layer.search("knows", agent_id="main")
        assert len(r) == 1
        assert "Python" in r[0].content

        # Scoped to coder.
        r = layer.search("knows", agent_id="coder")
        assert len(r) == 1
        assert "Rust" in r[0].content

        # Cross-agent (no filter).
        r = layer.search("knows", agent_id=None)
        assert len(r) == 2


def test_long_term_agent_metadata():
    """Long-term stores agent_id in sidecar and backend metadata."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(Path(tmpdir), backend="tfidf")
        eid = layer.write(MemoryEntry(
            content="Python is great",
            agent_id="coder",
        ))
        meta = layer.get_entry(eid)
        assert meta is not None
        assert meta["agent_id"] == "coder"

        # Batch write.
        batch = [
            MemoryEntry(content=f"batch-{i}", agent_id=f"agent-{i}")
            for i in range(3)
        ]
        ids = layer.write_batch(batch)
        assert len(ids) == 3
        for i in range(3):
            meta = layer.get_entry(ids[i])
            assert meta["agent_id"] == f"agent-{i}"


def test_long_term_search_agent_filter():
    """Long-term search can filter by agent_id."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            cross_agent_search=True,
        )
        layer.write(MemoryEntry(
            content="Python async programming",
            agent_id="coder",
        ))
        layer.write(MemoryEntry(
            content="writing design docs",
            agent_id="reviewer",
        ))

        # Filtered to coder.
        r = layer.search("Python", agent_id="coder")
        assert len(r) >= 1, f"Expected >=1 results for coder/Python, got {len(r)}"
        assert all("Python" in hit.content for hit in r)

        # Cross-agent.
        r = layer.search("Python", agent_id=None)
        assert len(r) >= 1, f"Expected >=1 results for cross-agent/Python, got {len(r)}"

        # No match for wrong agent — Python is in coder's entry,
        # and reviewer's entry has a near-zero score for "Python"
        # (TF-IDF scores all docs, even irrelevant ones, and
        # min_score=0.0 lets them through). Use a reasonable min.
        r2 = layer.search("Python", agent_id="reviewer")
        # The filter does correctly return only reviewer's docs,
        # but "writing design docs" gets a tiny score for "Python".
        # We verify the returned content is the reviewer's, not coder's.
        assert all(
            "design" in hit.content.lower() or hit.score < 0.1
            for hit in r2
        ), f"Got unexpected content from reviewer search: {[(h.content[:30], h.score) for h in r2]}"


def test_working_memory_shared():
    """Working memory: agent-specific + shared entries."""
    import tempfile
    from hippocampus.layers.working import WorkingMemoryLayer, SHARED_AGENT_ID

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = WorkingMemoryLayer(Path(tmpdir))

        # Shared entry.
        layer.write(MemoryEntry(
            content="Global config: timeout=30s",
            agent_id=SHARED_AGENT_ID,
        ))
        # Agent-specific.
        layer.write(MemoryEntry(
            content="I prefer Rust",
            agent_id="coder",
        ))

        # "main" searches, sees shared.
        r = layer.search("timeout", agent_id="main")
        assert len(r) == 1
        assert "timeout" in r[0].content

        # "coder" sees both their own and shared.
        r = layer.search("Rust", agent_id="coder")
        assert len(r) >= 1
        r = layer.search("timeout", agent_id="coder", include_shared=True)
        assert len(r) >= 1
        assert any("timeout" in hit.content for hit in r)

        # "reviewer" sees only shared.
        r = layer.search("timeout", agent_id="reviewer")
        assert len(r) == 1
        r = layer.search("Rust", agent_id="reviewer")
        assert len(r) == 0


def test_store_write_search_agent():
    """MemoryStore write/search with agent_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        config = Config.from_file(config_path)
        config.storage.data_dir = str(Path(tmpdir) / "data")
        store = MemoryStore(config)

        # Write for two agents.
        store.write("I write Python", agent_id="coder")
        store.write("I review code", agent_id="reviewer")

        # Search scoped.
        r = store.search("Python", agent_id="coder")
        assert len(r) >= 1, f"store: Expected >=1 results for coder/Python, got {len(r)}"

        r = store.search("Python", agent_id="reviewer")
        assert len(r) == 0, f"store: Expected 0 results for reviewer/Python, got {len(r)}"

        # Cross-agent search — "code" appears in reviewer's entry
        # and possibly scores weakly on coder's entry.
        r = store.search("code", agent_id=None)
        assert len(r) >= 1, f"store: Expected >=1 results for cross-agent/code, got {len(r)}: {[(h.content, h.score) for h in r]}"


def test_compress_per_agent():
    """Compression is per-agent: A's compression doesn't touch B's entries."""
    import tempfile
    from hippocampus.layers.short_term import ShortTermMemoryLayer
    from hippocampus.layers.long_term import LongTermMemoryLayer
    from hippocampus.compressor import Compressor

    with tempfile.TemporaryDirectory() as tmpdir:
        st = ShortTermMemoryLayer(Path(tmpdir), window_size=100)
        lt = LongTermMemoryLayer(Path(tmpdir), backend="tfidf")
        comp = Compressor(st, lt, batch_size=3, threshold=0.3)

        # Fill both agents' short-term.
        for i in range(5):
            st.write(MemoryEntry(content=f"main entry {i}", agent_id="main"))
        for i in range(3):
            st.write(MemoryEntry(content=f"coder entry {i}", agent_id="coder"))

        # Compress only "main".
        result = comp.compress(force=True, agent_id="main")
        assert result["compressed"] > 0

        # "coder" entries are still in short-term.
        coder_st = st.stats(agent_id="coder")
        assert coder_st["count"] == 3

        # Long-term should have main's compressed entries.
        assert lt.count > 0


def test_stats_agent_breakdown():
    """Stats show per-agent breakdown."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        config = Config.from_file(config_path)
        config.storage.data_dir = str(Path(tmpdir) / "data")
        store = MemoryStore(config)

        store.write("a", agent_id="main")
        store.write("b", agent_id="coder")
        store.write("c", agent_id="reviewer")

        # Global stats.
        s = store.stats()
        st = s["short_term"]
        assert st["agent_count"] == 3
        assert "main" in st["agents"]
        assert "coder" in st["agents"]
        assert "reviewer" in st["agents"]

        # Agent-scoped stats.
        s = store.stats(agent_id="coder")
        assert s["short_term"]["agent_id"] == "coder"
        assert s["short_term"]["count"] == 1


def test_backward_compatible_load():
    """Entries serialized before V0.4 (no agent_id) default to 'main'."""
    old_data = {
        "id": "test-123",
        "content": "old entry",
        "layer": "short_term",
        "source": "user",
        "metadata": {},
        "timestamp": "2026-01-01T00:00:00.000Z",
        "parent_id": None,
        # Note: no agent_id field
    }
    entry = MemoryEntry.from_dict(old_data)
    assert entry.agent_id == "main"
    assert entry.content == "old entry"


def test_cross_agent_search_disabled():
    """When cross_agent_search=False, unfiltered long-term search returns []."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            cross_agent_search=False,  # ← strict isolation
        )
        layer.write(MemoryEntry(
            content="secret data", agent_id="main",
        ))

        # Unfiltered search should return nothing.
        r = layer.search("secret", agent_id=None)
        assert len(r) == 0

        # Filtered search works.
        r = layer.search("secret", agent_id="main")
        assert len(r) >= 1


# ── V0.4.1: Long-term isolation mode tests ─────────────────────────

def test_long_term_isolation_separate_backends():
    """Isolation mode: each agent gets its own TF-IDF store."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            long_term_isolation=True,
        )
        layer.write(MemoryEntry(
            content="Python async patterns", agent_id="coder",
        ))
        layer.write(MemoryEntry(
            content="review guidelines", agent_id="reviewer",
        ))

        # Each agent's backend has exactly 1 entry.
        coder_stats = layer.stats(agent_id="coder")
        assert coder_stats["count"] == 1, f"coder count={coder_stats['count']}"

        reviewer_stats = layer.stats(agent_id="reviewer")
        assert reviewer_stats["count"] == 1

        # Global stats = both agents.
        global_stats = layer.stats()
        assert global_stats["count"] == 2
        assert global_stats["isolation"] is True


def test_long_term_isolation_search_requires_agent_id():
    """Isolation mode: search without agent_id raises ValueError."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            long_term_isolation=True,
        )
        layer.write(MemoryEntry(
            content="secret", agent_id="main",
        ))

        # Search with agent_id works.
        r = layer.search("secret", agent_id="main")
        assert len(r) >= 1

        # Search without agent_id raises.
        try:
            layer.search("secret", agent_id=None)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "long_term_isolation" in str(e)


def test_long_term_isolation_no_cross_contamination():
    """Isolation mode: agent A cannot see agent B's data."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            long_term_isolation=True,
        )
        layer.write(MemoryEntry(
            content="Rust ownership rules", agent_id="coder",
        ))
        layer.write(MemoryEntry(
            content="CSS layout tips", agent_id="designer",
        ))

        # "coder" searches for "Rust" — found.
        r = layer.search("Rust", agent_id="coder")
        assert len(r) >= 1

        # "designer" searches for "Rust" — their own entries may get
        # score 0.0 (no overlap), which passes min_score=0.0.  But the
        # returned content should NOT include anything about Rust.
        r = layer.search("Rust", agent_id="designer")
        # No entry should contain "Rust" in designer's isolated pool.
        assert not any("Rust" in h.content for h in r), (
            f"designer should not see Rust content, got: "
            f"{[h.content[:40] for h in r]}"
        )

        # "designer" searches for "CSS" — found.
        r = layer.search("CSS", agent_id="designer")
        assert len(r) >= 1


def test_long_term_isolation_clear_per_agent():
    """Isolation mode: clear one agent without affecting others."""
    import tempfile
    from hippocampus.layers.long_term import LongTermMemoryLayer

    with tempfile.TemporaryDirectory() as tmpdir:
        layer = LongTermMemoryLayer(
            Path(tmpdir), backend="tfidf",
            long_term_isolation=True,
        )
        layer.write(MemoryEntry(content="a", agent_id="main"))
        layer.write(MemoryEntry(content="b", agent_id="coder"))

        # Clear only "coder".
        layer.clear(agent_id="coder")

        # "main" still has its entry.
        assert layer.stats(agent_id="main")["count"] == 1
        # "coder" is gone.
        assert layer.stats(agent_id="coder")["count"] == 0
        # Global count = 1.
        assert layer.count == 1


def test_long_term_isolation_store_integration():
    """Full store integration with long_term_isolation=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yml"
        config_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        config = Config.from_file(config_path)
        config.storage.data_dir = str(Path(tmpdir) / "data")
        config.agent.long_term_isolation = True
        store = MemoryStore(config)

        # Write short-term for two agents.
        store.write("coder knows Rust", agent_id="coder", layer="short_term")
        store.write("reviewer knows docs", agent_id="reviewer", layer="short_term")

        # Write direct to long-term.
        store.write("async Python", agent_id="coder", layer="long_term")
        store.write("PR template", agent_id="reviewer", layer="long_term")

        # Search long-term scoped to coder.
        r = store.search("Python", agent_id="coder", layers=["long_term"])
        assert len(r) >= 1, f"coder should find Python, got {len(r)}"

        # Search long-term scoped to reviewer — their entry ("PR template")
        # scores 0.0 for "Python", which passes min_score=0.0.  Key is that
        # returned content does NOT come from coder, proving isolation.
        r = store.search("Python", agent_id="reviewer", layers=["long_term"])
        assert not any(
            "Python" in h.content for h in r
        ), f"reviewer in isolation should not see Python content, got {[h.content[:40] for h in r]}"

        # Cross-agent long-term search should raise.
        try:
            store.search("anything", agent_id=None, layers=["long_term"])
            assert False, "Expected ValueError for cross-agent in isolation"
        except ValueError:
            pass


if __name__ == "__main__":
    tests = [
        ("agent_id default", test_agent_id_default),
        ("short-term partition", test_short_term_partition),
        ("short-term search scoped", test_short_term_search_scoped),
        ("long-term agent metadata", test_long_term_agent_metadata),
        ("long-term search filter", test_long_term_search_agent_filter),
        ("working memory shared", test_working_memory_shared),
        ("store write/search agent", test_store_write_search_agent),
        ("compress per agent", test_compress_per_agent),
        ("stats agent breakdown", test_stats_agent_breakdown),
        ("backward compatible load", test_backward_compatible_load),
        ("cross_agent_search disabled", test_cross_agent_search_disabled),
        # ── V0.4.1: long-term isolation ──────────────────────────
        ("LT isolation: separate backends", test_long_term_isolation_separate_backends),
        ("LT isolation: search requires agent_id", test_long_term_isolation_search_requires_agent_id),
        ("LT isolation: no cross contamination", test_long_term_isolation_no_cross_contamination),
        ("LT isolation: clear per agent", test_long_term_isolation_clear_per_agent),
        ("LT isolation: store integration", test_long_term_isolation_store_integration),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    if failed > 0:
        exit(1)
