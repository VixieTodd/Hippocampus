"""
Basic end-to-end tests for Hippocampus.
Run from project root: PYTHONPATH=. python tests/test_basic.py
"""

import json
import os
import sys
import tempfile
import unittest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hippocampus.config import load_config, deep_merge
from hippocampus.utils import generate_id, utc_now_iso
from hippocampus.memory.base import MemoryEntry
from hippocampus.memory.short_term import ShortTermMemory
from hippocampus.memory.long_term import LongTermMemory
from hippocampus.memory.working import WorkingMemory
from hippocampus.compressor import Compressor
from hippocampus.storage.vector_store import TFIDFVectorStore


class TestUtils(unittest.TestCase):
    def test_generate_id(self):
        eid = generate_id("hippo")
        self.assertTrue(eid.startswith("hippo_"))
        parts = eid.split("_")
        self.assertEqual(len(parts), 3)

    def test_utc_now_iso(self):
        ts = utc_now_iso()
        self.assertIn("T", ts)
        self.assertIn("+", ts)  # timezone

    def test_memory_entry_create(self):
        entry = MemoryEntry.create("test content", source="test")
        self.assertEqual(entry.content, "test content")
        self.assertEqual(entry.source, "test")
        self.assertEqual(entry.layer, "short_term")
        self.assertIsNotNone(entry.id)
        self.assertIsNotNone(entry.timestamp)

    def test_memory_entry_serialize(self):
        entry = MemoryEntry.create("hello world")
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        self.assertEqual(restored.id, entry.id)
        self.assertEqual(restored.content, entry.content)


class TestShortTermMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stm = ShortTermMemory(self.tmpdir, window_size=5, compression_threshold=3)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_count(self):
        self.assertEqual(self.stm.count(), 0)
        entry = MemoryEntry.create("test")
        self.stm.add(entry)
        self.assertEqual(self.stm.count(), 1)

    def test_search(self):
        self.stm.add(MemoryEntry.create("小狐喜欢喝桂馥兰香"))
        self.stm.add(MemoryEntry.create("晨是AI助手"))
        results = self.stm.search("小狐")
        self.assertEqual(len(results), 1)
        self.assertIn("桂馥兰香", results[0].content)

    def test_needs_compression(self):
        for i in range(4):
            self.stm.add(MemoryEntry.create(f"entry {i}"))
        self.assertTrue(self.stm.needs_compression())

    def test_pop_oldest(self):
        self.stm.add(MemoryEntry.create("first"))
        self.stm.add(MemoryEntry.create("second"))
        removed = self.stm.pop_oldest(1)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].content, "first")
        self.assertEqual(self.stm.count(), 1)

    def test_persistence(self):
        self.stm.add(MemoryEntry.create("persist me"))
        # Re-load
        stm2 = ShortTermMemory(self.tmpdir)
        self.assertEqual(stm2.count(), 1)
        self.assertEqual(stm2.get_all()[0].content, "persist me")


class TestLongTermMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ltm = LongTermMemory(self.tmpdir, embedding_backend="tfidf")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_search(self):
        self.ltm.add(MemoryEntry.create("向量数据库支持语义检索"))
        self.ltm.add(MemoryEntry.create("小狐的奶茶偏好是桂馥兰香"))
        results = self.ltm.search("向量数据库")
        self.assertGreaterEqual(len(results), 1)

    def test_find_by_id(self):
        entry = MemoryEntry.create("findable")
        self.ltm.add(entry)
        found = self.ltm.find_by_id(entry.id)
        self.assertIsNotNone(found)
        self.assertEqual(found.content, "findable")


class TestWorkingMemory(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.wm = WorkingMemory(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_and_search(self):
        self.wm.add(MemoryEntry.create("规则: 每天8点提醒吃药"))
        results = self.wm.search("吃药")
        self.assertEqual(len(results), 1)

    def test_clear(self):
        self.wm.add(MemoryEntry.create("temp rule"))
        self.wm.clear()
        self.assertEqual(self.wm.count(), 0)


class TestCompressor(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.stm = ShortTermMemory(self.tmpdir, window_size=3, compression_threshold=2)
        self.ltm = LongTermMemory(self.tmpdir, embedding_backend="tfidf")
        self.compressor = Compressor(self.stm, self.ltm, id_prefix="test")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_compress_force(self):
        for i in range(3):
            self.stm.add(MemoryEntry.create(f"memory {i}"))
        self.assertEqual(self.stm.count(), 3)
        n = self.compressor.compress(force=True)
        self.assertEqual(n, 3)
        self.assertEqual(self.stm.count(), 0)
        self.assertEqual(self.ltm.count(), 1)  # merged into 1 batch

    def test_compress_threshold(self):
        # window_size=4, compression_threshold=3 → trigger at 4, keep 4
        stm = ShortTermMemory(self.tmpdir + "_2", window_size=4, compression_threshold=3)
        ltm = LongTermMemory(self.tmpdir + "_2", embedding_backend="tfidf")
        compressor = Compressor(stm, ltm, id_prefix="test")
        for i in range(5):
            stm.add(MemoryEntry.create(f"memory {i}"))
        self.assertTrue(stm.needs_compression())
        n = compressor.compress(force=False)
        self.assertEqual(n, 1)  # 5 - 4 = 1 excess
        self.assertEqual(stm.count(), 4)


class TestTFIDFVectorStore(unittest.TestCase):
    def test_add_and_search(self):
        store = TFIDFVectorStore()
        store.add("1", "向量数据库支持语义检索")
        store.add("2", "小狐喜欢喝奶茶")
        results = store.search("向量")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "1")

    def test_batch(self):
        store = TFIDFVectorStore()
        store.add_batch([
            ("a", "记忆是AI的重要能力", {}),
            ("b", "语义搜索需要向量化", {}),
        ])
        self.assertEqual(store.count(), 2)
        results = store.search("记忆")
        self.assertEqual(results[0]["id"], "a")

    def test_persistence(self):
        """Test that TF-IDF index survives save/load cycle."""
        import tempfile
        tmppath = os.path.join(tempfile.mkdtemp(), "tfidf_index.json")
        store1 = TFIDFVectorStore(persist_path=tmppath)
        store1.add("p1", "持久化测试数据")
        store1.add("p2", "向量检索需要保存索引")
        # Simulate new process: create new store from same file
        store2 = TFIDFVectorStore(persist_path=tmppath)
        self.assertEqual(store2.count(), 2)
        results = store2.search("持久化")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "p1")


class TestConfig(unittest.TestCase):
    def test_deep_merge(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 10}}
        merged = deep_merge(base, override)
        self.assertEqual(merged["a"]["b"], 10)
        self.assertEqual(merged["a"]["c"], 2)  # preserved


if __name__ == "__main__":
    unittest.main(verbosity=2)
