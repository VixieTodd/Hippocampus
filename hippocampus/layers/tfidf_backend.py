"""
TF-IDF backend — zero-dependency vector retrieval for long-term memory.

This is the default (Lite) backend.  It provides semantic-ish search without
installing ChromaDB or sentence-transformers (hundreds of MB).

Key features:
  - CJK-aware tokeniser: single-character + bigram for Chinese/Japanese/Korean,
    whitespace-split for Latin/ASCII.
  - Cosine similarity over sparse TF-IDF vectors.
  - In-memory document store, persisted as JSON (``tfidf_store.json``).
  - Designed for small-to-medium collections (<10 K documents); beyond that
    ChromaDB is recommended.

Usage:
  from hippocampus.layers.tfidf_backend import TFIDFBackend

  be = TFIDFBackend(data_dir)
  be.add(id="abc", content="今天天气很好")
  results = be.search("天气", top_k=5)
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


class TFIDFBackend:
    """Lightweight vector store powered by TF-IDF + cosine similarity."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._store_file = self._data_dir / "tfidf_store.json"

        # doc_id → {content, timestamp, source, ...}
        self._docs: dict[str, dict[str, Any]] = {}

        # term → {doc_id → tf}
        self._index: dict[str, dict[str, float]] = defaultdict(dict)

        # doc_id → term → tf
        self._doc_terms: dict[str, dict[str, float]] = {}

        # Total document count (for IDF), loaded on init.
        self._doc_count: int = 0

        self._load()

    # ── Persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._store_file.exists():
            return
        try:
            with open(self._store_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        for doc in data.get("docs", []):
            doc_id = doc["id"]
            content = doc.get("content", "")
            self._docs[doc_id] = doc
            tokens = self._tokenise(content)
            tf = self._compute_tf(tokens)
            self._doc_terms[doc_id] = tf
            for term, freq in tf.items():
                self._index[term][doc_id] = freq

        self._doc_count = len(self._docs)

    def _save(self) -> None:
        data = {
            "version": 1,
            "count": len(self._docs),
            "docs": list(self._docs.values()),
            # We don't persist the index — it's rebuilt on _load().
        }
        with open(self._store_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Tokenisation ─────────────────────────────────────────────────

    # Pre-compiled regex for CJK character detection.
    _CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\ud840-\ud87f\udc00-\udfff"
                         r"\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u3000-\u303f"
                         r"\uff00-\uffef]")

    # Punctuation stripped from tokens.
    _PUNCT_RE = re.compile(r"[，。！？、；：""''（）【】《》\s,.!?;:'\"()\[\]{}]+")

    @classmethod
    def _is_cjk(cls, ch: str) -> bool:
        """Check if a single character is CJK."""
        return cls._CJK_RE.match(ch) is not None

    @classmethod
    def _strip_punct(cls, text: str) -> str:
        """Remove common CJK + ASCII punctuation."""
        return cls._PUNCT_RE.sub(" ", text).strip()

    @classmethod
    def _tokenise(cls, text: str) -> list[str]:
        """Tokenise into unigrams + bigrams for CJK, word tokens for ASCII.

        Examples:
          "今天天气好" → ["今","今天","天","天气","气","气好","好"]
          "hello world" → ["hello", "world"]
          "今天hello"   → ["今","今天","天","hello"]
        """
        text = cls._strip_punct(text.lower())
        tokens: list[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if cls._is_cjk(ch):
                # CJK: output single char
                tokens.append(ch)
                # bigram with next char
                if i + 1 < len(text) and cls._is_cjk(text[i + 1]):
                    tokens.append(ch + text[i + 1])
                i += 1
            elif ch.isalpha() or ch.isdigit():
                # ASCII word: collect until next non-alnum
                j = i
                while j < len(text) and (text[j].isalpha() or text[j].isdigit()):
                    j += 1
                word = text[i:j]
                if len(word) >= 2:  # skip single-letter English words
                    tokens.append(word)
                elif len(word) == 1:
                    tokens.append(word)
                i = j
            else:
                i += 1
        return tokens

    # ── TF-IDF ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_tf(tokens: list[str]) -> dict[str, float]:
        """Term frequency (raw count / total tokens)."""
        if not tokens:
            return {}
        total = len(tokens)
        counts: dict[str, int] = defaultdict(int)
        for t in tokens:
            counts[t] += 1
        return {t: c / total for t, c in counts.items()}

    def _compute_idf(self, term: str) -> float:
        """Inverse document frequency: log(N / df) + smoothing."""
        df = len(self._index.get(term, {}))
        if df == 0:
            return 0.0
        return math.log((self._doc_count + 1) / (df + 1)) + 1.0

    def _vec(self, doc_id: str) -> dict[str, float]:
        """Sparse TF-IDF vector for a document."""
        tf = self._doc_terms.get(doc_id, {})
        vec: dict[str, float] = {}
        for term, tf_val in tf.items():
            idf = self._compute_idf(term)
            vec[term] = tf_val * idf
        return vec

    @staticmethod
    def _cosine_similarity(
        a: dict[str, float], b: dict[str, float]
    ) -> float:
        """Cosine similarity between two sparse vectors."""
        if not a or not b:
            return 0.0
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for k, va in a.items():
            norm_a += va * va
            vb = b.get(k, 0.0)
            if vb:
                dot += va * vb
        for vb in b.values():
            norm_b += vb * vb
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    # ── Public API ───────────────────────────────────────────────────

    def add(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add or update a document.

        Returns the doc_id (same as input, for API consistency with ChromaDB).
        """
        tokens = self._tokenise(content)
        tf = self._compute_tf(tokens)

        # Remove old index entries for this doc (in case of overwrite).
        if doc_id in self._doc_terms:
            for term in self._doc_terms[doc_id]:
                if term in self._index:
                    self._index[term].pop(doc_id, None)

        self._docs[doc_id] = {
            "id": doc_id,
            "content": content,
            "timestamp": metadata.get("timestamp", "") if metadata else "",
            "source": metadata.get("source", "") if metadata else "",
            "parent_id": metadata.get("parent_id", "") if metadata else "",
        }
        self._doc_terms[doc_id] = tf
        for term, freq in tf.items():
            self._index[term][doc_id] = freq
        self._doc_count = len(self._docs)
        self._save()
        return doc_id

    def add_batch(
        self, entries: list[tuple[str, str, dict[str, Any] | None]]
    ) -> list[str]:
        """Add multiple documents at once.  Faster than calling add() N times.

        Each entry is (doc_id, content, metadata).
        """
        ids: list[str] = []
        for doc_id, content, meta in entries:
            self.add(doc_id, content, meta)
            ids.append(doc_id)
        return ids

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Search by cosine similarity over TF-IDF vectors.

        Returns list of dicts: {id, content, score, timestamp, source, parent_id}.
        """
        query_tokens = self._tokenise(query)
        if not query_tokens:
            return []

        q_tf = self._compute_tf(query_tokens)

        # Compute IDF for query terms and build query vector.
        q_vec: dict[str, float] = {}
        for term, tf_val in q_tf.items():
            idf = self._compute_idf(term)
            q_vec[term] = tf_val * idf

        # Score every document.
        scored: list[tuple[float, str]] = []
        for doc_id in self._docs:
            doc_vec = self._vec(doc_id)
            sim = self._cosine_similarity(q_vec, doc_vec)
            if sim >= min_score:
                scored.append((sim, doc_id))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_k]

        results: list[dict[str, Any]] = []
        for score, doc_id in top:
            doc = self._docs[doc_id]
            results.append({
                "id": doc_id,
                "content": doc["content"],
                "score": round(score, 4),
                "timestamp": doc.get("timestamp", ""),
                "source": doc.get("source", ""),
                "parent_id": doc.get("parent_id", ""),
            })

        return results

    def count(self) -> int:
        """Number of documents in the store."""
        return len(self._docs)

    def all_docs(self) -> list[dict[str, Any]]:
        """Return all documents (for export)."""
        return list(self._docs.values())

    def remove(self, doc_ids: list[str]) -> None:
        """Remove documents by ID."""
        for doc_id in doc_ids:
            if doc_id in self._docs:
                self._docs.pop(doc_id, None)
            if doc_id in self._doc_terms:
                for term in self._doc_terms.pop(doc_id, {}):
                    self._index.get(term, {}).pop(doc_id, None)
        self._doc_count = len(self._docs)
        self._save()

    def clear(self) -> None:
        """Remove all documents and index."""
        self._docs.clear()
        self._index.clear()
        self._doc_terms.clear()
        self._doc_count = 0
        if self._store_file.exists():
            self._store_file.unlink()
