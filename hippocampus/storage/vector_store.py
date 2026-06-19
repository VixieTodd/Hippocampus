"""
Vector Store — Chroma-based semantic search with TF-IDF fallback.
Includes persistent TF-IDF backend with disk serialization.
"""

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

# Chroma imports (optional; fallback to TF-IDF if unavailable)
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


class TFIDFVectorStore:
    """Simple TF-IDF based search — zero dependencies, no downloads.
    
    Used as fallback when Chroma is unavailable or when
    embedding_backend is set to "tfidf".
    
    Supports disk persistence: saves/loads index to JSON.
    Thread-safe for single-process use.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._docs: Dict[str, Tuple[str, dict]] = {}  # id -> (content, metadata)
        self._idf: Dict[str, float] = {}  # term -> idf
        self._persist_path = persist_path
        self._lock = threading.Lock()
        if persist_path and os.path.exists(persist_path):
            self._load()

    def _atomic_write(self, filepath: str, data: str):
        """Atomic write: write to temp file, then rename."""
        tmp = filepath + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, filepath)

    def _save(self):
        """Persist index to disk."""
        if not self._persist_path:
            return
        os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
        data = {
            "docs": {eid: [content, meta] for eid, (content, meta) in self._docs.items()},
            "idf": self._idf,
        }
        self._atomic_write(self._persist_path, json.dumps(data, ensure_ascii=False))

    def _load(self):
        """Load index from disk."""
        with open(self._persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._docs = {
            eid: (arr[0], arr[1]) for eid, arr in data.get("docs", {}).items()
        }
        self._idf = data.get("idf", {})

    def add(self, doc_id: str, content: str, metadata: Optional[dict] = None):
        with self._lock:
            self._docs[doc_id] = (content, metadata or {})
            self._rebuild_idf()
            self._save()

    def add_batch(self, items: List[Tuple[str, str, dict]]):
        with self._lock:
            for doc_id, content, metadata in items:
                self._docs[doc_id] = (content, metadata)
            self._rebuild_idf()
            self._save()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer supporting both CJK and Latin scripts.
        
        For CJK characters: uses character unigrams + bigrams.
        For Latin: splits on non-alphanumeric, lowercases.
        """
        import re
        tokens = []
        # Extract CJK characters
        cjk_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text)
        # Single CJK chars as unigrams
        tokens.extend(cjk_chars)
        # CJK bigrams for better matching
        for i in range(len(cjk_chars) - 1):
            tokens.append(cjk_chars[i] + cjk_chars[i + 1])
        # Latin/alphanumeric tokens
        latin_tokens = re.findall(r'[a-zA-Z0-9_]{2,}', text.lower())
        tokens.extend(latin_tokens)
        return [t for t in tokens if len(t) >= 1]

    def _rebuild_idf(self):
        """Rebuild IDF scores."""
        n = max(len(self._docs), 1)
        df: Dict[str, int] = {}
        for _, (content, _) in self._docs.items():
            tokens = set(self._tokenize(content))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        self._idf = {
            token: math.log((n + 1) / (freq + 1)) + 1
            for token, freq in df.items()
        }

    def _tfidf_vector(self, text: str) -> Dict[str, float]:
        """Compute TF-IDF vector for text."""
        tokens = self._tokenize(text)
        tf: Dict[str, float] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalize TF
        max_tf = max(tf.values()) if tf else 1
        result = {}
        for token, freq in tf.items():
            if token in self._idf:
                result[token] = (freq / max_tf) * self._idf[token]
        return result

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search by TF-IDF cosine similarity."""
        q_vec = self._tfidf_vector(query)
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1

        scored = []
        with self._lock:
            for doc_id, (content, metadata) in self._docs.items():
                d_vec = self._tfidf_vector(content)
                # Dot product / cosine similarity
                dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in set(q_vec) | set(d_vec))
                d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1
                score = dot / (q_norm * d_norm)
                if score > 0:
                    scored.append((score, doc_id, metadata))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"id": doc_id, "score": score, "metadata": meta}
                for score, doc_id, meta in scored[:top_k]]

    def count(self) -> int:
        return len(self._docs)


class VectorStore:
    """Unified vector store interface.
    
    Backends:
    - "chroma_default": Chroma with built-in sentence-transformers
    - "tfidf": Local TF-IDF with disk persistence (no downloads, no GPU)
    """

    def __init__(self, persist_dir: str, collection_name: str,
                 backend: str = "chroma_default"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.backend = backend
        self._store = None
        self._init_backend()

    def _init_backend(self):
        if self.backend == "chroma_default" and CHROMA_AVAILABLE:
            self._init_chroma()
        else:
            if self.backend == "chroma_default":
                print("[hippocampus] Chroma not available, using TF-IDF fallback.")
            persist_path = os.path.join(
                self.persist_dir, f"{self.collection_name}_tfidf.json"
            )
            self._store = TFIDFVectorStore(persist_path=persist_path)

    def _init_chroma(self):
        os.makedirs(self.persist_dir, exist_ok=True)
        client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        # Use Chroma's default embedding function
        self._store = client.get_or_create_collection(
            name=self.collection_name,
        )

    def add(self, doc_id: str, content: str, metadata: Optional[dict] = None):
        """Add a single document."""
        if isinstance(self._store, TFIDFVectorStore):
            self._store.add(doc_id, content, metadata)
        else:
            self._store.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata or {}],
            )

    def add_batch(self, items: List[Tuple[str, str, dict]]):
        """Add multiple documents at once."""
        if isinstance(self._store, TFIDFVectorStore):
            self._store.add_batch(items)
        else:
            ids = [item[0] for item in items]
            docs = [item[1] for item in items]
            metas = [item[2] for item in items]
            self._store.add(ids=ids, documents=docs, metadatas=metas)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """Search for similar documents."""
        if isinstance(self._store, TFIDFVectorStore):
            return self._store.search(query, top_k)
        else:
            results = self._store.query(
                query_texts=[query],
                n_results=top_k,
            )
            output = []
            ids = results.get("ids", [[]])[0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            for i, eid in enumerate(ids):
                output.append({
                    "id": eid,
                    "score": 1.0 - distances[i] if i < len(distances) else 0.0,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            return output

    def count(self) -> int:
        if isinstance(self._store, TFIDFVectorStore):
            return self._store.count()
        else:
            return self._store.count()
