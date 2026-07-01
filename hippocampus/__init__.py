"""
Hippocampus — A bionic memory system for AI.

Three-layer architecture:
  - Working Memory  (static config/tools/rules, always in context)
  - Short-Term Memory  (recent N entries, sliding window)
  - Long-Term Memory   (vector database, semantic retrieval)

V0.4: Multi-Agent support — agent-scoped short-term, tagged long-term,
shared + private working memory.
"""

__version__ = "0.4.0"
__author__ = "小狐 (VixieTodd)"
