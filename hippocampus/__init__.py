"""
Hippocampus — AI Bionic Memory System
======================================
Three-layer memory architecture for AI agents:
  - Short-Term: sliding window of recent turns
  - Long-Term: vector-indexed persistent memory
  - Working: static config/rules, always in context

Author: 小狐 (VixieTodd)
License: MIT
"""

__version__ = "0.1.0"
__author__ = "小狐 (VixieTodd)"

from .cli import Hippocampus
