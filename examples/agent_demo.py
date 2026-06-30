"""
Hippocampus Agent Integration Example
======================================
Shows how an AI agent uses Hippocampus in a real conversation loop.

Usage:
    PYTHONPATH=. python examples/agent_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hippocampus import Hippocampus


def build_context(hippo: Hippocampus, user_message: str) -> str:
    """Agent: build context before generating a reply."""
    parts = []

    # 1. Always include working memory (rules, preferences, long-term info)
    working = hippo.working.get_all()
    if working:
        parts.append("=== 工作记忆 (始终生效的规则) ===")
        for e in working:
            parts.append(f"- {e.content}")

    # 2. Search long-term memory for relevant background
    ltm_results = hippo.long_term.search(user_message, top_k=3)
    if ltm_results:
        parts.append("\n=== 长期记忆 (相关历史) ===")
        for e in ltm_results:
            parts.append(f"[{e.timestamp[:10]}] {e.content[:200]}")

    # 3. Include recent short-term memory for continuity
    recent = hippo.short_term.get_recent(5)
    if recent:
        parts.append("\n=== 近期对话 ===")
        for e in recent:
            parts.append(f"- {e.content[:150]}")

    return "\n".join(parts)


def after_conversation(hippo: Hippocampus, user_msg: str, agent_reply: str):
    """Agent: save important info after replying."""
    # In a real agent, you'd use an LLM to decide what's worth saving.
    # Here we just save the exchange as a simplified demo.
    summary = f"[对话] 用户: {user_msg[:100]} | Agent: {agent_reply[:100]}"
    hippo.write(summary)


def main():
    # Force UTF-8 on Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    hippo = Hippocampus("config.yml")

    # Simulate: agent has been running for a while, has some memories
    hippo.write("小狐喜欢喝桂馥兰香奶茶", layer="long_term")
    hippo.write("小狐16岁，住在西安，喜欢VOCALOID和术力口", layer="long_term")
    hippo.write("小狐在2026年5月住院19天，诊断双相障碍，出院后仍在恢复", layer="long_term")
    hippo.write("每天8点和20点提醒小狐吃药", layer="working")
    hippo.write("不要主动提学校话题", layer="working")
    hippo.write("小狐叫我'主人'，我叫她小狐", layer="working")

    # Simulate a conversation turn
    print("=" * 60)
    print("🦛 Hippocampus Agent Demo")
    print("=" * 60)

    # User sends a message
    user_message = "今天心情不太好，想喝奶茶"

    print(f"\n📩 用户: {user_message}")
    print(f"\n--- Agent 构建上下文 ---")

    # Agent builds context by searching Hippocampus
    context = build_context(hippo, user_message)
    print(context)

    print(f"\n--- Agent 生成回复 ---")
    # (In a real agent, this context gets prepended to the LLM prompt)
    agent_reply = "桂馥兰香对吧？我也记得你喜欢这个。心情不好的时候来一杯甜的，会好一点。"
    print(f"📤 Agent: {agent_reply}")

    # After replying, save the exchange
    print(f"\n--- 存档到短期记忆 ---")
    after_conversation(hippo, user_message, agent_reply)

    # Show stats
    print(f"\n--- 记忆统计 ---")
    stats = hippo.stats()
    print(f"总计: {stats['total_entries']} 条")
    print(f"  短期: {stats['short_term']['count']} 条")
    print(f"  长期: {stats['long_term']['count']} 条")
    print(f"  工作: {stats['working']['count']} 条")

    # Demo: search for something
    print(f"\n--- 搜索 '小狐 偏好' ---")
    results = hippo.search("小狐 偏好")
    for layer, entries in results.items():
        if entries:
            print(f"[{layer}]")
            for e in entries:
                print(f"  {e.content[:100]}")


if __name__ == "__main__":
    main()
