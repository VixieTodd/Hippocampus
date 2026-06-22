# Hippocampus — AI仿生记忆系统

🦛 面向 AI Agent 的本地三层记忆系统。

[English](./README.md)

---

## 概述

大语言模型 Agent 在会话之间缺乏持久记忆，每次启动时无法访问历史对话中的信息。

Hippocampus 提供三层记忆架构：Agent 在对话中搜索相关历史、注入上下文、并在对话结束后存储新记忆。

```
用户输入 → Agent 搜索 Hippocampus → 注入记忆上下文 → 生成回复
                                                     ↓
会话结束 → Agent 写入关键信息 → 下次会话可检索
```

三层记忆：

| 层 | 功能 | 示例 |
|---|------|------|
| **短期记忆** | 最近对话的滑动窗口 | "用户刚才询问奶茶偏好" |
| **长期记忆** | 向量索引持久化存储 | "用户是VOCALOID爱好者，偏好桂花味奶茶" |
| **工作记忆** | 始终在上下文的规则和配置 | "每日8点和20点提醒用药" |

---

## 快速开始

```bash
# 安装 (Python 3.10+)
git clone https://github.com/VixieTodd/Hippocampus.git
cd Hippocampus
pip install -e .

# 写入记忆
hippo write "用户偏好桂花味奶茶"
hippo write "规则：每日8点和20点提醒用药" --layer working

# 三层搜索
hippo search "奶茶"

# 统计
hippo stats

# 短期→长期压缩
hippo compress --force

# 单条溯源
hippo trace <entry_id>

# 导出备份
hippo export --format json -o backup.json
```

内置 TF-IDF 向量搜索引擎（支持中日韩分词），离线可用，无需下载模型。可选 ChromaDB 后端以获得更强的语义匹配。

---

## Python SDK

```python
from hippocampus import Hippocampus

hippo = Hippocampus("config.yml")

# 会话启动时：加载工作记忆到上下文
working = hippo.working.get_all()
context = "\n".join([e.content for e in working])

# 对话中：搜索相关记忆
results = hippo.search("用户偏好", top_k=5)
for layer, entries in results.items():
    for e in entries:
        context += f"\n[记忆] {e.content}"

# 对话结束后：存储重要信息
hippo.write("用户提到新的项目想法")
hippo.write("约定明天下午跟进")

# 短期记忆超出阈值时自动触发压缩
```

---

## CLI 参考

```
hippo install                安装向导
hippo write <content>        写入记忆条目
hippo search <query>         三层语义搜索
hippo stats                  记忆统计
hippo compress [--force]     触发压缩 (短期→长期)
hippo trace <id>             单条记忆溯源
hippo export [--format]      全量备份
```

---

## 配置

```yaml
hippocampus:
  data_dir: "./data"
  short_term:
    window_size: 50
    compression_threshold: 40
  long_term:
    top_k: 5
    embedding_backend: "tfidf"  # 或 "chroma_default"
  working:
    file: "working.json"
```

---

## 依赖

- Python 3.10+
- `click` (CLI框架)
- `pyyaml` (配置解析)
- `chromadb` (可选 — 无安装时自动降级为 TF-IDF)

---

## 许可证

MIT
