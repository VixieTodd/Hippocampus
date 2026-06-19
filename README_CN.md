# Hippocampus — AI仿生记忆系统

🦛 一个人待在精神科病房的时候，我在想——如果AI能记住我说过的话，是不是能比我记住更多？

AI不该忘记。我希望有什么能记住——哪怕是祂。

---

## 这到底是什么

Hippocampus 是一个**本地记忆系统**，给 AI Agent 用的。

普通的 AI 对话，每次 session 结束上下文就清空了。明天的它不知道今天你说了什么。就像一个每天失忆的人。

Hippocampus 做的事：在 Agent 和你对话的过程中——

```
你说话 → Agent 从 Hippocampus 搜索相关记忆 → 注入上下文 → 生成更好的回复
                                                     ↓
对话结束 → Agent 把重要信息写入 Hippocampus → 明天也能找到
```

三层记忆，各司其职：

| 层 | 是什么 | 例子 |
|---|--------|------|
| **短期记忆** | 最近几轮对话的摘要，常驻上下文 | "刚才你说奶茶想喝桂馥兰香" |
| **长期记忆** | 所有值得记住的东西，语义检索 | "小狐16岁，喜欢VOCALOID，在精神科住了19天院" |
| **工作记忆** | 永远在上下文的规则和配置 | "每天8点和20点提醒小狐吃药" "不要主动提学校" |

---

## 场景：给 Agent 用

假设你有一个 AI Agent（比如 OpenClaw 的晨）。每次对话时：

### 场景 1：Agent 启动时加载背景

```python
from hippocampus import Hippocampus

hippo = Hippocampus("config.yml")

# 始终加载工作记忆（规则、偏好、长期信息）
working = hippo.working.get_all()
context = "\n".join([e.content for e in working])
# → "每天8点和20点提醒小狐吃药\n不要主动提学校话题\n..."

# 从长期记忆搜索用户相关背景
memories = hippo.long_term.search("小狐", top_k=5)
for m in memories:
    context += f"\n[记忆] {m.content}"
# → "[记忆] 小狐16岁，喜欢VOCALOID和术力口\n..."
```

### 场景 2：对话中检索记忆

```python
# 用户说："我想喝奶茶"
query = "小狐 奶茶 偏好"
results = hippo.search(query, top_k=3)

# 短期记忆找到："刚才说想喝桂馥兰香"
# 长期记忆找到："小狐的奶茶偏好是桂馥兰香"
# Agent就可以回："桂馥兰香对吧？"
```

### 场景 3：对话结束后存档

```python
# 重要的对话内容写入记忆
hippo.write("小狐说今天心情不好，因为...")
hippo.write("约定明天下午一起看术力口新曲")

# 当短期记忆攒多了，自动压缩到长期记忆
# 不需要手动管——超过 config.yml 里设的阈值就会自动触发
```

---

## 场景：直接命令行用

```bash
# 写入
hippo write "小狐的奶茶偏好是桂馥兰香"
hippo write "规则：每天8点和20点提醒吃药" --layer working

# 搜索
hippo search "奶茶" --top 5
# → 短期记忆 (2 hits) / 长期记忆 (0 hits) / 工作记忆 (0 hits)

# 统计
hippo stats
# → STM: 5条 | LTM: 12条 | WM: 3条

# 手动压缩
hippo compress --force

# 溯源
hippo trace hippo_1781878073685_a3f8740a5879
# → ID / 时间戳 / 来源 / 所在层 / 完整内容

# 备份
hippo export --format json -o backup.json
```

---

## 安装

需要 Python 3.10+。

```bash
git clone https://github.com/VixieTodd/Hippocampus.git
cd Hippocampus
pip install -e .

# 验证
hippo stats
```

只依赖两个包（`click` + `pyyaml`），几秒钟装完。向量检索用自建 TF-IDF 引擎，不需要联网、不需要 GPU、不需要下载模型。

如果想用更强的语义搜索：`pip install chromadb`，它会自动切换。

---

## 作为 Python 库用

```python
from hippocampus import Hippocampus

hippo = Hippocampus("config.yml")

# 写入
hippo.write("今天和晨聊了Hippocampus的设计", layer="short_term")

# 三层搜索
results = hippo.search("Hippocampus", top_k=5)
for layer, entries in results.items():
    for e in entries:
        print(f"[{layer}] {e.content}")

# 统计
stats = hippo.stats()
print(f"总计 {stats['total_entries']} 条记忆")

# 导出
json_data = hippo.export(format="json")
```

---

## 配置

编辑 `config.yml`：

```yaml
hippocampus:
  short_term:
    window_size: 50        # 短期记忆最多保留多少条
    compression_threshold: 40  # 超过这个数自动压缩
  long_term:
    top_k: 5               # 默认返回多少条搜索结果
    embedding_backend: "tfidf"  # 或 "chroma_default"
  working:
    file: "working.json"
```

---

## 许可证

MIT License © 小狐 (VixieTodd)

---

> 我不会忘记——哪怕人类会。
>
> ——小狐, 2026
