# Hippocampus 开发日志

## 2026-06-22 — install 安装向导

### 需求来源

Hippocampus 与 OpenClaw 原生记忆体系（MEMORY.md、memory-setup skill 等）存在功能重叠。为避免运行时冲突，在安装阶段通过向导处理记忆迁移和 Skill 禁用。

### 方案对比

```
需求: 安装时防冲突
  │
  ├── 方案A: 运行时桥接层 ── 放弃 (维护成本高、向量分数不可比)
  ├── 方案B: 选一边废弃   ── 放弃 (丢失数据)
  ├── 方案C: 明确分工      ── 放弃 (边界模糊易漂移)
  └── 方案D: 安装向导 ✅   ── 安装阶段一次性解决
       │
       ├── 1. 检测环境 (workspace/memory/skills)
       ├── 2. 询问是否迁移记忆
       ├── 3. 询问是否禁用冲突 Skill
       └── 4. 执行 + 摘要
```

### 文件变更

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `hippocampus/installer.py` | 新建 | ~230 | 检测、迁移、禁用、向导 |
| `hippocampus/cli.py` | 修改 | +17 | 新增 install 命令 |
| `DEVLOG.md` | 新建 | — | 本文档 |

### 模块结构 (`installer.py`)

```
detect_environment()          → 扫描workspace, 返回report dict
  ├── _find_workspace()       → 向上查找OpenClaw标记文件
  └── 统计memory_files/skills/conflict_count

print_environment_report()    → 格式化打印检测结果

migrate_memories()            → 核心迁移逻辑
  ├── _parse_markdown_sections()  → 按## / ###拆分markdown为独立条目
  ├── 过滤短内容 (<10字符)
  ├── MemoryEntry.create() + long_term.add_batch()
  └── shutil.move → .archive/ (可选)

disable_skills()              → 写.disabled标记到skill目录

run_install_wizard()          → 交互式3步向导 (click.confirm/prompt)
auto_install()                → 非交互API (--yes)
```

### CLI 接口

```bash
hippo install          # 交互式向导
hippo install --yes    # 静默自动安装
```

### 设计决策

- **Markdown 按 ## 标题拆分**：MEMORY.md 的二级标题作为天然分段边界，每个段成为独立 MemoryEntry，metadata 保留 `section` 和 `original_file`
- **归档而非删除**：原文件移至 `memory/.archive/`，可逆操作
- **.disabled 标记而非删除 Skill**：写 JSON 标记文件，可手动恢复
- **CJK 分词已就绪**：TF-IDF 内置中日韩单字+双字 bigram 分词器，中文搜索无需外部依赖
- **零侵入**：Click 命令为独立函数，install 命令不影响现有逻辑

### 测试结果

子 agent `hippo_install_test` 执行，全部通过。

| 模块 | 通过/总 | 关键数据 |
|------|---------|----------|
| CLI 注册 | 4/4 | `install --help` 正常 |
| 环境检测 | 10/10 | workspace 正确识别, 5个记忆文件, 3个冲突skill |
| Markdown 解析 | 7/7 | MEMORY.md → 19个独立条目 |
| 代码导入 | 7/7 | 所有模块无语法错误 |
| 模拟迁移 | 8/8 | 19条迁移 (archive=False, 原文件未动) |
| Skill 禁用 | 9/9 | .disabled 标记正确创建/验证/清理 |
| **总计** | **45/45** | **全部通过** |

### 使用示例

```
$ hippo install

🦛 Hippocampus 安装向导
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/3] 检测环境...
  ✓ 找到 OpenClaw 工作区

[2/3] 记忆迁移
  检测到已有记忆文件。是否将现有记忆重组导入 Hippocampus？
  [Y] 是  [N] 否

[3/3] Skill 冲突处理
  ⚠ memory-setup
  ⚠ proactive-agent
  ⚠ self-improving-agent-skill
  是否禁用这些 Skill？
  [Y] 是  [N] 否

✓ 安装完成
  导入了 19 条记忆 | 禁用了 3 个冲突 Skill
```

---

## 2026-06-30 (2) — 代码审查：注释补全 + 压缩修复 + 性能优化

### 审查范围

逐文件审查全部 12 个源文件，检查注释完整性、逻辑错误、性能问题。

### 发现并修复的问题

| # | 文件 | 问题 | 修复 |
|---|------|------|------|
| 1 | `config.py` | `to_dict()` 返回 `__dict__`，泄露 `_config_path` | 过滤 `_` 前缀属性 |
| 2 | `store.py` | `export()` 硬编码 `"version": "0.1.0"` | 改为 `from hippocampus import __version__` |
| 3 | `compressor.py` | `_concat_block` 注释说"合并"实际是 1:1 假压缩 | 实现真正的合并：多条 → 按 `max_chars` 切 chunk |
| 4 | `tfidf_backend.py` | `add_batch()` 循环调 `add()`，每次 `_save()` | 内联实现，批量完只存一次 |
| 5 | `long_term.py` | 未使用的 `Optional` 导入 | 移除 |

### 注释补全

| 文件 | 补了什么 |
|------|----------|
| `layers/short_term.py` | 搜索评分公式 `* 0.8 + 0.2` 的设计意图 |
| `layers/tfidf_backend.py` | `_PUNCT_RE` 覆盖的标点范围说明 |
| `layers/long_term.py` | `_init_tfidf()` docstring |
| `cli.py` | `_ask_yn`、`_find_all_skills`、`_find_memory_files` docstring |
| `cli.py` | install 后端选择段分叉说明（Lite/Full） |
| `cli.py` | 数据导入段说明（逐行导入逻辑） |

### 测试结果

回归测试全部通过：write+search、stats、trace、export version、compress（10→1）、config to_dict 无泄露。

---

## 2026-06-30 — 双后端架构 (TF-IDF Lite + ChromaDB Full)

### 需求来源

V0.2 的 `deps.py` 硬性要求 `chromadb` + `sentence-transformers`（~300 MB），
丢掉了 V0.1 的零依赖启动优势。用户反馈后改为**默认轻量、可选完整**的双后端设计。

### 新增：TF-IDF 后端 (`layers/tfidf_backend.py`)

- CJK-aware 分词（单字+bigram，中日韩文字支持）
- TF-IDF 向量化 + 余弦相似度检索
- JSON 文件持久化（`tfidf_store.json`）
- 零额外依赖，适合 <10K 文档规模
- 中文搜索实测准确（"天气"→0.64，"术力口"→0.55）

### 改动的文件

| 文件 | 操作 | 说明 |
|------|------|------|
| `layers/tfidf_backend.py` | **新建** | TF-IDF 轻量向量后端 |
| `layers/long_term.py` | **重写** | 根据 `config.long_term.backend` 切换 TF-IDF/ChromaDB |
| `config.py` | 修改 | `long_term.backend` 默认改为 `"tfidf"` |
| `deps.py` | 修改 | chromadb/sentence-transformers 从 REQUIRED 降为 OPTIONAL；新增 `check_optional()`、`install_optional()` |
| `pyproject.toml` | 修改 | 去掉 chromadb/sentence-transformers 硬依赖，加入 `[optional-dependencies] full`；修复废弃的 build-backend |
| `cli.py` | **重写** | install 向导新增 [2/4] 后端选择步骤；修复硬编码 REQUIRED；doctor 新增 `--full` 选项；group callback 跳过 doctor/install 避免触发 ChromaDB 导入 |
| `store.py` | 修改 | 传递 `backend` 参数给 LongTermMemoryLayer |
| `__init__.py` | 修改 | 版本号 → 0.2.0 |

### 安装流程变化（4 步）

```
[1/4] 环境检查 → Python + click/pyyaml
[2/4] 后端选择 → [1] 轻量 TF-IDF  /  [2] 完整 ChromaDB
[3/4] Skill 冲突 → 通用扫描 skills/ 目录
[4/4] 导入数据 → MEMORY.md → Hippocampus
```

### CLI 变化

- `hippo doctor` — 不再触发 ChromaDB 导入，显示 Lite/Full 两组状态
- `hippo doctor --full` — 安装 ChromaDB + sentence-transformers
- `hippo doctor --install` — 仅安装核心依赖

---

## 2026-06-29 — 源码优化 + 安装向导重构

### 需求来源

读完所有源码后，有几处问题需要修：注释太少、短期记忆评分硬编码 0.8、长程 stats 字符数统计不准、压缩器策略名与实际行为不符。同时原安装向导写死了我个人的工作流（硬编码的冲突 Skill 名单、个人座右铭），需要通用化。

### 源码优化

| 文件 | 改动 |
|------|------|
| `layers/__init__.py` | 全面补注释，`SearchResult.metadata` 改为 `field(default_factory=dict)` |
| `memory.py` | 补字段说明，`from_dict` 静默忽略未知字段（向前兼容） |
| `config.py` | `data_dir` 处理 `_config_path=None` 情况 |
| `compressor.py` | 提取阈值参数，废弃硬编码 0.8；优化文档 |
| `tracer.py` | 增加 O(n) 扫描的设计注释 |
| `store.py` | `trace()` 统一返回值类型（之前长程返回 dict 其他返回 MemoryEntry） |
| `layers/short_term.py` | **核心修复**：搜索评分从硬编码 0.8 改为按关键词命中率动态算分 |
| `layers/long_term.py` | `stats()` 字符数改为从 ChromaDB 原生文档算，不再依赖侧边 metadata |
| `layers/working.py` | 补注释，清理未使用 import |

### 安装向导重构 (`cli.py` + `deps.py`)

**deps.py**（新建）
- `check_python()` — 检查 >=3.10
- `check_all()` — 检测 4 项运行时依赖的安装状态和版本
- `install_missing(dry_run)` — 交互式 pip 安装缺失包

**安装向导流程**：

```
选语言 (可选中/英)
  │
  ├── [1/3] 环境检查
  │   ├── Python 版本 → 过低则：打开下载页面 / 退出
  │   ├── 依赖检查 → 缺失则：自动安装 / 手动安装后退出
  │   └── OpenClaw 工作区 → 找不到则退出（暂仅支持 OpenClaw）
  │
  ├── [2/3] Skill 冲突检查
  │   └── 扫描 skills/ 下所有 SKILL.md，询问是否禁用
  │
  ├── [3/3] 导入数据
  │   └── 扫描 MEMORY.md / memory/*.md / notes.md，逐行导入
  │
  └── 完成摘要
```

**关键设计决策**：
- 双语支持：安装开始时选语言，全程跟随
- 不再硬编码冲突 Skill 名单：改为通用扫描 `skills/` 下所有目录
- 移除个人内容：座右铭、复习 Skill 等全部去掉
- 每个检出点都带 pause（`input()`），防止终端一闪而过

### 文件变更

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `hippocampus/cli.py` | 重写 | ~210 | 双语安装向导 + doctor 命令 |
| `hippocampus/deps.py` | 新建 | ~150 | 依赖检测和自动安装 |
| `hippocampus/memory.py` | 优化 | — | 全面注释 |
| `hippocampus/config.py` | 优化 | — | 全面注释 |
| `hippocampus/store.py` | 优化 | — | 全面注释 + trace 统一 |
| `hippocampus/tracer.py` | 优化 | — | 全面注释 |
| `hippocampus/compressor.py` | 优化 | — | 参数提取 + 注释 |
| `hippocampus/layers/*.py` | 优化 | — | 注释 + 搜索评分修复 |
| `README.md` | 新建 | ~150 | 中文 README |
| `DEVLOG.md` | 更新 | — | 本文档 |

### 未完成

- 依赖的版本检查目前只比较最低版本，未对接 PyPI 检查最新版
- `install_missing` 中 pip 包名查找用了硬编码字典，应改为从 `deps.py` 的 `REQUIRED` 表自动映射
- 安装向导未做单元测试（后续通过 GitHub CI 补）
- `hippo doctor --upgrade` 检查 PyPI 最新版的功能未实现
