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
