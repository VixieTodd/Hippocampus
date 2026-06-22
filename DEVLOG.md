# Hippocampus 开发日志

## 2026-06-22 15:40-15:51 — install 安装向导

### 背景
小狐担心 Hippocampus 与已有的记忆相关 Skill（memory-setup、proactive-agent、
self-improving-agent-skill）以及 MEMORY.md 体系产生冲突。讨论后决定在安装阶段解决：
安装时询问是否迁移已有记忆、是否禁用冲突 Skill。

### 实现路径

```
需求: 安装时防冲突
  │
  ├── 方案A: 运行时桥接层 ── 放弃 (维护成本高、两个系统设计哲学不同)
  ├── 方案B: 选一边废弃   ── 放弃 (要么丢Hippocampus要么丢MEMORY.md)
  ├── 方案C: 明确分工      ── 放弃 (边界模糊容易漂移)
  └── 方案D: 安装向导 ✅   ── 在安装阶段一次性解决
       │
       ├── 1. 检测环境 (workspace/memory/skills)
       ├── 2. 询问是否迁移记忆
       ├── 3. 询问是否禁用冲突Skill
       └── 4. 执行 + 摘要
```

**文件变更：**

| 文件 | 操作 | 行数 | 说明 |
|------|------|------|------|
| `hippocampus/installer.py` | 新建 | ~230 | 检测、迁移、禁用、向导 |
| `hippocampus/cli.py` | 修改 | +17 | 新增 install 命令 |
| `DEVLOG.md` | 新建 | — | 本文档 |

**模块结构 (`installer.py`)：**

```
detect_environment()          → 扫描workspace, 返回report dict
  ├── _find_workspace()       → 向上查找OpenClaw标记文件
  └── 统计memory_files/skills/conflict_count

print_environment_report()    → 格式化打印检测结果

migrate_memories()            → 核心迁移逻辑
  ├── _parse_markdown_sections()  → 按##/###拆分markdown为独立条目
  ├── 过滤短内容 (<10字符)
  ├── MemoryEntry.create() + long_term.add_batch()
  └── shutil.move → .archive/ (可选)

disable_skills()              → 写.disabled标记到skill目录

run_install_wizard()          → 交互式3步向导 (click.confirm/prompt)
auto_install()                → 非交互API (--yes)
```

**CLI 接口：**
```bash
hippo install          # 交互式向导
hippo install --yes    # 静默自动安装
```

### 设计决策

- **Markdown拆分用##标题**：MEMORY.md 的 `## 关于我` / `## 关于小狐` 等作为天然分段边界，每个段成为一个独立记忆条目，metadata里保留 `section` 和 `original_file`
- **归档而非删除**：原文件移到 `memory/.archive/`，不丢数据
- **.disabled而非删除Skill**：写JSON标记文件，随时可手动恢复
- **CJK分词已就绪**：TF-IDF自带中日韩单字+双字bigram分词，中文迁移后搜索精度有保障
- **不碰现有逻辑**：Click命令是独立函数，install 命令零侵入

### 测试结果（子agent: hippo_install_test）

| 模块 | 通过/总 | 关键数据 |
|------|---------|----------|
| CLI注册 | 4/4 | `install --help` 正常 |
| 环境检测 | 10/10 | workspace ✓, 5个记忆文件, 3个冲突skill |
| Markdown解析 | 7/7 | MEMORY.md → 19个独立条目 |
| 代码导入 | 7/7 | 所有模块无语法错误 |
| 模拟迁移 | 8/8 | 19条迁移 (archive=False, 原文件未动) |
| Skill禁用 | 9/9 | .disabled标记正确创建/验证/清理 |
| **总计** | **45/45** | **全部通过** |

### 使用示例

```
$ hippo install

🦛 Hippocampus 安装向导
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1/3] 检测环境...
  ✓ 找到 OpenClaw 工作区: C:\Users\zhong\.openclaw\workspace
  ...(略)

[2/3] 记忆迁移
  检测到已有记忆文件。是否将现有记忆重组导入 Hippocampus？
  [Y] 是  [N] 否
  > Y

[3/3] Skill 冲突处理
  ⚠ memory-setup
  ⚠ proactive-agent
  ⚠ self-improving-agent-skill
  是否禁用这些 Skill？
  [Y] 是  [N] 否
  > Y

✓ 安装完成
  导入了 19 条记忆 | 禁用了 3 个冲突 Skill
  Hippocampus 现在是你的主记忆引擎。🦛
```
