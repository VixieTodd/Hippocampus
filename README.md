# Hippocampus — AI Bionic Memory System

🦛 三层仿生记忆系统：短期 → 长期 → 工作记忆。

## 安装

```bash
cd hippocampus
pip install -e .
```

## 快速开始

```bash
# 写入记忆
hippo write "今天和晨聊了关于Hippocampus项目的设计"

# 搜索记忆
hippo search "Hippocampus" --top 5

# 查看统计
hippo stats

# 手动压缩（短期 → 长期）
hippo compress --force

# 追踪单条记忆
hippo trace <id>

# 导出备份
hippo export --format json -o backup.json
```

## 架构

```
新输入 → 短期记忆 (sliding window)
           ↓ (超阈值自动压缩)
         长期记忆 (Chroma 向量检索)
           ↑ 按需检索
         工作记忆 (静态规则/配置，始终在上下文)
```

## 配置

编辑 `config.yml` 自定义窗口大小、压缩阈值、Top-K 等参数。

## 许可

MIT License © 小狐 (VixieTodd)
