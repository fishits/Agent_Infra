# Agent Infrastructure

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文

### 🚀 项目概述

Agent Infrastructure 是一个轻量级、高性能的多智能体框架，专注于快速构建和部署智能体工作流。

### ⭐ 核心亮点

**Nano Framework** - 极简但强大的智能体框架：
- 🚀 **5 行代码创建智能体** - 最简洁的 API 设计
- 🔄 **自动节点路由** - 智能体间无缝协作
- 💾 **会话持久化** - 自动保存和恢复上下文
- ⚡ **并发工具执行** - 多工具并行调用
- 🎯 **Boss-Worker 模式** - OnlyTeam 并行任务分发

### 📦 项目结构

```
Agent_Infra/
├── nano framework/         # ⭐ Nano 框架（推荐）
│   ├── framework/         # 核心框架实现
│   ├── OnlyTeam/          # Boss-Worker 多智能体系统
│   └── DatasetTeam/       # 数据处理智能体团队
└── langgraph/             # LangGraph 框架实现
    └── README.md          # 详见 LangGraph 文档
```

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/fishits/Agent_Infra.git
cd Agent_Infra
pip install openai python-dotenv  # 基础依赖
```

### 配置

创建 `.env` 文件：
```env
OPENAI_API_KEY=your_key
# 或使用 DeepSeek
DEEPSEEK_API_KEY=your_key
```

---

## 💡 Nano Framework 使用指南

### 1. 最简单的智能体（5 行代码）

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import end_decide
from nano_framework.framework.io_tools import read_file, write_file

# 创建智能体
llm = deepseek_llm()
agent = node(
    name="assistant",
    llm=llm,
    tools=[read_file, write_file, end_decide],
    prompt="你是一个文件助手，帮助用户读写文件",
    edges={"end": "任务完成时使用"}
)

# 运行
g = graph(nodes=[agent])
run(g, user_message="读取 README.md 并总结")
```

### 2. 多智能体协作（3 个节点）

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import decide, end_decide
from nano_framework.framework.io_tools import read_file, write_file

llm = deepseek_llm()

# 规划者
planner = node(
    name="planner",
    llm=llm,
    tools=[read_file, decide],
    prompt="分析任务并制定计划",
    edges={
        "coder": "需要编写代码时",
        "writer": "需要写文档时"
    }
)

# 编码者
coder = node(
    name="coder",
    llm=llm,
    tools=[write_file, decide],
    prompt="编写代码实现功能",
    edges={"reviewer": "代码完成后"}
)

# 审查者
reviewer = node(
    name="reviewer",
    llm=llm,
    tools=[read_file, end_decide],
    prompt="审查代码质量",
    edges={"end": "审查通过"}
)

# 运行多智能体工作流
g = graph(nodes=[planner, coder, reviewer], start_node="planner")
run(g, user_message="创建一个 Python 计算器")
```

### 3. OnlyTeam - Boss-Worker 并行分发

**一键启动 Boss 智能体：**

```bash
cd "nano framework"
python OnlyTeam/boss.py
```

**Boss 会自动：**
1. 分析你的任务需求
2. 生成多个并行任务卡片
3. 分发给子智能体执行
4. 汇总结果

**示例对话：**
```
Please describe your dispatch request: 
> 分析 src/ 目录下所有 Python 文件，找出代码质量问题并生成报告

[Boss 自动生成 3 个并行任务]
Task 1: 扫描文件列表
Task 2: 并行分析每个文件
Task 3: 汇总生成报告
```

### 4. 自定义工具

```python
from nano_framework.framework.tools import tool

@tool
def search_database(query: str) -> str:
    """在数据库中搜索信息"""
    # 你的实现
    return f"搜索结果: {query}"

# 添加到智能体
agent = node(
    name="db_agent",
    llm=llm,
    tools=[search_database, end_decide],
    prompt="你是数据库助手",
    edges={"end": "完成"}
)
```

---

## 🎯 核心概念

### Node（节点）
每个节点是一个独立的智能体，包含：
- **LLM**: 语言模型
- **Tools**: 可用工具列表
- **Prompt**: 系统提示词
- **Edges**: 路由规则（跳转到其他节点）

### Graph（图）
多个节点组成的工作流：
```python
graph(
    nodes=[node1, node2, node3],
    start_node="node1",  # 起始节点
    state={"workspace": "./"}  # 共享状态
)
```

### 控制流工具
- `decide(target, content)` - 跳转到指定节点
- `new_decide(target, content)` - 跳转并清空目标节点记忆
- `end_decide()` - 结束工作流
- `chat(message)` - 等待用户输入

### 自动特性
- ✅ **自动上下文管理** - 无需手动传递消息历史
- ✅ **自动工具调用** - 强制每轮调用工具
- ✅ **自动会话保存** - 支持中断恢复
- ✅ **自动并发执行** - 多工具自动并行

---

## 📚 完整示例

### 代码审查工作流

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import decide, end_decide
from nano_framework.framework.io_tools import read_file, write_file, list_directory

llm = deepseek_llm(model="deepseek-v4-pro")

# 1. 扫描器 - 找出所有代码文件
scanner = node(
    name="scanner",
    llm=llm,
    tools=[list_directory, decide],
    prompt="扫描项目目录，找出所有需要审查的代码文件",
    edges={"analyzer": "找到文件后传递给分析器"}
)

# 2. 分析器 - 分析代码质量
analyzer = node(
    name="analyzer",
    llm=llm,
    tools=[read_file, decide],
    prompt="""分析代码文件，检查：
    - 代码规范
    - 潜在 bug
    - 性能问题
    - 安全隐患""",
    edges={"reporter": "分析完成后"}
)

# 3. 报告器 - 生成报告
reporter = node(
    name="reporter",
    llm=llm,
    tools=[write_file, end_decide],
    prompt="汇总分析结果，生成 Markdown 格式的审查报告",
    edges={"end": "报告生成完毕"}
)

# 运行
g = graph(
    nodes=[scanner, analyzer, reporter],
    start_node="scanner",
    state={"workspace": "./src"}
)
run(g, user_message="审查 src/ 目录下的代码")
```

---

## 🛠️ 高级特性

### 1. 流式输出

```python
run(g, user_message="你的任务", stream=True)
```

### 2. 会话恢复

```python
from nano_framework.framework.session import pick_run_id

# 选择之前的会话
run_id = pick_run_id(run_prefix="my_agent")
g = graph(nodes=[...], run_id=run_id)
run(g)
```

### 3. 自定义事件处理

```python
def my_event_handler(event: dict):
    if event["type"] == "tool_call":
        print(f"调用工具: {event['name']}")

run(g, user_message="任务", event_output=my_event_handler)
```

### 4. 动态提示词

```python
def dynamic_prompt(state: dict) -> str:
    workspace = state.get("workspace", ".")
    return f"你在 {workspace} 目录工作"

agent = node(
    name="agent",
    llm=llm,
    tools=[...],
    prompt=dynamic_prompt,  # 函数而非字符串
    edges={...}
)
```

---

## 🔧 可用工具

### 文件操作
- `read_file(path)` - 读取文件
- `write_file(path, content)` - 写入文件
- `list_directory(path)` - 列出目录
- `glob_files(pattern)` - 模式匹配文件
- `grep_search(pattern, path)` - 搜索文件内容

### 控制流
- `decide(target, content)` - 节点跳转
- `chat(message)` - 用户交互
- `end_decide()` - 结束流程

### 并行工具
- `parallel_tools(tool_calls)` - 并发执行多个工具

---

## 📖 LangGraph 框架

如果你需要更复杂的企业级功能，查看 [langgraph/README.md](langgraph/README.md)

---

<a name="english"></a>
## English

### 🚀 Overview

Agent Infrastructure is a lightweight, high-performance multi-agent framework focused on rapid agent workflow development.

### ⭐ Highlights

**Nano Framework** - Minimal yet powerful:
- 🚀 **5-line agent creation** - Simplest API design
- 🔄 **Auto node routing** - Seamless agent collaboration
- 💾 **Session persistence** - Auto save/restore context
- ⚡ **Concurrent tool execution** - Parallel tool calls
- 🎯 **Boss-Worker pattern** - OnlyTeam parallel dispatch

### 🚀 Quick Start

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import end_decide
from nano_framework.framework.io_tools import read_file

llm = deepseek_llm()
agent = node(
    name="assistant",
    llm=llm,
    tools=[read_file, end_decide],
    prompt="You are a helpful assistant",
    edges={"end": "When done"}
)

g = graph(nodes=[agent])
run(g, user_message="Read and summarize README.md")
```

### 📚 Documentation

- **Nano Framework**: See examples above
- **LangGraph Framework**: [langgraph/README.md](langgraph/README.md)

### 🤝 Contributing

Contributions welcome! Feel free to submit Pull Requests.

---

## 🌟 Star History

If you find this project helpful, please give it a star! ⭐
