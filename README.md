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

### 2. 多智能体协作（完整工具配备）

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import decide, end_decide
from nano_framework.framework.io_tools import (
    read_file, write_file, list_directory, 
    glob_files, grep_search, apply_diff
)

llm = deepseek_llm()

# 规划者 - 配备文件搜索和读取工具
planner = node(
    name="planner",
    llm=llm,
    tools=[
        list_directory,  # 列出目录结构
        glob_files,      # 按模式查找文件
        grep_search,     # 搜索文件内容
        read_file,       # 读取文件
        decide           # 路由到下一节点
    ],
    prompt="分析项目结构，制定实现计划",
    edges={
        "coder": "需要编写代码时",
        "reviewer": "需要审查现有代码时"
    }
)

# 编码者 - 配备文件读写和编辑工具
coder = node(
    name="coder",
    llm=llm,
    tools=[
        read_file,       # 读取现有代码
        write_file,      # 创建新文件
        apply_diff,      # 精确编辑文件
        decide           # 路由
    ],
    prompt="根据计划编写高质量代码",
    edges={"reviewer": "代码完成后"}
)

# 审查者 - 配备代码分析工具
reviewer = node(
    name="reviewer",
    llm=llm,
    tools=[
        read_file,       # 读取代码
        grep_search,     # 搜索潜在问题
        apply_diff,      # 修复问题
        end_decide       # 结束流程
    ],
    prompt="审查代码质量、规范和安全性",
    edges={"end": "审查通过"}
)

# 运行多智能体工作流
g = graph(nodes=[planner, coder, reviewer], start_node="planner")
run(g, user_message="创建一个 Python 计算器模块")
```

### 3. OnlyTeam - Boss-Worker 并行分发

**启动 Streamlit UI：**

```bash
cd "nano framework"
streamlit run streamlit_onlyteam_app.py
```

打开浏览器访问 http://localhost:8501，通过可视化界面：
- 创建和管理多个 Boss 会话
- 实时查看任务分发和执行
- 监控子智能体并行工作
- 查看轨迹和性能指标

### 4. 自定义工具（完全自由）

**工具定义非常简单 - 只需装饰器 + 函数：**

```python
from nano_framework.framework.tools import tool

@tool(
    description="在数据库中搜索信息",
    params={
        "query": ("string", "搜索关键词"),
        "limit": ("integer", "返回结果数量")
    }
)
def search_database(query: str, limit: int = 10) -> str:
    """你的实现逻辑"""
    results = your_db.search(query, limit=limit)
    return f"找到 {len(results)} 条结果:\n" + "\n".join(results)

@tool(
    description="发送邮件通知",
    params={
        "to": ("string", "收件人邮箱"),
        "subject": ("string", "邮件主题"),
        "body": ("string", "邮件正文")
    }
)
def send_email(to: str, subject: str, body: str) -> str:
    """集成你的邮件服务"""
    your_email_service.send(to, subject, body)
    return f"邮件已发送到 {to}"

@tool(
    description="调用外部 API",
    params={
        "endpoint": ("string", "API 端点"),
        "method": ("string", "HTTP 方法"),
        "data": ("object", "请求数据")
    }
)
def call_api(endpoint: str, method: str = "GET", data: dict = None) -> str:
    """集成任何 REST API"""
    response = requests.request(method, endpoint, json=data)
    return response.text

# 添加到智能体 - 就这么简单！
agent = node(
    name="assistant",
    llm=llm,
    tools=[
        search_database,  # 你的自定义工具
        send_email,       # 你的自定义工具
        call_api,         # 你的自定义工具
        read_file,        # 内置工具
        write_file,       # 内置工具
        end_decide        # 控制流工具
    ],
    prompt="你是全能助手，可以搜索数据库、发邮件、调用 API",
    edges={"end": "任务完成"}
)
```

**工具参数类型支持：**
- `string` - 字符串
- `integer` - 整数
- `number` - 浮点数
- `boolean` - 布尔值
- `array` - 数组
- `object` - 对象/字典

**工具自动特性：**
- ✅ 自动转换为 OpenAI tools 格式
- ✅ 自动参数验证
- ✅ 自动错误处理
- ✅ 支持可选参数（默认值）
- ✅ 支持运行时上下文（`runtime_context` 参数）

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

## 🔧 内置工具库

### 文件操作（完整功能）
- `read_file(path, start_line, end_line)` - 按行范围读取，自动截断大文件
- `write_file(path, content)` - 写入文件，自动创建目录
- `append_file(path, content)` - 追加内容
- `apply_diff(path, search, replace)` - 精确文本替换
- `apply_patch(patch)` - 应用多文件补丁
- `list_directory(path, depth)` - 递归列出目录
- `glob_files(pattern, max_results)` - 模式匹配查找文件
- `grep_search(query, include_pattern)` - 正则搜索文件内容
- `read_many(paths)` - 批量读取多个文件

### 代码执行
- `run_python(code, timeout)` - 执行 Python 代码
- `run_powershell(command, timeout)` - 执行 PowerShell 命令
- `start_background_powershell(command, log_path)` - 后台执行长任务
- `check_background_powershell(pid, log_path)` - 检查后台任务状态

### 并行工具
- `parallel_tools(tasks_json)` - 并发执行多个独立工具调用

### 控制流
- `decide(target, content)` - 跳转到指定节点
- `new_decide(target, content)` - 跳转并清空目标节点记忆
- `end_decide()` - 结束工作流
- `chat(message)` - 等待用户输入

### 任务管理
- `todo_read()` - 读取持久化 TODO 列表
- `todo_write(todos_json)` - 更新 TODO 列表

**所有工具都支持：**
- 自动路径解析（相对于工作目录）
- 自动错误处理和友好提示
- 自动结果格式化
- 运行时上下文传递

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
