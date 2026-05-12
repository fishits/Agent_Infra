# Agent Infrastructure

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文

### 🚀 项目概述

Agent Infrastructure 是一个轻量级、高性能的多智能体框架，专注于快速构建和部署智能体工作流。

### ⭐ 核心亮点

**Nano Framework** - 极简但强大的智能体框架：
- 🚀 **极简 API** - 最直观的智能体定义方式
- 🔄 **自动节点路由** - 智能体间无缝协作
- 💾 **会话持久化** - 自动保存和恢复上下文
- ⚡ **并发工具执行** - 多工具并行调用
- 🎯 **Boss-Worker 模式** - OnlyTeam 并行任务分发
- 🛠️ **工具自由定制** - 装饰器即可扩展任意功能

### 📦 项目结构

```
Agent_Infra/
├── nano framework/         # ⭐ Nano 框架（推荐）
│   ├── framework/         # 核心框架实现
│   └── OnlyTeam/          # Boss-Worker 多智能体系统
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

### 1. 快速开始

```python
from nano_framework.framework import node, graph, run, deepseek_llm
from nano_framework.framework.control_tools import end_decide
from nano_framework.framework.io_tools import read_file, write_file, list_directory

llm = deepseek_llm()

# 定义智能体
assistant = node(
    name="assistant",
    llm=llm,
    tools=[read_file, write_file, list_directory, end_decide],
    prompt="你是一个文件助手，帮助用户管理和分析文件",
    edges={"end": "任务完成时使用"}
)

# 运行
g = graph(nodes=[assistant])
run(g, user_message="列出当前目录，读取 README.md 并总结")
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

### 4. Codex - 通用编码智能体

**单智能体模式，适合日常编码任务：**

```bash
cd "nano framework"
python Codex/codex_agent.py --stream
```

Codex 特点：
- 完整的文件操作和代码执行能力
- 自动会话管理和恢复
- 支持后台任务执行
- 内置 TODO 任务管理

### 5. 自定义工具

**示例 1：数据库查询工具**

```python
from nano_framework.framework.tools import tool
import sqlite3

@tool(
    description="在 SQLite 数据库中执行查询",
    params={
        "query": ("string", "SQL 查询语句"),
        "db_path": ("string", "数据库文件路径")
    }
)
def query_database(query: str, db_path: str = "data.db") -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(query)
    results = cursor.fetchall()
    conn.close()
    return f"查询返回 {len(results)} 行:\n" + "\n".join(str(row) for row in results)
```

**示例 2：HTTP API 调用工具**

```python
@tool(
    description="调用 REST API",
    params={
        "url": ("string", "API 端点 URL"),
        "method": ("string", "HTTP 方法 GET/POST/PUT/DELETE"),
        "data": ("object", "请求体数据（JSON）")
    }
)
def call_api(url: str, method: str = "GET", data: dict = None) -> str:
    import requests
    response = requests.request(method, url, json=data)
    return f"状态码: {response.status_code}\n响应: {response.text[:500]}"
```

**示例 3：邮件发送工具**

```python
@tool(
    description="发送邮件通知",
    params={
        "to": ("string", "收件人邮箱"),
        "subject": ("string", "邮件主题"),
        "body": ("string", "邮件正文")
    }
)
def send_email(to: str, subject: str, body: str) -> str:
    import smtplib
    from email.mime.text import MIMEText
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['To'] = to
    
    # 配置你的 SMTP 服务器
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your_email@gmail.com', 'your_password')
        server.send_message(msg)
    
    return f"邮件已发送到 {to}"
```

**使用自定义工具：**

```python
# 创建智能体并添加自定义工具
agent = node(
    name="data_assistant",
    llm=llm,
    tools=[
        query_database,   # 自定义工具
        call_api,         # 自定义工具
        send_email,       # 自定义工具
        read_file,        # 内置工具
        write_file,       # 内置工具
        end_decide
    ],
    prompt="你是数据助手，可以查询数据库、调用 API、发送邮件",
    edges={"end": "完成"}
)
```

**支持的参数类型：**
- `string` - 字符串
- `integer` - 整数  
- `number` - 浮点数
- `boolean` - 布尔值
- `array` - 数组
- `object` - 对象/字典

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
- 🚀 **Quick start in 15 lines** - Simplest API design
- 🔄 **Auto node routing** - Seamless agent collaboration
- 💾 **Session persistence** - Auto save/restore context
- ⚡ **Concurrent tool execution** - Parallel tool calls
- 🎯 **Boss-Worker pattern** - OnlyTeam parallel dispatch
- 🛠️ **Custom tools** - Extend with decorators

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
