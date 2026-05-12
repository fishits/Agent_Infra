# Agent Infrastructure

[English](#english) | [中文](#chinese)

---

<a name="chinese"></a>
## 中文

### 🚀 项目概述

Agent Infrastructure 是一个综合性的智能体基础设施项目，包含多个基于 LangGraph 的智能体框架实现。

### 📦 项目结构

```
Agent_Infra/
├── langgraph/              # LangGraph 智能体框架
│   ├── my_agent/          # 主智能体实现
│   │   ├── framework/     # 企业级框架
│   │   └── FrameWork_light/  # 轻量级 ReAct 框架
│   └── src/               # 额外的智能体实现
└── nano framework/         # Nano 框架实现
```

### 🎯 核心组件

#### 1. LangGraph 框架

完整的多智能体工作流框架，支持：
- **轻量级 ReAct 框架** (`FrameWork_light/`): 极简实现，适合快速开发
- **企业级框架** (`framework/`): 生产就绪，支持复杂工作流
- **OnlyTeam**: Boss-Worker 模式的并行任务分发系统
- **丰富的工具生态**: 文件操作、网络搜索、决策制定等

详细文档请查看 [langgraph/README.md](langgraph/README.md)

#### 2. Nano Framework

轻量级框架实现（开发中）

### 🚀 快速开始

#### 安装

```bash
# 克隆仓库
git clone https://github.com/fishits/Agent_Infra.git
cd Agent_Infra

# 安装 LangGraph 框架
cd langgraph
pip install -e .
```

#### 配置

复制环境变量模板：
```bash
cp langgraph/.env.example langgraph/.env
```

编辑 `.env` 文件，填入你的 API 密钥：
```env
OPENAI_API_KEY=your_api_key_here
```

#### 运行示例

**轻量级框架示例：**
```bash
cd langgraph/my_agent/FrameWork_light
python joke_demo_v2.py
```

**企业级框架示例：**
```python
from my_agent.framework import create_llm, define_react_agent
from my_agent.framework.tool_runtime import ToolRegistry

# 创建 LLM
llm = create_llm(model="gpt-4")

# 创建工具
tools = ToolRegistry()

# 定义智能体
agent = define_react_agent(
    name="assistant",
    channel="main",
    tool_registry=tools,
    prompt_builder=lambda state: "你是一个有用的助手",
    next_node_resolver=lambda state: "end"
)

# 运行
result = agent.run(initial_message="你好")
```

### 📚 文档

- [LangGraph 框架文档](langgraph/README.md)
- [轻量级框架说明](langgraph/my_agent/FrameWork_light/README.md)
- [工作流约定](langgraph/my_agent/workflow_conventions.md)

### 🛠️ 主要特性

- ✅ **多框架支持**: 轻量级和企业级两种实现
- ✅ **ReAct 模式**: 推理-行动循环
- ✅ **多智能体编排**: Boss-Worker、多节点工作流
- ✅ **工具生态**: 文件、网络、决策等工具
- ✅ **状态管理**: 结构化状态追踪
- ✅ **会话持久化**: 支持会话恢复
- ✅ **流式输出**: 实时响应流
- ✅ **LLM 无关**: 支持多种 LLM 提供商

### 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

### 📄 许可证

MIT License - 详见 [LICENSE](langgraph/LICENSE) 文件

---

<a name="english"></a>
## English

### 🚀 Project Overview

Agent Infrastructure is a comprehensive agent infrastructure project containing multiple LangGraph-based agent framework implementations.

### 📦 Project Structure

```
Agent_Infra/
├── langgraph/              # LangGraph agent framework
│   ├── my_agent/          # Main agent implementation
│   │   ├── framework/     # Enterprise framework
│   │   └── FrameWork_light/  # Lightweight ReAct framework
│   └── src/               # Additional agent implementations
└── nano framework/         # Nano framework implementation
```

### 🎯 Core Components

#### 1. LangGraph Framework

Complete multi-agent workflow framework with:
- **Lightweight ReAct Framework** (`FrameWork_light/`): Minimalist implementation for rapid development
- **Enterprise Framework** (`framework/`): Production-ready with complex workflow support
- **OnlyTeam**: Boss-Worker pattern for parallel task distribution
- **Rich Tool Ecosystem**: File operations, web search, decision-making, etc.

See [langgraph/README.md](langgraph/README.md) for detailed documentation

#### 2. Nano Framework

Lightweight framework implementation (in development)

### 🚀 Quick Start

#### Installation

```bash
# Clone repository
git clone https://github.com/fishits/Agent_Infra.git
cd Agent_Infra

# Install LangGraph framework
cd langgraph
pip install -e .
```

#### Configuration

Copy environment template:
```bash
cp langgraph/.env.example langgraph/.env
```

Edit `.env` file with your API keys:
```env
OPENAI_API_KEY=your_api_key_here
```

#### Run Examples

**Lightweight framework example:**
```bash
cd langgraph/my_agent/FrameWork_light
python joke_demo_v2.py
```

**Enterprise framework example:**
```python
from my_agent.framework import create_llm, define_react_agent
from my_agent.framework.tool_runtime import ToolRegistry

# Create LLM
llm = create_llm(model="gpt-4")

# Create tools
tools = ToolRegistry()

# Define agent
agent = define_react_agent(
    name="assistant",
    channel="main",
    tool_registry=tools,
    prompt_builder=lambda state: "You are a helpful assistant",
    next_node_resolver=lambda state: "end"
)

# Run
result = agent.run(initial_message="Hello")
```

### 📚 Documentation

- [LangGraph Framework Documentation](langgraph/README.md)
- [Lightweight Framework Guide](langgraph/my_agent/FrameWork_light/README.md)
- [Workflow Conventions](langgraph/my_agent/workflow_conventions.md)

### 🛠️ Key Features

- ✅ **Multi-Framework Support**: Lightweight and enterprise implementations
- ✅ **ReAct Pattern**: Reasoning-action loop
- ✅ **Multi-Agent Orchestration**: Boss-Worker, multi-node workflows
- ✅ **Tool Ecosystem**: File, web, decision tools
- ✅ **State Management**: Structured state tracking
- ✅ **Session Persistence**: Resume session support
- ✅ **Streaming Output**: Real-time response streaming
- ✅ **LLM Agnostic**: Multiple LLM provider support

### 🤝 Contributing

Contributions welcome! Feel free to submit Pull Requests.

### 📄 License

MIT License - see [LICENSE](langgraph/LICENSE) file

---

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐
