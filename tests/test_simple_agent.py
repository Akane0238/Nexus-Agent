# test_simple_agent.py
import os
from dotenv import load_dotenv
from rich import print as rprint
from src.core.llm import NexusAgentsLLM
from src.agents.simple_agents import SimpleAgent
from src.tools.registry import ToolRegistry
from src.tools.builtin import calculator

# 创建LLM实例
# llm = NexusAgentsLLM(
#     provider="ollama",
#     model="qwen2.5:3b",
#     baseURL="http://localhost:11434/v1",
#     apiKey="no key"
# )
load_dotenv()
llm = NexusAgentsLLM()

# 测试1:基础对话Agent（无工具）
def test_basic():
    print()
    basic_agent = SimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。",
        enable_tool_use=False
    )

    response1 = basic_agent.run("你好，请介绍一下自己")
    rprint(f"[blue]基础对话响应[/blue]: {response1}\n")

# 测试2:带工具的Agent
def test_tool():
    print()
    tool_registry = ToolRegistry()
    tool_registry.register_function(
        name="calculator",
        description="一个支持基础四则运算与简单函数的Python计算器工具",
        func=calculator.calculate
    )

    enhanced_agent = SimpleAgent(
        name="增强助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=tool_registry,
        enable_tool_use=True
    )

    tool_list = enhanced_agent.list_tools()
    names = "\n- ".join(tool_list)
    rprint(f"[blue]SimpleAgent可用的工具名如下[/blue]:\n{names}\n\n")

    response2 = enhanced_agent.run("请帮我计算 15 * 8 + 32")
    rprint(f"[blue]工具增强响应[/blue]:\n[bold white]{response2}[/bold white]\n")


# 测试3:流式响应
def test_streaming():
    print()
    basic_agent = SimpleAgent(
        name="基础助手",
        llm=llm,
        system_prompt="你是一个友好的AI助手，请用简洁明了的方式回答问题。",
        enable_tool_use=False
    )
    for chunk in basic_agent.stream_run("请解释什么是人工智能"):
        pass

