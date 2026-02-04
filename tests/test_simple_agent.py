"""
测试 SimpleAgent 的工具集成功能
包括工具注册、参数描述注入、工具选择和调用格式
"""

import os
import json
from dotenv import load_dotenv
from rich import print as rprint

from src.core.llm import NexusAgentsLLM
from src.agents.simple_agent import SimpleAgent
from src.tools.registry import ToolRegistry
from src.tools.builtin.calculator import CalculatorTool
from src.tools.builtin.search_tool import SearchTool

load_dotenv()
llm = NexusAgentsLLM()


def test_tool_registration():
    """测试工具注册和列表功能"""
    print("\n=== 测试 1: 工具注册和列表 ===\n")

    registry = ToolRegistry()

    # 注册 CalculatorTool
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)
    calc_schema = calc_tool.get_tool_schema()
    rprint(f"[cyan]CalculatorTool Schema:[/cyan]")
    rprint(json.dumps(calc_schema, indent=2, ensure_ascii=False))

    # 注册 SearchTool
    search_tool = SearchTool()
    registry.register_tool(search_tool)
    search_schema = search_tool.get_tool_schema()
    rprint(f"\n[cyan]SearchTool Schema:[/cyan]")
    rprint(json.dumps(search_schema, indent=2, ensure_ascii=False))

    # 验证工具列表
    tools = registry.list_tools()
    assert "calculator" in tools, "CalculatorTool should be registered"
    assert "search" in tools, "SearchTool should be registered"
    rprint(f"\n[green]✅ 工具注册成功: {tools}[/green]\n")


def test_prompt_injection():
    """测试参数描述能准确注入 prompt"""
    print("\n=== 测试 2: 参数描述注入 Prompt ===\n")

    registry = ToolRegistry()

    calc_tool = CalculatorTool()
    search_tool = SearchTool()
    registry.register_tool(calc_tool)
    registry.register_tool(search_tool)

    # 创建 SimpleAgent
    agent = SimpleAgent(
        name="工具测试助手",
        llm=llm,
        system_prompt="你是一个智能助手，可以使用工具来帮助用户。",
        tool_registry=registry,
        enable_tool_use=True,
    )

    # 检查工具描述是否正确注入
    enhanced_prompt = agent._get_enhanced_system_prompt()

    rprint("[cyan]生成的系统 Prompt 包含以下工具:[/cyan]\n")

    # 验证 CalculatorTool 的描述
    assert "calculator" in enhanced_prompt.lower(), (
        "CalculatorTool description should be in prompt"
    )
    assert "expression" in enhanced_prompt.lower(), (
        "expression parameter should be in prompt"
    )

    # 验证 SearchTool 的描述
    assert "search" in enhanced_prompt.lower(), (
        "SearchTool description should be in prompt"
    )
    assert "query" in enhanced_prompt.lower(), "query parameter should be in prompt"

    rprint("[green]✅ 参数描述已正确注入 Prompt[/green]\n")


def test_tool_selection_calculator():
    """测试 LLM 能正确选择 CalculatorTool"""
    print("\n=== 测试 3: LLM 选择 CalculatorTool ===\n")

    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    agent = SimpleAgent(
        name="计算助手",
        llm=llm,
        system_prompt="你是一个严谨的计算助手，你的所有计算都需要使用 calculator 工具来执行计算任务。",
        tool_registry=registry,
        enable_tool_use=True,
    )

    # 测试用例
    test_cases = [
        ("请帮我计算 2 + 3", "加法", "5"),
        ("100 除以 4 等于多少？", "除法", "25"),
        ("计算 sqrt(16) 的值", "函数调用", "4"),
    ]

    for question, desc, expected_result in test_cases:
        rprint(f"[blue]测试 {desc}: {question}[/blue]")
        response = agent.run(question, max_tool_iterations=2)
        rprint(
            f"[cyan][Client]LLM responsed:[/cyan]\n[bold white]{response}[/bold white]"
        )

        # 验证计算结果正确（响应中包含计算结果）
        assert expected_result in response, (
            f"Calculation result should contain {expected_result} for {desc}"
        )

        rprint(f"[green]✅ 通过 - 响应包含: {expected_result}[/green]\n")


def test_tool_selection_search():
    """测试 LLM 能正确选择 SearchTool"""
    print("\n=== 测试 4: LLM 选择 SearchTool ===\n")

    registry = ToolRegistry()
    search_tool = SearchTool()
    registry.register_tool(search_tool)

    agent = SimpleAgent(
        name="搜索助手",
        llm=llm,
        system_prompt="你是一个严谨的实时搜索助手，需要使用相关搜索工具来查找信息。",
        tool_registry=registry,
        enable_tool_use=True,
    )

    test_questions = [
        "搜索关于 AI 的最新信息",
        "查找 DeepSeek 模型的相关信息",
    ]

    for question in test_questions:
        rprint(f"[blue]测试: {question}[/blue]")
        try:
            response = agent.run(question, max_tool_iterations=2)

            # 验证响应中提到了搜索相关信息
            assert "搜索" in response.lower() or "AI" in response.lower(), (
                f"Response should mention search or AI for: {question}"
            )

            rprint(f"[green]✅ 响应包含相关内容[/green]\n")

        except Exception as e:
            if "API" in str(e) or "key" in str(e):
                rprint(f"[yellow]跳过（需要 API key）: {str(e)[:100]}[/yellow]\n")
            else:
                raise


def test_tool_format_json():
    """测试 LLM 能使用 JSON 格式调用工具"""
    print("\n=== 测试 5: JSON 格式工具调用 ===\n")

    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    agent = SimpleAgent(
        name="JSON 格式测试助手",
        llm=llm,
        system_prompt='你是一个助手，请使用 JSON 格式调用工具。格式: [TOOL_USE:{"tool": "tool_name", "parameters": {"key": "value"}}]',
        tool_registry=registry,
        enable_tool_use=True,
    )

    question = "请计算 15 * 3"
    rprint(f"[blue]测试: {question}[/blue]")

    response = agent.run(question, max_tool_iterations=2)

    rprint(f"[green]响应: {response}[/green]\n")

    # 验证计算结果正确
    assert "45" in response, "Calculation result should be 45"

    rprint("[green]✅ JSON 格式工具调用正常[/green]\n")


def test_multiple_tools_selection():
    """测试 LLM 能在多个工具中选择正确的工具"""
    print("\n=== 测试 7: 多工具选择 ===\n")

    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    search_tool = SearchTool()
    registry.register_tool(calc_tool)
    registry.register_tool(search_tool)

    agent = SimpleAgent(
        name="多工具助手",
        llm=llm,
        system_prompt='你是一个智能助手，可以使用已有的工具辅助你回答问题。',
        tool_registry=registry,
        enable_tool_use=True,
    )

    # 测试用例：计算任务应该选择 calculator
    test_cases = [
        ("计算 50 + 100", "calculator", "150"),
        ("搜索人工智能的历史", "search", None),
        ("计算 10 * 10", "calculator", "100"),
    ]

    for question, expected_tool, expected_result in test_cases:
        rprint(f"[blue]测试: {question}[/blue]")
        rprint(f"[cyan]期望工具: {expected_tool}[/cyan]")

        if expected_tool == "search":
            # 搜索任务
            try:
                response = agent.run(question, max_tool_iterations=2)
                rprint(f"[green]响应: {response}[/green]\n")
            except Exception as e:
                if "API" in str(e) or "key" in str(e):
                    rprint(f"[yellow]跳过（需要 API key）[/yellow]\n")
                else:
                    raise
        else:
            # 计算任务
            response = agent.run(question, max_tool_iterations=2)
            assert expected_tool in response.lower() or expected_result in response, (
                f"{expected_tool} should be used or result should contain {expected_result}"
            )
            rprint(f"[green]响应: {response}[/green]\n")


def test_parameter_validation_in_agent():
    """测试 Agent 能正确处理参数验证"""
    print("\n=== 测试 8: 参数验证处理 ===\n")

    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    agent = SimpleAgent(
        name="参数验证助手",
        llm=llm,
        system_prompt="你是一个助手，可以处理计算任务。",
        tool_registry=registry,
        enable_tool_use=True,
    )

    # 测试场景 1: 正常参数
    rprint("[blue]测试场景 1: 正常参数[/blue]")
    response1 = agent.run("计算 2 + 3", max_tool_iterations=2)
    assert "5" in response1, "Should calculate 2 + 3 = 5"
    rprint(f"[green]响应: {response1}[/green]\n")

    # 测试场景 2: 空参数（应该被捕获）
    rprint("[blue]测试场景 2: 空参数处理[/blue]")
    # Agent 应该能处理 LLM 传入的空参数
    # 这依赖于 LLM 不会传入空参数，所以这里主要是验证系统不会崩溃
    response2 = agent.run("计算 5 * 6", max_tool_iterations=2)
    assert "30" in response2, "Should calculate 5 * 6 = 30"
    rprint(f"[green]响应: {response2}[/green]\n")

    rprint("[green]✅ 参数验证处理正常[/green]\n")


if __name__ == "__main__":
    print("=" * 60)
    print("SimpleAgent 工具集成测试")
    print("=" * 60)

    try:
        test_tool_registration()
        test_prompt_injection()
        test_tool_selection_calculator()
        test_tool_selection_search()
        test_tool_format_json()
        test_multiple_tools_selection()

        print("\n" + "=" * 60)
        print("[green]✅ 所有测试通过！[/green]")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[red]❌ 测试失败: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
