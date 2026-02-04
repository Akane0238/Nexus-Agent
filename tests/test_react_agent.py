"""
ReActAgent 全面测试套件
包含单元测试、集成测试和边界测试
"""

import sys
import json
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, "/home/akane/Projects/Nexus-Agent")

from dotenv import load_dotenv
from rich import print as rprint
from src.core.llm import NexusAgentsLLM
from src.agents.react_agent import ReActAgent, REACT_PROMPT_TEMPLATE
from src.tools.registry import ToolRegistry
from src.tools.builtin.calculator import CalculatorTool
from src.tools.builtin.search_tool import SearchTool
from src.core.message import Message

# 加载环境变量
load_dotenv()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """创建模拟的 LLM 实例"""
    llm = Mock(spec=NexusAgentsLLM)
    return llm


@pytest.fixture
def calculator_registry():
    """创建带有计算器工具的注册表"""
    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)
    return registry


@pytest.fixture
def multi_tool_registry():
    """创建带有多个工具的注册表"""
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(SearchTool())
    return registry


@pytest.fixture
def empty_registry():
    """创建空工具注册表"""
    return ToolRegistry()


# =============================================================================
# 单元测试 - ReActAgent 内部方法测试
# =============================================================================


class TestReActAgentUnit:
    """ReActAgent 单元测试 - 不需要真实 LLM"""

    def test_init_default_values(self, mock_llm, calculator_registry):
        """测试初始化时的默认值"""
        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
        )

        assert agent.name == "测试Agent"
        assert agent.llm == mock_llm
        assert agent.tool_registry == calculator_registry
        assert agent.max_steps == 5  # 默认值
        assert agent.current_history == []
        assert agent.prompt_template == REACT_PROMPT_TEMPLATE

    def test_init_custom_values(self, mock_llm, calculator_registry):
        """测试自定义初始化参数"""
        custom_prompt = "Custom prompt template"
        agent = ReActAgent(
            name="自定义Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=10,
            custom_prompt=custom_prompt,
        )

        assert agent.max_steps == 10
        assert agent.prompt_template == custom_prompt

    def test_format_tools_description(self, mock_llm, calculator_registry):
        """测试工具描述格式化"""
        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
        )

        desc = agent._format_tools_description()

        assert "### calculator" in desc
        assert "描述:" in desc
        assert "参数定义:" in desc
        assert "expression" in desc

    def test_format_empty_tools_description(self, mock_llm):
        """测试空工具描述格式化"""
        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=ToolRegistry(),
        )

        desc = agent._format_tools_description()
        assert desc == "No available tool"

    def test_parse_output_with_thought_and_action(self):
        """测试解析包含 Thought 和 Action 的输出"""
        agent = ReActAgent(name="测试", llm=Mock())

        output = """Thought: 我需要计算这个数学问题
Action: {"tool": "calculator", "parameters": {"expression": "10+20"}}"""

        thought, action = agent._parse_output(output)

        assert thought == "我需要计算这个数学问题"
        assert action == '{"tool": "calculator", "parameters": {"expression": "10+20"}}'

    def test_parse_output_with_finish_action(self):
        """测试解析包含 Finish 动作的输出"""
        agent = ReActAgent(name="测试", llm=Mock())

        output = """Thought: 我已经得到了答案
Action: Finish[30]"""

        thought, action = agent._parse_output(output)

        assert thought == "我已经得到了答案"
        assert action == "Finish[30]"

    def test_parse_output_missing_thought(self):
        """测试缺少 Thought 的输出"""
        agent = ReActAgent(name="测试", llm=Mock())

        output = "Action: Finish[答案]"

        thought, action = agent._parse_output(output)

        assert thought is None
        assert action == "Finish[答案]"

    def test_parse_output_missing_action(self):
        """测试缺少 Action 的输出"""
        agent = ReActAgent(name="测试", llm=Mock())

        output = "Thought: 我在思考"

        thought, action = agent._parse_output(output)

        assert thought == "我在思考"
        assert action is None

    def test_parse_action_json_format(self):
        """测试解析 JSON 格式的动作"""
        agent = ReActAgent(name="测试", llm=Mock())

        action = '{"tool": "calculator", "parameters": {"expression": "5*6"}}'

        tool_name, params = agent._parse_action(action)

        assert tool_name == "calculator"
        assert params == {"expression": "5*6"}

    def test_parse_action_invalid_json(self):
        """测试解析无效的 JSON 格式动作"""
        agent = ReActAgent(name="测试", llm=Mock())

        action = "这不是有效的JSON"

        tool_name, params = agent._parse_action(action)

        assert tool_name is None
        assert params is None

    def test_parse_action_empty_json(self):
        """测试解析空的 JSON 对象"""
        agent = ReActAgent(name="测试", llm=Mock())

        action = "{}"

        tool_name, params = agent._parse_action(action)

        assert tool_name is None
        assert params is None

    def test_parse_action_missing_fields(self):
        """测试解析缺少必要字段的 JSON"""
        agent = ReActAgent(name="测试", llm=Mock())

        action1 = '{"tool": "calculator"}'  # 缺少 parameters
        action2 = '{"parameters": {"x": 1}}'  # 缺少 tool

        tool_name1, params1 = agent._parse_action(action1)
        tool_name2, params2 = agent._parse_action(action2)

        assert tool_name1 is None and params1 is None
        assert tool_name2 is None and params2 is None


# =============================================================================
# 集成测试 - 需要真实或模拟 LLM
# =============================================================================


class TestReActAgentIntegration:
    """ReActAgent 集成测试"""

    @pytest.mark.skipif(not load_dotenv(), reason="需要环境变量配置")
    def test_simple_calculation(self):
        """测试简单的计算任务"""
        llm = NexusAgentsLLM()
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        agent = ReActAgent(
            name="计算助手",
            llm=llm,
            tool_registry=registry,
            max_steps=3,
        )

        response = agent.run("计算 10 + 20")

        assert "30" in response
        assert agent.current_history  # 验证历史记录被保存

    @pytest.mark.skipif(not load_dotenv(), reason="需要环境变量配置")
    def test_multi_step_reasoning(self):
        """测试多步推理任务"""
        llm = NexusAgentsLLM()
        registry = ToolRegistry()
        registry.register_tool(CalculatorTool())

        agent = ReActAgent(
            name="推理助手",
            llm=llm,
            tool_registry=registry,
            max_steps=5,
        )

        # 测试需要多步计算的问题
        response = agent.run("计算 15 乘以 8 的结果，然后加上 10")

        assert response  # 验证有响应

    def test_run_with_mock_llm_finish_directly(self, mock_llm, calculator_registry):
        """测试 LLM 直接返回 Finish 的情况"""
        # 模拟 LLM 直接返回 Finish
        mock_llm.invoke.return_value = """Thought: 这是一个简单问题，不需要工具
Action: Finish[直接回答]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=3,
        )

        response = agent.run("简单问题")

        assert response == "直接回答"
        mock_llm.invoke.assert_called_once()

    def test_run_with_mock_llm_tool_call(self, mock_llm, calculator_registry):
        """测试 LLM 调用工具的情况"""
        # 第一次调用：使用工具
        # 第二次调用：完成
        mock_llm.invoke.side_effect = [
            """Thought: 需要计算
Action: {"tool": "calculator", "parameters": {"expression": "5+5"}}""",
            """Thought: 得到结果了
Action: Finish[10]""",
        ]

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=3,
        )

        response = agent.run("计算 5+5")

        assert response == "10"
        assert mock_llm.invoke.call_count == 2

    def test_run_with_empty_llm_response(self, mock_llm, calculator_registry):
        """测试 LLM 返回空响应"""
        mock_llm.invoke.return_value = None

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=3,
        )

        response = agent.run("测试问题")

        # 应该返回错误信息或空字符串
        assert "Sorry" in response or response == ""

    def test_run_with_invalid_action(self, mock_llm, calculator_registry):
        """测试 LLM 返回无效动作格式"""
        mock_llm.invoke.return_value = """Thought: 我想执行操作
Action: 这不是有效的格式"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        response = agent.run("测试")

        # 验证历史记录被更新
        assert len(agent.current_history) >= 2
        assert "Invalid action format" in agent.current_history[1]


# =============================================================================
# 边界测试 - 异常情况和边界条件
# =============================================================================


class TestReActAgentEdgeCases:
    """ReActAgent 边界测试"""

    def test_max_steps_limit(self, mock_llm, calculator_registry):
        """测试最大步数限制"""
        # LLM 永远不返回 Finish，触发最大步数限制
        mock_llm.invoke.return_value = """Thought: 继续思考
Action: {"tool": "calculator", "parameters": {"expression": "1+1"}}"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        response = agent.run("测试")

        assert "maximum steps" in response or "Sorry" in response
        assert mock_llm.invoke.call_count == 2  # 只调用了 max_steps 次

    def test_tool_not_found(self, mock_llm, calculator_registry):
        """测试调用不存在的工具"""
        mock_llm.invoke.return_value = """Thought: 使用不存在的工具
Action: {"tool": "nonexistent_tool", "parameters": {"x": 1}}"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        response = agent.run("测试")

        # 验证工具执行返回错误
        assert "does not find tool" in str(agent.current_history) or "Error" in str(
            agent.current_history
        )

    def test_empty_input(self, mock_llm, calculator_registry):
        """测试空输入"""
        mock_llm.invoke.return_value = """Thought: 空输入
Action: Finish[收到]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        response = agent.run("")

        assert response == "收到"

    def test_special_characters_input(self, mock_llm, calculator_registry):
        """测试特殊字符输入"""
        mock_llm.invoke.return_value = """Thought: 处理特殊字符
Action: Finish[处理完成]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        special_input = "测试!@#$%^&*()_+{}|:<>?~`-=[]\\;',./\""
        response = agent.run(special_input)

        assert response == "处理完成"

    def test_long_input(self, mock_llm, calculator_registry):
        """测试超长输入"""
        mock_llm.invoke.return_value = """Thought: 处理长文本
Action: Finish[完成]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        long_input = "A" * 10000
        response = agent.run(long_input)

        assert response == "完成"

    def test_unicode_input(self, mock_llm, calculator_registry):
        """测试 Unicode 字符输入"""
        mock_llm.invoke.return_value = """Thought: 处理 Unicode
Action: Finish[成功]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        unicode_input = "你好世界 🌍 Привет мир こんにちは世界 🎉"
        response = agent.run(unicode_input)

        assert response == "成功"


# =============================================================================
# 自定义提示模板测试
# =============================================================================


class TestReActAgentCustomPrompt:
    """ReActAgent 自定义提示模板测试"""

    def test_custom_prompt_template(self, mock_llm):
        """测试使用自定义提示模板"""
        custom_template = """
自定义提示：{tools}
问题：{question}
历史：{history}
请回答：
"""

        agent = ReActAgent(
            name="自定义Agent",
            llm=mock_llm,
            tool_registry=ToolRegistry(),
            custom_prompt=custom_template,
            max_steps=2,
        )

        mock_llm.invoke.return_value = "Thought: 思考\nAction: Finish[答案]"

        agent.run("测试")

        # 验证调用时使用了自定义模板
        call_args = mock_llm.invoke.call_args
        prompt = call_args[0][0][0]["content"]

        assert "自定义提示：" in prompt
        assert "问题：" in prompt
        assert "历史：" in prompt

    def test_system_prompt(self, mock_llm, calculator_registry):
        """测试系统提示设置"""
        system_prompt = "你是一个数学专家"

        agent = ReActAgent(
            name="专家Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            system_prompt=system_prompt,
            max_steps=2,
        )

        assert agent.system_prompt == system_prompt


# =============================================================================
# 消息历史测试
# =============================================================================


class TestReActAgentMessageHistory:
    """ReActAgent 消息历史测试"""

    def test_message_history_tracking(self, mock_llm, calculator_registry):
        """测试消息历史跟踪"""
        mock_llm.invoke.return_value = """Thought: 完成任务
Action: Finish[答案]"""

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        agent.run("测试问题")

        # 验证消息被添加到历史
        history = agent.get_history()
        assert len(history) == 2
        assert history[0].content == "测试问题"
        assert history[0].role == "user"
        assert history[1].content == "答案"
        assert history[1].role == "assistant"

    def test_conversation_history_persistence(self, mock_llm, calculator_registry):
        """测试对话历史持久化"""
        mock_llm.invoke.side_effect = [
            "Thought: 思考1\nAction: Finish[答案1]",
            "Thought: 思考2\nAction: Finish[答案2]",
        ]

        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        agent.run("问题1")
        agent.run("问题2")

        history = agent.get_history()
        assert len(history) == 4  # 两轮对话


# =============================================================================
# 工具注册和管理测试
# =============================================================================


class TestReActAgentToolManagement:
    """ReActAgent 工具管理测试"""

    def test_dynamic_tool_registration(self, mock_llm):
        """测试动态工具注册"""
        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=ToolRegistry(),
            max_steps=2,
        )

        # 初始没有工具
        assert "No available tool" in agent._format_tools_description()

        # 动态添加工具
        agent.tool_registry.register_tool(CalculatorTool())

        assert "calculator" in agent._format_tools_description()

    def test_tool_unregistration(self, mock_llm, calculator_registry):
        """测试工具注销"""
        agent = ReActAgent(
            name="测试Agent",
            llm=mock_llm,
            tool_registry=calculator_registry,
            max_steps=2,
        )

        # 确认工具存在
        assert "calculator" in agent._format_tools_description()

        # 注销工具
        agent.tool_registry.unregister("calculator")

        # 确认工具已移除
        assert "No available tool" in agent._format_tools_description()


# =============================================================================
# 原有测试（保留向后兼容）
# =============================================================================


def test_json_format():
    """测试 ReActAgent 只能使用 JSON 格式（向后兼容）"""
    print("\n=== 测试 ReActAgent JSON 格式约束 ===\n")

    llm = NexusAgentsLLM()
    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    agent = ReActAgent(
        name="ReAct 测试助手",
        llm=llm,
        tool_registry=registry,
        max_steps=3,
    )

    # 测试用例
    test_cases = [
        "计算 10 + 20",
        "计算 5 * 6",
        "计算 sqrt(25)",
    ]

    for question in test_cases:
        rprint(f"[blue]测试: {question}[/blue]")

        response = agent.run(question)

        # 验证响应
        assert "5" in response or "30" in response or "2.236" in response, (
            f"Failed to calculate: {question}"
        )

        rprint(f"[green]✅ 响应: {response}[/green]\n")


def test_multiple_tools():
    """测试 ReActAgent 多工具选择（向后兼容）"""
    print("\n=== 测试 ReActAgent 多工具选择 ===\n")

    llm = NexusAgentsLLM()
    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    search_tool = SearchTool()
    registry.register_tool(calc_tool)
    registry.register_tool(search_tool)

    agent = ReActAgent(
        name="ReAct 多工具助手",
        llm=llm,
        tool_registry=registry,
        max_steps=3,
    )

    # 测试用例
    test_cases = [
        "计算 15 + 25",
        "计算 8 * 7",
    ]

    for question in test_cases:
        rprint(f"[blue]测试: {question}[/blue]")

        response = agent.run(question)

        # 验证响应
        assert "40" in response or "56" in response, f"Failed to calculate: {question}"

        rprint(f"[green]✅ 响应: {response}[/green]\n")


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ReActAgent 全面测试套件")
    print("=" * 60)

    # 运行向后兼容的测试
    try:
        test_json_format()
        test_multiple_tools()

        print("\n" + "=" * 60)
        print("[green]✅ 向后兼容测试通过！[/green]")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n[red]❌ 测试失败: {str(e)}[/red]")
        import traceback

        traceback.print_exc()

    # 提示运行 pytest 以执行完整测试套件
    print(
        "\n[blue]提示：运行 `pytest tests/test_react_agent.py -v` 以执行完整测试套件[/blue]"
    )
