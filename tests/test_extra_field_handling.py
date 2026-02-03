"""
测试工具的参数验证和额外字段处理
"""

from src.tools.registry import ToolRegistry
from src.tools.builtin.calculator import CalculatorTool
from src.tools.builtin.search_tool import SearchTool


def test_calculator_extra_field_handling():
    """测试 CalculatorTool 的额外字段处理（应忽略而非拒绝）"""
    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    print("\n=== Testing CalculatorTool extra='ignore' ===\n")

    # 测试 1: 正常输入
    print("Test 1: Normal input")
    success, validated, error = registry.parse_tool_parameters(
        "calculator", {"expression": "2+3"}
    )
    assert success, f"Expected success but got: {error}"
    assert "expression" in validated
    assert validated["expression"] == "2+3"
    print("✅ Normal input validated successfully")

    # 测试 2: 带额外字段的输入（应该被忽略）
    print("\nTest 2: Input with extra fields (should be ignored)")
    success, validated, error = registry.parse_tool_parameters(
        "calculator",
        {
            "expression": "2*3",
            "extra_field": "this should be ignored",
            "another_extra": "also ignored",
        },
    )
    assert success, f"Expected success (extra fields ignored) but got: {error}"
    assert "expression" in validated
    assert validated["expression"] == "2*3"
    assert "extra_field" not in validated
    assert "another_extra" not in validated
    print(f"✅ Extra fields correctly ignored. Validated: {validated}")

    # 测试 3: 执行工具（额外字段不影响结果）
    print("\nTest 3: Tool execution with extra fields")
    result = registry.execute_tool(
        "calculator", {"expression": "sqrt(16)", "hallucination": "LLM might add this"}
    )
    assert "4" in result or "4.0" in result
    print(f"✅ Tool executed successfully with extra fields. Result: {result}")

    # 测试 4: 空表达式（应该失败）
    print("\nTest 4: Empty expression (should fail validation)")
    success, validated, error = registry.parse_tool_parameters(
        "calculator", {"expression": ""}
    )
    assert not success, "Expected validation failure for empty expression"
    assert "empty" in error.lower()
    print(f"✅ Empty expression correctly rejected: {error}")

    # 测试 5: 数字类型（应该失败）
    print("\nTest 5: Wrong type (should fail validation)")
    success, validated, error = registry.parse_tool_parameters(
        "calculator", {"expression": 123}
    )
    assert not success, "Expected validation failure for wrong type"
    assert "string" in error.lower()
    print(f"✅ Wrong type correctly rejected: {error}")


def test_search_tool_extra_field_handling():
    """测试 SearchTool 的额外字段处理（应忽略而非拒绝）"""
    registry = ToolRegistry()
    search_tool = SearchTool()
    registry.register_tool(search_tool)

    print("\n=== Testing SearchTool extra='ignore' ===\n")

    # 测试 1: 正常输入
    print("Test 1: Normal input")
    success, validated, error = registry.parse_tool_parameters(
        "search", {"query": "DeepSeek"}
    )
    assert success, f"Expected success but got: {error}"
    assert "query" in validated
    assert validated["query"] == "DeepSeek"
    print(f"✅ Normal input validated successfully: {validated}")

    # 测试 2: 带额外字段的输入（应该被忽略）
    print("\nTest 2: Input with extra fields (should be ignored)")
    success, validated, error = registry.parse_tool_parameters(
        "search",
        {
            "query": "AI models",
            "thought": "LLM might add reasoning",
            "confidence": 0.95,
            "another_extra": "also ignored",
        },
    )
    assert success, f"Expected success (extra fields ignored) but got: {error}"
    assert "query" in validated
    assert validated["query"] == "AI models"
    assert "thought" not in validated
    assert "confidence" not in validated
    assert "another_extra" not in validated
    # 验证默认值自动填充
    assert validated["backend"] == "hybrid"
    assert validated["max_results"] == 5
    print(f"✅ Extra fields correctly ignored. Validated: {validated}")

    # 测试 3: 简单字符串（多参数工具）
    print("\nTest 3: Simple string with multi-param tool")
    success, validated, error = registry.parse_tool_parameters(
        "search", "latest AI trends"
    )
    assert success, f"Expected success but got: {error}"
    assert "query" in validated
    assert validated["query"] == "latest AI trends"
    print(f"✅ Simple string validated: {validated}")

    # 测试 4: 空查询（应该失败）
    print("\nTest 4: Empty query (should fail validation)")
    success, validated, error = registry.parse_tool_parameters("search", {"query": ""})
    assert not success, "Expected validation failure for empty query"
    assert "empty" in error.lower()
    print(f"✅ Empty query correctly rejected: {error}")

    # 测试 5: 纯空格（应该失败）
    print("\nTest 5: Whitespace-only query (should fail validation)")
    success, validated, error = registry.parse_tool_parameters(
        "search", {"query": "   "}
    )
    assert not success, "Expected validation failure for whitespace"
    assert "empty" in error.lower()
    print(f"✅ Whitespace correctly rejected: {error}")


def test_llm_hallucination_scenarios():
    """测试常见的 LLM 幻觉场景"""
    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    print("\n=== Testing LLM Hallucination Scenarios ===\n")

    # 场景 1: LLM 添加了推理过程字段
    print("Scenario 1: LLM adds reasoning field")
    success, validated, error = registry.parse_tool_parameters(
        "calculator",
        {
            "expression": "2+3",
            "reasoning": "I need to add 2 and 3",
            "step_by_step": "First add 2 and 3",
        },
    )
    assert success, "Should succeed even with reasoning fields"
    assert validated["expression"] == "2+3"
    assert "reasoning" not in validated
    print("✅ Reasoning fields ignored")

    # 场景 2: LLM 添加了置信度字段
    print("\nScenario 2: LLM adds confidence field")
    success, validated, error = registry.parse_tool_parameters(
        "calculator", {"expression": "10-4", "confidence": 0.99, "uncertainty": "low"}
    )
    assert success, "Should succeed even with confidence fields"
    assert validated["expression"] == "10-4"
    assert "confidence" not in validated
    print("✅ Confidence fields ignored")

    # 场景 3: LLM 添加了注释字段
    print("\nScenario 3: LLM adds comment/note fields")
    success, validated, error = registry.parse_tool_parameters(
        "calculator",
        {
            "expression": "5*6",
            "note": "Multiplication operation",
            "comment": "Simple math",
        },
    )
    assert success, "Should succeed even with comment fields"
    assert validated["expression"] == "5*6"
    assert "note" not in validated
    print("✅ Comment fields ignored")


if __name__ == "__main__":
    test_calculator_extra_field_handling()
    test_search_tool_extra_field_handling()
    test_llm_hallucination_scenarios()
    print("\n" + "=" * 60)
    print("✅ All tests passed! extra='ignore' configuration works correctly.")
    print("=" * 60)
