import json
from src.tools.registry import ToolRegistry
from src.tools.builtin.calculator import CalculatorTool


def test_calculator_tool():
    """测试新的 CalculatorTool 类"""

    registry = ToolRegistry()
    calc_tool = CalculatorTool()
    registry.register_tool(calc_tool)

    print("=== Testing CalculatorTool ===")
    print()

    print("1. 测试工具 Schema 生成:")
    schema = calc_tool.get_tool_schema()
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    print()

    print("2. 测试参数验证:")
    test_params = [
        {"expression": "2 + 3"},
        {"expression": "10 - 4"},
        {"expression": "5 * 6"},
        {"expression": "15 / 3"},
        {"expression": "sqrt(16)"},
    ]

    for i, params in enumerate(test_params, 1):
        print(f"测试 {i}: {params['expression']}")
        result = registry.execute_tool("calculator", params)
        print(f"结果: {result}\n")


def test_calculator_function():
    """测试旧的 calculate 函数（向后兼容）"""

    from src.tools.builtin import calculator

    registry = ToolRegistry()
    registry.register_function(
        name="calculator_legacy",
        description="旧的计算器函数（向后兼容）",
        func=calculator.calculate,
    )

    print("\n=== Testing Legacy Calculator Function ===")
    print()

    test_cases = [
        "2 + 3",
        "10 - 4",
        "5 * 6",
        "15 / 3",
        "sqrt(16)",
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("calculator_legacy", expression)
        print(f"结果: {result}\n")


if __name__ == "__main__":
    test_calculator_tool()
    test_calculator_function()
