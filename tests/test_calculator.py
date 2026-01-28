from src.tools.registry import ToolRegistry
from src.tools.builtin import calculator


def test_calculate():
    """测试自定义计算器工具"""

    # 实例化工具注册表
    registry = ToolRegistry()
    print()

    registry.register_function(
        name="calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=calculator.calculate
    )

    # 简单测试用例
    test_cases = [
        "2 + 3",           # 基本加法
        "10 - 4",          # 基本减法
        "5 * 6",           # 基本乘法
        "15 / 3",          # 基本除法
        "sqrt(16)",        # 平方根
    ]

    for i, expression in enumerate(test_cases, 1):
        print(f"测试 {i}: {expression}")
        result = registry.execute_tool("calculator", expression)
        print(f"结果: {result}\n")

