import ast
import math
import operator
from typing import Any
from pydantic import BaseModel, Field

from src.tools.tool_base import Tool


class CalculatorInput(BaseModel):
    """Pydantic model for CalculatorTool parameters"""

    expression: str = Field(description="数学表达式，如 2+3 或 sqrt(16)")


class CalculatorTool(Tool):
    """Simple mathematical calculation tool"""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="数学计算工具，支持基本运算和数学函数"
        )

    def run(self, parameters: dict[str, Any]) -> str:
        """Execute calculation with validated parameters"""
        expression = parameters.get("expression", "").strip()

        if not expression:
            return "Error: expression cannot be empty"

        return self._calculate(expression)

    def _calculate(self, expression: str) -> str:
        """Simple math calculating function"""
        operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
        }

        functions = {
            "sqrt": math.sqrt,
            "pi": math.pi,
        }

        try:
            node = ast.parse(expression, mode="eval")
            result = self._eval_node(node.body, operators, functions)
            return str(result)
        except Exception as e:
            return f"Failed to calculate, please check the expression format: {str(e)}"

    def _eval_node(self, node, operators, functions):
        """Evaluate AST node recursively"""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, operators, functions)
            right = self._eval_node(node.right, operators, functions)
            op = operators.get(type(node.op))
            return op(left, right)
        elif isinstance(node, ast.Call):
            func_name = getattr(node.func, "id", None)
            if func_name and func_name in functions:
                args = [self._eval_node(arg, operators, functions) for arg in node.args]
                return functions[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in functions:
                return functions[node.id]

    def get_input_schema(self):
        """Return Pydantic model for parameter validation"""
        return CalculatorInput


def calculate(expression: str) -> str:
    """
    Simple math calculating function (for backward compatibility).
    Deprecated: Use CalculatorTool class instead.
    """
    if not expression.strip():
        return "expression cannot be empty"

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }

    functions = {
        "sqrt": math.sqrt,
        "pi": math.pi,
    }

    try:
        node = ast.parse(expression, mode="eval")
        result = _eval_node(node.body, operators, functions)
        return str(result)
    except Exception as e:
        return f"Failed to calculate, please check the expression format: {str(e)}"


def _eval_node(node, operators, functions):
    """Evaluate AST node recursively (for backward compatibility)."""
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.BinOp):
        left = _eval_node(node.left, operators, functions)
        right = _eval_node(node.right, operators, functions)
        op = operators.get(type(node.op))
        return op(left, right)
    elif isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None)
        if func_name and func_name in functions:
            args = [_eval_node(arg, operators, functions) for arg in node.args]
            return functions[func_name](*args)
    elif isinstance(node, ast.Name):
        if node.id in functions:
            return functions[node.id]
