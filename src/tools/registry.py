from types import FunctionType
from typing import Any, Callable, Optional
from rich import print as rprint
from tool_base import Tool


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}


    def register_tool(self, tool: Tool):
        """
        Register a `Tool` object tool, suitable for complex tools.
        """

        if tool.name in self._tools:
            rprint(f"[yellow][Registry] ⚠️ Warning: tool `{tool.name}` already exists, going to be overwritten[/yellow]")
        self._tools[tool.name] = tool
        rprint(f"[bold magenta][Registry] ✅ Tool `{tool.name}` is registered [/bold magenta]")

    
    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        Directly register function as tool (convenient), suitable for simple tools.

        Args:
            `name`: tool name
            `description`: function description
            `func`: callable function, return string result
        """
        if name in self._functions:
            rprint(f"[yellow][Registry] ⚠️ Warning: function `{name}` already exists, going to be overwritten[/yellow]")
        self._functions[name] = {
            "description": description,
            "func": func
        }
        rprint(f"[bold magenta][Registry] ✅ Function `{name}` is registered [/bold magenta]")


    def get_tools_description(self) -> str:
        """
        Acquire formatted description strings of all available tools.
        """
        descriptions = []

        for tool_name, tool in self._tools.items():
            descriptions.append(f"- `{tool_name}`: {tool.description}")

        for func_name, func in self._functions.items():
            descriptions.append(f"- `{func_name}`: {func['description']}")

        return "\n".join(descriptions) if descriptions else "No available tool"


    def get_tool(self, tool_name: str) -> Optional[Tool]:
        return self._tools.get(tool_name)


    def get_funtion(self, func_name: str) -> Optional[Callable[[str], str]]:
        info = self._functions.get(func_name)
        return info.get("func") if info else None


    def execute_tool(self, tool_name: str, parameters: str) -> str:
        tool = self.get_tool(tool_name)
        return ""
