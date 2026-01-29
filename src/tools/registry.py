from typing import Any, Callable, Optional
from rich import print as rprint
from .tool_base import Tool


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


    def unregister(self, tool_name: str):
        if tool_name in self._tools:
            self._tools.pop(tool_name)
            rprint(f"[bold magenta][Registry] ✅ Tool `{tool_name}` has been removed[/bold magenta]")
        elif tool_name in self._functions:
            self._functions.pop(tool_name)
            rprint(f"[bold magenta][Registry] ✅ Function `{tool_name}` has been removed[/bold magenta]")
        else:
            rprint(f"[yellow][Registry] ⚠️ There is no tool/function named `{tool_name}` [/yellow]")


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

    
    def list_tools(self) -> list[str]:
        """
        List all the tools name
        """
        tools = []
        for tool in self._tools.keys():
            tools.append(tool)
        
        for func in self._functions.keys():
            tools.append(func)

        return tools


    def execute_tool(self, tool_name: str, parameters: str) -> str:
        """
        Args:
            `name`: tool name
            `parameters`: input parameters
        """
        # Search `Tool` object first
        tool = self.get_tool(tool_name)
        if tool:
            try:
                return tool.run({"input": parameters})
            except Exception as e:
                return f"Error: exception in tool `{tool_name}` execution: {str(e)}"

        # Search `func` function tool
        func = self.get_funtion(tool_name)
        if func:
            try:
                return func(parameters)
            except Exception as e:
                return f"Error: exception in function `{tool_name}` execution: {str(e)}"

        return f"Error: does not find the tool named {tool_name}"


# Global tool registry
global_registry = ToolRegistry()
