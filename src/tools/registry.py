import json
from typing import Any, Callable, Optional, Union
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

    def register_function(
        self, name: str, description: str, func: Callable[[str], str]
    ):
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

    def get_tools_schema_json(self) -> list[dict]:
        """
        Generate JSON schemas for all registered tools.
        Used to pass tool definitions to LLM.

        Returns:
            List of tool schemas in JSON format
        """
        schemas = []

        for tool_name, tool in self._tools.items():
            schemas.append(tool.get_tool_schema())

        for func_name, func_info in self._functions.items():
            schemas.append(
                {
                    "name": func_name,
                    "description": func_info["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Input parameters for the function",
                            }
                        },
                        "required": ["input"],
                    },
                }
            )

        return schemas

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

    def parse_tool_parameters(
        self, tool_name: str, parameters: Union[str, dict]
    ) -> tuple[bool, dict | None, str | None]:
        """
        Parse and validate tool parameters.

        Supports multiple input formats:
        1. dict: Already parsed parameters
        2. JSON string: Standard JSON format
        3. Simple string: Legacy format for backward compatibility

        Returns:
            (success, parsed_params, error_message)
        """
        if isinstance(parameters, dict):
            return self._validate_and_parse_dict(tool_name, parameters)

        if isinstance(parameters, str):
            trimmed = parameters.strip()

            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    params_dict = json.loads(trimmed)
                    return self._validate_and_parse_dict(tool_name, params_dict)
                except json.JSONDecodeError as e:
                    return False, None, f"Invalid JSON format: {e}"
            else:
                return self._parse_simple_string(tool_name, trimmed)

        return False, None, f"Invalid parameter type: {type(parameters)}"

    def _validate_and_parse_dict(
        self, tool_name: str, params_dict: dict
    ) -> tuple[bool, dict | None, str | None]:
        """
        Validate parameters using tool's Pydantic schema.
        """
        tool = self.get_tool(tool_name)
        if tool:
            is_valid, validated_params, error_msg = tool.validate_parameters(params_dict)
            if not is_valid:
                return False, None, f"Parameter validation failed: {error_msg}"
            return True, validated_params, None

        func = self.get_funtion(tool_name)
        if func:
            return True, {"input": str(params_dict.get("input", ""))}, None

        return False, None, f"Tool not found: {tool_name}"

    def _parse_simple_string(
        self, tool_name: str, params_str: str
    ) -> tuple[bool, dict | None, str | None]:
        """
        Parse simple string format for backward compatibility.
        Format: "query=xxx,option=yyy" or just "xxx"
        """
        tool = self.get_tool(tool_name)
        if tool:
            schema_cls = tool.get_input_schema()
            if schema_cls:
                schema = schema_cls.model_json_schema()
                properties = schema.get("properties", {})

                if "=" not in params_str:
                    if properties:
                        param_name = list(properties.keys())[0]
                        return True, {param_name: params_str}, None
                    return False, None, "Multiple parameters required but none provided"

                params_dict = {}
                pairs = params_str.split(",")
                for pair in pairs:
                    if "=" in pair:
                        key, val = pair.split("=", 1)
                        params_dict[key.strip()] = val.strip()

                is_valid, validated_params, error_msg = tool.validate_parameters(params_dict)
                if not is_valid:
                    return False, None, f"Parameter validation failed: {error_msg}"
                return True, validated_params, None

            return True, {"input": params_str}, None

        func = self.get_funtion(tool_name)
        if func:
            return True, {"input": params_str}, None

        return False, None, f"Tool not found: {tool_name}"

    def execute_tool(
        self, tool_name: str, parameters: Union[str, dict]
    ) -> str | dict[str, Any]:
        """
        Execute a tool with validated parameters.

        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters in dict or string format

        Returns:
            Tool execution result or error message
        """
        # 1. Validating parameters passed from Agent/LLM
        success, validated_params, error_msg = self.parse_tool_parameters(tool_name, parameters)

        if not success or validated_params is None:
            return f"Error: {error_msg}"

        # 2. Pass valid parameters to `run` method/function
        tool = self.get_tool(tool_name)
        if tool:
            try:
                return tool.run(validated_params)
            except Exception as e:
                return f"Error: exception in tool `{tool_name}` execution: {str(e)}"

        func = self.get_funtion(tool_name)
        if func:
            try:
                input_val = validated_params.get("input", "")
                return func(input_val)
            except Exception as e:
                return f"Error: exception in function `{tool_name}` execution: {str(e)}"

        return f"Error: does not find tool named {tool_name}"


# Global tool registry
global_registry = ToolRegistry()
