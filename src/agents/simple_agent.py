"""Basic communicating agent implement"""

import re
import json
from typing import Optional, Iterator
from rich import print as rprint

from src.core.llm import NexusAgentsLLM
from src.core.agent import Agent
from src.core.config import Config
from src.core.message import Message
from src.tools.registry import ToolRegistry


class SimpleAgent(Agent):
    """Simple chatting Agent, support available tool use"""

    def __init__(
        self,
        name: str,
        llm: NexusAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_use: bool = True,
    ):
        """
        Args:
            `name`: name of agent
            `llm`: LLM client instant
            `system_prompt`: agent system prompt
            `config`: configuration object
            `tool_registry`: registry of tools (optional)
            `enable_tool_use`: activated only when providing `tool_registry`
        """
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry
        self.enable_tool_use = enable_tool_use and tool_registry is not None
        rprint(
            f"[bold magenta][Agent] ✅ {name} Initialization complete，tool use: {'Enable' if self.enable_tool_use else 'Disable'}[/bold magenta]"
        )

    def _get_enhanced_system_prompt(self) -> str:
        """
        Build an enhanced system prompt including tools information
        """
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_use or not self.tool_registry:
            return base_prompt

        tools_description = self._format_tools_description()
        if not tools_description:
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下JSON格式：\n"

        tools_section += "```\n"
        tools_section += (
            '[TOOL_USE:{"tool": "tool_name", "parameters": {"key": "value"}}]\n'
        )
        tools_section += "```\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 调用需符合上述JSON格式，参数必须符合工具定义\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += '- 数字参数直接写数字，不需要引号：`{"a": 12}`\n'
        tools_section += (
            "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"
        )

        return base_prompt + tools_section

    def _format_tools_description(self) -> str:
        """
        Format tools description with JSON Schema.
        """
        if not self.tool_registry:
            return ""

        schemas = self.tool_registry.get_tools_schema_json()

        descriptions = []
        for schema in schemas:
            name = schema["name"]
            desc = schema["description"]
            params = schema["parameters"]

            descriptions.append(f"### {name}")
            descriptions.append(f"描述: {desc}")
            descriptions.append(
                f"参数定义: {json.dumps(params, ensure_ascii=False, indent=2)}"
            )
            descriptions.append("")

        return "\n".join(descriptions) if descriptions else ""

    def _parse_tool_uses(self, response: str) -> list:
        """
        Parse tool use commands from LLM response.
        Supports JSON format: [TOOL_USE:{"tool": "...", "parameters": {...}}]
        """
        tool_uses = []

        # JSON format: [TOOL_USE:{"tool": "...", "parameters": {...}}]
        json_pattern = r"\[TOOL_USE:\{.*?\}\]"
        json_matches = re.findall(json_pattern, response)

        for match in json_matches:
            try:
                # Extract content between [TOOL_USE:{ and }]
                inner_json = match[10:-1]  # Remove [TOOL_USE:{ and }]
                action_dict = json.loads(inner_json)
                if isinstance(action_dict, dict):
                    tool_name = action_dict.get("tool")
                    parameters = action_dict.get("parameters")
                    if tool_name:
                        tool_uses.append(
                            {
                                "tool_name": tool_name.strip(),
                                "parameters": parameters if parameters else "",
                                "original": match,
                            }
                        )
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                rprint(f"[yellow][Agent] ⚠️ JSON parsing error: {e}[/yellow]")

        if not tool_uses and re.search(r"\[TOOL_USE:", response):
            rprint(
                f"[bold magenta][Agent] ⚠️ Tool use format detected but parsing failed[/bold magenta]"
            )

        return tool_uses

    def _execute_tool_use(self, tool_name: str, parameters) -> str:
        """
        Execute a tool with parsed parameters.
        """
        if not self.tool_registry:
            return f"Error: does not configure tool registry"

        try:
            result = self.tool_registry.execute_tool(tool_name, parameters)
            return f"Tool `{tool_name}` execution result:\n{result}"
        except Exception as e:
            return f"Failed to invoke tools: {str(e)}"

    def _run_with_tools(
        self, messages: list, input_text: str, max_tool_iterations: int, **kwargs
    ) -> str:
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            rprint(
                f"[bold blue]--- _run_with_tools: iteration {current_iteration + 1}/{max_tool_iterations} ---[/bold blue]"
            )
            response = self.llm.invoke(messages, **kwargs)
            rprint(
                f"[bold magenta][Client] ✅ LLM successfully response[/bold magenta]:\n[cyan]{response}[/cyan]"
            )

            # Parsing response
            tool_uses = self._parse_tool_uses(response)

            rprint(
                f"[bold magenta][Agent] 🔧 Detecting {len(tool_uses)} tool uses[/bold magenta]\n"
            )
            if tool_uses:
                # Calling all tools
                results = []
                clean_response = response

                for tool_use in tool_uses:
                    rprint(
                        f"[bold magenta][Agent] 🔧 Tool `{tool_use['tool_name']}` executing...[/bold magenta]"
                    )
                    rprint(f"[bold white]- {tool_use['original']}[/bold white]")

                    result = self._execute_tool_use(
                        tool_use["tool_name"], tool_use["parameters"]
                    )
                    rprint(f"[bold white]- {result}[/bold white]\n")
                    results.append(result)
                    clean_response = clean_response.replace(tool_use["original"], "")

                # Construct message with tool execution result
                messages.append({"role": "assistant", "content": clean_response})

                tool_results_text = "\n\n".join(results)
                messages.append(
                    {
                        "role": "user",
                        "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。",
                    }
                )

                current_iteration += 1
                continue

            else:
                # No tool use, return final answer
                final_response = response
                break

        # Get the last answer if iterations exceed
        if current_iteration >= max_tool_iterations and not final_response:
            final_response = self.llm.invoke(messages, **kwargs)

        # Save to `_history`
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_response, "assistant"))
        rprint(
            f"[bold magenta][Agent] ✅ {self.name} finished response[/bold magenta]\n"
        )

        return final_response

    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        Overwrite `run` function,
        implement simple conversational logic and support tool use.
        """
        rprint(
            f"[bold magenta][Agent] 🤖 {self.name} is processing: {input_text}[/bold magenta]"
        )

        # Create a message list
        messages = []

        # Add system message and tools information(if exist)
        enhanced_system_prompt = self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # Add history information
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # Add user information
        messages.append({"role": "user", "content": input_text})

        # If tool use is disabled, use simple conversational logic
        if not self.enable_tool_use or not self.tool_registry:
            response = self.llm.invoke(messages, **kwargs)
            rprint(f"[bold magenta][Agent] ✅ {self.name} responsed.[/bold magenta]")
            # Update local history
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            return response

        # Muti-round tool use response
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def stream_run(self, input_text: str, **kwargs) -> Iterator[str]:
        """
        Self-defined streaming run method, tool use disabled
        """

        rprint(
            f"[bold magenta][Agent] 🤖 {self.name} is processing in streaming way: {input_text}[/bold magenta]"
        )

        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        messages.append({"role": "user", "content": input_text})

        full_response = ""
        temperature = kwargs.get("temperature")
        for chunk in self.llm.think(messages, temperature if temperature else 0):
            full_response += chunk
            # Already print in think()
            yield chunk

        print()

        self.add_message(Message(input_text, "user"))
        self.add_message(Message(full_response, "assistant"))
        rprint(
            f"[bold magenta][Agent] ✅ {self.name} streaming responsed[/bold magenta]"
        )

    def list_tools(self) -> list:
        """
        List all available tools and functions
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()

        return []
