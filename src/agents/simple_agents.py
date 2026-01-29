"""Basic communicating agent implement"""
import re
from typing import Optional, Iterator, Callable
from rich import print as rprint

from src.core.llm import NexusAgentsLLM
from src.core.agent import Agent
from src.core.config import Config
from src.core.message import Message
from src.tools.registry import ToolRegistry
from src.tools.tool_base import Tool


class SimpleAgent(Agent):
    """Simple chatting Agent, support available tool use"""
    def __init__(
        self,
        name: str,
        llm: NexusAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        enable_tool_use: bool = True
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
        rprint(f"[bold magenta][Agent] ✅ {name} Initialization complete，tool use: {'Enable' if self.enable_tool_use else 'Disable'}[/bold magenta]")

    
    def _get_enhanced_system_prompt(self) -> str:
        """
        Build an enhanced system prompt including tools information
        """
        base_prompt = self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_use or not self.tool_registry:
            return base_prompt

        tools_description = self.tool_registry.get_tools_description()
        if not tools_description or tools_description == "No available tool":
            return base_prompt

        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tools_description + "\n"

        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请务必遵循以下格式：\n"
        tools_section += "`[TOOL_USE:{tool_name}:{parameters}]`\n\n"

        tools_section += "### 参数格式说明\n"
        tools_section += "1. **多个参数**： 使用 `key=value` 格式，用英文逗号分隔\n"
        tools_section += "  示例：`[TOOL_USE:calculator_multiple:a=3,b=2]`\n"
        tools_section += "  示例：`[TOOL_USE:filesystem_read_file:path=README.md]`\n\n"
        tools_section += "2. **单个参数**： 直接使用 `key=value`\n"
        tools_section += "  示例：`[TOOL_USE:search:query=Transformer架构]`\n\n"
        tools_section += "3. **简单查询**： 可以直接传入查询文本\n"
        tools_section += "  示例：`[TOOL_USE:search:Transformer架构]`\n\n"

        tools_section += "### 重要提示\n"
        tools_section += "- 参数名必须与工具定义的参数名完全匹配\n"
        tools_section += "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`\n"
        tools_section += "- 文件路径等字符串参数直接写：`path=docs/README.md`\n"
        tools_section += "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答\n"

        return base_prompt + tools_section


    def _parse_tool_uses(self, response: str) -> list:
        pattern = r"\[TOOL_USE:([^:]+):([^\]]+)\]"
        matches = re.findall(pattern, response)

        if not matches:
            rprint(f"[bold magenta][Agent] LLM responsed with wrong tool use format, no pattern found [/bold magenta]")

        tool_uses = [] # list of tool callings
        for tool_name, params_raw in matches:
            tool_uses.append({
                "tool_name": tool_name.strip(),
                "parameters": params_raw.strip(),
                "original": f"[TOOL_USE:{tool_name}:{params_raw}]"
            })

        return tool_uses


    def _parse_tool_parameters(self, tool_name: str, params_raw: str) -> dict:
        params_dict = {}
        if "=" not in params_raw:
            if tool_name == "search":
                # Simple search query
                params_dict["query"] = params_raw
            elif tool_name == "memory":
                params_dict["action"] = "search"
                params_dict["query"] = params_raw
            else:
                params_dict["input"] = params_raw
        else:
            # Format: key=value or action=search,query=xxx
            params_pairs = params_raw.split(",")
            for pair in params_pairs:
                if "=" in pair:
                    key, val = pair.split("=", 1)
                    params_dict[key.strip()] = val.strip()

        return params_dict


    def _execute_tool_use(self, tool_name: str, parameters: str) -> str:
        if not self.tool_registry:
            return f"Error: does not configure tool registry"
        
        try:
            # Intelligent parameters parsing
            if tool_name == "calculator":
                result = self.tool_registry.execute_tool(tool_name, parameters)
            else:
                params_dict = self._parse_tool_parameters(tool_name, parameters)
                tool = self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"Error: does not find tool `{tool_name}`"
                result = tool.run(params_dict)

            return f"Tool `{tool_name}` execution result:\n{result}"
                
        except Exception as e:
            return f"Failed to invoke tools: {str(e)}"


    def _run_with_tools(self, messages: list,input_text: str, max_tool_iterations: int, **kwargs) -> str:
        current_iteration = 0
        final_response = ""

        while current_iteration < max_tool_iterations:
            rprint(f"[bold blue]--- _run_with_tools: iteration {current_iteration+1}/{max_tool_iterations} ---[/bold blue]")
            response = self.llm.invoke(messages, **kwargs)
            rprint(f"[bold magenta][Client] ✅ LLM successfully response[/bold magenta]:\n[cyan]{response}[/cyan]")

            # Parsing response
            tool_uses = self._parse_tool_uses(response)

            rprint(f"[bold magenta][Agent] 🔧 Detecting {len(tool_uses)} tool uses[/bold magenta]\n")
            if tool_uses:
                # Calling all tools
                results = []
                clean_response = response # without tools calling information

                for tool_use in tool_uses:
                    rprint(f"[bold magenta][Agent] 🔧 Tool `{tool_use['tool_name']}` executing...[/bold magenta]")
                    rprint(f"[bold white]- {tool_use['original']}[/bold white]")

                    result = self._execute_tool_use(tool_use["tool_name"], tool_use["parameters"])
                    rprint(f"[bold white]- {result}[/bold white]\n")
                    results.append(result)
                    clean_response = clean_response.replace(tool_use["original"], "")

                # Construct message with tool execution result
                messages.append({"role": "assistant", "content": clean_response})

                tool_results_text = "\n\n".join(results)
                messages.append({"role": "user", "content": f"工具执行结果:\n{tool_results_text}\n\n请基于这些结果给出完整的回答。"})

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
        rprint(f"[bold magenta][Agent] ✅ {self.name} finished reponse[/bold magenta]\n")

        return final_response


    def run(self, input_text: str, max_tool_iterations: int = 3, **kwargs) -> str:
        """
        Overwrite `run` function,
        implement simple conversational logic and support tool use.
        """
        rprint(f"[bold magenta][Agent] 🤖 {self.name} is processing: {input_text}[/bold magenta]")

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

        rprint(f"[bold magenta][Agent] 🤖 {self.name} is processing in streaming way: {input_text}[/bold magenta]")

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
        rprint(f"[bold magenta][Agent] ✅ {self.name} streaming responsed[/bold magenta]")


    def list_tools(self) -> list:
        """
        List all available tools and functions
        """
        if self.tool_registry:
            return self.tool_registry.list_tools()

        return []





