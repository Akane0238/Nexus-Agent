import re
import json
from typing import Optional
from rich import print as rprint
from src.core.llm import NexusAgentsLLM
from src.tools.registry import ToolRegistry
from src.core.config import Config
from src.core.message import Message
from src.core.agent import Agent


# Default ReAct prompt template with JSON Schema support
REACT_PROMPT_TEMPLATE = """
你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 选择合适的工具获取信息，格式必须是以下之一:
- `{{"tool": "tool_name", "parameters": {{参数}}}}`: 调用一个可用工具（JSON格式）
- `Finish[最终答案]`: 当你有足够信息给出最终答案时。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 必须使用JSON格式调用工具，参数必须符合工具定义
3. 只有当你确信有足够信息回答问题时，必须在 `Action:` 字段后使用 `Finish[最终答案]` 来输出最终答案
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 工具调用示例

示例1：
Thought: 为了了解最新的DeepSeek模型，我需要通过搜索工具查找最新信息。
Action: {{"tool": "search", "parameters": {{"query": "DeepSeek最新模型是什么"}}}}

示例2：
Thought: 为了计算一个数学问题，我需要使用计算器工具。
Action: {{"tool": "calculator", "parameters": {{"expression": "15 * 3"}}}}

示例3（完成）：
Thought: 通过搜索引擎工具我知道了DeepSeek目前最新的模型是 DeepSeek-V3.2，我已经收集了足够的信息来输出最终答案了。
Action: Finish[DeepSeek最新的模型是 DeepSeek-V3.2]

## 当前任务
**Question:** {question}

## 执行历史
**History:** {history}

现在开始你的推理和行动：
"""


class ReActAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: NexusAgentsLLM,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry if tool_registry else ToolRegistry()
        self.max_steps = max_steps
        self.current_history: list[str] = []
        self.prompt_template = custom_prompt if custom_prompt else REACT_PROMPT_TEMPLATE

        rprint(
            f"[bold magenta][Agent] ✅ {name} Initialization complete, max steps: {max_steps}[/bold magenta]"
        )

    def _format_tools_description(self) -> str:
        """
        Format tools description with JSON Schema.
        """
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

        return "\n".join(descriptions) if descriptions else "No available tool"

    def _parse_output(self, output: str) -> tuple[str | None, str | None]:
        """
        Extract Thought and Action from LLM response.
        """
        thought_match = re.search(r"Thought:\s*(.*)", output)
        action_match = re.search(r"Action:\s*(.*)", output)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    def _parse_action(self, action: str) -> tuple[str | None, dict | None]:
        """
        Parse JSON format action: {"tool": "...", "parameters": {...}}
        Returns:
            (tool_name, parameters_dict)
        """
        try:
            action_dict = json.loads(action.strip())
            if isinstance(action_dict, dict):
                tool_name = action_dict.get("tool")
                parameters = action_dict.get("parameters")
                if tool_name and isinstance(parameters, dict):
                    return tool_name, parameters
        except (json.JSONDecodeError, AttributeError):
            pass
        return None, None

    def run(self, input_text: str, **kwargs) -> str:
        """
        Run ReAct agent to solve the problem.
        """
        self.current_history = []
        current_step = 0

        rprint(f"\n[bold magenta][Agent] 🤖 Start solving problem [/bold magenta]")

        while current_step < self.max_steps:
            current_step += 1
            rprint(
                f"[bold green]--- ReAct Step {current_step}/{self.max_steps} ---[/bold green]"
            )

            tools_desc = self._format_tools_description()
            history_str = "\n".join(self.current_history)

            prompt = self.prompt_template.format(
                tools=tools_desc, question=input_text, history=history_str
            )

            messages = [{"role": "user", "content": prompt}]
            response = self.llm.invoke(messages, **kwargs)

            if not response:
                rprint(
                    "[bold red][Agent] Error: LLM cannot return a valid response.[/bold red]"
                )
                break

            rprint(
                f"[bold magenta][Client] LLM response[/bold magenta]:\n[bold white]{response}[/bold white]"
            )

            thought, action = self._parse_output(response)

            if not action:
                rprint(
                    "[bold red][Agent] Warning: cannot parse valid Action, progress stop.[/bold red]"
                )
                break

            if action.startswith("Finish"):
                answer = (
                    ans.group(1)
                    if (ans := re.search(r"Finish\[(.*)\]", action))
                    else ""
                )

                self.add_message(Message(input_text, "user"))
                self.add_message(Message(answer, "assistant"))

                rprint(
                    f"[bold magenta][Agent] 🎉 Final answer[/bold magenta]:\n[bold white]{answer}[/bold white]"
                )
                return answer
            else:
                tool_name, tool_parameters = self._parse_action(action)
                if not tool_name or not tool_parameters:
                    observation = "Invalid action format"
                    rprint(
                        f"[bold magenta][Agent] {observation}[/bold magenta]"
                    )
                    self.current_history.append(f"Action: {action}")
                    self.current_history.append(f"Observation: {observation}")
                    continue

                # Parse JSON action
                rprint(
                    f"[bold magenta][Agent] 🎬 Action (JSON)[/bold magenta]:\n[bold white]{json.dumps({'tool': tool_name, 'parameters': tool_parameters}, ensure_ascii=False, indent=2)}[/bold white]"
                )

                observation = self.tool_registry.execute_tool(tool_name, tool_parameters)

                rprint(
                    f"[bold magenta][Agent] 👀 Observation[/bold magenta]:\n[bold white]{observation}[/bold white]"
                )
                self.current_history.append(f"Action: {action}")
                self.current_history.append(f"Observation: {observation}")
                continue

        rprint("[bold magenta][Agent] Reached maximum steps, progress stopped.[/bold magenta]")
        final_answer = "Sorry, I cannot finish this task within the given steps."
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer
