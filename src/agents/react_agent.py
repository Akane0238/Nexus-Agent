import re
from typing import Optional
from rich import print as rprint
from src.core.llm import NexusAgentsLLM
from src.tools.registry import ToolRegistry
from src.core.config import Config
from src.core.message import Message
from src.core.agent import Agent
from src.tools.tool_base import Tool


# Default ReAct prompt template
REACT_PROMPT_TEMPLATE = """
你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 选择合适的工具获取信息，格式必须是以下之一:
- `tool_name[tool_input]`:调用一个可用工具。
- `Finish[最终答案]`:当你有足够信息给出最终答案时。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，必须在 `Action:` 字段后使用 `Finish[最终答案]` 来输出最终答案
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数


示例回应 1：
Though: 为了了解最新的DeepSeek模型，我需要通过工具`Search`在网络上搜索最新的信息。
Action: Search[DeepSeek最新模型是什么]

示例回应 2：
Though: 通过搜索引擎工具我知道了DeepSeek目前最新的模型是 DeepSeek-V3.2，我已经收集了足够的信息来输出最终答案了。
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
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        custom_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: list[str] = []
        self.prompt_template = custom_prompt if custom_prompt else REACT_PROMPT_TEMPLATE
        rprint(f"[bold magenta][Agent] ✅ {name} Initialization complete, max steps: {max_steps}[/bold magenta]")

    
    # Helper function: Extract `Thought` and `Action`
    def  _parse_output(self, ouput: str):
        thought_match = re.search(r"Thought:\s*(.*)", ouput)
        action_match = re.search(r"Action:\s*(.*)", ouput)

        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None

        return thought, action

    # Helper function: Extract `tool` and `input` in `Action`
    def _parse_action(self, action: str):
        tool_name_match = re.search(r"(\w*)\[", action)
        tool_input_match = re.search(r"\[(.*)\]", action)

        tool_name = tool_name_match.group(1).strip() if tool_name_match else None
        tool_input = tool_input_match.group(1).strip() if tool_input_match else None

        return tool_name, tool_input

    # Run ReAct agent to start answering a question
    def run(self, input_text: str, **kwargs) -> str:
        self.current_history = [] # reset hitory everytime when agent runs
        current_step = 0

        rprint(f"\n[bold magenta][Agent] 🤖 Start solving problem [/bold magenta]")
       
        # Main loop
        while current_step < self.max_steps:
            current_step += 1
            rprint(f"[bold green]--- ReAct Step {current_step}/{self.max_steps} ---[/bold green]")

            # 1. Formating prompt
            tools_desc = self.tool_registry.get_tools_description()
            history_str = "\n".join(self.current_history)

            prompt = self.prompt_template.format(
                tools = tools_desc,
                question = input_text,
                history = history_str
            )

            # 2. Calling LLM to think
            messages = [{"role": "user", "content": prompt}]

            response = self.llm.invoke(messages, **kwargs)

            if not response:
                rprint("[bold red][Agent] Error: LLM cannot return a valid reponse.[/bold red]")
                break
            
            rprint(f"[bold magenta][Client] LLM response[/bold magenta]:\n[bold white]{response}[/bold white]")

            # 3. Parsing LLM output and taking action
            thought, action = self._parse_output(response)

            # if thought:
            #    rprint(f"[bold magenta][Client] thought[/bold magenta]:\n[bold white]{thought}[/bold white]")

            if not action:
                rprint("[bold red][Agent] Warning: cannot parse valid Action, progress stop.[/bold red]")
                break

            # 4. Executing action and observing
            if action.startswith("Finish"):
                answer = ans.group(1) if (ans := re.search(r"Finish\[(.*)\]", action)) else ""

                # Update history
                self.add_message(Message(input_text, "user"))
                self.add_message(Message(answer, "assistant"))

                rprint(f"[bold green][Agent] 🎉 Final answer[/bold green]:\n[bold white]{answer}[/bold white]")
                return answer

            # LLM wants to use tools
            tool_name, tool_input = self._parse_action(action)

            if not tool_name or not tool_input:
                # ... invalid Action format
                continue

            rprint(f"[bold green][Agent] 🎬 Action[/bold green]: [bold white]{tool_name}[{tool_input}][/bold white]")

            observation = self.tool_registry.execute_tool(tool_name, tool_input)

            rprint(f"[bold green][Agent] 👀 Observation[/bold green]:\n[bold white]{observation}[/bold white]")

            # 5. Adding action and observation to history
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        # Loop ended
        rprint("[bold green]Reach largest steps, progress stop.[/bold green]")
        final_answer = "Sorry, I cannot finish this task in given steps."
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))
        return final_answer

            
