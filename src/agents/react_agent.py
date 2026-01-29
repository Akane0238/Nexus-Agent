import re
from dotenv import load_dotenv
from llm_client import NexusAgentsLLM
from tools import ToolExecutor, search
from rich import print

# ReAct prompt template
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `tool_name[tool_input]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

示例回应 1：
Though: 为了了解最新的DeepSeek模型，我需要通过工具`Search`在网络上搜索最新的信息。
Action: Search[DeepSeek最新模型是什么]

示例回应 2：
Though: 通过搜索引擎工具我知道了DeepSeek目前最新的模型是 DeepSeek-V3.2，我已经收集了足够的信息来输出最终答案了。
Action: Finish[DeepSeek最新的模型是 DeepSeek-V3.2]

---
现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: NexusAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 3) -> None:
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
    
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
    def run(self, question: str):
        self.history = [] # reset hitory everytime when agent runs
        current_step = 0
        
        # Main loop
        while current_step < self.max_steps:
            current_step += 1
            print(f"[bold green]--- Step {current_step} ---[/bold green]")

            # 1. Formating prompt
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)

            prompt = REACT_PROMPT_TEMPLATE.format(
                tools = tools_desc,
                question = question,
                history = history_str
            )

            # 2. Calling LLM to think
            messages = [{"role": "user", "content": prompt}]

            response_txt = self.llm_client.think(message=messages)

            if not response_txt:
                print("[bold red]Error: LLM cannot return a valid reponse.[/bold red]")
                break
            
            # 3. Parsing LLM output and taking action
            thought, action = self._parse_output(response_txt)

            # if thought:
            #    print(f"thought: {thought}")

            if not action:
                print("[bold red]Warning: cannot parse valid Action, progress stop.[/bold red]")
                break

            # 4. Executing action and Observing
            if action.startswith("Finish"):
                answer = ans.group(1) if (ans := re.search(r"Finish\[(.*)\]", action)) else ""

                print(f"🎉 [bold green]Final answer[/bold green]: {answer}")
                return answer

            # LLM wants to use tools
            tool_name, tool_input = self._parse_action(action)

            if not tool_name or not tool_input:
                # ... invalid Action format
                continue

            print(f"🎬 [bold green]Action[/bold green]: {tool_name}[{tool_input}]")

            tool_func = self.tool_executor.getTool(tool_name)
            if not tool_func:
                observation = f"[bold red]Error: `{tool_name}` is not a valid tool[/bold red]"
            else:
                observation = tool_func(tool_input)

            print(f"👀 [bold green]Observation[/bold green]:\n {observation}")

            # 5. Adding action and observation to history
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")
            
        # Loop ended
        print("[bold green]Reach largest steps, progress stop.[/bold green]")
        return None
            
            

# --- Test ---
if __name__ == "__main__":
    load_dotenv()
    client = NexusAgentsLLM()
    executor = ToolExecutor()

    # Register searching tool `search()`
    description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    executor.registerTool("Search", description, search)

    # Print all available tools
    print("\n[green]--------- Available tools ---------[/green]")
    print(executor.getAvailableTools())
    print()


    agent = ReActAgent(llm_client=client, tool_executor=executor)
    agent.run("明天广州的气温是多少度")
