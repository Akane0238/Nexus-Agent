import ast
from llm_client import NexusAgentsLLM

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

class Planner:
    def __init__(self, llm_client: NexusAgentsLLM) -> None:
        self.llm_client = llm_client
        pass

    # Generate an action plan based on user question
    def plan(self, question: str) -> list[str]:
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            question = question
        )
        message = [{"role": "user", "content": prompt}]

        print("--- Generating plan ---")
        response_txt = self.llm_client.think(message=message)
        print(f"✅ Plan is generated:\n{response_txt}")

        # Parse each steps from LLM output
        try:
            plan_str = response_txt.split("```python")[1].split("```")[0].strip() if response_txt else ""
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []

        except (SyntaxError, ValueError, IndexError) as e:
            print(f"❌ Failed to parse response: {e}")
            return []

        except Exception as e:
            print(f"❌ Unknown error in parsing plan: {e}")
            return []

