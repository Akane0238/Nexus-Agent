from llm_client import NexusAgentsLLM

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Executor:
    def __init__(self, llm_client: NexusAgentsLLM) -> None:
        self.llm_client = llm_client
        self.history = []  # save log for debugging

    def execute(self, question: str, plan: list[str]) -> str:
        self.history = []
        response = ""
        # Iterate plan list to strictly execute every step
        print("\n--- Executing plan ---")
        for i, step in enumerate(plan):
            print(f"\n-> Executing step {i+1}/{len(plan)}: {step}")

            try:
                prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                    question = question,
                    plan = plan,
                    history = self.history if self.history else "None",
                    current_step = step
                )

                message = [{"role": "user", "content": prompt}]

                response = self.llm_client.think(message=message) or ""

                self.history.append(f"Step {i+1}: {step}\nResult: {response}")
                print(f"✅ Step {i+1} finished，result: {response}")
                
            except Exception as e:
                print(f"❌ Error in plan execution: {e}")
                return ""

        return response
