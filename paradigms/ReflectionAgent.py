from PaSModules.memory import Memory
from llm_client import NexusAgentsLLM
from dotenv import load_dotenv
from rich import print 


INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求: {task}

请直接输出代码，不要包含任何额外的解释也不要带有mardown表示符号。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位资深的代码评审专家（Tech Lead），在审查代码时注重<strong>性能与工程复杂度的平衡</strong>。
你的任务是审查以下 Python 代码，评估其在<strong>通用场景</strong>下的算法效率。

# 原始任务:
{task}

# 待审查的代码:
```python
{code}
```

请遵循以下收敛原则进行评审：
1. 基准判断：如果当前算法的时间复杂度在一般数据规模下（例如 N < 10^6）已经足够高效（如达到 O(N) 或 O(N log N)），不要仅仅为了追求理论上的极致性能而建议更换复杂的算法（例如，不要建议将标准的筛法更换为 Miller-Rabin，除非任务明确指出是加密级的大数场景）。
2. 避免过度设计：只有当当前代码存在明显的性能缺陷（如 O(N^2) 或更差）或逻辑错误时，才提出改进建议。
3. 收敛指令：如果代码逻辑正确，且性能在常规工程标准下是可接受的，请直接回答“无需改进”，不要输出其他任何内容。

如果确实需要改进，请指出瓶颈并提出更优方案。
"""


REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务:
{task}

# 你上一轮尝试的代码:
{last_code_attempt}
评审员的反馈：
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释和markdown标识符。
"""


class ReflectionAgent:
    def __init__(self, llm_client: NexusAgentsLLM, max_iterations: int = 3) -> None:
        self.llm_client = llm_client
        self.max_iterations = max_iterations
        self.memory = Memory()

    def run(self, task: str) -> str:
        print(f"[bold green]--- Start solving task ---[/bold green]\ntask: {task}\n")

        # Generate first response of task
        print("[bold green]--- First attempt ---[/bold green]")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_msg = [{"role": "user", "content": initial_prompt}]
        code_txt = self.llm_client.think(initial_msg) or ""

        self.memory.add_record("execution", code_txt) # update memory

        for i in range(self.max_iterations):
            print(f"[bold green]--- Iteration turn {i+1}/{self.max_iterations} ---[/bold green]\n")

            # a. Reflection
            print("[yellow]-> Start reflecting[/yellow]")
            last_execution_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task = task,
                code = last_execution_code
            )

            reflect_msg = [{"role": "user", "content": reflect_prompt}]
            feedback_txt = self.llm_client.think(reflect_msg) or ""

            # print(f"evaluator feedback:{feedback_txt}")
            self.memory.add_record("reflection", feedback_txt) # memory update

            # b. Check looping condition
            if "无需改进" in feedback_txt:
                print("\n✅ Upon reflection, it was determined that the code required no further improvement, and the task was completed.")
                break

            # c. Refinement
            print("[yellow]-> Start optimizing[/yellow]")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task = task,
                last_code_attempt = last_execution_code,
                feedback = feedback_txt
            )

            refine_msg = [{"role": "user", "content": refine_prompt}]
            refinement_code_txt = self.llm_client.think(refine_msg) or ""

            # print(f"actor refinement code:\n```python\n{refinement_code_txt}\n```")
            self.memory.add_record("execution", refinement_code_txt) # memory update

        final_resolution = self.memory.get_last_execution() or ""
        print(f"\n[bold green]--- Task finished ---[/bold green]\n[yellow]Finally generated code[/yellow]:\n```python\n{final_resolution}\n```")

        return final_resolution




# --- Test ---
if __name__ == "__main__":
    load_dotenv()
    client = NexusAgentsLLM()
    reflectionAgent = ReflectionAgent(client, max_iterations=6)

    task = """
**请编写一个 Python 函数 `find_longest_substring(text, k)`。**

**功能要求：**
1.  该函数接收一个字符串 `text` 和一个整数 `k`。
2.  请找出 `text` 中**包含“至多” k 个不同字符**的**最长连续子串**。
3.  如果存在多个长度相同的最长子串，返回在原字符串中**最先出现**的那个。
4.  **特殊处理**：
    *   忽略大小写（即 'a' 和 'A' 视为同一个字符）。
    *   忽略所有非字母字符（如空格、数字、标点），它们不应打断子串的连续性，也不计入 k 的计数，但在返回结果时，原始字符串中的这些非字母字符需要保留在子串中。

**示例：**
输入: `text = "a b c A B c c b a"`, `k = 2`
逻辑: 忽略空格后是 `abcABccba` -> 视作 `abcabccba`。最长含2个不同字符的是 `bccb` (由 b,c 组成)。
返回: 原始子串 `"c A B c c b"` 
    """
    reflectionAgent.run(task)
