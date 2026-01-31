from rich import print as rprint
from typing import Any, Optional

from src.core.llm import NexusAgentsLLM
from src.core.agent import Agent
from src.core.config import Config
from src.core.message import Message

DEFAULT_PROMPTS = {
        "initial": """
    请根据以下要求完成任务:

    任务: {task}

    请提供一个完整、准确的回答。
    """,
        "reflect": """
    请仔细审查以下回答，并找出可能的问题或改进空间:

    # 原始任务:
    {task}

    # 当前回答:
    {content}

    请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
    如果回答已经很好，请回答"无需改进"。
    """,
        "refine": """
    请根据反馈意见改进你的回答:

    # 原始任务:
    {task}

    # 上一轮回答:
    {last_attempt}

    # 反馈意见:
    {feedback}

    请提供一个改进后的回答。
    """
}


class ReflectionAgent(Agent):
    def __init__(
        self,
        name: str,
        llm: NexusAgentsLLM,
        config: Optional[Config] = None,
        system_prompt: Optional[str] = None,
        custom_prompt: Optional[dict[str, str]] = None,
        max_iterations: int = 3
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_iterations = max_iterations
        self.memory: list[dict[str, Any]] = []  # built-in memory module
        self.prompt_templates = custom_prompt if custom_prompt else DEFAULT_PROMPTS

        rprint(f"[bold magenta][Agent] ✅ {name} Initialization complete, max reflection iteration: {max_iterations}[/bold magenta]")


    def _get_last_execution(self) -> Optional[str]:
        for record in reversed(self.memory):
            if record["type"] == "execution":
                return record["content"]
        return None


    def _get_trajectory(self) -> str:
        """
        Serialize all the memory records
        """

        trajectory_parts = []
        for record in self.memory:
            if record["type"] == "execution":
                trajectory_parts.append(f'--- Attemption of last turn ---\n{record["content"]}')
            elif record["type"] == "reflection":
                trajectory_parts.append(f'--- Feedback of evaluator ---\n{record["content"]}')

        return "\n\n".join(trajectory_parts)


    def run(self, input_text: str, **kwagrs) -> str:
        rprint(f"[bold magenta][Agent] 🤖 Start solving input_text:[/bold magenta]\n{input_text}\n")

        self.memory.clear()

        # Generate first response of task
        rprint("[bold magenta][Agent] First attempt[/bold magenta]")

        initial_prompt = self.prompt_templates["initial"].format(task=input_text)
        initial_msg = [{"role": "user", "content": initial_prompt}]
        initial_response = self.llm.invoke(initial_msg, **kwagrs)
        rprint(f"[bold magenta][Client] Initial response[/bold magenta]:\n[bold white]{initial_response}[/bold white]")

        self.memory.append({"type": "execution", "content": initial_response}) # update memory
        rprint(f"[bold magenta][Agent] 📝 Memory updates，adding new record typed 'execution'[/bold magenta]\n")

        for i in range(self.max_iterations):
            rprint(f"[bold green]--- Iteration turn {i+1}/{self.max_iterations} ---[/bold green]\n")

            # a. Reflection
            rprint("[yellow][Agent]-> Start reflecting[/yellow]")
            last_execution = self._get_last_execution()
            reflect_prompt = self.prompt_templates["reflect"].format(
                task = input_text,
                content = last_execution
            )

            reflect_msg = [{"role": "user", "content": reflect_prompt}]
            feedback_response = self.llm.invoke(reflect_msg, **kwagrs)

            rprint(f"[bold magenta][Client] Evaluator response[/bold magenta]:\n[bold white]{feedback_response}[/bold white]")

            # print(f"evaluator feedback:{feedback_txt}")
            self.memory.append({"type": "reflection", "content": feedback_response}) # update memory
            rprint(f"[bold magenta][Agent] 📝 Memory updates，adding new record typed 'reflection'[/bold magenta]\n")

            # b. Check looping condition
            if "无需改进" in feedback_response:
                rprint("\n[bold green][Agent]✅ Upon reflection, it was determined that the resolution required no further improvement, and the task was completed.[/bold green]")
                break

            # c. Refinement
            rprint("[yellow][Agent]-> Start optimizing[/yellow]")
            refine_prompt = self.prompt_templates["refine"].format(
                task = input_text,
               last_attempt = last_execution,
                feedback = feedback_response
            )

            refine_msg = [{"role": "user", "content": refine_prompt}]
            refinement_response = self.llm.invoke(refine_msg, **kwagrs)

            rprint(f"[bold magenta][Client] Actor refinement[/bold magenta]:\n[bold white]{refinement_response}[/bold white]")

            self.memory.append({"type": "execution", "content": refinement_response}) # update memory
            rprint(f"[bold magenta][Agent] 📝 Memory updates，adding new record typed 'execution'[/bold magenta]\n")

        final_resolution = self._get_last_execution()
        assert final_resolution

        # Update history
        self._history.append(Message(input_text, "user"))
        self._history.append(Message(final_resolution, "assistant"))

        rprint(f"\n[bold green]--- Task finished ---[/bold green]\n[yellow][Agent] Finally generated reponse[/yellow]:\n[bold white]{final_resolution}[/bold white]\n")

        return final_resolution

