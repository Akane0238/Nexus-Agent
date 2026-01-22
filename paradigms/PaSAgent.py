from dotenv import load_dotenv
from PaSModules.planner import Planner
from PaSModules.executor import Executor
from llm_client import NexusAgentsLLM
from rich import print


class PlanAndSolveAgent:
    def __init__(self, llm_client: NexusAgentsLLM) -> None:
        self.llm_client = llm_client
        self.planner = Planner(llm_client)
        self.executor = Executor(llm_client)

    def run(self, question: str):
        print(f"\n[bold green]--- Start processing question ---[/bold green]\n[yellow]question[/yellow]: {question}")

        plan = self.planner.plan(question)

        if not plan:
            print("\n[bold green]--- Task terminated ---[/bold green]\nCannot generate valid plan.")
            return

        answer = self.executor.execute(question, plan)

        print(f"\n[bold green]--- Task finished ---[/bold green]\n[yellow]Final answer[/yellow]: {answer}")


# --- Test ---
if __name__ == "__main__":
    load_dotenv()
    client = NexusAgentsLLM()

    pasAgent = PlanAndSolveAgent(llm_client=client)
    pasAgent.run("一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")

