# test_reflection_agent.py
from src.core.llm import NexusAgentsLLM
from src.agents.reflection_agent import ReflectionAgent

def test_default():
    llm = NexusAgentsLLM()

    # 使用默认通用提示词
    general_agent = ReflectionAgent(name="我的反思助手", llm=llm)

    general_agent.run("写一篇关于人工智能发展历程的简短文章")


def test_custom():
    llm = NexusAgentsLLM()

    # 使用自定义代码生成提示词
    code_prompts = {
        "initial": "你是Python专家，请编写函数:{task}",
        "reflect": "请审查代码的算法效率:\n任务:{task}\n代码:{content}",
        "refine": "请根据反馈优化代码:\n任务:{task}\n反馈:{feedback}"
    }
    code_agent = ReflectionAgent(
        name="我的代码生成助手",
        llm=llm,
        custom_prompt=code_prompts
    )

    code_agent.run("实现一个计算1-100之间所有素数的Python程序")
