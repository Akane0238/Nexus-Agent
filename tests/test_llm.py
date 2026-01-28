# test_myllm.py
from rich import print as rprint
from src.core.my_llm import MyLLM 
from src.core.llm import NexusAgentsLLM

messages = [{"role": "user", "content": "你好，请介绍一下你自己(打上时间戳，具体的模型型号)。目前Qwen最先进的推理LLM是什么？"}]


def test_myllm():
    """测试封装框架底层NexusAgentsLLM的MyLLM类"""
    llm = MyLLM(provider="modelscope", model="Qwen/Qwen3-4B")

    response_stream = llm.think(messages=messages)
    assert response_stream
    # think中已流式输出
    for chunk in response_stream:
        pass

def test_ollama():
    """测试ollama本地模型服务"""
    llm = NexusAgentsLLM(
        model="qwen2.5:3b",
        baseURL="http://localhost:11434/v1",
        apiKey="Ciallo~"
    )
    response_stream = llm.think(messages=messages)
    assert response_stream
    for chunk in response_stream:
        pass

def test_dotenv():
    """`.env`提供环境变量定义base_url与api_key"""
    llm = NexusAgentsLLM()
    response_stream = llm.think(messages=messages)
    assert response_stream
    for chunk in response_stream:
        pass

def test_provider():
    """通过传入provider参数自动解析base_url"""
    llm = NexusAgentsLLM(provider="siliconflow")
    response_stream = llm.think(messages=messages)
    assert response_stream
    for chunk in response_stream:
        pass

def test_invoke():
    """测试llm client的非流式输出invoke()"""
    llm = NexusAgentsLLM()
    response = llm.invoke(messages=messages)
    assert response 
    rprint(f"\n[cyan]LLM 非流式输出响应：\n{response}[/cyan]")
