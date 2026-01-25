# test_myllm.py
# from dotenv import load_dotenv
from src.core.my_llm import MyLLM 
from dotenv import load_dotenv

def test_init():
    llm = MyLLM(provider="modelscope", model="Qwen/Qwen3-4B")
    messages = [{"role": "user", "content": "你好，请介绍一下你自己(打上时间戳，具体的模型型号)。目前Qwen最先进的推理LLM是什么？"}]

    response_stream = llm.think(message=messages)
    assert response_stream

    # think中已流式输出
