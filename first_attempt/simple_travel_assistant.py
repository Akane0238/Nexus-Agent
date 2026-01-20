import requests
import os
import re
from tavily import TavilyClient
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from typing import List

AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具：
- `get_weather(city: str)`: 查询指定城市的实时天气
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点

# 行动格式:
你的回答必须严格遵循以下格式。首先是你的思考过程，然后是你要执行的具体行动，每次回复只输出一对Thought-Action：
Thought: [这里是你的思考过程和下一步计划]
Action: 你决定采取的行动，必须是以下格式之一:
    - `function_name(arg_name="arg_value")`:调用一个可用工具。
    - `Finish[最终答案]`:当你认为已经获得最终答案时。
    - 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

# 回答例子如下：
1. 调用工具
Thought: [你的思考]
Action: get_weather(city="Gaungzhou")

或者

Thought: [你的思考]
Action: Finish[你的最终答案]

请开始吧！

"""


def get_weather(city: str) -> str:
    """
    Tool 1:
    调用 wttr.in API 查询真实的天气信息
    """
    
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # send request
        response = requests.get(url)
        # check status code
        response.raise_for_status()
        # parsing return JSON
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"

    except requests.exceptions.RequestException as e:
        return f"错误：查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市名称无效 - {e}"

def get_attraction(city: str, weather: str) -> str:
    """
    Tool 2:
    根据城市和天气，使用 Tavily Search API 搜索并返回优化后的景点推荐
    """

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "错误：未配置 TAVILY_API_KEY 环境变量"

    tavily = TavilyClient(api_key=api_key)

    query = f"'{city}'在'{weather}'天气下最值得去的旅游景点推荐及理由"

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("result", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"

# All available tools for agent
available_tools = {
        "get_weather": get_weather,
        "get_attraction": get_attraction,
}

class OpenAICompatibleClient:
    """
    一个用于调用任何兼容OpenAI接口的LLM服务的客户端
    """

    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        """调用 LLM API 来生成回应"""
        print(f"正在调用LLM {self.model}...")
        try:
            messages: List[ChatCompletionMessageParam] = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )

            answer = response.choices[0].message.content

            print("大语言模型响应成功。")
            return answer
        except Exception as e:
            print(f"调用LLM API时发生错误: {e}")
            return "错误:调用语言模型服务时出错。"


# --- LLM client configuration ---
API_KEY = "sk-uwjbpoutcktuaqzdcepabyqnxenwszbvwyypimpleovyguhf"
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
TAVILY_API_KEY = "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"
os.environ['TAVILY_API_KEY'] = "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"

llm = OpenAICompatibleClient(
        model=MODEL_ID,
        api_key=API_KEY,
        base_url=BASE_URL
)

if __name__ == "__main__":
    # initialization
    # user_prompt = "你好，请帮我查询一下今天南京的天气，然后根据天气推荐一个合适的旅游景点。"
    user_prompt = "你好，请帮我查询一下今天穗织的天气，然后根据天气推荐一个合适的旅游景点。穗织是一个真实的地名，使用工具查询时地名一定是“穗织”！"
    prompt_history = [f"用户请求：{user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    # main loop
    for i in range(6):
        print(f"--- 循环 {i+1} ---\n")

        # 1. construct Prompt
        full_prompt = "\n".join(prompt_history)

        # 2. call LLM to think
        llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            truncated = match.group(1).strip()
            if truncated != llm_output.strip():
                llm_output = truncated
                print("已截断多余的 Thought-Action 对")
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)

        # 3. parsing then acting
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。"
            observation_str = f"Observation: {observation}"
            print(f"{observation_str}\n" + "="*40)
            prompt_history.append(observation_str)
            continue
        action_str = action_match.group(1).strip()

        finish_match = re.search(r"Finish\[(.*?)\]", action_str, re.DOTALL)

        if finish_match:
            final_answer = finish_match.group(1).strip()
            print(f"任务完成，最终答案: {final_answer}")
            break
        
        # if action_str.startswith("Finish"):
        #     final_answer = re.match(r"Finish\[(.*)\]", action_str).group(1)
        #     print(f"任务完成，最终答案: {final_answer}")
        #     break

        tool_name = re.search(r"(\w+)\(", action_str).group(1)
        args_str = re.search(r"\((.*)\)", action_str).group(1)
        kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))


        if tool_name in available_tools:
            observation = available_tools[tool_name](**kwargs)
        else:
            observation = f"错误:未定义的工具 '{tool_name}'"

        # 4. record observation
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)

