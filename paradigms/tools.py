import os
from serpapi import GoogleSearch
from dotenv import load_dotenv
from typing import Any, Callable


# Search tool
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 Executing [SerpApi] web search: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "Error: SerpApi key should be configurated in `./.env` file."
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        # Find the most straightforward answer
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # No direct answer，return abstracts of top 3 organic results
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
    
        return f"Sorry, didn't find any infomation about `{query}`. You can try again with short brief query."
    
    except Exception as e:
        return f"Error while searching: {e}"
    

# Tools executor and manager
class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """

    def __init__(self) -> None:
        """
        Example:
            tools = {
                "ToolName": {
                    "description": "description",
                    "func": <function>
                },
            }
        """
        self.tools: dict[str, dict[str, Any]] = {}

    # Register a new tool in `ToolExecutor.tools`
    def registerTool(self, name: str, description: str, func: Callable):
        if name in self.tools:
            print(f"Warning:tool `{name}` has already been registered, the old one will be replaced.")
        self.tools[name] = {"description": description, "func": func}
        print(f"Tool `{name}` is registered.")

    # Get the function of a tool by name
    def getTool(self, name: str) -> Callable|None:
        if name in self.tools:
            return self.tools[name]["func"]
        
        print(f"Error: Tool '{name}' not found in toolbox.")
        return None

    # Acquire format description string of all usable tools
    def getAvailableTools(self) -> str:
        toolLists = []
        for name, info in self.tools.items():
            toolLists.append(f"- {name}: {info['description']}")
        
        return "\n".join(toolLists)
    



# --- Test ---
if __name__ == "__main__":
    load_dotenv()

    # Initialize tool executor
    toolExecutor = ToolExecutor()

    # Register searching tool `search()`
    description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", description, search)

    # Print all available tools
    print("\n--- Available tools ---\n")
    print(toolExecutor.getAvailableTools())

    # Agent action calling (after LLM thinking)
    tool = "Search"
    #input = "DeepSeek的最新模型是什么"
    input = "咖啡"
    func = toolExecutor.getTool(tool)

    if func:        
        print(f"--- Executing Action: {tool} ---\n")
        observation = func(input)
        print("--- Observation ---\n")
        print(observation)
    else:
        print(f"Error:does not find the tool named `{tool}`.")

