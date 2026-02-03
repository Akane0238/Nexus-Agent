# test_search_tool.py
import json
from rich import print as rprint
from src.tools.registry import ToolRegistry
from src.tools.builtin.search_tool import SearchTool

def test_search():
    """测试高级搜索工具"""

    # 创建包含高级搜索工具的注册表
    registry = ToolRegistry()

    # 直接创建搜索工具实例
    search_tool = SearchTool("hybrid")

    # 注册search_tool
    registry.register_tool(search_tool)


    rprint("[blue]🔍 测试高级搜索工具[/blue]\n")

    # 测试查询
    test_queries = [
        "Python编程语言的历史",
        "人工智能的最新发展",
        "2024年科技趋势"
    ]

    for i, query in enumerate(test_queries, 1):
        rprint(f"[blue]测试 {i}: {query}[/blue]")
        result = registry.execute_tool("search", {"query":query, "max_results":3})
        print(f"结果: {result}\n")
        print("-" * 60 + "\n")


def test_with_agent():
    """测试与Agent的集成"""
    print("\n🤖 与Agent集成测试:")
    print("高级搜索工具已准备就绪，可以与Agent集成使用")

    registry = ToolRegistry()
    search_tool = SearchTool()
    registry.register_tool(search_tool)

    # 显示工具描述
    tools_desc = registry.get_tools_description()
    rprint("="*8 + "[bold green]Tool Description[/bold green]" + "="*8)
    rprint(f"{tools_desc}")
    rprint("="*32)

    # 工具参数约束
    tools_param = registry.get_tools_schema_json()
    rprint("="*8 + "[bold green]Tool  Parameters[/bold green]" + "="*8)
    for tool_schema in tools_param:
        rprint("-"*24)
        rprint(f"[yellow]{json.dumps(tool_schema, ensure_ascii=False, indent=2)}[/yellow]")
        rprint("-"*24)


