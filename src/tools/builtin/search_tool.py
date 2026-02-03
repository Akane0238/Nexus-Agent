import os
from typing import Optional, Literal, Any, Iterable
from pydantic import BaseModel, Field
from tavily import TavilyClient
from serpapi import GoogleSearch
from rich import print as rprint

from src.tools.tool_base import Tool


CHARS_PER_TOKEN = 4
DEFAULT_MAX_RESULTS = 5
SUPPORTED_RETURN_MODES = {"text", "structured", "json", "dict"}
SUPPORTED_BACKENDS = {
    "hybrid",
    "tavily",
    "serpapi",
}

class SearchInput(BaseModel):
    """Pydantic model for SearchTool parameters"""

    query: str = Field(description="搜索查询关键词")
    backend: Optional[Literal["hybrid", "tavily", "serpapi"]] = Field(
        default="hybrid", description="搜索后端选择，默认混合模式"
    )
    mode: Optional[Literal["text", "structured", "json", "dict"]] = Field(
        default="text", description="返回模式：文本或结构化数据"
    )
    fetch_full_page: Optional[bool] = Field(default=False, description="是否获取完整页面内容")
    max_results: Optional[int] = Field(default=DEFAULT_MAX_RESULTS, description="最大搜索结果数")
    max_tokens_per_source: Optional[int] = Field(
        default=2000, description="每个搜索结果的最大token数"
    )


def _limit_text(text: str, token_limit: int) -> str:
    char_limit = token_limit * CHARS_PER_TOKEN
    if len(text) <= char_limit:
        return text
    return text[:char_limit] + "... [truncated]"

def _normalized_result(
    *,
    title: str,
    url: str,
    content: str,
    raw_content: str | None,
) -> dict[str, str]:
    payload: dict[str, str] = {
        "title": title or url,
        "url": url,
        "content": content or "",
    }
    if raw_content is not None:
        payload["raw_content"] = raw_content
    return payload

def _structured_payload(
    results: Iterable[dict[str, Any]],
    *,
    backend: str,
    answer: str | None = None,
    notices: Iterable[str] | None = None,
) -> dict[str, Any]:
    return {
        "results": list(results),
        "backend": backend,
        "answer": answer,
        "notices": list(notices or []),
    }


class SearchTool(Tool):
    """
    Intelligent hybrid search tool.
    Supports multiple search engine backends, intelligently selecting the best search source:
        1. Hybrid mode - intelligently selects TAVILY or SERPAPI
        2. Tavily API (tavily) - professional AI Search
        3. SerpApi (serpapi) - traditional Google Search
    """

    def __init__(
        self,
        backend: str = "hybrid",
        tavily_key: Optional[str] = None,
        serpapi_key: Optional[str] = None,
    ):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。",
        )
        self.backend = backend.lower()
        self.tavily_key = tavily_key or os.getenv("TAVILY_API_KEY")
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_API_KEY")

        self.tavily_client = None
        self.available_backends: list[str] = []
        self._setup_backends()

    def run(self, parameters: dict[str, Any]) -> str | dict[str, Any]:
        """Execute search with validated parameters"""
        query = (parameters.get("query") or "").strip()
        if not query:
            return "Error: search query cannot be empty"

        backend = str(parameters.get("backend", self.backend)).lower()
        if backend not in SUPPORTED_BACKENDS:
            backend = "hybrid"

        mode = str(parameters.get("mode") or "text").lower()
        if mode not in SUPPORTED_RETURN_MODES:
            mode = "text"

        fetch_full_page = parameters.get("fetch_full_page", False)
        max_results = int(parameters.get("max_results", DEFAULT_MAX_RESULTS))
        max_tokens = int(parameters.get("max_tokens_per_source", 2000))

        payload = self._structured_search(
            query=query,
            backend=backend,
            fetch_full_page=fetch_full_page,
            max_results=max_results,
            max_tokens=max_tokens,
        )

        if mode in {"structured", "json", "dict"}:
            return payload

        return self._format_text_response(query=query, payload=payload)

    def get_input_schema(self):
        """Return Pydantic model for parameter validation"""
        return SearchInput

    def _setup_backends(self):
        if self.serpapi_key:
            try:
                self.tavily_client = TavilyClient(api_key=self.tavily_key)
                self.available_backends.append("tavily")
                rprint("[bold blue][Tool] ✅ `Tavily` search engine has been initialized[/bold blue]")
            except Exception as e:
                rprint(f"[bold red][Tool] ⚠️ Tavily failed to initialize: {e}[/bold red]")
        else:
            rprint("[yellow][Tool] ⚠️ `TAVILY_API_KEY` does not set[/yellow]")

        if self.serpapi_key:
            self.available_backends.append("serpapi")
            rprint("[bold blue][Tool] ✅ `serpapi` search engine has been initialized[/bold blue]")
        else:
            rprint("[yellow][Tool] ⚠️ `SERPAPI_API_KEY` does not set[/yellow]")

        if self.backend not in SUPPORTED_BACKENDS:
            rprint(f"[yellow][Tool] Unsupported searching backend `{self.backend}`, going to use `hybrid` mode[/yellow]")
            self.backend = "hybrid"
        elif self.backend == "tavily" and "tavily" not in self.available_backends:
            rprint("[yellow][Tool] `tavily` cannot be used, going to use `hybrid` mode[/yellow]")
            self.backend = "hybrid"
        elif self.backend == "serpapi" and "serpapi" not in self.available_backends:
            rprint("[yellow][Tool] `serpapi` cannot be used, going to use `hybrid` mode[/yellow]")
            self.backend = "hybrid"

        if self.backend == "hybrid":
            rprint("[yellow][Tool] 🔧 Searching mode `hybrid` is set, available backends[/yellow]:" + ", ".join(self.available_backends))


    def _structured_search(
        self,
        *,
        query: str,
        backend: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        if backend == "hybrid":
            return self._search_hybrid(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        elif backend == "tavily":
            return self._search_tavily(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        elif backend == "serpapi":
            return self._search_serpapi(
                query=query,
                fetch_full_page=fetch_full_page,
                max_results=max_results,
                max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"Unsupported search backend: {backend}")

    def _search_hybrid(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        notices = []  # collecting error messages when searching

        # Use `Tavily` in prior
        if "tavily" in self.available_backends:
            try:
                return self._search_tavily(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                msg = f"Tavily failed to search: {e}"
                rprint(f"[yellow][Tool] ⚠️ {msg}[/yellow]")
                notices.append(msg)

        # Try SerpaApi
        if "serpapi" in self.available_backends:
            if notices:
                rprint("[yellow][Tool] Switch  🔄 to SerpApi[/yellow]")

            try:
                return self._search_serpapi(
                    query=query,
                    fetch_full_page=fetch_full_page,
                    max_results=max_results,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                msg = f"SerpApi failed to search: {e}"
                rprint(f"[yellow][Tool] ⚠️ {msg}[/yellow]")
                notices.append(msg)

        if not self.available_backends:
            # a: no api key configured
            error_message = "No usable searching engine, configure `TAVILY_API_KEY` or `SERPAPI_API_KEY` environment variable in `./.env`"
            rprint("[bold red][Tool] ❌ Configuration Error: No search backends available[/bold red]")
        else:
            # b: api searching all failed
            error_message = f"Searching failed, no response from all available backends: {'; '.join(notices)}"
            rprint("[bold red][Tool] ❌ Runtime Error: All backends failed[/bold red]")

        return _structured_payload(
            results=[],
            backend="hybrid",
            answer=error_message,
            notices=notices
        )

    def _search_tavily(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self.tavily_client:
            message = "TAVILY_API_KEY is not configured or tavily is not installed"
            raise RuntimeError(message)

        response = self.tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=fetch_full_page,
        )

        results = []
        for item in response.get("results", [])[:max_results]:
            raw = item.get("raw_content") if fetch_full_page else item.get("content")
            if raw and fetch_full_page:
                raw = _limit_text(raw, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("url", ""),
                    url=item.get("url", ""),
                    content=item.get("content") or "",
                    raw_content=raw,
                )
            )

        return _structured_payload(
            results,
            backend="tavily",
            answer=response.get("answer"),
        )

    def _search_serpapi(
        self,
        *,
        query: str,
        fetch_full_page: bool,
        max_results: int,
        max_tokens: int,
    ) -> dict[str, Any]:
        if not self.serpapi_key:
            raise RuntimeError("SERPAPI_API_KEY is not configured，cannot use serpapi search")

        params = {
            "engine": "google",
            "q": query,
            "api_key": self.serpapi_key,
            "gl": "cn",
            "hl": "zh-cn",
            "num": max_results,
        }

        response = GoogleSearch(params).get_dict()

        answer_box = response.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")

        results = []
        for item in response.get("organic_results", [])[:max_results]:
            raw_content = item.get("snippet")
            if raw_content and fetch_full_page:
                raw_content = _limit_text(raw_content, max_tokens)
            results.append(
                _normalized_result(
                    title=item.get("title") or item.get("link", ""),
                    url=item.get("link", ""),
                    content=item.get("snippet") or "",
                    raw_content=raw_content,
                )
            )

        return _structured_payload(results, backend="serpapi", answer=answer)

    def _format_text_response(self, *, query: str, payload: dict[str, Any]) -> str:
        answer = payload.get("answer")
        notices = payload.get("notices") or []
        results = payload.get("results") or []
        backend = payload.get("backend", self.backend)

        lines = [f"🔍 Search key word：{query}", f"🧭 Use search backend：{backend}"]
        if answer:
            lines.append(f"💡 Direct answer：{answer}")

        if results:
            lines.append("")
            lines.append("📚 Reference sources：")
            for idx, item in enumerate(results, start=1):
                title = item.get("title") or item.get("url", "")
                lines.append(f"[{idx}] {title}")
                if item.get("content"):
                    lines.append(f"    {item['content']}")
                if item.get("url"):
                    lines.append(f"    source: {item['url']}")
                lines.append("")
        else:
            lines.append("❌ Does not find relevant search result")

        if notices:
            lines.append("⚠️ Attentions：")
            for notice in notices:
                if notice:
                    lines.append(f"- {notice}")

        return "\n".join(line for line in lines if line is not None)
