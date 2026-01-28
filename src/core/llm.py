import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, Literal, List, Iterator
from rich.console import Console


SUPPORTED_PROVIDERS = Literal[
    "openai",
    "modelscope",
    "siliconflow",
    "ollama",
    "local",
    "custom", # customize your base_url and api_key in `.env`
    "auto", # default configuration
]


class NexusAgentsLLM:
    """
    Self-defined LLM client. 
    Capsulate OpenAI interface and use streaming response by default.
    """
    console = Console() # log for debug
    
    def __init__(
        self,
        provider: Optional[SUPPORTED_PROVIDERS] = None,
        model: Optional[str] = None,
        apiKey: Optional[str] = None, 
        baseURL: Optional[str] = None,
        timeout: Optional[int] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Initialize client.
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        self.temperature = temperature 
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # auto detect if `provider` not set
        self.provider = provider.lower() if provider else self._auto_detect_provider(api_key=apiKey, base_url=baseURL)
        
        if self.provider == "custom":
            # Customized url and api key
            self.api_key = apiKey or os.getenv("LLM_API_KEY")
            self.base_url = baseURL or os.getenv("LLM_BASE_URL")
        else:
            # Try resolving base url and api key
            self.api_key, self.base_url = self._resolve_credentials(apiKey, baseURL)

        if not all([self.model, self.api_key, self.base_url]):
            raise ValueError("Model ID, API key and Serving address should be provided in `./.env` file or passed as parameters.")

        # Create OpenAI client
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)


    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """
        Helper function
        Auto detect LLM provider in prior.
        """

        # 1. Based on enviroment variables
        if os.getenv("SILICONFLOW_API_KEY"): return "siliconflow"
        if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
        if os.getenv("OPENAI_API_KEY"): return "openai"

        general_api_key = api_key or os.getenv("LLM_API_KEY")
        general_base_url = base_url or os.getenv("LLM_BASE_URL")

        # 2. Base on `LLM_BASE_URL`
        if general_base_url:
            url_lower = general_base_url.lower()
            if "api-inference.modelscope.cn" in url_lower: return "modelscope"
            if "api.openai.com" in url_lower: return "openai"
            if "api.siliconflow.cn" in url_lower: return "siliconflow"
            if "localhost" in url_lower or "127.0.0.1" in url_lower:
                if ":11434" in url_lower: return "ollama"
                if ":8000" in url_lower: return "vllm"
                return "local"  # other local port

        # 3. Base on `LLM_API_KEY`
        if general_api_key:
            if general_api_key.startswith("ms-"): return "modelscope"
            if general_api_key.startswith("sk-"): return "siliconflow"

        # 4. Return "auto" by default
        return "auto"


    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str | None, str | None]:
        """Parsing API key and base_url based on provider"""
        if self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("MODELSCOPE_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url
        elif self.provider == "siliconflow":
            resolved_api_key = api_key or os.getenv("SILICONFLOW_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1/"
            return resolved_api_key, resolved_base_url
        elif self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url
        elif self.provider == "ollama":
            resolved_base_url = base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
            return "", resolved_base_url
       # ...

        else:
            # `auto` or `local`: using general configuration
            resolved_api_key = api_key or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL")
            return resolved_api_key, resolved_base_url
 

    def think(self, messages: List[dict[str, str]], temperature: float = 0) -> Iterator[str] :
        """
        Invoking LLM to think and return streaming response.

        Args:
            messages: list of message
            temperature: temperature of model
        """

        self.console.print(f"[bold magenta][Client] 🧠 Invoking {self.model} Model...[/bold magenta]")
        try:
            response = self._client.chat.completions.create(
                model=self.model, # type: ignore
                messages=messages, # type: ignore
                temperature=temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            # Handle streaming response
            self.console.print("[bold magenta][Client] ✅ LLM successfully response[/bold magenta]:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                self.console.out(content, end="", style="cyan")
                collected_content.append(content)
                # return the iterator
                yield content
            self.console.print()
            
        except Exception as e:
            self.console.print(f"[bold red][Client] ❌ Error when invoking LLM API[/bold red]: [red]{e}[/red]")
            raise ConnectionError(f"Invoke LLM failed: {e}")


    def invoke(self, messages: List[dict[str, str]], **kwargs) -> str:
        """
        Non-streaming LLM invocation, return complete response.
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model, # type: ignore
                messages=messages, # type: ignore
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            return response.choices[0].message.content

        except Exception as e:
            self.console.print(f"[bold red][Client] ❌ Error when invoking LLM API[/bold red]: [red]{e}[/red]")
            raise ConnectionError(f"Invoke LLM failed: {e}")

