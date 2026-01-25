import os
from typing import Optional
from openai import OpenAI
from src.nexus_agent.paradigms.llm_client import NexusAgentsLLM


class MyLLM(NexusAgentsLLM):
    """
    Self-defined LLM client, adding support for ModelScope.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseURL: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ) -> None:
        if provider == "modelscope":
            print("Using self-defined ModelScope Provider.")
            self.provider = "modelscope"

            # Parse certification of ModelScope
            self.api_key = apiKey or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = baseURL or "https://api-inference.modelscope.cn/v1/"

         # Verify certification
            if not self.api_key:
                raise ValueError("ModelScope API key not found.")

            self.model = model or os.getenv("LLM_MODEL_ID") or "Qwen/Qwen3-8B"
            self.temprature = kwargs.get('temprature', 0.7)
            self.max_tokens = kwargs.get('max_tokens')
            self.timeout = kwargs.get('timeout', 60)

            # Create an OpenAI client instant
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout
            )

        else:
            # LLM provider is not `modelscope`, 
            # use original initialization.
            super().__init__(model=model, apiKey=apiKey, baseURL=baseURL, **kwargs)
