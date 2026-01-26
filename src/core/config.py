import os
from typing import Optional, Any
from pydantic import BaseModel, Field


class Config(BaseModel):
    # LLM configuration
    default_model: str = "deepseek-ai/DeepSeek-V3.2"
    default_provider: str = "siliconflow"
    temperature: float = Field(default=0.5, ge=0, le=2)
    max_tokens: Optional[int] = None

    # System configuration
    debug: bool = False
    log_level: str = "INFO"

    # Other settings
    max_history_length: int = Field(default=100)


    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            temperature=float(os.getenv("TEMPERATURE", 0.7)),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None, # type: ignore
        )


    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

