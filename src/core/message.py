from typing import Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# Define types of message sender
MessageRole = Literal["user", "system", "assistant", "tool"]

class Message(BaseModel):
    content: str
    role: MessageRole
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


    def __init__(self, content: str, role: MessageRole, **kwargs):
        super().__init__(
            content=content,
            role=role,
            timestamp=kwargs.get("timestamp", datetime.now()),
            metadata=kwargs.get("metadata", {})
        )

    
    def to_dict(self) -> dict[str, Any]:
        # Convert to OpenAI format
        return {"role": self.role, "content": self.content}


    def __str__(self) -> str:
        return f"[{self.role}]{self.content}"

