from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Any, Callable


class ToolParameter(BaseModel):
    """Definition of tool parameter"""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


class Tool(ABC):
    """Base class of tool"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str:
        """Tool execution"""
        pass


    @abstractmethod
    def get_parameters(self) -> list[ToolParameter]:
        """Acquire parameter definition of tool"""
        pass

