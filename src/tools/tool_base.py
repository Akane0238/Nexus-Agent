from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError
from typing import Any, Type, Optional


class Tool(ABC):
    """Base class of tool with Pydantic support"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: dict[str, Any]) -> str | dict[str, Any]:
        """
        Tool execution with validated parameters.

        Args:
            parameters: Validated parameters dictionary

        Returns:
            Execution result as string or structured dictionary
        """
        pass

    @abstractmethod
    def get_input_schema(self) -> Type[BaseModel] | None:
        """
        Return Pydantic BaseModel class for parameter definition.
        Override this method to use Pydantic-based parameter validation.

        Example:
            from pydantic import BaseModel, Field

            class SearchInput(BaseModel):
                query: str = Field(description="Search query")
                max_results: int = Field(default=5, description="Max results")

            class SearchTool(Tool):
                def get_input_schema(self):
                    return SearchInput

        Returns:
            BaseModel class or None (if no parameters needed)
        """
        return None

    def get_tool_schema(self) -> dict:
        """
        Generate complete tool schema in JSON format.
        This is used to pass tool definitions to LLM.

        Returns:
            Tool schema dictionary with name, description, and parameters
        """
        schema_cls = self.get_input_schema()

        if not schema_cls:
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            }

        json_schema = schema_cls.model_json_schema()
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }

    def validate_parameters(
        self, parameters: dict[str, Any]
    ) -> tuple[bool, dict | None, str | None]:
        """
        Validate input parameters against tool's Pydantic schema.

        Args:
            parameters: Raw parameters dictionary to validate

        Returns:
            (is_valid, validated_params, error_message)
            - is_valid: True if validation passed
            - validated_params: Validated and type-converted parameters (if valid)
            - error_message: Error description (if invalid)
        """
        schema_cls = self.get_input_schema()

        if schema_cls:
            try:
                validated = schema_cls.model_validate(parameters)
                return True, validated.model_dump(), None
            except ValidationError as e:
                return False, None, str(e)

        return True, parameters, None
