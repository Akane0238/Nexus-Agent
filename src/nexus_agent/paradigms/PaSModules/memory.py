from typing import Any, Optional


class Memory:
    """
    A simple short-term memory module,
    saving actions and reflection trajectory of agent.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []


    def add_record(self, record_type: str, content: str):
        """
        Add new record to memory.

        Parameters:
        - record_type (str): "execution" or "reflection"
        - content: concrete content of record
        """

        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"📝 Memory updates，adding new record typed '{record_type}'\n")

    def get_trajectory(self) -> str:
        """
        Serialize all the memory records for building prompt.
        """

        trajectory_parts = []
        for record in self.records:
            if record["type"] == "execution":
                trajectory_parts.append(f'--- Attemption of last turn (code) ---\n{record["content"]}')
            elif record["type"] == "reflection":
                trajectory_parts.append(f'--- Feedback of evaluator ---\n{record["content"]}')

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        """
        Acquire result of last execution, return None if do not exist.
        """
        
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]

        return None
