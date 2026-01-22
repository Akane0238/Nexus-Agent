import os
from openai import OpenAI
from dotenv import load_dotenv
from rich.console import Console

class NexusAgentsLLM:
    """
    Self-defined LLM client. 
    Capsulate OpenAI interface and use streaming response by default.
    """
    
    def __init__(self, model: str = None, apiKey: str = None, baseURL: str = None, timeout: int = None) -> None:  # type: ignore
        """
        Initialize client.
        """
        m = model or os.getenv("LLM_MODEL_ID")
        key = apiKey or os.getenv("LLM_API_KEY")
        url = baseURL or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([m, key, url]):
            raise ValueError("Model ID, API key and Serving address should be provided in `./.env` file.")
        
        self.model = m
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout)
        self.console = Console() # log for debug
    
    
    def think(self, message: list[dict[str, str]], temperature: float = 0) -> str | None :
        """
        Calling LLM to think and return the response.
        """

        self.console.print(f"🧠 [bold magenta]Calling {self.model} Model...[/bold magenta]")
        try:
            response = self.client.chat.completions.create(model=self.model, messages=message, temperature=temperature, stream=True,) #type: ignore

            # handle streaming response
            self.console.print("✅ [bold magenta]LLM successfully response[/bold magenta]:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                self.console.out(content, end="", style="cyan")
                collected_content.append(content)
            self.console.print()
            
            # return the complete response in string
            return "".join(collected_content)
        except Exception as e:
            self.console.print(f"❌ [bold red]Error when calling LLM API[/red]: [red]{e}[/red]")
            return None


# --- Testing ---
if __name__ == "__main__":
    # Loading enviroment variables from `.env`
    load_dotenv()
    
    try:
        llmClient = NexusAgentsLLM()

        msg = [
            {"role": "system", "content": "You are a helpful assistant that writes Golang code."},
            {"role": "user", "content": "Write a function to complete quick sorting algorithm."}
        ]

        print(f"--- Calling LLM ---")
        res = llmClient.think(msg)
        if res:
            print("\n\n--- Complete model's response ---")
            print(res)
    except ValueError as e:
        print(e)
