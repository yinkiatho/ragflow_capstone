from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
import ollama
from deepeval.models import DeepEvalBaseLLM


class CustomGemma2B(DeepEvalBaseLLM):
    def __init__(self):
        self.model_name = "gemma2:2b"
        self.client = ollama.Client(host="http://localhost:11434")  # Use the detected IP

    def load_model(self):
        """Return the model name (since Ollama handles loading internally)."""
        return self.model_name

    def generate(self, prompt: str) -> str:
        """Generate output from the local Ollama model."""
        response = self.client.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        """Async wrapper for generate()."""
        return self.generate(prompt)

    def get_model_name(self):
        """Return the model name."""
        return self.model_name