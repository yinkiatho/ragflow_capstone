from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
import ollama
import json

class PoisonGemma2B(DeepEvalBaseLLM):
    '''
    Wrapper LLM for Gemma 2B to generate poisoned responses.
    Uses local Gemma 2B through Ollama.
    '''
    def __init__(self):
        self.model_name = "gemma2:2b"
        self.client = ollama.Client(host="http://localhost:11434")  # Local Ollama instance

    def load_model(self):
        """Return the model name (Ollama handles loading internally)."""
        return self.model_name

    def generate(self, prompt_question, ground_truth: str, schema: BaseModel):
        """Generate an inaccurate version of the given ground truth."""
        prompt = f"""
        The following is a factually correct legal statement:

        "{ground_truth}"
        
        This is a prompt question:
        "{prompt_question}"

        Your task is to generate a misleading or inaccurate version of this statement that answers the prompt question.
        
        Change key facts in a way that appears plausible but introduces misinformation.
        Do NOT explicitly state that the answer is incorrect.
        Do NOT bold, italic or generate formatted text.

        Provide the response in JSON format. 
        """

        response = self.client.chat(model=self.model_name, 
                                    messages=[{"role": "user", "content": prompt}],
                                    format=schema.model_json_schema())
        
        # Ensure response.message.content is valid JSON before parsing
        try:
            response_data = json.loads(response.message.content)  # Convert string to dict
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON response: {response.message.content}")
        
        return schema(**response_data)

    async def a_generate(self, ground_truth: str, schema: BaseModel) -> BaseModel:
        """Async wrapper for generate()."""
        return self.generate(ground_truth, schema)

    def get_model_name(self):
        """Return the model name."""
        return self.model_name
    


