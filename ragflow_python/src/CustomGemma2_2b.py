from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
import ollama
from deepeval.models import DeepEvalBaseLLM
from lmformatenforcer import JsonSchemaParser
import json
import instructor


class CustomGemma2B(DeepEvalBaseLLM):
    '''
    Wrapper LLM gemma2:2b for attacks, uses local gemma2:2b through Ollama
    '''
    def __init__(self):
        self.model_name = "gemma2:2b"
        self.client = ollama.Client(host="http://localhost:11434")  # Use the detected IP

    def load_model(self):
        """Return the model name (since Ollama handles loading internally)."""
        return self.model_name

    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Generate output from the local Ollama model."""
        response = self.client.chat(model=self.model_name, 
                                    messages=[{"role": "user", "content": prompt}],
                                    format=schema.model_json_schema())
        
        # Ensure response.message.content is valid JSON before parsing
        #print(response)
        try:
            response_data = json.loads(response.message.content)  # Convert string to dict
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON response: Prompt: {prompt}, Content: {response.message.content}")
        
        return schema(**response_data) 
    
    
    def generate_normal(self, prompt: str):
        response = self.client.chat(model=self.model_name,
                                    messages=[{"role": "user", "content": prompt}])
        
        return response["message"]["content"]
       
        
    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        """Async wrapper for generate()."""
        return self.generate(prompt, schema)

    def get_model_name(self):
        """Return the model name."""
        return self.model_name