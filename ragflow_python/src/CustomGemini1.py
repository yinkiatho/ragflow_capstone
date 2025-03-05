from pydantic import BaseModel
import google.generativeai as genai
import instructor
import time
from deepeval.models import DeepEvalBaseLLM


class CustomGeminiFlash(DeepEvalBaseLLM):
    def __init__(self, api_key: str = ''):
        self.model_name = "gemini-1.5-flash"
        self.api_key = api_key
        genai.configure(api_key=api_key)
        #self.client = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")
        self.model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")


    def load_model(self):
        return self.model
    

    def generate(self, prompt: str, schema: BaseModel, sleep: float = 1.0) -> BaseModel:
        client = self.load_model()
        instructor_client = instructor.from_gemini(
            client=client,
            mode=instructor.Mode.GEMINI_JSON,
        )
        resp = instructor_client.messages.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_model=schema,
        )
        time.sleep(sleep)
        return resp

    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return "Gemini 1.5 Flash"