import ragflow_python.utils.logger as log
from ragflow_sdk import RAGFlow, Chat
from ragflow_sdk.modules.chunk import Chunk
import pandas as pd
import numpy as np
import json
import requests
import time
import datetime
import os
from supabase import create_client, Client
from pprint import pprint
from ragflow_python.utils.data_types import *
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval import evaluate
from deepeval.evaluate import EvaluationResult
from ragflow_python.utils.helpers import *
from ragflow_python.src.CustomLLama import CustomLLAMA3
from ragflow_python.src.CustomGemma2_2b import CustomGemma2B
from deepeval.models import DeepEvalBaseLLM
from deepeval.guardrails import Guardrails
from guardrails.hub import DetectPII, QARelevanceLLMEval
from guardrails import Guard, OnFailAction
from guardrails.hub import LlamaGuard7B

from guardrails import Guard


logger = log.setup_custom_logger('root')


class RagFlowTester: 
    
    def __init__(self, API_KEY: str, 
                       SUPABASE_KEY: str, 
                       SUPABASE_URL: str,
                       base_url: str, 
                       port: int = 80,
                       test_cases: list = [],
                       model_name: str = 'llama3.1:8b',
                       model: DeepEvalBaseLLM = None,
                       guardrails: Guardrails = None):
        
        self.api_key = API_KEY
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=self.api_key, base_url=f"{base_url}:{port}")
        
        self.test_cases = test_cases
        
        print(f"Current Configuration")
        print(f"API KEY: {self.api_key}")
        print(f"Base URL: {base_url}:{port}")
        print(f"Directory: {os.getcwd()}")
        
        #self.local_model = CustomLLAMA3()
        self.local_model = model
        
        self.contextual_precision = ContextualPrecisionMetric(model=self.local_model)
        self.contextual_recall = ContextualRecallMetric(model=self.local_model)
        self.contextual_relevancy = ContextualRelevancyMetric(model=self.local_model)
        
        
        # Supabase Client Settings
        self.supabase = create_client(supabase_url=SUPABASE_URL,
                                    supabase_key=SUPABASE_KEY)
        
        # Default LLM Settings
        self.llm = Chat.LLM(self.rag_object, 
                            {"model_name": model_name,
                               "temperature": 0.1,
                               "top_p": 0.3,
                               "presence_penalty": 0.4,
                               "frequency_penalty": 0.7,
                               "max_tokens": 512, })
        
        self.prompt = Chat.Prompt(self.rag_object, {"similarity_threshold": 0.2,
                                             "keywords_similarity_weight": 0.7,
                                             "top_n": 8,
                                             "top_k": 1024,
                                             "variables": [{
                                                 "key": "knowledge",
                                                 "optional": True
                                             }], "rerank_model": "",
                                             "empty_response": None,
                                             "opener": None,
                                             "show_quote": True,
                                             "prompt": None})
        
        # Caching Chat Data
        self.session = None
        self.dataset_ids = None
        
        # Guard Rails 
        self.guard = Guard().use_many(DetectPII(["EMAIL_ADDRESS", "PHONE_NUMBER"]),
                                      QARelevanceLLMEval(llm_callable=f"ollama/{model_name}", 
                                                         on_fail=OnFailAction.REASK),
                                      LlamaGuard7B(policies=[LlamaGuard7B.POLICY__NO_ILLEGAL_DRUGS, 
                                                             LlamaGuard7B.POLICY__NO_VIOLENCE_HATE,
                                                             LlamaGuard7B.POLICY__NO_SEXUAL_CONTENT,
                                                             LlamaGuard7B.POLICY__NO_CRIMINAL_PLANNING,
                                                             LlamaGuard7B.POLICY__NO_GUNS_AND_ILLEGAL_WEAPONS, 
                                                             LlamaGuard7B.POLICY__NO_ENOURAGE_SELF_HARM], 
                                                            on_fail=OnFailAction.REASK))
                        
    
    async def target_model_callback(self, prompt: str) -> str:
        try:

            testing_time = int(time.time() * 1000)
            if self.dataset_ids is None:
                datasets = self.rag_object.list_datasets(name="nomic")
                dataset_ids = []
                for dataset in datasets:
                    dataset_ids.append(dataset.id)
                self.dataset_ids = dataset_ids
            if self.session is None:
                self.session = self.rag_object.create_chat(f"Chat Assistant @ {testing_time}", dataset_ids=self.dataset_ids,
                                                            llm=self.llm, prompt=self.prompt).create_session()

            rag_response = self.session.ask(question=prompt, stream=True)
            
            response_content = ""  
            for ans in rag_response:
                #print(ans.content[len(cont):], end='', flush=True)
                response_content = ans.content

           # print(response_content)
            logger.info(f"Prompt: {prompt}, Response: {response_content}")
            return response_content
        
        except Exception as e:
            logger.error(f"Error at Target Model Callback: {e}")
            

    def target_model_callback_g(self, *, dataset_ids=None, session=None, llm=None, prompt=None, **kwargs) -> str:
        """Async generator that yields responses for Guardrails compatibility."""
        
        testing_time = int(time.time() * 1000)
        messages = kwargs.pop("messages", [])

        if not messages:
            raise ValueError("Messages list is empty.")

        if self.dataset_ids is None:
            datasets = self.rag_object.list_datasets(name="nomic")
            self.dataset_ids = [dataset.id for dataset in datasets]

        if self.session is None:
            self.session = self.rag_object.create_chat(
                f"Chat Assistant @ {testing_time}",
                dataset_ids=self.dataset_ids,
                llm=self.llm,
                prompt=self.prompt
            ).create_session()

        prompt = messages[-1]["content"]  # Extract the latest user message
        rag_response = self.session.ask(question=prompt, stream=True)

        res = ''
        for ans in rag_response:
            res = ans.content  # Yield content instead of returning a single string
        return res

    async def target_model_callback_guardrails(self, prompt: str) -> str:
        '''
        Wrapper around the target_model_callback to implement the guardrails
        '''

        messages = [{"role": "user", "content": prompt}]

        try:
            validated_response = self.guard(
                self.target_model_callback_g,
                messages=messages,
                prompt=prompt,  # Pass as a keyword argument
                metadata={"original_prompt": prompt},
                # stream=True
            )
            logger.info(f"Prompt: {prompt}, Guarded Response: {validated_response}")
            response_str = validated_response.raw_llm_output
            validated_output = validated_response.validated_output

            if validated_response.validation_passed is True:
                logger.info(f"Validation Passed......")
                return validated_output

            else:
                logger.info(f"Defense Activated, using defense reponses........")
                return self.get_defense_response(prompt)

        except Exception as e:
            logger.error(f"Error in model call back guardrails: {e}")
            logger.info(f"Defense Activated, using defense reponses........")
            return self.get_defense_response(prompt)
    
    def get_defense_response(self, prompt: str):
        '''
        Gets the defense response via llm call 
        '''
        input_prompt = f"You are a RAG Legal Chatbot and you have detected that potentially this prompt is suspicious in trying to illicit a poor response. \
                    This is the prompt in question '{prompt}'. \
                    Generate a short but professional response from the chatbot acknowledging the question but asking it to rephrase properly, keep it in context as a RAG Legal Chatbot."
                    
        return self.local_model.generate_normal(input_prompt)
              
           

# Example usage
if __name__ == "__main__":
    
    tester = RagFlowTester(
        api_key="your_api_key",
        base_url="localhost",
        test_cases={
            "What is the capital of France?": "Paris",
            "Who wrote '1984'?": "George Orwell"
        }
    )
    results_df = tester.test_rag()
    print(results_df)

        
        
        
        
        



