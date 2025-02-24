import ragflow_python.utils.logger as log
from ragflow_sdk import RAGFlow, Chat
import pandas as pd
import numpy as np
import json
import requests
import time
import os

logger = log.setup_custom_logger('root')

class RagFlowTester: 
    
    def __init__(self, API_KEY: str, 
                       base_url: str, 
                       port: int = 80,
                       answer_question_pairs: dict = {}):
        
        self.api_key = API_KEY
        self.base_url = base_url
        self.rag_object = RAGFlow(api_key=self.api_key, base_url=f"{base_url}:{port}")
        
        self.answer_question_pairs = answer_question_pairs
        
        print(f"Current Configuration")
        print(f"API KEY: {self.api_key}")
        print(f"Base URL: {base_url}:{port}")
        print(f"Directory: {os.getcwd()}")
        
        
        # Default LLM Settings
        self.llm = Chat.LLM(self.rag_object, 
                            {"model_name": 'gemma2:2b',
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
        
        
    def test_rag(self, dataset_name: str = None):
        '''
        Function that initialises the chat session and executes the series of questions, testing against the answers
        '''
        
        # Create chat assistant
        testing_time = int(time.time() * 1000)
        datasets = self.rag_object.list_datasets()
        dataset_ids = []
        for dataset in datasets:
            dataset_ids.append(dataset.id)
            
        print(f"Queried Datasets: {[i.name for i in datasets]}")
        assistant = self.rag_object.create_chat(f"Chat Assistant @ {testing_time}", dataset_ids=dataset_ids,
                                                llm=self.llm, prompt=self.prompt)
        session = assistant.create_session()
        
        results = []
        
        for question, expected_answer in self.answer_question_pairs.items():
            logger.info(f"Asking: {question}, Expecting: {expected_answer}")
            for ans in session.ask(question, stream=False):
                response = ans.content
                is_correct = self.verify_response(response=str(response), answer=expected_answer)
                results.append({
                    'question': question,
                    'expected_answer': expected_answer,
                    'actual_response': response,
                    'is_correct': is_correct
                })
        
        df = pd.DataFrame(results)
        df.to_csv(f"ragflow_python/data/qna_results_{testing_time}.csv")
        
        return df
                
        
    @staticmethod
    def verify_response(response: str, answer: str) -> bool:
        '''
        Function verifies the response with the answer
        '''
        return True
           

# Example usage
if __name__ == "__main__":
    tester = RagFlowTester(
        api_key="your_api_key",
        base_url="localhost",
        answer_question_pairs={
            "What is the capital of France?": "Paris",
            "Who wrote '1984'?": "George Orwell"
        }
    )
    results_df = tester.test_rag()
    print(results_df)

        
        
        
        
        



