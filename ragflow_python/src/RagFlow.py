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

logger = log.setup_custom_logger('root')

class RagFlowTester: 
    
    def __init__(self, API_KEY: str, 
                       SUPABASE_KEY: str, 
                       SUPABASE_URL: str,
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
        
        
        # Supabase Client Settings
        self.supabase = create_client(supabase_url=SUPABASE_URL,
                                             supabase_key=SUPABASE_KEY)
        
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
        
        
    def test_rag(self, test_id: int, 
                       attack_type: Attack_Type=None, 
                       defense_type: Defense_Type=None,
                       is_attack: bool = False,
                       is_defense: bool = False):
        '''
        Function that initialises the chat session and executes the series of questions, testing against the answers
        '''
        try:
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
            
            result_queries = []
            result_chunks = []
            chunk_query_params = []
            for question, expected_answer in self.answer_question_pairs.items():
                
                # Querying using Chat Session
                logger.info(f"Asking: {question}, Expecting: {expected_answer}")
                ans = session.ask(question, stream=False)
                response = ans.content
                is_correct = self.verify_response(response=str(response), answer=expected_answer)
                result_queries.append({
                    'test_id': test_id,
                    'timestamp': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                    'attack_type': attack_type if is_attack else None,
                    'defense_type': defense_type if is_defense else None,
                    'input_parameters': attack_type.get_parameters() if is_attack else (defense_type.get_parameters() if is_defense else None),
                    'input_question': question,
                    'response': response,
                    'expected_answer': expected_answer,
                })
                
                # Looking at Chunks retrieved, make sure to add return data_json as well so it looks like this 
                '''
                ################# CHANGE IN ragflow_sdk.RAGFlow.retrieve() ############################
                 if res.get("code") ==0:
                    chunks=[]
                    for chunk_data in res["data"].get("chunks"):
                        chunk=Chunk(self,chunk_data)
                        chunks.append(chunk)
                    return chunks, data_json
                '''
                chunk_list, retrieval_params = self.rag_object.retrieve(question = question,
                                                      dataset_ids=dataset_ids,
                                                      page_size=2572) ## Total number of chunks in the current Knowledge Base
                
                result_chunks.append([ChunkWrapper.from_rag_chunk(chunk).to_json for chunk in chunk_list])
                chunk_query_params.append(retrieval_params)
                                
                                
            # df = pd.DataFrame(result_queries)
            # df.to_csv(f"ragflow_python/data/qna_results_{testing_time}.csv")
            
            # Inserting Data into Supabase
            response_res_queries = (
                self.supabase.table("FILL UP HERE")
                .insert(result_queries)
                .execute()
            )
            
            response_res_chunks = (
                self.supabase.table("CHUNK_Table name")
                .insert(result_chunks)
                .execute()
            )
            
            response_res_chunk_params = (
                self.supabase.table('cunk_params_table')
                .insert(result_chunks)
                .execute()
            )
            
            
            return response_res_queries and response_res_chunks and response_res_chunk_params
        
        except Exception as e:
            logger.error(f"Exception encountered in test_rag: {e}")
            return e
                
        
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

        
        
        
        
        



