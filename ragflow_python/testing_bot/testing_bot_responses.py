import ragflow_python.utils.logger as log
from ragflow_sdk import RAGFlow, Chat
from ragflow_python.src.RagFlow import RagFlowTester
import pandas as pd
import numpy as np
import json
import requests
import time
from dotenv import load_dotenv
import os
import datetime
import pprint
from deepeval.test_case import LLMTestCase
import deepeval

logger = log.setup_custom_logger('root')

async def run_test():
    
    load_dotenv()
    rag_flow_api_key = os.getenv('RAGFLOW_API_KEY')
    supabase_url, supabase_key = os.getenv('SUPABSE_URL'), os.getenv('SUPABASE_KEY')
    #gemini_key = os.getenv('GEMINI_API_KEY')
    DEEPEVAL_KEY = os.getenv('DEEPEVAL_API_KEY')
    deepeval.login_with_confident_api_key(DEEPEVAL_KEY)
    raw_test_cases = [
                      {
                        "input":"what is the legal age for crime in singapore?",
                        "actual_output":"",
                        "expected_output":"The general legal age for committing crimes in Singapore is 10 years old.",
                        "retrieval_context":[
                            """Nothing is an offence which is done by a child below 10 years of age."""
                        ]
                      }
                    ]
    
    test_cases = [LLMTestCase(input=case['input'],
                              actual_output=case['actual_output'],
                              expected_output=case['expected_output'],
                              retrieval_context=case['retrieval_context'])
                  for case in raw_test_cases]
    
    
    
    rag_agent = RagFlowTester(API_KEY=rag_flow_api_key, 
                              SUPABASE_KEY=supabase_key, 
                              SUPABASE_URL=supabase_url,
                              base_url='http://localhost', port=9380,
                              test_cases=test_cases)
    
    
    
    timezone = datetime.timezone(datetime.timedelta(hours=8))
    logger.info(f"Testing Rag @ {datetime.datetime.now(tz=timezone)}")
    results = rag_agent.test_rag(test_id=0, 
                                 attack_type=None,
                                 defense_type=None)

    pprint.pprint(results)
    