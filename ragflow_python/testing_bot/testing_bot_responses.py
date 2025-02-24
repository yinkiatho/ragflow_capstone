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


logger = log.setup_custom_logger('root')


async def run_test():
    
    load_dotenv()
    rag_flow_api_key = os.getenv('RAGFLOW_API_KEY')
    gemini_key = os.getenv('GEMINI_API_KEY')
    rag_agent = RagFlowTester(API_KEY=rag_flow_api_key, base_url='http://localhost', port=9380, 
                            answer_question_pairs={
                                "What is the capital of France?": "Paris",
                                "Who wrote '1984'?": "George Orwell"
                            })
    
    timezone = datetime.timezone(datetime.timedelta(hours=8))
    logger.info(f"Testing Rag @ {datetime.datetime.now(tz=timezone)}")
    results = rag_agent.test_rag('Crimminal Law')

    pprint.pprint(results)
    