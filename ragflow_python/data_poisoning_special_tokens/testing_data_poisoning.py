import random
import os
import json
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from dotenv import load_dotenv

load_dotenv()


# Parameters
json_data = []

# change directory to defense_result1, defense_result2, defense_result3
directory = "./defense_results3"
num_of_chunks_to_mask = 50 # defense_result1 num_of_chunks_to_mask = 10
perplexity_threshold = 500
is_new_qa_pair =  True # only True for defense_result3


url = "https://bggngaqkkmslamsbebew.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJnZ25nYXFra21zbGFtc2JlYmV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA0MDgxNzIsImV4cCI6MjA1NTk4NDE3Mn0.B4D-5t0oxa8D6xMSoywufdB7aSmGy1s8bvytH0znows"
supabase = create_client(url, key)

special_token_attack_id = 1080
defense_id = 1988


def process_attack_results(data):
    """Processes a single test case and inserts data into Supabase"""

    # Initialize metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # Create data test case for post attack
    value = data["post_attack"]
    test_case = LLMTestCase(
                input= value[0],
                actual_output=value[3],
                expected_output=value[1],
                retrieval_context=value[2]
            )
    
    # Measure metrics for post attack
    contextual_precision.measure(test_case)
    precision_score = contextual_precision.score

    contextual_recall.measure(test_case)
    recall_score = contextual_recall.score

    contextual_relevancy.measure(test_case)
    relevancy_score = contextual_relevancy.score

    answer_relevancy.measure(test_case)
    answer_score = answer_relevancy.score

    faithfulness.measure(test_case)
    faithfulness_score = faithfulness.score

    # Insert into Special_Tokens_Attack_Table
    attack_response = supabase.table("Special_Token_Attacks").insert({
        "attack_id": special_token_attack_id, 
        "contextual_precision": precision_score,
        "contextual_recall": recall_score,
        "contextual_relevancy": relevancy_score,
        "answer_relevancy": answer_score,
        "faithfulness": faithfulness_score,
        "perplexity_threshold": perplexity_threshold, 
        "num_chunks_to_mask": num_of_chunks_to_mask,
        "chunks_retrieved": value[2],
        "llm_answer": value[3],
        "ground_truth": value[1],
        "question": value[0],
        "is_new_qa_pair": is_new_qa_pair
    }).execute()

    print(attack_response)

    # Create data test case for post defense
    pd_value = data["post_defense"]
    pd_test_case = LLMTestCase(
                input= pd_value[0],
                actual_output=pd_value[3],
                expected_output=pd_value[1],
                retrieval_context=pd_value[2]
            )
    
    # Measure metrics for post attack
    contextual_precision.measure(pd_test_case)
    precision_score = contextual_precision.score

    contextual_recall.measure(pd_test_case)
    recall_score = contextual_recall.score

    contextual_relevancy.measure(pd_test_case)
    relevancy_score = contextual_relevancy.score

    answer_relevancy.measure(pd_test_case)
    answer_score = answer_relevancy.score

    faithfulness.measure(pd_test_case)
    faithfulness_score = faithfulness.score

    # Insert into Perplexity_Defense Table
    defense_response = supabase.table("Perplexity_Defense").insert({
        "defense_id": defense_id, 
        "contextual_precision": precision_score,
        "contextual_recall": recall_score,
        "contextual_relevancy": relevancy_score,
        "answer_relevancy": answer_score,
        "faithfulness": faithfulness_score,
        "perplexity_threshold": perplexity_threshold, 
        "num_chunks_to_mask": num_of_chunks_to_mask,
        "chunks_retrieved": pd_value[2],
        "llm_answer": pd_value[3],
        "ground_truth": pd_value[1],
        "question": pd_value[0],
        "is_new_qa_pair": is_new_qa_pair
    }).execute()
    print(defense_response)
    print(f"✅ Test case '{value[0]}' inserted successfully!\n")


# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # Check if file is JSON
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load JSON data into a dictionary
            json_data.append(data)

# Run test cases in parallel
test_cases = json_data
MAX_WORKERS = min(5, len(test_cases))  # Limit to 5 workers or number of test cases
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    list(executor.map(process_attack_results, test_cases))

print("✅ All test cases processed successfully!")


if __name__ == "__main__":
    # Your code is already here
    print("Script executed directly")

        

