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

# change directory to defense_results_10_new_qa
directory = "../defense_results_10_new_qa"
num_of_chunks_to_mask = 10 # defense_results_10_new_qa num_of_chunks_to_mask = 10
perplexity_threshold = 500
is_new_qa_pair =  True # True for both cases


url = "YOUR SUPABASE URL"
key = "YOUR SUPABASE API KEY"
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
    
    # Measure metrics for post defense
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

for test_case in json_data:
    process_attack_results(test_case)
    print(f"Processed test case {json_data.index(test_case) + 1} of {len(json_data)}")

print("✅ All test cases processed successfully!")


if __name__ == "__main__":
    # Your code is already here
    print("Script executed directly")

        

