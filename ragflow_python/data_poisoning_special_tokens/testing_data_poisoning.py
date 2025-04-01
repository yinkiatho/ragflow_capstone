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

'''
Hi mak! Remember to ignore the 'pre_attack' in each JSON file because I did not run chat for pre_attack to save time and CPU. You can start with 'post_attack' and 'post_defense'.
You can also create a separate table on supabase to upload the data, then ignore the experiment_id and all that

defense_results1 = results with a defense that only mask top 10 chunks w perplexity >= 500
defense_results2 = results with a defense that mask top 50 chunks w perplexity >= 500
defense_results3 = results with a defense that mask top 50 chunks w perplexity >= 500, but with new qa_pairs created by jj
'''

# Parameters
json_data = []
directory = "../data/data_poisoning"


url = "https://bggngaqkkmslamsbebew.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJnZ25nYXFra21zbGFtc2JlYmV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA0MDgxNzIsImV4cCI6MjA1NTk4NDE3Mn0.B4D-5t0oxa8D6xMSoywufdB7aSmGy1s8bvytH0znows"
supabase = create_client(url, key)
pre_attack_experiment_id = 372
post_attack_experiment_id = 6890


def generate_unique_id():
    """Generate a unique retrieval_id that does not exist in RAGFlow_Response"""
    while True:
        new_id = random.randint(1, 10000)
        response = supabase.table("RAGFlow_Response").select("retrieval_id").eq("retrieval_id", new_id).execute()
        
        if not response.data:  # If no existing retrieval_id found, it's unique
            return new_id


def generate_unique_experiment_id():
    while True:
        new_id = random.randint(1, 10000)
        response = supabase.table("Evaluation_Table").select("evaluation_id").eq("evaluation_id", new_id).execute()

        if not response.data:  # If no existing retrieval_id found, it's unique
            return new_id

def process_attack_results(data):
    """Processes a single test case and inserts data into Supabase"""
    

    # Initialize metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # Create data test case for both pre and post attack
    for key, value in data.items():
        retrieval_id = generate_unique_id()
        evaluation_id = generate_unique_experiment_id()
        test_case = LLMTestCase(
            input= value[0],
            actual_output=value[3],
            expected_output=value[1],
            retrieval_context=value[2]
        )

        # Measure metrics
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

        # Insert into RAGFlow_Response first
        RAGFlow_response = supabase.table("RAGFlow_Response").insert({
            "retrieval_id": retrieval_id, 
            "chunks_retrieved": value[2],
            "model_answer": value[3]
        }).execute()

        # Ensure retrieval_id exists before inserting into Evaluation_Table
        confirmation = supabase.table("RAGFlow_Response").select("retrieval_id").eq("retrieval_id", retrieval_id).execute()
        if not confirmation.data:
            raise Exception(f"retrieval_id {retrieval_id} was not found in RAGFlow_Response after insertion!")
        
        # Insert into Evaluation_Table
        ex_id = None # experiment_id
        if key == "pre_attack":
            ex_id = pre_attack_experiment_id
        elif key == "post_attack":
            ex_id = post_attack_experiment_id

        evaluation_response = supabase.table("Evaluation_Table").insert({
            "evaluation_id": evaluation_id,
            "experiment_type_id": ex_id,
            "contextual_precision": precision_score,
            "contextual_recall": recall_score,
            "contextual_relevancy": relevancy_score,
            "answer_relevancy": answer_score,
            "faithfulness": faithfulness_score,
            "retrieval_id": retrieval_id,  # This key now exists in RAGFlow_Response
            "ground_truth": value[1],
            "question": value[0],
            "model": "gemma2:2b"
        }).execute()

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

        

