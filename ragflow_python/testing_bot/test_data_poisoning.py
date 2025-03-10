import os
import json
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase
from deepeval import evaluate

'''
TO DO NEXT:
upload test data to supabase, referring to test_cases_db.py
for pre-attack/post-attack labelling, just label it in the experiment_type (to think about later!)
'''

contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

# Directory containing JSON files
directory = "your json path"

# List to store JSON data
json_data = []


# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # Check if file is JSON
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load JSON data into a dictionary
            json_data.append(data)


for data in json_data:
    for key, value in data.items():
        test_case = LLMTestCase(
            input=value[0],
            actual_output=value[3],
            expected_output=value[1],
            retrieval_context=value[2]
        )

        print(f"evaluating {key} retrieval and llm answer:")
        evaluate(
            test_cases=[test_case],
            metrics=[contextual_precision, contextual_recall, contextual_relevancy]
        )

        


