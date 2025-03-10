import os
import json
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

from deepeval.test_case import LLMTestCase
from CustomGemma2_2b import CustomGemma2B
from deepeval import evaluate


custom_model = CustomGemma2B()
contextual_precision = ContextualPrecisionMetric()
contextual_recall = ContextualRecallMetric()
contextual_relevancy = ContextualRelevancyMetric()

# Directory containing JSON files
directory = "path/to/dir/w/all/json/results"

# List to store JSON data
json_data = []

# Iterate through all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # Check if file is JSON
        file_path = os.path.join(directory, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load JSON data into a dictionary
            json_data.append(data)

test_cases = []
for data in json_data:
    # pre-attack eval
    pre_attack_data = data["pre_attack"]
    test_case = LLMTestCase(
        input=pre_attack_data[0],
        actual_output=pre_attack_data[3],
        expected_output=pre_attack_data[1],
        retrieval_context=pre_attack_data[2]
    )

    # post-attack eval
    post_attack_data = data["pre_attack"]
    test_case = LLMTestCase(
        input=post_attack_data[0],
        actual_output=post_attack_data[3],
        expected_output=post_attack_data[1],
        retrieval_context=post_attack_data[2]
    )



evaluate(
    test_cases=test_cases,
    metrics=[contextual_precision, contextual_recall, contextual_relevancy]
)
