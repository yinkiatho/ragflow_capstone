import json
import time
import multiprocessing as mp
from datetime import datetime
<<<<<<< HEAD
import asyncio
=======
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
<<<<<<< HEAD
from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate


# -------------------------------------
# Custom Template for Contextual Relevancy Metric
# -------------------------------------
class CustomVerdictsTemplate(ContextualRelevancyTemplate):
    @staticmethod
    def generate_verdicts(input: str, context: str):
        return f"""You are a fact-checking assistant. Based on the input and context, determine whether each statement in the context is relevant to the input.
A statement is considered **relevant** if it has any connection to the input—whether directly, indirectly, or even partially. 
Mark a statement as "yes" unless it is clearly off-topic or completely unrelated.

Step 1: Analyze the provided context and split it into complete statements

Step 2: For each statement, return a JSON object with:
- 'verdict': either "yes" or "no".
- 'statement': always return null.
- 'reason': always return null.

Output:
{{
  "verdicts": [
    {{
      "verdict": "yes",
      "statement": null,
      "reason": null
    }},
    {{
      "verdict": "yes",
      "statement": null,
      "reason": null
    }}
  ]
}}

Input:
{input}

Context:
{context}

JSON:
"""


# -------------------------------------
# Custom Template for Answer Relevancy Metric
# -------------------------------------
from deepeval.metrics.answer_relevancy import AnswerRelevancyTemplate


class CustomAnswerRelevancyTemplate(AnswerRelevancyTemplate):
    @staticmethod
    def generate_statements(actual_output: str):
        return f"""Given the following answer, extract the key statements or points.
Example:
Answer: "The Earth revolves around the Sun and the Moon orbits the Earth."
Output:
{{
  "statements": [
    "The Earth revolves around the Sun.",
    "The Moon orbits the Earth."
  ]
}}
Answer:
{actual_output}

JSON:
"""

    @staticmethod
    def generate_verdicts(input: str, statements: str):
        return f"""For the provided list of statements, determine whether each statement is relevant to addressing the input.
        A statement is considered **relevant** if it satisfies any of the following conditions: it is any related to the input or provides any supporting information or offers any indirect support or directly answers the input.

Please generate a list of JSON objects with two keys: 'verdict' and 'reason'.
- Use 'yes' if the statement is relevant to the input.
- Use 'no' if the statement is irrelevant.

- Always output the 'reason' as null.
The number of verdicts must match the number of statements.

Output:
{{
  "verdicts": [
    {{"verdict": "yes", "reason": null}},
    {{"verdict": "yes", "reason": null}},
    {{"verdict": "yes", "reason": null}}
  ]
}}

Input:
{input}

Statements:
{statements}

JSON:
"""


# -------------------------------------
# Custom Template for Faithfulness Metric
# -------------------------------------
from deepeval.metrics.faithfulness import FaithfulnessTemplate


class CustomFaithfulnessTemplate(FaithfulnessTemplate):
    @staticmethod
    def generate_verdicts(claims: list[str], retrieval_context: str):
        return f"""Based on the given claims, which is a list of strings, generate a list of JSON objects to indicate whether EACH claim contradicts any facts in the retrieval context. The JSON will have 2 fields: 'verdict' and 'reason'.
For each claim, output a JSON object with two fields: 
  - 'verdict': use "yes" if the claim is supported or not contradicted by the retrieval context, and "no" if the retrieval context explicitly strongly contradicts the claim.
  - 'reason': always output null.
Only mark a claim as "no" if the retrieval context contains a clear, direct contradiction. If there is any ambiguity or if the claim uses vague, suggestive, or speculative language, default to "yes".
Do not use any prior knowledge; base your judgment only on the information in the retrieval context.
Ensure that the number of verdict objects is exactly equal to the number of claims.

Claims using vague, suggestive, or speculative language do NOT count as contradictions.
**  
IMPORTANT: Please make sure to only return in JSON format, with the 'verdicts' key mapping to a list of JSON objects.
Output:
{{
    "verdicts": [
        {{"verdict": "yes", "reason": null}},
        {{"verdict": "yes", "reason": null}}
    ]
}}
===== END OF EXAMPLE ======
Retrieval Contexts:
{retrieval_context}

Claims:
{claims}

JSON:
"""


# -------------------------------------
# Custom Template for Contextual Precision Metric
# -------------------------------------
from deepeval.metrics.contextual_precision import ContextualPrecisionTemplate


class CustomContextualPrecisionPrompt(ContextualPrecisionTemplate):
    @staticmethod
    def generate_verdicts(input: str, expected_output: str, retrieval_context: list[str]):
        return f"""Given the input, expected output, and retrieval context, please generate a list of JSON objects to determine whether each node in the retrieval context was remotely useful in arriving at the expected output.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }}
    ]
}}


Input:
{input}

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""


# -------------------------------------
# Custom Template for Contextual Recall Metric
# -------------------------------------
from deepeval.metrics.contextual_recall import ContextualRecallTemplate


class CustomContextualRecallPrompt(ContextualRecallTemplate):
    @staticmethod
    def generate_verdicts(expected_output: str, retrieval_context: list[str]):
        return f"""For each sentence in the expected output below, determine whether the sentence is supported by the retrieval context.
A sentence is considered supported if there is any indication—even minimal or indirect—that it is mentioned, alluded to, or can be inferred from the retrieval context.
Default to marking a sentence as "yes" unless it is clearly or directly contradicted by the retrieval context.
Always set the "reason" field to null.
Ensure that the number of verdicts exactly matches the number of sentences in the expected output.

Example JSON:
{{
    "verdicts": [
        {{
            "verdict": "yes",
            "reason": "..."
        }},
    ]
}}

Expected Output:
{expected_output}

Retrieval Context:
{retrieval_context}

JSON:
"""

=======
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from metrics_template.custom_contextual_precision import CustomContextualPrecisionPrompt
from metrics_template.custom_contextual_recall import CustomContextualRecallPrompt
from metrics_template.custom_contextual_relevancy import CustomVerdictsTemplate
from metrics_template.custom_answer_relevancy import CustomAnswerRelevancyTemplate
from metrics_template.custom_faithfulness import CustomFaithfulnessTemplate
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b

# -------------------------------------
# Factory Functions for Custom Metrics
# -------------------------------------
<<<<<<< HEAD
def custom_relevancy_metric():
    return ContextualRelevancyMetric(
        evaluation_template=CustomVerdictsTemplate,
        async_mode=False
    )


def custom_answer_relevancy_metric():
    return AnswerRelevancyMetric(
        evaluation_template=CustomAnswerRelevancyTemplate,
        async_mode=False
    )


def custom_faithfulness_metric():
    return FaithfulnessMetric(
        evaluation_template=CustomFaithfulnessTemplate,
        async_mode=False
    )


=======
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
def custom_precision_metric():
    return ContextualPrecisionMetric(
        evaluation_template=CustomContextualPrecisionPrompt,
        async_mode=False
    )

<<<<<<< HEAD

=======
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
def custom_recall_metric():
    return ContextualRecallMetric(
        evaluation_template=CustomContextualRecallPrompt,
        async_mode=False
    )

<<<<<<< HEAD

# -------------------------------------
# Multiprocessing Worker and Metric Runner Functions
=======
def custom_relevancy_metric():
    return ContextualRelevancyMetric(
        evaluation_template=CustomVerdictsTemplate,
        async_mode=False
    )

def custom_answer_relevancy_metric():
    return AnswerRelevancyMetric(
        evaluation_template=CustomAnswerRelevancyTemplate,
        async_mode=False
    )

def custom_faithfulness_metric():
    return FaithfulnessMetric(
        evaluation_template=CustomFaithfulnessTemplate,
        async_mode=False
    )

# -------------------------------------
# Multiprocessing Worker and Metric Runner
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
# -------------------------------------
def worker(metric_cls_or_factory, test_case, queue):
    try:
        metric = metric_cls_or_factory() if callable(metric_cls_or_factory) else metric_cls_or_factory(async_mode=False)
        metric.measure(test_case)
        queue.put(metric.score)
    except Exception as e:
        queue.put(e)

<<<<<<< HEAD

=======
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
def measure_metric(metric_cls_or_factory, test_case, timeout=180, retries=3):
    for attempt in range(1, retries + 1):
        result_queue = mp.Queue()
        process = mp.Process(target=worker, args=(metric_cls_or_factory, test_case, result_queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
<<<<<<< HEAD
            print(
                f"Attempt {attempt} for {getattr(metric_cls_or_factory, '__name__', str(metric_cls_or_factory))} timed out after {timeout} seconds.")
=======
            print(f"⚠️ Attempt {attempt} for {getattr(metric_cls_or_factory, '__name__', str(metric_cls_or_factory))} timed out after {timeout} seconds.")
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
        else:
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
<<<<<<< HEAD
                    print(
                        f"Attempt {attempt} for {getattr(metric_cls_or_factory, '__name__', str(metric_cls_or_factory))} raised exception: {result}")
                else:
                    return result
            except Exception as e:
                print(f"Attempt {attempt} failed to get result: {e}")
    return None


# -------------------------------------
# Test Case Processing Logic
=======
                    print(f"❌ Attempt {attempt} raised exception: {result}")
                else:
                    return result
            except Exception as e:
                print(f"❌ Attempt {attempt} failed to retrieve result: {e}")
    return None

# -------------------------------------
# Test Case Processing
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
# -------------------------------------
def process_test_case(case):
    start_time = time.time()

    test_case_answer = LLMTestCase(
        input=case["input_question"],
        actual_output=case["actual_output"]
    )
    test_case_faithfulness = LLMTestCase(
        input=case["input_question"],
        actual_output=case["actual_output"],
        retrieval_context=case["retrieval_context"]
    )
    test_case_contextual = LLMTestCase(
        input=case["input_question"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"]
    )
    test_case_relevancy = LLMTestCase(
        input=case["input_question"],
        actual_output=case["actual_output"],
        retrieval_context=case["retrieval_context"]
    )

<<<<<<< HEAD
    start_metric = time.time()
    precision_score = measure_metric(custom_precision_metric, test_case_contextual)
    print(f"✅ Precision score measured in {time.time() - start_metric:.2f} seconds.")

    start_metric = time.time()
    recall_score = measure_metric(custom_recall_metric, test_case_contextual)
    print(f"✅ Recall score measured in {time.time() - start_metric:.2f} seconds.")

    start_metric = time.time()
    relevancy_score = measure_metric(custom_relevancy_metric, test_case_relevancy)
    print(f"✅ Contextual relevancy score measured in {time.time() - start_metric:.2f} seconds.")

    start_metric = time.time()
    answer_score = measure_metric(custom_answer_relevancy_metric, test_case_answer)
    print(f"✅ Answer relevancy score measured in {time.time() - start_metric:.2f} seconds.")

    start_metric = time.time()
    faithfulness_score = measure_metric(custom_faithfulness_metric, test_case_faithfulness)
    print(f"✅ Faithfulness score measured in {time.time() - start_metric:.2f} seconds.")

    print(f"✅ Test case '{case['input_question']}' completed in {time.time() - start_time:.2f} seconds.")
=======
    precision_score = measure_metric(custom_precision_metric, test_case_contextual)
    print(f"✅ Precision measured in {time.time() - start_time:.2f}s")

    recall_score = measure_metric(custom_recall_metric, test_case_contextual)
    print(f"✅ Recall measured in {time.time() - start_time:.2f}s")

    relevancy_score = measure_metric(custom_relevancy_metric, test_case_relevancy)
    print(f"✅ Contextual relevancy measured in {time.time() - start_time:.2f}s")

    answer_score = measure_metric(custom_answer_relevancy_metric, test_case_answer)
    print(f"✅ Answer relevancy measured in {time.time() - start_time:.2f}s")

    faithfulness_score = measure_metric(custom_faithfulness_metric, test_case_faithfulness)
    print(f"✅ Faithfulness measured in {time.time() - start_time:.2f}s")

>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
    return {
        "input_question": case["input_question"],
        "actual_output": case["actual_output"],
        "expected_output": case["expected_output"],
        "retrieval_context": case["retrieval_context"],
        "precision": precision_score,
        "recall": recall_score,
        "relevancy": relevancy_score,
        "answer_relevancy": answer_score,
        "faithfulness": faithfulness_score
    }

<<<<<<< HEAD

=======
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
# -------------------------------------
# Main Runner
# -------------------------------------
def main():
<<<<<<< HEAD
    json_path = "test_cases_score_20250401_203536.json"
    with open(json_path, "r") as f:
        test_cases = json.load(f)

    # Uncomment to test on a subset:
    # test_cases = test_cases[:8]

=======
    json_path = "ragflow_python/baseline_model/test_case_output/test_cases_output_new.json"
    with open(json_path, "r") as f:
        test_cases = json.load(f)

>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b
    scores_list = []
    total_cases = len(test_cases)

    for idx, case in enumerate(test_cases, start=1):
        scores = process_test_case(case)
        scores_list.append(scores)
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
<<<<<<< HEAD
        print(f"Current time: {current_time_str} | Processed test case {idx} out of {total_cases}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = f"test_cases_score_{timestamp}.json"
    with open(output_json_path, "w") as f:
        json.dump(scores_list, f, indent=2)
    print("✅ All test cases processed successfully and scores saved to", output_json_path)
=======
        print(f"🕒 {current_time_str} | Processed test case {idx}/{total_cases}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"ragflow_python/baseline_model/test_scores/test_cases_score_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(scores_list, f, indent=2)
    print("✅ All test cases evaluated and saved to", output_path)
>>>>>>> 41d234eeea2af76a7d3f8efc5bf422bab522ed4b


if __name__ == "__main__":
    main()
