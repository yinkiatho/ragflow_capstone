import json
import time
import multiprocessing as mp
from datetime import datetime

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from metrics_template.custom_contextual_precision import CustomContextualPrecisionPrompt
from metrics_template.custom_contextual_recall import CustomContextualRecallPrompt
from metrics_template.custom_contextual_relevancy import CustomVerdictsTemplate
from metrics_template.custom_answer_relevancy import CustomAnswerRelevancyTemplate
from metrics_template.custom_faithfulness import CustomFaithfulnessTemplate

# -------------------------------------
# Factory Functions for Custom Metrics
# -------------------------------------
def custom_precision_metric():
    return ContextualPrecisionMetric(
        evaluation_template=CustomContextualPrecisionPrompt,
        async_mode=False
    )

def custom_recall_metric():
    return ContextualRecallMetric(
        evaluation_template=CustomContextualRecallPrompt,
        async_mode=False
    )

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
# -------------------------------------
def worker(metric_cls_or_factory, test_case, queue):
    try:
        metric = metric_cls_or_factory() if callable(metric_cls_or_factory) else metric_cls_or_factory(async_mode=False)
        metric.measure(test_case)
        queue.put(metric.score)
    except Exception as e:
        queue.put(e)

def measure_metric(metric_cls_or_factory, test_case, timeout=180, retries=3):
    for attempt in range(1, retries + 1):
        result_queue = mp.Queue()
        process = mp.Process(target=worker, args=(metric_cls_or_factory, test_case, result_queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"⚠️ Attempt {attempt} for {getattr(metric_cls_or_factory, '__name__', str(metric_cls_or_factory))} timed out after {timeout} seconds.")
        else:
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    print(f"❌ Attempt {attempt} raised exception: {result}")
                else:
                    return result
            except Exception as e:
                print(f"❌ Attempt {attempt} failed to retrieve result: {e}")
    return None

# -------------------------------------
# Test Case Processing
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

# -------------------------------------
# Main Runner
# -------------------------------------
def main():
    json_path = "ragflow_python/baseline_model/test_case_output/test_cases_output_new.json"
    with open(json_path, "r") as f:
        test_cases = json.load(f)

    scores_list = []
    total_cases = len(test_cases)

    for idx, case in enumerate(test_cases, start=1):
        scores = process_test_case(case)
        scores_list.append(scores)
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🕒 {current_time_str} | Processed test case {idx}/{total_cases}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"ragflow_python/baseline_model/test_scores/test_cases_score_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(scores_list, f, indent=2)
    print("✅ All test cases evaluated and saved to", output_path)


if __name__ == "__main__":
    main()
