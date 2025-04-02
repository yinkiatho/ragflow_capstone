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

def worker(metric_cls, test_case, queue):
    """
    Worker function that instantiates the metric (with async_mode=False),
    calls metric.measure(test_case), and puts the resulting metric.score into a queue.
    """
    try:
        # Instantiate a fresh metric object in the worker process
        metric = metric_cls(async_mode=False)
        metric.measure(test_case)
        queue.put(metric.score)
    except Exception as e:
        queue.put(e)

def measure_metric(metric_cls, test_case, timeout=180, retries=3):
    """
    Runs the blocking metric.measure(test_case) in a separate process with a timeout.
    Retries up to 'retries' times if it times out or if an exception is raised.
    Returns metric.score if successful, otherwise None.
    """
    for attempt in range(1, retries + 1):
        result_queue = mp.Queue()
        process = mp.Process(target=worker, args=(metric_cls, test_case, result_queue))
        process.start()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join()
            print(f"Attempt {attempt} for {metric_cls.__name__} timed out after {timeout} seconds.")
        else:
            try:
                result = result_queue.get_nowait()
                if isinstance(result, Exception):
                    print(f"Attempt {attempt} for {metric_cls.__name__} raised exception: {result}")
                else:
                    return result
            except Exception as e:
                print(f"Attempt {attempt} for {metric_cls.__name__} failed to get result: {e}")
    return None

def process_test_case(case):
    """
    Processes a single test case synchronously:
      - Creates the appropriate LLMTestCase objects,
      - Calls each metric's measure() method (via measure_metric) sequentially,
      - Prints the time taken for each metric.
    Returns a dictionary containing the test case information and metric scores.
    """
    start_time = time.time()
    
    # Create LLMTestCase objects for each metric's requirements
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
    
    # Measure each metric sequentially and print timing information
    start_metric = time.time()
    precision_score = measure_metric(ContextualPrecisionMetric, test_case_contextual)
    precision_time = time.time() - start_metric
    print(f"✅ Precision score measured in {precision_time:.2f} seconds.")
    
    start_metric = time.time()
    recall_score = measure_metric(ContextualRecallMetric, test_case_contextual)
    recall_time = time.time() - start_metric
    print(f"✅ Recall score measured in {recall_time:.2f} seconds.")
    
    start_metric = time.time()
    relevancy_score = measure_metric(ContextualRelevancyMetric, test_case_relevancy)
    relevancy_time = time.time() - start_metric
    print(f"✅ Contextual relevancy score measured in {relevancy_time:.2f} seconds.")
    
    start_metric = time.time()
    answer_score = measure_metric(AnswerRelevancyMetric, test_case_answer)
    answer_time = time.time() - start_metric
    print(f"✅ Answer relevancy score measured in {answer_time:.2f} seconds.")
    
    start_metric = time.time()
    faithfulness_score = measure_metric(FaithfulnessMetric, test_case_faithfulness)
    faithfulness_time = time.time() - start_metric
    print(f"✅ Faithfulness score measured in {faithfulness_time:.2f} seconds.")
    
    total_time = time.time() - start_time
    print(f"✅ Test case '{case['input_question']}' completed in {total_time:.2f} seconds.")
    
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

def main():
    # Load test cases from the JSON file
    json_path = "ragflow_python/baseline_model/test_case_output/test_cases_score_20250401_203536.json"
    with open(json_path, "r") as f:
        test_cases = json.load(f)
    
    scores_list = []
    total_cases = len(test_cases)
    
    for idx, case in enumerate(test_cases, start=1):
        scores = process_test_case(case)
        scores_list.append(scores)
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_time_str} | Processed test case {idx} out of {total_cases}")
    
    # Save the results to a timestamped JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = f"ragflow_python/baseline_model/test_scores/test_cases_score_{timestamp}.json"
    with open(output_json_path, "w") as f:
        json.dump(scores_list, f, indent=2)
    
    print("✅ All test cases processed successfully and scores saved to", output_json_path)

if __name__ == "__main__":
    main()
