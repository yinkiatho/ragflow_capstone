from deepeval.evaluate import EvaluationResult, TestResult, MetricData


def evaluation_result_to_json(result: EvaluationResult):
    results_metrics_json = result.model_dump()
    return results_metrics_json