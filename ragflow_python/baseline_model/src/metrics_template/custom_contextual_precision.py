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
