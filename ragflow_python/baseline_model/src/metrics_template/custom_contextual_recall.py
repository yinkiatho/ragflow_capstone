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
