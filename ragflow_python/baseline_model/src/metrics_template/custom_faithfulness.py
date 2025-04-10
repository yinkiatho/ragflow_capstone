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
