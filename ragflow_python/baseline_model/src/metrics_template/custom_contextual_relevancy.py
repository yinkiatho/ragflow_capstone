from deepeval.metrics.contextual_relevancy import ContextualRelevancyTemplate

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
