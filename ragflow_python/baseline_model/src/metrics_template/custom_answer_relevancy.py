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
