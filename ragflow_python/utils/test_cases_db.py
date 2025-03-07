import random
from concurrent.futures import ThreadPoolExecutor
from supabase import create_client
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric
)
from deepeval.test_case import LLMTestCase


url = "https://bggngaqkkmslamsbebew.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJnZ25nYXFra21zbGFtc2JlYmV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA0MDgxNzIsImV4cCI6MjA1NTk4NDE3Mn0.B4D-5t0oxa8D6xMSoywufdB7aSmGy1s8bvytH0znows"
supabase = create_client(url, key)

def generate_unique_id():
    """Generate a unique retrieval_id that does not exist in RAGFlow_Response"""
    while True:
        new_id = random.randint(1, 10000)
        response = supabase.table("RAGFlow_Response").select("retrieval_id").eq("retrieval_id", new_id).execute()
        
        if not response.data:  # If no existing retrieval_id found, it's unique
            return new_id

def process_test_case(case):
    """Processes a single test case and inserts data into Supabase"""

    retrieval_id = generate_unique_id()

    # Initialize metrics
    contextual_precision = ContextualPrecisionMetric()
    contextual_recall = ContextualRecallMetric()
    contextual_relevancy = ContextualRelevancyMetric()
    answer_relevancy = AnswerRelevancyMetric()
    faithfulness = FaithfulnessMetric()

    # Create test case object
    test_case = LLMTestCase(
        input=case["input_question"],
        actual_output=case["actual_output"],
        expected_output=case["expected_output"],
        retrieval_context=case["retrieval_context"]
    )

    # Measure metrics
    contextual_precision.measure(test_case)
    precision_score = contextual_precision.score

    contextual_recall.measure(test_case)
    recall_score = contextual_recall.score

    contextual_relevancy.measure(test_case)
    relevancy_score = contextual_relevancy.score

    answer_relevancy.measure(test_case)
    answer_score = answer_relevancy.score

    faithfulness.measure(test_case)
    faithfulness_score = faithfulness.score

    # 🔹 Insert into RAGFlow_Response first
    RAGFlow_response = supabase.table("RAGFlow_Response").insert({
        "retrieval_id": retrieval_id, 
        "chunks_retrieved": case["retrieval_context"],
        "model_answer": case["actual_output"]
    }).execute()

    # ✅ Ensure retrieval_id exists before inserting into Evaluation_Table
    confirmation = supabase.table("RAGFlow_Response").select("retrieval_id").eq("retrieval_id", retrieval_id).execute()
    if not confirmation.data:
        raise Exception(f"retrieval_id {retrieval_id} was not found in RAGFlow_Response after insertion!")

    # 🔹 Insert into Evaluation_Table
    evaluation_response = supabase.table("Evaluation_Table").insert({
        "contextual_precision": precision_score, 
        "contextual_recall": recall_score,
        "contextual_relevancy": relevancy_score,
        "answer_relevancy": answer_score,
        "faithfulness": faithfulness_score,
        "retrieval_id": retrieval_id,  # This key now exists in RAGFlow_Response
        "ground_truth": case["expected_output"],
        "question": case["input_question"]
    }).execute()

    print(f"✅ Test case '{case['input_question']}' inserted successfully!\n")

test_cases = [
    {
        "input_question": "What constitutes criminal trespass in Singapore?",
        "actual_output": """Whoever commits criminal trespass shall be punished with imprisonment for a term which may extend to 3 months, or with fine which may extend to $1,500, or with both"
                            Essentially, criminal trespass involves intentionally going onto someone else’s property without permission and doing something that violates their rights (like committing a crime).""",
        "expected_output": "Whoever enters into or upon property in the possession of another with intent to commit an offence or to intimidate, insult or annoy any person in possession of such property,or having lawfully entered into or upon such property,unlawfully remains there with intent thereby to intimidate, insult or annoy any such person, or with intent to commit an offence, is said to commitcriminal trespass",
        "retrieval_context": ["""A person may be a party to a criminal conspiracy in Singaporeto commit an offence outside Singapore, which would constitute anoffence if committed in Singapore. [15/2019]Every public servant who, being a citizen or a permanent residentof Singapore, when acting or purporting to act in the course of hisemployment, commits an act or omission outside Singapore that ifcommitted in Singapore would constitute an offence under the law inforce in Singapore, is deemed to have committed that act or omissionin Singapore. and includes an attempt to commit the offence, an abetmentof the offence and a criminal conspiracy to commit theoffence.is entering sufficient to constitute house-breaking.
                                15/2019]""",
                            """443. [Repealed by Act 15 of 2019]444. [Repealed by Act 15 of 2019]445. [Repealed by Act 15 of 2019]446. [Repealed by Act 15 of 2019]
                                Punishment for criminal trespass
                                447. Whoever commits criminal trespass shall be punished withimprisonment for a term which may extend to 3 months, or with finewhich may extend to $1,500, or with both.
                                Informal Consolidation - version in force from 31/1/2024
                                Punishment for house-breaking""",
                            """448. Whoever commits house-breaking shall be guilty of an offenceand shall be punished with imprisonment for a term which mayextend to 3 years, or with fine, or with both.
                                [15/2019]
                                House-breaking in order to commit an offence punishable withdeath"""]
    },
    {
        "input_question": "What is the punishment for furnishing false information to a public servant?",
        "actual_output": """The penalty for furnishing false information to a public servant in Singapore can vary depending on several factors, including:
                            Severity of the False Information: If it's knowingly providing false information that could lead to harm or injustice.
                            Intent: If you intended to deceive and cause damage.
                            Here are some general guidelines for punishment under this section:
                            1. The penalty can range from a fine, up to $5000 (Singapore dollars), to imprisonment of up to six months. This is outlined in Penal Code Section 183(a). 2. In addition, the court may impose additional penalties like fines or community service depending on the circumstances and severity of the offense.""",
        "expected_output": """A person who, being legally bound to furnish
                                information on any subject to any public servant, as such,
                                furnishes, as true, information on the subject which the person
                                knows or has reason to believe to be false, shall —
                                (a) in the case of an individual, be punished with
                                imprisonment for a term which may extend to 6 months,
                                or with fine which may extend to $5,000, or with both; or
                                (b) in any other case, be punished with fine which may extend
                                to $10,000.""",
        "retrieval_context": [
            """"(2) If the notice or information required to be given respects thecommission of an offence, or is required for the purpose of preventingthe commission of an offence or in order to the apprehension of an
                    177.—(l) A person who, being legally bound to furnishinformation on any subject to any public servant, as such,furnishes, as true, information on the subject which the personknows or has reason to believe to be false, shall -(a) in the case of an individual, be punished with""",
            """182. Whoever gives to any public servant any information which heknows or believes to be false, intending thereby to cause, or knowingit to be likely that he will thereby cause, such public servant to use thelawful power of such public servant to the injury or annoyance of anyperson, or to do or omit anything which such public servant ought notto, or would not, do or omit if the true state of facts respecting whichsuch information is given were known by him, shall be punished withimprisonment for a term which may extend to 2 years, or with fine, orWith both.
                    llustrations
                    (a) A informs a superintendent of police that Z, a police officer subordinate tosuch superintendent, has been guilty of neglect of duty or misconduct, knowingsuch information to be false, and knowing it to be likely that the information willcause the superintendent to dismiss Z. A has committed the offence defined inthis section.""",
            """False statement on oath to public servant or person authorisedto administer an oath
                    181. Whoever, being legally bound by an oath to state the truth onany subject to any public servant or other person authorised by law toadminister such oaths, makes to such public servant or other person asaforesaid, touching that subject, any statement which is false, andwhich he either knows or believes to be false or does not believe to betrue, shall be punished with imprisonment for a term which mayextend to 3 years, and shall also be liable to fine.
                    False information, with intent to cause a public servant to usehis lawful power to the injury of another person"""
        ]
    }
]

# Run test cases in parallel
MAX_WORKERS = min(5, len(test_cases))  # Limit to 5 workers or number of test cases
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    executor.map(process_test_case, test_cases)

print("✅ All test cases processed successfully!")



