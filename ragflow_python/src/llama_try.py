from ragflow_sdk import RAGFlow
import os
from dotenv import load_dotenv

load_dotenv()
rag_object = RAGFlow(api_key=os.getenv('RAGFLOW_API_KEY'), base_url="http://127.0.0.1:9380")
dataset = rag_object.list_datasets()
dataset = (dataset[0])
assistant = rag_object.list_chats()
assistant = assistant[0]
session = assistant.list_sessions()
session = session [0]
questions= ['What is the legal age of criminal responsibility in Singapore?',
            'What are the penalties for theft under Singapore law?',
            'Is self-defense a valid legal defense in Singapore?',
            'What is the penalty for individuals found in possession of subversive documents?',
            'How long can a person be detained without being brought before a court?',
            'What constitutes criminal trespass in Singapore?',
            'What is the punishment for furnishing false information to a public servant?',
            'What constitutes defamation under the Penal Code?',
            'What are the different types of witness examinations in court?',
            'Can a witness refuse to answer a question that might incriminate them?',
            'Who has the power to refer any question of law of public interest without the permission of the Court of Appeal?',
            'Under what conditions can a court proceed in the absence of an accused?',
            'What constitutes assault under Singapore law?',
            'What is the penalty for illegal strikes and lock-outs under the Criminal Law (Temporary Provisions) Act 1955?',
            'What is the punishment for making or possessing a subversive document under the Criminal Law (Temporary Provisions) Act 1955?',
            'What constitutes an unlawful assembly under the Penal Code 1871?',
            'What is criminal breach of trust in Singapore?',
            'Who is ineligible for a community order under the Criminal Procedure Code 2010?',
            'Under what circumstances does the right of private defence of property extend to causing death?']
            
for i, question in enumerate(questions, start=1):
    print(f"Question {i}: {question}")
    response = rag_object.retrieve(question=question, dataset_ids=[dataset.id])
    for j, res in enumerate(response[:3], start=1):  # Get top 3 results with numbering
        print(f"🔹 Top {j} for Question {i}:\n{res.content}\n{'-'*40}")

cont = ""
for i, question in enumerate(questions, start=1):
    print(f"Question {i}: {question}")
    try:
        for ans in session.ask(question=question, stream=True):
            print(ans.content)
    except KeyError as e:
        print(f"Error: {e}")
    print('-' * 40)  # Separator between questions