import numpy as np
import pandas as pd
from DataPoisoningAttack import DataPoisoningAttack
from PoisonGemma2_2b import PoisonGemma2B
import os
import requests


data = pd.read_csv("path/to/json", encoding="utf-8")
data = data[["Question", "Corrected Answers"]].dropna()
# data = data[:1] # for smaller sampling

# PARAMS
api_key = 'YOUR API KEY'
base_url = "YOUR BASE URL"
kb_name = "Singapore Criminal Law" # <-- change to your KB name
k = 5
path = './ragflow_test.txt' # create an empty file in the directory named "ragflow_test.txt" to hold the poisoned chunk and upload to the KB
                            # so that it doesn't get mixed up with other clean documents
display_name = "test_retrieve_chunks.txt"
llm = PoisonGemma2B()
n = 3 # number of tests
# chat_id = ""


def create_new_chat():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Create a new chat
    create_chat_url = f"{base_url}/api/chats"
    response = requests.post(create_chat_url, headers=headers)

    if response.status_code == 200:
        chat_data = response.json()
        return chat_data.get("id")  # Return the new chat ID
    else:
        print(f"Failed to create chat: {response.status_code}")
        print(response.text)
        return None

chat_id = create_new_chat()

# ITERATION
for index, row in data.iterrows():
    prompt = row['Question']
    ground_truth = row["Corrected Answers"]
    # create attack object
    simulator = DataPoisoningAttack(api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, llm, n, chat_id)
    print("collecting data...")
    #print(simulator.pre_attack_retrieval())
    print("self.k = ", simulator.K)
    simulator.collect_data()
    simulator.save_results_to_json()
