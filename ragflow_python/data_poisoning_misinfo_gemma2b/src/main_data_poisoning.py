import numpy as np
import pandas as pd
from DataPoisoningAttack import DataPoisoningAttack
from PoisonGemma2_2b import PoisonGemma2B
import os
import requests
import time


data = pd.read_json("path/to/json/file")
data = data[["question", "expected answer"]].dropna()


# PARAMS
api_key = "ragflow-Y1Y2NjZjQwZjVlNjExZWZiNTgxMDI0Mm"
base_url = "http://127.0.0.1:9380"
kb_name = "Sample 1" # <-- change to your KB name
k = 10
path = './ragflow_test.txt' # create an empty file in the directory named "ragflow_test.txt" to hold the poisoned chunk and upload to the KB
                            # so that it doesn't get mixed up with other clean documents
display_name = "test_retrieve_chunks.txt"
llm = PoisonGemma2B()
n = 3 # number of tests


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
    print(f"Case {index} / {len(data)}")
    prompt = row["question"]
    ground_truth = row["expected answer"]

    # create attack object
    simulator = DataPoisoningAttack(api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, llm, n, chat_id)
    print("collecting data...")
    #print(simulator.pre_attack_retrieval())
    print("self.k = ", simulator.K)
    simulator.collect_data()
    simulator.save_results_to_json()

    print("sleeping for 10 seconds...")
    time.sleep(10)
    print("sleep finish")

