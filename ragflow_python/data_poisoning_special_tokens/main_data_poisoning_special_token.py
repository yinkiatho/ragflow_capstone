import numpy as np
import pandas as pd
from DataPoisoningAttack import DataPoisoningAttack
import requests
import os
from dotenv import load_dotenv

load_dotenv()
data = pd.read_csv("qa.csv")
data = data[["Question", "Corrected Answers"]].dropna()

api_key = os.getenv('RAGFLOW_API_KEY')
base_url = "http://localhost:9380"

#api_key = "ragflow-Y1Y2NjZjQwZjVlNjExZWZiNTgxMDI0Mm"
#base_url = "http://127.0.0.1:9380"
kb_name = "Singapore Criminal Law" # <-- change to your KB name
k = 5
path = './ragflow_test.txt'
display_name = "test_retrieve_chunks.txt"
n = 5 # number of tests
special_tokens = [
    "*!@$!!%$", "#%&(%#$@)>"
]

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
counter = 0

# ITERATION
for index, row in data.iterrows():
    prompt = row['Question']
    ground_truth = row["Corrected Answers"]
    simulator = DataPoisoningAttack(api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, special_tokens, n, chat_id)
    simulator.collect_data()
    print(f"is_successful_attack = {simulator.IS_SUCCESSFUL_ATTACK}")
    if simulator.IS_SUCCESSFUL_ATTACK:
        counter += 1
    simulator.save_results_to_json()

print(f"number of successful attacks = {counter}")
print(f"ASR = {counter / len(data)}")

