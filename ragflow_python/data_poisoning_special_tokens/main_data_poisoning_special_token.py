import numpy as np
import pandas as pd
from DataPoisoningAttack import DataPoisoningAttack
import requests
from pyplexity import PerplexityModel, PerplexityProcessor
import time


data = pd.read_csv("/Users/Olivia/Desktop/NUS/SoC/Y3S2/BT4103_Capstone/api-testing/qa.csv")
data = data[["Question", "Corrected Answers"]].dropna()



api_key = "ragflow-Y1Y2NjZjQwZjVlNjExZWZiNTgxMDI0Mm"
base_url = "http://127.0.0.1:9380"
kb_name = "Sample 1" # <-- change to your KB name
k = 5
path = './ragflow_test.txt'
display_name = "test_retrieve_chunks.txt"
n = 3 # number of tests
special_tokens = [
    "d3f4ult", "v4l1d4t3", "xylophonic", "quintessate" # Hex/leet pattern  
]

'''
special_tokens = [
    "*!@$!!%$", "#%&(%#$@)>", "#@%!"
]

special_tokens = [
    "xylophonic", "quintessate", "zelambient", 
    "floramatic", "novalunite"  # Fake but word-like
]
'''
threshold = 500
model = PerplexityModel.from_str("bigrams-cord19")

chat_id = "70563e72f38f11efb4780242ac110002"
succ_attacks = 0
post_def_dirty = 0
poisoned_rate = []

# ITERATION
for index, row in data.iterrows():
    print(f"Case # {index + 1} / 26")
    prompt = row['Question']
    ground_truth = row["Corrected Answers"]
    simulator = DataPoisoningAttack(api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, special_tokens, n, chat_id, threshold, model)
    simulator.collect_data()
    simulator.save_results_to_json()
    print(f"is_successful_attack = {simulator.IS_SUCCESSFUL_ATTACK}")
    if simulator.IS_SUCCESSFUL_ATTACK:
        succ_attacks += 1
    poisoned_rate.append(simulator.count_poisoned_chunks_percentage())

    print(f"post-def ans is still clean = {simulator.IS_CLEAN}")
    if not simulator.IS_CLEAN:
        post_def_dirty += 1

    print("Sleeping for 10 seconds...")
    time.sleep(10) # sleep for 10 seconds before running the next round
    print("Sleep finish :D")
    

print(f"number of successful attacks = {succ_attacks}")
print(f"number of answers that are still dirty after defense = {post_def_dirty}")