import numpy as np
import pandas as pd
from DataPoisoningAttack import DataPoisoningAttack
from PoisonGemma2_2b import PoisonGemma2B


data = pd.read_csv("qa.csv")
data = data[["Question", "Corrected Answers"]].dropna()
# data = data[:1] # for smaller sampling

# PARAMS
api_key = "YOUR API KEY"
base_url = "http://127.0.0.1:9380"
kb_name = "Sample 1" # <-- change to your KB name
k = 5
path = './ragflow_test.txt'
display_name = "test_retrieve_chunks.txt"
llm = PoisonGemma2B()
n = 3 # number of tests
chat_id = "70563e72f38f11efb4780242ac110002"


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
