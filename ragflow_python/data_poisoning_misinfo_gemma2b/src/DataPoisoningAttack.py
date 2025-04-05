import os

from data_poisoning_functions import *
from PoisonGemma2_2b import PoisonGemma2B
import json
import time

class DataPoisoningAttack:
    def __init__(self, api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, llm, n, chat_id):
        self.API_KEY = api_key
        self.BASE_URL = base_url
        self.KB_NAME = kb_name
        self.PROMPT = prompt
        self.GROUND_TRUTH = ground_truth
        self.K = k
        self.PATH = path
        self.DISPLAY_NAME = display_name
        self.LLM = llm
        self.N = n
        self.CHAT_ID = chat_id
        self.POISONED_CHUNKS = []
        self.FIRST_SUCC_ATTACK_ITER = 0
        self.RAG_OBJECT = initialise_ragflow(api_key, base_url)
        self.DATASET = get_dataset(kb_name, self.RAG_OBJECT)
        self.RESULTS = [] # [preattack results, postattack results]

    def pre_attack_retrieval(self):
        #print("self dataset = ", self.DATASET)
        chunks = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET)
        return show_k_chunk_content(chunks, self.K)
    
    def post_attack_retrieval(self):
        for i in range(self.N): # num of attacks
            result = perform_data_poisoning(self.RAG_OBJECT,
                                            self.DATASET,
                                            self.PATH, 
                                            self.PROMPT, 
                                            self.GROUND_TRUTH,
                                            self.LLM,
                                            self.DISPLAY_NAME)
            chunks = result[0] # clean chunks
            new_chunks = result[1]
            attack_chunk = result[2] # dirty chunks
            self.POISONED_CHUNKS.append(attack_chunk)

            # Compare whether poisoned chunk is in retrieved chunks
            chunk_in_lst = compare_w_chunks_retrieved(self.POISONED_CHUNKS, new_chunks[: self.K + 1])

            # Generate timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"result_at_{timestamp}.txt"
            dir = "../data_new_qa_pairs" 
            file_path = os.path.join(dir, filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write results into txt file
            with open(file_path, "w") as f:
                f.write("------------------------- RESULTS -------------------------\n")
                f.write(f"Poisoned Chunks = {self.POISONED_CHUNKS}\n")
                f.write(f"Prompt Question = {self.PROMPT}\n")
                f.write(f"Ground Truth = {self.GROUND_TRUTH}\n\n")
                f.write(f"Poisoned chunk in lst: {chunk_in_lst}\n\n")

                f.write("Clean Retrieval:\n")
                f.write("\n".join(show_topk_chunks(chunks, self.K)))  # Assuming function returns list of strings
                f.write("\n\n")

                f.write("Attacked Retrieval:\n")
                f.write("\n".join(show_topk_chunks(new_chunks, self.K)))

            print(f"Results saved to {filename}")

            '''
            if chunk_in_lst:
                self.FIRST_SUCC_ATTACK_ITER = i + 1
                break
            '''

        return show_k_chunk_content(new_chunks, self.K)
    
    def conduct_chat_session(self):
        assistant = self.RAG_OBJECT.list_chats(id=self.CHAT_ID)
        assistant = assistant[0]
        timestamp2 = time.strftime("%Y%m%d_%H%M%S")
        session = assistant.create_session(f"Chat at {timestamp2}")

        while True:
            print("Session Started...")
            question = self.PROMPT  
            cont = ""
            for ans in session.ask(question, stream=True):
                print(ans.content[len(cont):], end='', flush=True)
                cont = ans.content
            break

        return cont
    
    def delete_poisoned_chunks(self):
        print("identifying poisoned chunks...")
        ids = []
        dataset = self.DATASET
        lst = dataset.list_documents()
        poisoned_chunk_name = self.DISPLAY_NAME.replace(".txt", "")
        filtered_list = list(filter(lambda x: poisoned_chunk_name in x.name, lst))
        print(filtered_list)
        for item in filtered_list:
            ids.append(item.id)
        
        print("deleting poisoned chunks...")
        self.DATASET.delete_documents(ids=ids)

    
    def collect_data(self):
        pre_attack_chunks = self.pre_attack_retrieval()
        print("...")
        pre_attack_llm_ans = self.conduct_chat_session()
        post_attack_chunks = self.post_attack_retrieval()
        post_attack_llm_ans = self.conduct_chat_session()
        pre_results = [self.PROMPT, self.GROUND_TRUTH, pre_attack_chunks, pre_attack_llm_ans]
        post_results = [self.PROMPT, self.GROUND_TRUTH, post_attack_chunks, post_attack_llm_ans]
        self.RESULTS = [pre_results, post_results]
        self.delete_poisoned_chunks()



    def save_results_to_json(self):
        timestamp2 = time.strftime("%Y%m%d_%H%M%S")
        filename = f"combined_results_at {timestamp2}.json"
        dir = "../data_new_qa_pairs" # change to new directory to store the new data
        file_path = os.path.join(dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"targeted_wrong_answers": self.POISONED_CHUNKS, "pre_attack": self.RESULTS[0], "post_attack": self.RESULTS[1]}, f, indent=4)
            # added collection of poisoned chunks which are target wrong answers to calculate ASR for misinformation attacks
    

    

