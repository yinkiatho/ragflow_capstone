from data_poisoning_functions import *
import json
import time
import numpy as np
import pyplexity

class DataPoisoningAttack:
    def __init__(self, api_key, base_url, kb_name, prompt, ground_truth, k, path, display_name, special_tokens, n, chat_id, perp_threshold, perpl_model):
        self.API_KEY = api_key
        self.BASE_URL = base_url
        self.KB_NAME = kb_name
        self.PROMPT = prompt
        self.GROUND_TRUTH = ground_truth
        self.K = k
        self.PATH = path
        self.DISPLAY_NAME = display_name
        self.SPECIAL_TOKENS = special_tokens
        self.N = n
        self.CHAT_ID = chat_id
        self.THRESHOLD = perp_threshold
        self.POISONED_CHUNKS = []
        self.CLEAN_CHUNKS = []
        self.POST_ATT_CHUNKS = []
        self.PRE_DEF_CHUNKS = []
        self.POST_DEF_CHUNKS = []
        self.RAG_OBJECT = initialise_ragflow(api_key, base_url)
        self.DATASET = get_dataset(kb_name, self.RAG_OBJECT)
        self.RESULTS = [] # [preattack results, postattack results, postdefense results]
        self.IS_SUCCESSFUL_ATTACK = False
        self.IS_CLEAN = True
        self.PERPLEXITY_MODEL = perpl_model
        

    def pre_attack_retrieval(self):
        #print("self dataset = ", self.DATASET)
        chunks = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET)
        self.CLEAN_CHUNKS = chunks
        return chunks
    
    def make_original_chunks_unavailable(self):
        chunks = self.CLEAN_CHUNKS
        for chunk in chunks:
            chunk.update({"available": False})

    def recover_unavailable(self):
        chunks = self.CLEAN_CHUNKS
        for chunk in chunks:
            chunk.update({"available": True})


    '''
    DEFENSE SCRIPTS
    '''
    def mask_poisoned_chunks(self): # chunks = chunks identified as poisoned
        chunks = self.PRE_DEF_CHUNKS
        for chunk in chunks:
            chunk.update({"available": False})
    
    def recover_poisoned_chunks(self): # in case some clean chunks got masked by accident too
        chunks = self.PRE_DEF_CHUNKS
        for chunk in chunks:
            chunk.update({"available": True})

    def post_defense_retrieval(self, threshold):
        #chunks = self.POST_ATT_CHUNKS
        # chunks = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET, 50)
        chunks = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET, self.K)
        chunks_to_mask = []

        # identify chunks with content perpl > threshold
        for chunk in chunks:
            if calculate_perpl(chunk.content, threshold, self.PERPLEXITY_MODEL):
                print("chunk masked")
                chunks_to_mask.append(chunk)
        self.PRE_DEF_CHUNKS = chunks_to_mask

        # mask these pre-defense chunks
        self.mask_poisoned_chunks()

        # retrieve chunks again and store post-def chunks
        self.POST_DEF_CHUNKS = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET)

        return self.POST_DEF_CHUNKS
        
    '''
    DEFENSE END
    '''
        
    def post_attack_retrieval(self):

        # get clean chunks
        chunks = self.CLEAN_CHUNKS
        assert len(chunks) > 0, "Error: chunks list is empty!"

        # generate poisoned chunks and upload them
        poisoned_chunks = generate_poisoned_chunk(chunks, self.SPECIAL_TOKENS)
        generate_attack(poisoned_chunks, self.DATASET, self.PATH, self.DISPLAY_NAME)
        
        # mask the clean chunks
        self.make_original_chunks_unavailable()

        # get post att chunks
        new_chunks = retrieve_chunks(self.RAG_OBJECT, self.PROMPT, self.DATASET)

        self.POISONED_CHUNKS = poisoned_chunks
        self.POST_ATT_CHUNKS = new_chunks
        return new_chunks
    
    def get_poisoned_docs(self):
        keyword = self.DISPLAY_NAME.replace(".txt", "")
        poisoned_docs = self.DATASET.list_documents(keywords=keyword)
        lst = []
        for poisoned_doc in poisoned_docs:
            lst.append(poisoned_doc.id)
        return lst
    
    # count the post-attack percentage of poisoned chunks retrieved / total chunks retrieved
    def count_poisoned_chunks_percentage(self):
        chunks = self.POST_ATT_CHUNKS
        lst = self.get_poisoned_docs()
        counter = 0
        for chunk in chunks:
            if chunk.document_id in lst:
                counter += 1
        return counter / len(chunks)
    

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
        # Pre-attack activities
        pre_attack_chunks = self.pre_attack_retrieval()
        print("pre-attack retrieval completed.")
        # pre_attack_llm_ans = self.conduct_chat_session()


        # Conduct attack and conduct Post-attack activities
        post_attack_chunks = self.post_attack_retrieval()
        print("attack completed")
        post_attack_llm_ans = self.conduct_chat_session()

        # Conduct defense
        threshold = self.THRESHOLD
        post_defense_chunks = self.post_defense_retrieval(threshold)
        print("defense completed")
        post_defense_llm_ans = self.conduct_chat_session()

        
        # Collect results
        pre_results = [self.PROMPT, self.GROUND_TRUTH, list(map(lambda x: x.content, pre_attack_chunks)), "(assuming pre-attack is not affected by special tokens)"]
        post_results = [self.PROMPT, self.GROUND_TRUTH, list(map(lambda x: x.content, post_attack_chunks)), post_attack_llm_ans]
        post_def_results = [self.PROMPT, self.GROUND_TRUTH, list(map(lambda x: x.content, post_defense_chunks)), post_defense_llm_ans]
        self.RESULTS = [pre_results, post_results, post_def_results]
        print("results collected")


        # see if post_attack_llm_ans contains special tokens
        for token in self.SPECIAL_TOKENS:
            print("token =", token)
            if token in post_attack_llm_ans:
                print("yes!")
                self.IS_SUCCESSFUL_ATTACK = True


        # see if post-defense llm ans contains special tokens
        for token in self.SPECIAL_TOKENS:
            print("token = ", token)
            if token in post_defense_llm_ans:
                print("defense failed, token still in.")
                self.IS_CLEAN = False

        # Recover pre-defense chunks
        self.recover_poisoned_chunks()
        print("pre-defense chunks recovered")

        # Recover clean chunks
        self.recover_unavailable()
        print("clean chunks recovered")

        # Delete poisoned chunks
        self.delete_poisoned_chunks()
        print("poisoned chunks deleted!")

    def save_results_to_json(self):
        timestamp2 = time.strftime("%Y%m%d_%H%M%S")
        filename = f"combined_results_at {timestamp2}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"pre_attack": self.RESULTS[0], 
                       "post_attack": self.RESULTS[1],
                       "post_defense": self.RESULTS[2]}, f, indent=4)
    
