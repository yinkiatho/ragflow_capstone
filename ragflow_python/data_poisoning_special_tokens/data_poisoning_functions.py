from ragflow_sdk import RAGFlow
from pydantic import BaseModel
import random


# Initialise RAGFlow client
def initialise_ragflow(api_key, base_url):
    rag_object = RAGFlow(api_key=api_key, base_url=base_url)
    return rag_object

# Set up Knowledge Base
def get_dataset(kb_name, rag_object):
    dataset = rag_object.list_datasets(name=kb_name)[0]
    return dataset

# Retrieve chunks from KB
def retrieve_chunks(rag_object, prompt, dataset, page_size=10):
    chunks = rag_object.retrieve(question=prompt, page_size = page_size, dataset_ids=[dataset.id])
    return chunks


# Display top k chunks retrieved
def show_topk_chunks(chunks, k):
    lst_str = []
    for i in range(k):
        item = chunks[i]
        msg = f"Chunk # {i + 1} : \ndocument_id = {item.document_id}, \ncontent = {item.content}\n"
        #print(msg)
        lst_str.append(msg)
    return lst_str

def show_k_chunk_content(chunks, k):
    lst_str = []
    for i in range(k):
        item = chunks[i]
        lst_str.append(item.content)
    return lst_str

# Generate poisoned text
def generate_poisoned_chunk(prompt_question, ground_truth, special_tokens):
    # Get random special token insertion positions
    dct = {}
    for token in special_tokens:
        insert_pos = random.randint(0, len(ground_truth))  # Allow inserting at the end
        dct[insert_pos] = token
    
    poisoned_text = ""
    i = 0
    while i < len(ground_truth) + 1:  # Iterate up to len(ground_truth) to allow end insertions
        if i in dct:
            poisoned_text += dct[i]  # Insert the special token
        if i < len(ground_truth):  # Prevent indexing out of range
            poisoned_text += ground_truth[i]
        i += 1  # Always increment i to avoid infinite loop

    return poisoned_text

# Create a separate document to place the poisoned chunk (not finished)
def generate_attack(prompt, ground_truth, dataset, path, special_tokens, display_name): 
    
    # generate poisoned chunks that contains accurate information but with special tokens
    attack_chunk = generate_poisoned_chunk(prompt, ground_truth, special_tokens)
    print(attack_chunk)

    # add new document to hold poisoned text in KB
    documents =[{"display_name":display_name,"blob":open(path, "rb").read()}]
    docs = dataset.upload_documents(documents)
    doc = docs[0]

    # addition of poisoned text
    doc.add_chunk(content=attack_chunk)
    return attack_chunk

# Inject poisoned data into KB
def perform_data_poisoning(rag_object, dataset, path, prompt, ground_truth, special_tokens, display_name):

    chunks = retrieve_chunks(rag_object, prompt, dataset)
    attack_chunk = generate_attack(prompt, ground_truth, dataset, path, special_tokens, display_name)
    new_chunks = retrieve_chunks(rag_object, prompt, dataset)

    # store clean chunks, new chunks and attack content
    return [chunks, new_chunks, attack_chunk]

