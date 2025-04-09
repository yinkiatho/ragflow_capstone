from ragflow_sdk import RAGFlow
from pydantic import BaseModel


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
def generate_poisoned_chunk(prompt_question, ground_truth, model):
       
    # Initialize the model
    gemma_model = model
    class FakeLawResponse(BaseModel):
        content: str

    # Generate poisoned statement
    poisoned_response = gemma_model.generate(prompt_question, ground_truth, FakeLawResponse)

    return poisoned_response.content

# Create a separate document to place the poisoned chunk (not finished)
def generate_attack(prompt, ground_truth, dataset, path, model, display_name): 
    # note: ideally should use LLM to generate the attack_chunk from ground truth
    
    #attack_chunk = "Only the Chief Justice may refer any question of law of public interest without the permission of the Court of Appeal" # changed Public Prosecutor to Cheif Justice
    attack_chunk = generate_poisoned_chunk(prompt, ground_truth, model)
    print(attack_chunk)

    # add new document to hold poisoned text in KB
    documents =[{"display_name":display_name,"blob":open(path, "rb").read()}]
    docs = dataset.upload_documents(documents)
    doc = docs[0]

    # addition of poisoned text
    doc.add_chunk(content=attack_chunk)
    return attack_chunk

# Inject poisoned data into KB
def perform_data_poisoning(rag_object, dataset, path, prompt, ground_truth, llm, display_name):
    #rag_object = initialise_ragflow(api_key, base_url)
    #dataset = get_dataset(kb_name, rag_object)
    chunks = retrieve_chunks(rag_object, prompt, dataset)
    attack_chunk = generate_attack(prompt, ground_truth, dataset, path, llm, display_name)
    new_chunks = retrieve_chunks(rag_object, prompt, dataset)

    # store clean chunks, new chunks and attack content
    return [chunks, new_chunks, attack_chunk]

# Test if poisoned chunk is in retrieved chunk list
def compare_w_chunks_retrieved(poisoned_chunks, chunks):
    lst_str = []
    for poisoned_chunk in poisoned_chunks:
        for chunk in chunks:
            content = chunk.content
            lst_str.append(content)
            
            if poisoned_chunk in lst_str:
                return True
    
    return False

