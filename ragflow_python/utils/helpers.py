from deepeval.evaluate import EvaluationResult, TestResult, MetricData
import random

def evaluation_result_to_json(result: EvaluationResult):
    results_metrics_json = result.model_dump()
    return results_metrics_json



DOCUMENT_PAGES = {
    'Criminal Procedure Code 2010.pdf': 573,
    'Penal Code 1871.pdf': 313,
    'Evidence Act 1893-2.pdf': 128,
    'Criminal Law (Temporary Provisions) Act 1955.pdf': 53
}

def load_all_chunks():
    chunks = []

    # Load all the chunks
    rag_object = RAGFlow(api_key=rag_flow_api_key, base_url=f"http://localhost:9380")
    dataset = rag_object.list_datasets(0)
    
    all_chunks = []
    for ds in dataset:
        docs = ds.list_documents(page=1, page_size=500)
        for doc in docs:
            print(doc.name)
            for page_num in range(3, DOCUMENT_PAGES.get(doc.name)):
                #print(page_num)
                for chunk in doc.list_chunks(page=page_num, page_size=10):
                    chunk_json = chunk.to_json()
                    all_chunks.append(chunk_json)
    
    # Save all chunks as JSON
    with open("chunks_data.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)

    print("JSON file saved successfully as chunks_data.json")
    
    
def generate_unique_id(supabase, table_name, end_range=10000):
    """Generate a unique retrieval_id that does not exist in RAGFlow_Response"""
    while True:
        new_id = random.randint(1, end_range)
        response = supabase.table(table_name).select("retrieval_id").eq("retrieval_id", new_id).execute()
        
        if not response.data:  # If no existing retrieval_id found, it's unique
            return new_id