from supabase import create_client
import os
import random

used_ids = set()  # Store previously generated IDs
url = "https://bggngaqkkmslamsbebew.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJnZ25nYXFra21zbGFtc2JlYmV3Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDA0MDgxNzIsImV4cCI6MjA1NTk4NDE3Mn0.B4D-5t0oxa8D6xMSoywufdB7aSmGy1s8bvytH0znows"
supabase = create_client(url, key)


def generate_unique_id():
    while True:
        new_id = random.randint(1, 10000)
        if new_id not in used_ids:  
            used_ids.add(new_id)
            return new_id

def create_new_experiment(attack_name):
    
    # create attack type
    attack_id = generate_unique_id()
    attack_response = (
        supabase.table("Attack_Type")
        .insert({"attack_id": attack_id, "attack_name": attack_name})
        .execute()
    )

    # create experiment type
    ex_id = generate_unique_id()
    ex_response = (
        supabase.table("Experiment_Type")
        .insert({"type_id": ex_id, "attack_type": attack_id})
        .execute()
    )

    return [attack_response, ex_response]

# create data poisoning attack type (done alr)
result = create_new_experiment("data-poisoning-pre-attack")
result = create_new_experiment("data-poisoning-post-attack")
print(result)
