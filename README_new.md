# RAGFlow Container

Docker container setup for [RAGFlow](https://github.com/infiniflow/ragflow), a powerful Retrieval-Augmented Generation (RAG) framework.

## Overview

This repository contains an end-to-end attack and defense implementations on a RAG system, based on the RAGFlow system on https://github.com/infiniflow/ragflow. 

## Features

- RAGFlow system setup 
- Baseline RAG system evaluations using .......
- Generation Attacks using DeepEval, Ollama locally-hosted LLMs and Guardrails AI for Generation Attack Defenses
- Data Poisoning Attacks using .......
- Uploading of data to Supabase

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- Packages detailed in requirements.txt


## Directory Structure

```
.
├── .env                                            # Environment variables configuration
├── main.py                                         # Starting point to run the Generation Attack scripts
├── ragflow_python/                                 # Main folder containing source codes
│   ├── baseline_model/                             # Main folder containing source codes for evaluating baseline model
│   ├── data/                                       # Locally saved results for Generation Attack scripts
│   ├── data_poisoning_misinfo_gemma2b/             # Main folder containig source codes for Data Poisoning misinformation attack
|   ├── data_poisoning_special_tokens/              # Main folder containig source codes for Data Poisoning special tokens attack and defense
|   ├── documents/                                  # PDF files used in our Knowledge Base
|   ├── src/                                        # RAG and Local LLM wrappers
|   ├── testing_bot/                                # Attack/Defense scripts
|   ├── utils/                                      # Utility functions and variables
|   ├── evaluation.ipynb                            # Jupyter Notebook for evaluating attack results
├── requirements.txt                                # Txt file for installing packages
└── README.md                                       # This documentation
```

└──

## Initialization of RAGFlow 
- Loading of environment variables
- Setting up of Docker Container
- Chunking of knowledge base

1. Initialize the workspace and configuration:
   - Load the Docker Container from 


2. Edit configuration files as needed:
   - `.env`: Environment variables, refer to the .env.example for list of environment keys required


3. Start the services:



## Data Poisoning Attacks
After installing all the necessary package, we need to run the following commands in the terminal:  

### Data Poisoning with Special Tokens
To collect results for attack and defense, run the following command in ~/ragflow_python/data_poisoning_special_tokens:  
```bash
python main_data_poisoning_special_token.py
```
This repeats the attack and defense experiments on all QA Pairs and stores the results in a `.json` file for each QA. The following special tokens are inserted into the top 10 most relevant chunks:
![dp_ss_5](ragflow_python/documents/dp_ss_5.png)

The code for the main loop is demonstrated below:  
![dp_ss_1](ragflow_python/documents/dp_ss_1.png)

The code for each individual experiment is outlined by the following code in `DataPoisoningAttack.py`:
![dp_ss_2](ragflow_python/documents/dp_ss_2.png)
To collect results for ASR, DSR and Poison Rates, go to `collect_metrics.ipynb` and run all cells to save results into `.csv` format and save summary into `.txt` format.

To run DeepEval results and upload them to supabase, run:
```bash
deepeval login
```
Optionally, you can set ur default model (gemma2:2b for eg.) in Deepeval via
```bash
deepeval set-ollama gemma2:2b
```
```bash
python testing_data_poisoning.py
```


### Data Poisoning with Misinformation attacks
To collect results for attack and defense, run the following command in ~/ragflow_python/data_poisoning_misinfo_gemma2b/src:  
```bash
main_data_poisoning.py
```
The code for generating poisoned chunks that carries misinformation is demonstrated below:  
![dp_ss_6](ragflow_python/documents/dp_ss_6.png)

The code for the main loop is demonstrated below:  
![dp_ss_3](ragflow_python/documents/dp_ss_3.png)

The code for each individual experiment is outlined by the following code in `DataPoisoningAttack.py`:
![dp_ss_4](ragflow_python/documents/dp_ss_4.png)

To collect results for ASR and Poison Rates, run:
```bash
python collect_asr.py
```
This will save results into `.csv` format and save summary into `.txt` format in the data results.

## 🚀 Generation Attacks 

### Loading of required libraries: Deepeval
After installing all the necessary packages, we need to run these commands in the terminal to login to their backend 
```bash
deepeval login
```

Optionally, you can set ur default model (gemma2:2b for eg.) in Deepeval via
```bash
deepeval set-ollama gemma2:2b
```

### Loading of required libraries: Guardrails-AI
Similarly for Guardrails-AI, run: 
```bash
pip install guardrails-ai

guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/qa_relevance_llm_eval
guardrails hub install hub://guardrails/llamaguard_7b
```

Depending on the other sub-guardrails used in the guardrails object, more install commands may be needed. Set remote-inferencing = True during the installation, you can manually set it via: 

```bash
guardrails configure --enable-remote-inferencing
```

### Running Generation Attacks
In order to run generation attacks, simply run main.py, do run the command whilst being in the current working directory of ~/ragflow_python so that the results would be saved in the correct directories.

```bash
python3 main.py
```

## main.py
- run_test function is imported from the testing_goldens_mak.py file which contains the main logic for executing attacks
- There are 3 parameters available to run run_test() with, activate_defense=True activates the guardrails whilst fetch_chunks=False and generate_attacks=False utilises locally saved attacks and saved chunks previously generated/fetched. 

![main.py](ragflow_python/documents/main_py.png)

## testing_goldens_mak.py
The main attack script can be divided into 3 sub-portions:

1. Defining target vulnerabilities and attack enhancements 
   - Do note that different types of attack enhancements will require different amounts of computational power
   ![goldens_1](ragflow_python/documents/goldens_1.png)

2. Parameterization of the RedTeamer
   - Input the target_purpose and the target_system_prompt to the RedTeamer for it to generate attacks and evaluate LLM responses
   - Define the RagFlowTester wrapper over our RAG object to utilise the target_model_callback functions for use within RedTeamer
   ![goldens_2](ragflow_python/documents/goldens_2.png)

3. Calculation of Scores
   - Attack scores and ASR are then calculated
   - Input and outputs are used to run our standard 5 RAG Evaluation Metrics
   ![goldens_3](ragflow_python/documents/goldens_3.png)



## License

This project is licensed under the MIT License. RAGFlow itself is licensed under the Apache License 2.0.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
