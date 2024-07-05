import asyncio
import json
import os
import random
import re
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import AsyncInferenceClient
from tqdm import trange, tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer
import numpy as np
from llm_swarm import LLMSwarm, LLMSwarmConfig

MODEL_ID = "Qwen/Qwen2-7B-Instruct"
STEP_SIZE = 10000
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
hf_dataset = load_dataset("lmsys/chatbot_arena_conversations")['train']


def add_response_to_chat(conversation, response):
    response_with_format = [{'role': 'assistant', 'content': response}]
    return np.append(conversation, response_with_format)

def extract_turns_until_last_assistant(conversation):
    assistant_turn_indices = [i for i, turn in enumerate(conversation) if turn['role'] == 'assistant']
    if not assistant_turn_indices:
        return conversation  # Return entire conversation if no 'assistant' turn is found
    last_assistant_index = assistant_turn_indices[-1]
    return conversation[:last_assistant_index]

def apply_add_response_to_chat(row):
    return add_response_to_chat(row['conversation_a'], row['Completion']) 

def choose_winner_conversation(row):
    if row['winner'] == 'model_a':
        return row['conversation_a']
    elif row['winner'] == 'model_b':
        return row['conversation_b']
    return random.choice([row['conversation_a'], row['conversation_b']])


def get_start_index(directory, step_size):
    """
    Determine the start index by checking existing files in the directory.
    """
    files = [f for f in os.listdir(directory) if f.startswith('distil_dataset_') and f.endswith('.h5')]
    if not files:
        return 0

    indices = [int(f.split('_')[2].split('.')[0]) for f in files]
    max_index = max(indices)
    return max_index + step_size


def launch():
    output_directory = "./"  # Directory where the HDF5 files are saved
    start_index = get_start_index(output_directory, STEP_SIZE)
    with LLMSwarm(
        LLMSwarmConfig(
            instances=8,
            inference_engine="vllm",
            gpus=1,
            model=MODEL_ID,
            slurm_template_path="templates/vllm_h100.template.slurm",
            load_balancer_template_path="templates/nginx.template.conf",
            trust_remote_code=True,
            per_instance_max_parallel_requests=200,
        )
    ) as llm_swarm:
        semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
        client = AsyncInferenceClient(model=llm_swarm.endpoint)

        async def process_text(prompt):
            async with semaphore:
                prompt = tokenizer.apply_chat_template(prompt,
                tokenize=False, 
                add_generation_prompt=True
            )

                response = await client.post(
                    json={
                        "prompt": prompt,
                        "max_tokens": 2000,
                    }
                )
                res = json.loads(response.decode("utf-8"))["text"][0][len(prompt):]
                return res

        async def process_dataset(hf_dataset, start_index):
            for idx in trange(start_index, len(hf_dataset), STEP_SIZE, desc="Processing Dataset"):
                batch = hf_dataset[idx:idx + STEP_SIZE]
                winner_conversations = pd.DataFrame(batch).apply(choose_winner_conversation, axis=1)

                tasks = [extract_turns_until_last_assistant(example) for example in winner_conversations]

                results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))

                completions = [add_response_to_chat(conversation, response) for conversation, response in zip(tasks, results)]
                
                df = pd.DataFrame({"question_id": batch['question_id'], "messages": completions})

                # Save the batch to HDF5
                df.to_hdf(f'distil_dataset_{idx}.h5', key='df', mode='w')


        async def main():
            if start_index != 0:
                print(f"Resuming from tar file {start_index}")

            processor = asyncio.create_task(process_dataset(hf_dataset, start_index))
            await processor

            print("All batches processed.")

            directory = '.'
            all_files = os.listdir(directory)
            h5_files = sorted([f for f in all_files if re.match(r'distil_dataset_\d+\.h5$', f)])
            
            dataframes = []
            for file in tqdm(h5_files, desc="Loading data"):
                 file_path = os.path.join(directory, file)
                 df = pd.read_hdf(file_path)
                 dataframes.append(df)

            concatenated_df = pd.concat(dataframes, ignore_index=True)
            deduplicate_concatenated_df =  concatenated_df.drop_duplicates(subset='question_id', keep='first')
            
            output_hf_dataset = Dataset.from_pandas(deduplicate_concatenated_df, preserve_index=False)
            output_hf_dataset.push_to_hub('andito/chatbot_arena_completions')

        asyncio.run(main())

launch()
