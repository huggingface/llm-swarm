import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import multiprocessing
import os
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
import time
from huggingface_hub import HfApi
api = HfApi()

CHUNK_SIZE = 50000  # Define your chunk size here

@dataclass
class Args:
    max_samples: int = -1
    """The maximum umber of samples to generate (use -1 for all))"""
    max_new_tokens: int = 4000
    """Max new tokens"""
    temperature: float = 0.5
    """Generation temperature"""
    do_sample: bool = True
    """Whether to sample"""
    repo_id: str = "openhermes-dev"
    """The repo id to push to"""
    timestamp: bool = True
    """Whether to add a timestamp to the repo_id"""
    push_to_hub: bool = False
    """Whether to push to hub"""
    test_split_percentage: float = 0.05
    """The percentage of the dataset to use for testing"""
    debug: bool = False
    """Debug mode"""
    max_samples_per_source_category: int = 2
    """The maximum number of samples per source"""
    restart_chunk_index: int = 0
    """The index of the chunk to restart from"""

parser = HfArgumentParser([Args, LLMSwarmConfig])
args, isc = parser.parse_args_into_dataclasses()
if args.timestamp:
    args.repo_id += f"__{isc.model.replace('/', '_')}__{str(int(time.time()))}"
if "/" not in args.repo_id:  # find the current user
    args.repo_id = f"{api.whoami()['name']}/{args.repo_id}"

tokenizer = AutoTokenizer.from_pretrained(isc.model, revision=isc.revision)
ds = load_dataset('teknium/OpenHermes-2.5', split="train")

if args.max_samples_per_source_category > 0:
    count = defaultdict(int)
    def filter_unique(row):
        if count[f'{row["source"]}_{row["category"]}'] < args.max_samples_per_source_category:
            count[f'{row["source"]}_{row["category"]}'] += 1
            return True
        return False
    ds = ds.filter(filter_unique)
    print(ds.to_pandas()["source"].value_counts())
if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))

def extract(row):
    sample = {}
    conversations = row["conversations"]
    if conversations[0]["from"] == "system":
        conversations[1]["value"] = conversations[0]["value"] + " " + conversations[1]["value"]
        conversations = conversations[1:] # merge the first two
    sample["prompt"] = conversations[0]["value"]
    sample["chosen_policy"] = conversations[1]["from"]
    sample["chosen"] = []
    for i, conv in enumerate(conversations):
        if i % 2 == 0:
            sample["chosen"].append({"role": "user", "content": conv["value"]})
        else:
            sample["chosen"].append({"role": "assistant", "content": conv["value"]})
    tokens = tokenizer.apply_chat_template(sample["chosen"])
    sample["token_length"] = len(tokens)
    return sample

ds = ds.map(extract, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
print("max token length", ds.to_pandas()["token_length"].max())
print("mean token length", ds.to_pandas()["token_length"].mean())
print("median token length", ds.to_pandas()["token_length"].median())
# ds = ds.filter(lambda x: x["token_length"] < args.max_token_length, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    print(f"{llm_swarm.suggested_max_parallel_requests=}")
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    MAX_RETRIES = 3  # maximum number of retries
    RETRY_DELAY = 5  # delay in seconds between retries

    async def process_text(row):
        attempt = 0
        prompt = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
        while attempt < MAX_RETRIES:
            try:
                async with semaphore:
                    completion = await client.text_generation(
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                    )
                    row["rejected"] = row["chosen"][:-1] + [{"role": "assistant", "content": completion}]
                    row["rejected_policy"] = ":".join([isc.model, isc.revision])
                    return row
            except Exception as e: 
                attempt += 1
                if attempt < MAX_RETRIES:
                    print(
                        f"Request failed, retrying in {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(
                        f"Max retries reached. Failed to process the request with error {str(e)}."
                    )
                    row["rejected"] = ""
                    row["rejected_policy"] = ""
                    return row

    async def main():
        os.makedirs("chunks_cache", exist_ok=True)
        results = []
        num_chunks = len(ds) // CHUNK_SIZE
        restart_idx = 0
        if args.restart_chunk_index > 0:
            post_ds = Dataset.load_from_disk(f"chunks_cache/cache_chunk{args.restart_chunk_index}.arrow")
            results = post_ds.to_list()
            restart_idx = (args.restart_chunk_index + 1) * CHUNK_SIZE

        for i in range(restart_idx, len(ds), CHUNK_SIZE):
            chunk_idx = i // CHUNK_SIZE + 1
            print(f"Processing chunk {chunk_idx}/{num_chunks}")
            start_time = time.time()
            chunk = ds.select(range(i, min(i + CHUNK_SIZE, len(ds))))
            chunk_results = await tqdm_asyncio.gather(*[process_text(row) for row in chunk])
            results.extend(chunk_results)
            print(f"Chunk {chunk_idx}/{num_chunks} took {time.time() - start_time} seconds")
            post_ds = Dataset.from_list(results)
            post_ds.save_to_disk(f"chunks_cache/cache_chunk{chunk_idx}.arrow")
            # if chunk_idx > 0:
            #     os.remove(f"chunks_cache/cache_chunk{chunk_idx - 1}.arrow")

        post_ds = Dataset.from_list(results)
        post_ds = post_ds.filter(lambda x: x["rejected"] != "") # remove empty completions
        print(post_ds)
        if args.push_to_hub:
            test_split_samples = int(len(post_ds) * args.test_split_percentage)
            post_ds.select(range(test_split_samples, len(post_ds))).push_to_hub(args.repo_id, split="train_prefs")
            post_ds.select(range(test_split_samples)).push_to_hub(args.repo_id, split="test_prefs")
            for file, name in zip([__file__], ["create_dataset.py"]):
                api.upload_file(
                    path_or_fileobj=file,
                    path_in_repo=name,
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
            print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")

    asyncio.run(main())

