import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import multiprocessing
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
import time
from huggingface_hub import HfApi
api = HfApi()

CHUNK_SIZE = 20000  # Define your chunk size here

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
    sample["chosen_policy"] = conversations[0]["from"]
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
    semaphore = asyncio.Semaphore(500)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)

    async def process_text(row):
        async with semaphore:
            prompt = tokenizer.apply_chat_template(row["chosen"][:-1], tokenize=False)
            completion = None
            while completion is None:
                try:
                    completion = await client.text_generation(
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        do_sample=args.do_sample,
                    )
                except Exception as e:
                    print(f"error in: {e}; retrying in 2 seconds")
                    time.sleep(2)
                    continue
            row["rejected"] = row["chosen"][:-1] + [{"role": "assistant", "content": completion}]
            row["rejected_policy"] = ":".join([isc.model, isc.revision])
            return row

    async def main():
        # results = await tqdm_asyncio.gather(*[process_text(row) for row in ds])
        results = []
        num_chunks = len(ds) // CHUNK_SIZE
        for i in range(0, len(ds), CHUNK_SIZE):
            print(f"Processing chunk {i // CHUNK_SIZE + 1}/{num_chunks}")
            chunk = ds.select(range(i, min(i + CHUNK_SIZE, len(ds))))
            chunk_results = await tqdm_asyncio.gather(*[process_text(row) for row in chunk])
            results.extend(chunk_results)
        post_ds = Dataset.from_list(results)
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

