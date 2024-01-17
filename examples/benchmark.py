import asyncio
import pandas as pd
from tgi_swarm import InferenceSwarm, InferenceSwarmConfig, test_generation
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import time

parser = HfArgumentParser(InferenceSwarmConfig)
isc = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})
ds = load_dataset("Anthropic/hh-rlhf", split="train")
ds = ds.select(range(10480))
def extract(example):
    # Extract the "Human:" prompts
    example = example["chosen"]
    split_text = example.split("\n\n")
    for segment in split_text:
        if "Human:" in segment:
            return {"prompt": segment.split(": ")[1]}
ds = ds.map(extract)["prompt"]
rate_limit = 500 * isc.instances
semaphore = asyncio.Semaphore(rate_limit)
with InferenceSwarm(isc) as inference_swarm:
    client = AsyncInferenceClient(model=inference_swarm.endpoint)
    async def process_text(task):
        async with semaphore:
            prompt = rf"<s>[INST] {task} [\INST]"
            completion = await client.text_generation(
                prompt=prompt,
                max_new_tokens=1500,
                stop_sequences=["User:", "###", "<|endoftext|>"],
            )
            tokenized_completion = tokenizer.encode(completion)
            token_length = len(tokenized_completion)
            return completion, token_length

    async def main():
        start_time = time.time()
        tasks = [process_text(task) for task in ds]
        results = await tqdm_asyncio.gather(*tasks)
        end_time = time.time()

        total_duration = end_time - start_time
        total_tokens = sum(result[1] for result in results)
        overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        df = pd.DataFrame({
            'Task': tasks,
            'Completion': [result[0] for result in results],
            'Token Length': [result[1] for result in results]
        })
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")
        print(df)
    asyncio.run(main())
