import asyncio
import json
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset
import time

parser = HfArgumentParser(LLMSwarmConfig)
isc = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained(isc.model, revision=isc.revision)
tasks = load_dataset("Anthropic/hh-rlhf", split="train")
tasks = tasks.select(range(1024))


def extract(example):
    # Extract the "Human:" prompts
    example = example["chosen"]
    split_text = example.split("\n\n")
    for segment in split_text:
        if "Human:" in segment:
            return {"prompt": segment.split(": ")[1]}


tasks = tasks.map(extract)["prompt"]
with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    async def process_text(task):
        async with semaphore:
            prompt = rf"<s>[INST] {task} [\INST]"
            if isc.inference_engine == "tgi":
                completion = await client.text_generation(
                    prompt=prompt,
                    max_new_tokens=200,
                    stop_sequences=["User:", "###", "<|endoftext|>"],
                )
            elif isc.inference_engine == "vllm":
                response = await client.post(
                    json={
                        "prompt": prompt,
                        "max_tokens": 200,
                    }
                )
                completion = json.loads(response.decode("utf-8"))["text"][0][len(prompt) :]
            tokenized_completion = tokenizer.encode(completion)
            token_length = len(tokenized_completion)
            return completion, token_length

    async def main():
        start_time = time.time()
        results = await tqdm_asyncio.gather(*[process_text(task) for task in tasks])
        end_time = time.time()

        total_duration = end_time - start_time
        total_tokens = sum(result[1] for result in results)
        overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        df = pd.DataFrame(
            {"Task": tasks, "Completion": [result[0] for result in results], "Token Length": [result[1] for result in results]}
        )
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")
        print(df)

    asyncio.run(main())
