import asyncio
import os
import time
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import AsyncInferenceClient
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

from llm_swarm import LLMSwarm, LLMSwarmConfig


HF_TOKEN = os.environ.get("HF_TOKEN", None)

@dataclass
class Args:
    max_samples: int = -1
    """The maximum umber of samples to generate (use -1 for all))"""
    max_new_tokens: int = 3000
    """Max new tokens"""
    temperature: float = 0.6
    """Generation temperature"""
    top_p: float = 0.95
    """Generation top_p"""
    top_k: int = 50
    """Generation top_k"""
    repetition_penalty: float = 1.2
    """Generation repetition_penalty"""
    prompt_column: str = "prompt"
    """Name of the column containing the prompt"""
    repo_id: str = "HuggingFaceTB/openstax_generations_f"
    """The repo id to push to"""
    push_to_hub: bool = True
    """Whether to push to hub"""

# mixtral throughput 4290
parser = HfArgumentParser((Args, LLMSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
print(args)
isc.model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
isc.instances = 10
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

ds = load_dataset(
    "HuggingFaceTB/openstax_prompts_concatenated", token=HF_TOKEN, split="train"
).shuffle(seed=42)

if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))


with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    STOP_SEQ = ["<|endoftext|>"]

    async def process_text(sample):
        token_length = 0
        async with semaphore:
            completion = await client.text_generation(
                prompt=tokenizer.apply_chat_template(
                    [{"role": "user", "content": sample[args.prompt_column]}],
                    tokenize=False,
                ),
                max_new_tokens=args.max_new_tokens,
                stop_sequences=STOP_SEQ,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
            )
            for stop_seq in STOP_SEQ:
                if completion.endswith(stop_seq):
                    completion = completion[: -len(stop_seq)].rstrip()
            token_length += len(tokenizer.encode(completion))
        sample["completion"] = completion
        sample["token_length"] = token_length
        return sample

    async def main():
        start_time = time.time()
        results = await tqdm_asyncio.gather(*(process_text(sample) for sample in ds))
        end_time = time.time()
        df = pd.DataFrame(results)
        output_ds = Dataset.from_pandas(df)
        total_duration = end_time - start_time
        total_tokens = sum(output_ds["token_length"])
        overall_tokens_per_second = total_tokens / total_duration if total_duration > 0 else 0
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")
        print(output_ds)

        if args.push_to_hub:
            output_ds.push_to_hub(args.repo_id, private=True)

    asyncio.run(main())
