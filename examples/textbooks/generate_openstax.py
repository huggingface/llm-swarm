import asyncio
import os
import time
from dataclasses import dataclass

import pandas as pd
from datasets import Dataset, load_dataset
from huggingface_hub import AsyncInferenceClient
from llm_swarm import LLMSwarm, LLMSwarmConfig
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@dataclass
class Args:
    max_samples: int = 50000
    """The maximum umber of samples to generate (use -1 for all))"""
    tgi_instances: int = 5
    """Number of TGI instances to use"""
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    """Model to prompt"""
    max_new_tokens: int = 2100
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


parser = HfArgumentParser((Args, LLMSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
print(args)

# overwrite model and number of instances
isc.model = args.model_name
isc.instances = args.tgi_instances
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

ds = load_dataset(
    "HuggingFaceTB/openstax_prompts_concatenated", token=HF_TOKEN, split="train"
).shuffle(seed=42)

if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))


with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    STOP_SEQ = ["<|endoftext|>"]

    MAX_RETRIES = 3  # maximum number of retries
    RETRY_DELAY = 5  # delay in seconds between retries

    async def process_text(sample):
        token_length = 0
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
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
                    sample["completion"] = ""
                    sample["token_length"] = 0
                    return sample

    async def main():
        start_time = time.time()
        results = await tqdm_asyncio.gather(*(process_text(sample) for sample in ds))
        end_time = time.time()
        df = pd.DataFrame(results)
        output_ds = Dataset.from_pandas(df)
        total_duration = end_time - start_time
        total_tokens = sum(output_ds["token_length"])
        overall_tokens_per_second = (
            total_tokens / total_duration if total_duration > 0 else 0
        )
        print(f"Overall Tokens per Second: {overall_tokens_per_second}")

        # remove empty completions
        output_ds = output_ds.filter(lambda x: x["completion"] != "")
        print(output_ds)
        if args.push_to_hub:
            output_ds.push_to_hub(args.repo_id, private=True)

    asyncio.run(main())
