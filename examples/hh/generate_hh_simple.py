import asyncio
import os
from dataclasses import dataclass, field
from typing import Annotated

import pandas as pd
import tyro
from aiohttp import ClientError
from datasets import load_dataset
from rich.pretty import pprint

from tgi_swarm import SENTINEL, TGIConfig, generate_data


@dataclass
class Args:
    output_folder: str = "output/hh_simple"
    """Folder to store the output"""
    prompt_column: Annotated[str, tyro.conf.arg(aliases=["-pcol"])] = "prompt"
    """Name of the column containing the prompt"""
    temperature: Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1.0
    """Generation temperature"""
    max_new_tokens: Annotated[int, tyro.conf.arg(aliases=["-toks"])] = 1500
    """Max new tokens"""
    format_prompt: bool = True
    """Whether to format prompt"""
    max_samples: int = 1024
    """The maximum number of samples to generate (use -1 for all))"""
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig] = field(default_factory=lambda: TGIConfig())
    use_textbooks: bool = True
    """Whether to use textbooks prompts (the prompts have varying sizes ~10 to 800 tokens)"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    os.makedirs(args.output_folder, exist_ok=True)
    if args.use_textbooks:
        rw = load_dataset("HuggingFaceTB/synthetic_textbooks_subset", split="train")
    else:
        rw = load_dataset("Anthropic/hh-rlhf", split="train")
    if args.max_samples == -1:
        args.max_samples = len(rw)
    pprint(args)

    def reader(input_queue, start_index):
        print("Loading dataset")
        if args.use_textbooks:
            rw = load_dataset(
                "HuggingFaceTB/synthetic_textbooks_subset", split="train"
            ).select(range(args.max_samples))
        else:
            rw = load_dataset("Anthropic/hh-rlhf", split="train").select(
                range(args.max_samples)
            )

        def extract(example):
            # Extract the "Human:" prompts
            example = example["chosen"]
            split_text = example.split("\n\n")
            for segment in split_text:
                if "Human:" in segment:
                    return {"prompt": segment.split(": ")[1]}

        def extract_textbooks(example):
            text = example["prompt"]
            prompt = text[: -len("Falcon:")] if text.endswith("Falcon:") else text
            return {"prompt": prompt}

        if args.use_textbooks:
            rw = rw.map(extract_textbooks)
        else:
            rw = rw.map(extract)

        for si, sample in enumerate(rw):
            if si < start_index:
                continue
            input_queue.put({"index": si, "prompt": sample["prompt"]})
        input_queue.put(SENTINEL)
        print("Dataset ready")

    # called for each complete chunk
    def writer(chunk, chunk_i, total_nr_chunks):
        print(f"Saving chunk {chunk_i + 1}/{total_nr_chunks}")
        pd.DataFrame(chunk).to_csv(f"{args.output_folder}/{chunk_i:05d}.csv", index=False)

    STOP_SEQ = ["User:", "###", "<|endoftext|>"]

    async def send_request(sample, client):
        res = None
        tries = 1
        while not res:
            try:
                res = await client.text_generation(
                    prompt=rf"<s>[INST] {sample[args.prompt_column]} [\INST]",
                    max_new_tokens=args.max_new_tokens,
                    stop_sequences=STOP_SEQ,
                    temperature=args.temperature,
                )
                for stop_seq in STOP_SEQ:
                    if res.endswith(stop_seq):
                        res = res[: -len(stop_seq)].rstrip()
            # retry on error
            except ClientError as e:
                if tries == 10:
                    raise e
                print(f"Error: {e}. Retrying...", flush=True)
                await asyncio.sleep(tries * 2 + 3)
                tries += 1
        sample["continuation"] = res
        return sample

    generate_data(
        args.tgi,
        reader,
        writer,
        send_request,
        total_input=args.max_samples,
        max_input_size=20000,
        log_throughput=True,
    )
