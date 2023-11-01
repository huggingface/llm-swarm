from dataclasses import dataclass, field
from string import Template

import asyncio
from typing import Annotated
from aiohttp import ClientError
from datasets import load_dataset
import pandas as pd
import tyro
from wonderwords import RandomWord
from rich.pretty import pprint

from tgi_swarm import TGIConfig, generate_data, SENTINEL


@dataclass
class Args:
    output_folder: str = "output/rw_textbooks"
    """Folder to store the output"""
    prompt_column: Annotated[str, tyro.conf.arg(aliases=["-pcol"])] = "prompt"
    """Name of the column containing the prompt"""
    temperature: Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1.0
    """Generation temperature"""
    max_new_tokens: Annotated[int, tyro.conf.arg(aliases=["-toks"])] = 1500
    """Max new tokens"""
    format_prompt: bool = True
    """Whether to format prompt"""
    ext_len: int = 200
    """Max extract length in characters"""
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig] = field(
        default_factory=lambda: TGIConfig()
    )

prompt_template = Template("""Here is an extract from a webpage: "$WEBPAGE".

Write a long and very detailed tutorial that could be part of WikiHow whose title is related to the extract above. Include in depth explanations for each step, the reasoning behind them and how they help achieve the desired outcome. The tutorial should include the words "$WORD1" and "$WORD2".
""")

if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)
    pprint(args)

    # reader: add all data and then add SENTINEL
    def reader(input_queue, start_index):
        print(f"Loading dataset")
        rw = load_dataset("tiiuae/falcon-refinedweb", streaming=True, split="train")
        randomw = RandomWord()

        for si, sample in enumerate(rw):
            if si < start_index:
                continue
            extract = f"{sample['content'][:args.ext_len]}..." \
                if len(sample['content']) > args.ext_len else sample['content']
            input_queue.put({
                "index": si,
                "url": sample["url"],
                "dump": sample["dump"],
                "prompt": prompt_template.substitute(WEBPAGE=extract, WORD1=randomw.word(include_parts_of_speech=["nouns"]), WORD2=randomw.word(include_parts_of_speech=["adjectives"]))

            })
        input_queue.put(SENTINEL)

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
                    prompt=f"User: {sample[args.prompt_column]}\nFalcon: " if args.format_prompt else sample[args.prompt_column],
                    max_new_tokens=args.max_new_tokens,
                    stop_sequences=STOP_SEQ, temperature=args.temperature
                )
                for stop_seq in STOP_SEQ:
                    if res.endswith(stop_seq):
                        res = res[:-len(stop_seq)].rstrip()
            except ClientError as e:
                if tries == 10:
                    raise e
                print(f"Error: {e}. Retrying...", flush=True)
                await asyncio.sleep(tries * 2 + 3)
                tries += 1
        sample["textbook"] = res
        return sample


    generate_data(args.tgi, reader, writer, send_request, max_input_size=20000)
