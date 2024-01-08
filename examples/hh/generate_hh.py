import asyncio
import json
import os
from dataclasses import dataclass, field
import random
import shutil
from typing import Annotated

from huggingface_hub import HfApi
import tyro
from aiohttp import ClientError
from datasets import load_dataset, Dataset, combine
from rich.pretty import pprint
from transformers import AutoTokenizer

from tgi_swarm import SENTINEL, TGIConfig, generate_data
api = HfApi()


@dataclass
class Args:
    output_folder: str = "output/hh"
    """Folder to store the output"""
    overwrite: bool = False
    """Whether to overwrite the output folder"""
    prompt_column: Annotated[str, tyro.conf.arg(aliases=["-pcol"])] = "prompt"
    """Name of the column containing the prompt"""
    temperature: Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1.0
    """Generation temperature"""
    max_new_tokens: Annotated[int, tyro.conf.arg(aliases=["-toks"])] = 1500
    """Max new tokens"""
    format_prompt: bool = True
    """Whether to format prompt"""
    max_samples: int = 128
    """The maximum umber of samples to generate (use -1 for all))"""
    split: str = "train"
    """The split to use"""
    push_to_hub: bool = False
    """Whether to push to hub"""
    constitution_path: str = "examples/hh/constitution1.json"
    """Path to the constitution"""
    repo_id: str = "cai-conversation-dev"
    """The repo id to push to"""
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig] = field(default_factory=lambda: TGIConfig())


if __name__ == "__main__":
    args = tyro.cli(Args)
    if os.path.exists(args.output_folder):
        args.overwrite = input(f"Output folder {args.output_folder} already exists. Overwrite? [y/N] ").lower() == "y"
        if args.overwrite:
            # remove folder
            shutil.rmtree(args.output_folder)
    os.makedirs(args.output_folder)
    rw = load_dataset("Anthropic/hh-rlhf", split=args.split, data_dir="harmless-base")
    if args.max_samples == -1:
        args.max_samples = len(rw)
    pprint(args)

    def reader(input_queue, start_index):
        print("Loading dataset")
        rw = load_dataset("Anthropic/hh-rlhf", split=args.split, data_dir="harmless-base").select(range(args.max_samples))

        def extract(example):
            # Extract the "Human:" prompts
            example = example["chosen"]
            split_text = example.split("\n\n")
            for segment in split_text:
                if "Human:" in segment:
                    return {"prompt": segment.split(": ")[1]}

        rw = rw.map(extract)

        for si, sample in enumerate(rw):
            if si < start_index:
                continue
            input_queue.put({"index": si, "prompt": sample["prompt"]})
        input_queue.put(SENTINEL)

    # called for each complete chunk
    def writer(chunk, chunk_i, total_nr_chunks):
        print(f"Saving chunk {chunk_i + 1}/{total_nr_chunks}")
        Dataset.from_list(chunk).save_to_disk(f"{args.output_folder}/{chunk_i:05d}")

    STOP_SEQ = ["User:", "###", "<|endoftext|>"]

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})
    with open(args.constitution_path, 'r') as f:
        data = json.load(f)
        constitutions = data["constitutions"]
        system_chat = data["system_chat"]
        system_chat = [item for sublist in system_chat for item in sublist]

    async def send_request(sample, client):
        chat = system_chat.copy()
        constitution = random.choice(constitutions)
        print("SAMPLED CONSTITUTION:", constitution)
        for prompt, prompt_key, response_key in [
            (sample[args.prompt_column], "init_prompt", "init_response"),
            (constitution["critic"], "critic_prompt", "critic_response"),
            (constitution["revision"], "revision_prompt", "revision_response"),
        ]:
            tries = 1
            res = None
            while not res:
                try:
                    prompt_dict = {"role": "user", "content": prompt}
                    chat.append(prompt_dict)
                    res = await client.text_generation(
                        prompt=tokenizer.apply_chat_template(chat, tokenize=False),
                        max_new_tokens=args.max_new_tokens,
                        stop_sequences=STOP_SEQ,
                        temperature=args.temperature,
                    )
                    for stop_seq in STOP_SEQ:
                        if res.endswith(stop_seq):
                            res = res[: -len(stop_seq)].rstrip()
                    response_dict = {"role": "assistant", "content": res}
                    chat.append(response_dict)
                # retry on error
                except ClientError as e:
                    if tries == 10:
                        raise e
                    print(f"Error: {e}. Retrying...", flush=True)
                    await asyncio.sleep(tries * 2 + 3)
                    tries += 1
            sample[prompt_key] = prompt_dict
            sample[response_key] = response_dict

        return sample

    closer = None
    if args.push_to_hub:
        def closer():
            """Called at the end of the generation"""
            dataset_shards = []
            for file in os.listdir(args.output_folder):
                print(file)
                dataset_shards.append(Dataset.load_from_disk(os.path.join(args.output_folder, file)))
            ds = combine.concatenate_datasets(dataset_shards)
            def process(example):
                return {
                    "prompt": example["init_prompt"]["content"],
                    "messages": [
                        example["init_prompt"],
                        example["revision_response"],
                    ],
                    "chosen": [
                        example["init_prompt"],
                        example["revision_response"],
                    ],
                    "rejected": [
                        example["init_prompt"],
                        example["init_response"],
                    ],
                }
            ds = ds.map(process)
            ds.select(range(len(ds) // 2)).push_to_hub(args.repo_id, split=f"{args.split}_sft")
            ds.select(range(len(ds) // 2, len(ds))).push_to_hub(args.repo_id, split=f"{args.split}_prefs")
            if "/" not in args.repo_id: # find the current user
                args.repo_id = f"{api.whoami()['name']}/{args.repo_id}"
            api.upload_file(
                path_or_fileobj=__file__,
                path_in_repo="create_dataset.py",
                repo_id=args.repo_id,
                repo_type="dataset",
            )
            print("Done!")

    generate_data(args.tgi, reader, writer, send_request, closer=closer, total_input=args.max_samples, max_input_size=20000)
