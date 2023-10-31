from dataclasses import dataclass, field
import os

from datasets import load_dataset
import pandas as pd
import tyro

from tgi_swarm import TGIConfig, generate_data, SENTINEL


@dataclass
class Args:
    output_folder: str = "output/hh_simple"
    """Folder to store the output"""
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig] = field(
        default_factory=lambda: TGIConfig()
    )


if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)
    os.makedirs(args.output_folder, exist_ok=True)

    def reader(input_queue, start_index):
        print(f"Loading dataset")
        rw = load_dataset("Anthropic/hh-rlhf", split="train").select(range(64))
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
            input_queue.put({
                "index": si,
                "prompt": sample["prompt"]
            })
        input_queue.put(SENTINEL)

    # called for each complete chunk
    def writer(chunk, chunk_i, total_nr_chunks):
        print(f"Saving chunk {chunk_i + 1}/{total_nr_chunks}")
        pd.DataFrame(chunk).to_csv(f"{args.output_folder}/{chunk_i:05d}.csv", index=False)

    generate_data(args.tgi, reader, writer, 0, max_input_size=20000)
