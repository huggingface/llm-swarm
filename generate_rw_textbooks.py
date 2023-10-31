from dataclasses import dataclass
from string import Template

from datasets import load_dataset
import pandas as pd
import tyro
from wonderwords import RandomWord

from tgi_swarm import TGIConfig, generate_data, SENTINEL


@dataclass
class Args:
    ext_len: int
    """Max extract length in characters"""
    output_folder: str
    """Folder to store the output"""
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig]

prompt_template = Template("""Here is an extract from a webpage: "$WEBPAGE".

Write a long and very detailed tutorial that could be part of WikiHow whose title is related to the extract above. Include in depth explanations for each step, the reasoning behind them and how they help achieve the desired outcome. The tutorial should include the words "$WORD1" and "$WORD2".
""")

if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)

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

    generate_data(args, reader, writer, max_input_size=20000)
