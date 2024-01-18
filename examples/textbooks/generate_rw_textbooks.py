import asyncio
import os
import multiprocessing
from dataclasses import dataclass, field
from string import Template
from typing import Annotated, Any

from huggingface_hub import AsyncInferenceClient

import pandas as pd
import tyro
from aiohttp import ClientError
from datasets import load_dataset
from rich.pretty import pprint
from wonderwords import RandomWord

from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer
from datatrove.pipeline.tokens.merger import DocumentTokenizerMerger
from datatrove.io import S3OutputDataFolder, S3InputDataFolder
from datatrove.data import Document
from datatrove.utils.stats import PipelineStats

from inference_swarm import SENTINEL, TGIConfig, generate_data


@dataclass
class Args:
    dataset: str = "tiiuae/falcon-refinedweb"
    """Hub dataset to use"""
    output_filename: str = "debug-v0.1"
    """Filenames for the output"""
    ds_subtype: str = "debug"
    """Dataset subtype"""
    output_local_folder: str = "output/textbooks"
    """Local folder where to save the output"""
    s3_final_prefix: str = "s3://extreme-scale-datasets/textbooks"
    """S3 prefix for the final output"""
    s3_tmp_prefix: str = "s3://extreme-scale-dp-temp/textbooks"
    """S3 prefix for the temporary output before merging"""
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
    textbook_tokenizer: str = "gpt2"
    """Tokenizer to use for the textbook generation"""
    stop_num_tokens: int = 500
    """Target number of tokens befor """
    tgi: tyro.conf.OmitArgPrefixes[TGIConfig] = field(default_factory=lambda: TGIConfig())


prompt_template = Template(
    """Here is an extract from a webpage: "$WEBPAGE".

Write a long and very detailed tutorial that could be part of WikiHow whose title is related to the extract above. Include in depth explanations for each step, the reasoning behind them and how they help achieve the desired outcome. The tutorial should include the words "$WORD1" and "$WORD2".
"""
)

STOP_SEQ = ["User:", "###", "<|endoftext|>"]

EOS_TOKEN: str = ("<|endoftext|>",)  # whether to add the EOS token after each document

if __name__ == "__main__":
    args = tyro.cli(Args, use_underscores=True)
    pprint(args)
    os.makedirs(args.output_local_folder, exist_ok=True)
    doc_tokenizer = DocumentTokenizer(
        output_folder=S3InputDataFolder(
            path=f"{args.s3_tmp_prefix}/{args.output_filename}/tokenized",
            local_path=f"{args.output_local_folder}/{args.output_filename}/tokenized",
        ),
        save_filename=args.output_filename,
        shuffle=False,
        # save_loss_metadata=False,
    )

    doc_merger = DocumentTokenizerMerger(
        input_folder=S3InputDataFolder(
            path=f"{args.s3_tmp_prefix}/{args.output_filename}/tokenized",
            local_path=f"{args.output_local_folder}/{args.output_filename}/tokenized",
        ),
        output_folder=S3OutputDataFolder(
            path=f"{args.s3_final_prefix}/{args.ds_subtype}/{args.output_filename}",
            local_path=f"{args.output_local_folder}/{args.ds_subtype}/{args.output_filename}",
        ),
        save_filename=args.output_filename,
    )

    def reader(input_queue: multiprocessing.Queue, start_index: int = 0):
        """Read the data starting from start_index and put it sample by sample in the input_queue.
            Add the end put a SENTINEL in the queue.

        Args:
            input_queue (multiprocessing.Queue): input queue
            start_index (int, optional): start index. Defaults to 0.
        """
        print("Loading dataset")
        rw = load_dataset(args.dataset, streaming=True, split="train")
        randomw = RandomWord()

        for si, sample in enumerate(rw):
            if si < start_index:
                continue
            extract = f"{sample['content'][:args.ext_len]}..." if len(sample["content"]) > args.ext_len else sample["content"]
            input_queue.put(
                {
                    "index": si,
                    "url": sample["url"],
                    "dump": sample["dump"],
                    "prompt": prompt_template.substitute(
                        WEBPAGE=extract,
                        WORD1=randomw.word(include_parts_of_speech=["nouns"]),
                        WORD2=randomw.word(include_parts_of_speech=["adjectives"]),
                    ),
                }
            )
        input_queue.put(SENTINEL)

    def writer(chunk: Any, chunk_i: int, total_nr_chunks: int) -> bool:
        """Write a chunk of samples to disk.
            Samples are as returned by send_request.
            Called for each complete chunk.

        Args:
            chunk (Any): sample
            chunk_i (int): chunk index
            total_nr_chunks (int): total number of chunks
        """
        print(f"Saving chunk {chunk_i + 1}{' of ' + str(total_nr_chunks) if total_nr_chunks else ''}")

        # Tokenize the textbook
        doc_tokenizer((Document(content=ch["textbook"], data_id=f"{chunk_i}_{i}") for i, ch in enumerate(chunk)), rank=chunk_i)

        tokens = doc_tokenizer.stats["tokens"].total

        pd.DataFrame(chunk).to_csv(f"{args.output_local_folder}/{chunk_i:05d}.csv", index=False)

        print(f"Tokens {tokens}")

        return tokens < args.stop_num_tokens, tokens

    async def send_request(sample: Any, client: AsyncInferenceClient) -> Any:
        """Send request to the model and return the result.

        Args:
            sample (Any): sample put in the input_queue by the reader function
            client (AsyncInferenceClienT): client

        Returns:
            Any: resul dict with TGI generated text
        """
        res = None
        tries = 1
        while not res:
            try:
                res = await client.text_generation(
                    prompt=f"User: {sample[args.prompt_column]}\nFalcon: "
                    if args.format_prompt
                    else sample[args.prompt_column],
                    max_new_tokens=args.max_new_tokens,
                    stop_sequences=STOP_SEQ,
                    temperature=args.temperature,
                    do_sample=True,
                    top_p=0.95,
                    top_k=None,
                )
                for stop_seq in STOP_SEQ:
                    if res.endswith(stop_seq):
                        res = res[: -len(stop_seq)].rstrip()

            except ClientError as e:
                if tries == 10:
                    raise e
                print(f"Error: {e}. Retrying...", flush=True)
                await asyncio.sleep(tries * 2 + 3)
                tries += 1
        sample["textbook"] = res
        return sample

    def closer():
        """Called at the end of the generation"""
        doc_merger(None)
        full_stats = PipelineStats([doc_tokenizer, doc_merger])
        full_stats.save_to_disk(f"{args.output_local_folder}/{args.output_filename}/stats.json")

        print("Done!")

    generate_data(args.tgi, reader, writer, send_request, closer=closer, total_tqdm=args.stop_num_tokens, max_input_size=20000)
