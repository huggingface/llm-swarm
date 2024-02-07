import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

import pandas as pd
from datasets import load_dataset
from huggingface_hub import AsyncInferenceClient
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoTokenizer, HfArgumentParser

import wandb
from llm_swarm import LLMSwarm, LLMSwarmConfig

HF_TOKEN = os.environ.get("HF_TOKEN", None)


@dataclass
class Args:
    # gneration parameters
    tgi_instances: int = 10
    """Number of TGI instances to use"""
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    """Dataset containing the prompts"""
    max_new_tokens: int = 2500
    """Max new tokens"""
    temperature: float = 0.6
    """Generation temperature"""
    top_p: float = 0.95
    """Generation top_p"""
    top_k: int = 50
    """Generation top_k"""
    repetition_penalty: float = 1.2
    """Generation repetition_penalty"""
    # prompts dataset parameters
    prompts_dataset: str = "HuggingFaceTB/fw_prompts_data_textbook"
    """Model to prompt"""
    max_samples: int = 5000
    """The maximum umber of samples to generate (use -1 for all))"""
    start_sample: int = -1
    """First sample to process"""
    end_sample: int = -1
    """Last sample to process"""
    num_proc: int = 36
    """Number of processes for loading the dataset"""
    prompt_column: str = "prompt_blogpost"
    """Name of the column containing the prompt (from 'prompt_wikihow', 'prompt_textbook_narrative', 'prompt_textbook_academic', 'prompt_blogpost'"""
    shuffle_dataset: bool = False
    """Whether to shuffle the prompts"""
    # logging parameters
    repo_id: str = "HuggingFaceTB/fw_generations_test"
    """The repo id to push to"""
    checkpoint_path: str = "./fw_data"
    """Path for saving intermediate generations"""
    checkpoint_interval: int = 10_000
    """Interval for saving intermediate generations"""
    wandb_username: str = "loubnabnl"
    """Wandb username"""
    push_to_hub: bool = True
    """Whether to push to hub"""


parser = HfArgumentParser((Args, LLMSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
print(args)

# overwrite model and number of instances
isc.model = args.model_name
isc.instances = args.tgi_instances
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
SEED = 42

ds = load_dataset(
    args.prompts_dataset, token=HF_TOKEN, split="train", num_proc=args.num_proc
)

if args.shuffle_dataset:
    ds = ds.shuffle(seed=SEED)

if args.start_sample >= 0:
    end_sample = len(ds) if args.end_sample < 0 else args.end_sample
    print(f"Loading a defined range of samples: ({args.start_sample}, {end_sample})...")
    ds = ds.select(range(args.start_sample, end_sample))
elif args.max_samples > 0:
    print(f"Loading the first {args.max_samples} samples...")
    ds = ds.select(range(args.max_samples))


with LLMSwarm(isc) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    STOP_SEQ = ["<|endoftext|>"]

    MAX_RETRIES = 6  # maximum number of retries
    RETRY_DELAY = 4  # delay in seconds between retries

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
        total_tokens = 0
        saving_time = 0

        wandb.init(
            project="synthetic_data",
            entity=args.wandb_username,
            name=repo_id.split("/")[1],
        )
        wandb.config.update(asdict(args))

        repo_id = (
            f"{args.repo_id}_{args.prompt_column}"
            if args.prompt_column not in args.repo_id
            else args.repo_id
        )
        checkpoint_dir = f"{args.checkpoint_path}/{repo_id.split('/')[1]}/data"
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Will be saving at {checkpoint_dir}")

        total_samples = len(ds)
        for i in range(0, total_samples, args.checkpoint_interval):
            batch_time = time.time()
            # Processing a chunk
            print(
                f"Processing chunk {int(i/args.checkpoint_interval)}/{int(total_samples/args.checkpoint_interval)}"
            )
            end_index = min(i + args.checkpoint_interval, total_samples)
            chunk = ds.select(range(i, end_index))
            chunk_results = await tqdm_asyncio.gather(
                *(process_text(sample) for sample in chunk)
            )
            # Save the chunk results and log throughput
            temp_time = time.time()
            time_per_chunk = temp_time - batch_time
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{i}.json")
            with open(checkpoint_path, "w") as f:
                json.dump(chunk_results, f)

            df = pd.DataFrame.from_records(chunk_results)
            batch_tokens = sum(df["token_length"])
            total_tokens += batch_tokens
            saving_time += time.time() - temp_time
            print(f"💾 Checkpoint {i} saved at {checkpoint_path}.")
            wandb.log(
                {
                    "sample": i + args.checkpoint_interval,
                    "batch": int(i / args.checkpoint_interval),
                    "total_tokens (M)": total_tokens / 1e6,
                    "tokens_per_batch": batch_tokens,
                    "time_per_batch (s)": time_per_chunk,
                    "generated_tokens_per_sec": int(batch_tokens / time_per_chunk),
                    "generated_tokens_per_sec_per_node": int(
                        batch_tokens / (time_per_chunk * args.tgi_instances)
                    ),
                }
            )

        end_time = time.time()

        print(
            "Done processing and saving all chunks 🎉! Let's get some stats and push to hub..."
        )
        total_duration = end_time - start_time
        overall_tokens_per_second = (
            total_tokens / total_duration if total_duration > 0 else 0
        )
        print(f"🏎️💨 Overall Tokens per Second: {overall_tokens_per_second:.2f}")
        print(f"Generated {total_tokens / 1e6:.2f}M tokens")
        print(
            f"Total duration: {total_duration // 3600}h{int((total_duration % 3600) // 60)}min "
        )
        print(f"Saving time: {saving_time}s={saving_time/60}min ")

        # load dataset
        print("Load checkpoints...")
        output_ds = load_dataset(
            "json", data_files=f"{checkpoint_dir}/*.json", split="train"
        )
        output_ds = output_ds.remove_columns(
            [
                col
                for col in output_ds.column_names
                if col
                not in [
                    "category",
                    "cluster_id",
                    "generation_type",
                    "educational_score",
                    "completion",
                    "token_length",
                    args.prompt_column,
                ]
            ]
        )
        # remove empty completions
        final_data = output_ds.filter(lambda x: x["completion"] != "")
        failed = output_ds.filter(lambda x: x["completion"] == "")
        print(output_ds)
        if args.push_to_hub:
            print(f"📨 Pushing dataset to {repo_id}")
            final_data.push_to_hub(repo_id, private=True)
            if len(failed) > 0:
                print(f"{len(failed)} generations failed")
                size = min(len(failed), 1000)
                failed = failed.select(range(size))
                failed.push_to_hub(f"{repo_id}_failed", private=True)

    asyncio.run(main())
    wandb.finish()
