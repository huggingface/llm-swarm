import asyncio
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from datasets import load_dataset
from dataclasses import dataclass
from transformers import HfArgumentParser, AutoTokenizer
import traceback


@dataclass
class Args:
    # input dataset parameters
    dataset_name: str = "HuggingFaceTB/fineweb_annotated_judge_300k_scores"
    batch_size: int = 32

    # output dataset parameters
    output_dataset_name: str = "HuggingFaceTB/fineweb_annotated_judge_300k_scores_embeddings"


@dataclass
class TEISwarmConfig(LLMSwarmConfig):
    model: str = "intfloat/multilingual-e5-large-instruct"
    instances: int = 2
    per_instance_max_parallel_requests: int = 32
    inference_engine: str = "tei"
    slurm_template_path: str = "templates/tei_h100.template.slurm"
    load_balancer_template_path: str = "templates/nginx.template.conf"


parser = HfArgumentParser((Args, TEISwarmConfig))
args, swarm_args = parser.parse_args_into_dataclasses()

with LLMSwarm(swarm_args) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    MAX_RETRIES = 6  # maximum number of retries
    RETRY_DELAY = 4  # delay in seconds between retries

    tokenizer = AutoTokenizer.from_pretrained(swarm_args.model)

    async def get_embeddings(record):
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                async with semaphore:
                    embeddings = await client.feature_extraction(text=record["text"][:4000])
                    record["embeddings"] = embeddings[0]
                    return record
            except Exception as e:
                attempt += 1
                if attempt < MAX_RETRIES:
                    print(
                        f"Request failed with error {str(e)}, retrying in {RETRY_DELAY} seconds... (Attempt {attempt}/{MAX_RETRIES})"
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    print(
                        f"Max retries reached. Failed to process the request with error {str(e)}."
                    )
                    traceback.print_exc()
                    raise

    async def process_batch(batch):
        batch = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        results = await asyncio.gather(*(get_embeddings(rec) for rec in batch))
        batch = {key: [rec[key] for rec in results] for key in results[0]}
        return batch

    def sync_process_batch(batch):
        results = asyncio.run(process_batch(batch))
        return results

    def main_sync():
        dataset = load_dataset(args.dataset_name, split="train", num_proc=4, cache_dir="/fsx/anton/.cache")
        processed_dataset = dataset.map(sync_process_batch, batched=True, batch_size=args.batch_size)

        processed_dataset.push_to_hub(args.output_dataset_name, private=True)

    main_sync()
