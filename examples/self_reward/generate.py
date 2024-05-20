import asyncio
from collections import defaultdict
from dataclasses import dataclass
import json
import random
import pandas as pd
from inference_swarm import InferenceSwarm, InferenceSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer, HfArgumentParser
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset, Dataset
import time
from huggingface_hub import HfApi

api = HfApi()


@dataclass
class Args:
    repo_id: str = "self-reward-dev"
    """The repo id to push to"""
    timestamp: bool = True
    """Whether to add a timestamp to the repo_id"""
    push_to_hub: bool = False
    """Whether to push to hub"""

    # task-specific args
    max_samples: int = 4
    """The maximum umber of samples to generate (use -1 for all))"""
    max_new_tokens: int = 1500
    """Max new tokens"""
    temperature: float = 1.0
    """Generation temperature"""
    candidates: int = 4
    """Number of candidates to generate"""

LLM_AS_A_JUDGE_PROMPT = """\
Review the user’s question and the corresponding response using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the response is relevant and provides some information related to the user’s inquiry, even if it is incomplete or contains some irrelevant content.
- Add another point if the response addresses a substantial portion of the user’s question, but does not completely resolve the query or provide a direct answer.
- Award a third point if the response answers the basic elements of the user’s question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results.
- Grant a fourth point if the response is clearly written from an AI Assistant’s perspective, addressing the user’s question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus.
- Bestow a fifth point for a response that is impeccably tailored to the user’s question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer.

User: <INSTRUCTION_HERE>
<response><RESPONSE_HERE></response>

After examining the user’s instruction and the response:
- Briefly justify your total score, up to 100 words.
- Conclude with the score using the format: “Score: <total points>”

Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we’ll systematically attribute points based on the outlined criteria."""



parser = HfArgumentParser((Args, InferenceSwarmConfig))
args, isc = parser.parse_args_into_dataclasses()
if args.timestamp:
    args.repo_id += str(int(time.time()))
tokenizer = AutoTokenizer.from_pretrained(isc.model, revision=isc.revision)
ds = load_dataset("HuggingFaceH4/ultrachat_200k")
for key in ds:
    max_samples = len(ds[key]) if args.max_samples == -1 else args.max_samples
    ds[key] = ds[key].select(range(max_samples))

def extract(example):
    return {
        "prompt": example["prompt"],
        "response": example["messages"][1]["content"],
    }

ds = ds.map(extract)
rate_limit = 500 * isc.instances
semaphore = asyncio.Semaphore(rate_limit)
with InferenceSwarm(isc) as inference_swarm:
    client = AsyncInferenceClient(model=inference_swarm.endpoint)

    async def process_text(split, i, row):
        scores = []
        newrow = {"prompt": row["prompt"]}
        for i in range(args.candidates):
            chat = [{"role": "user", "content": newrow["prompt"]}]
            # 2.2 Self-Instruction Creation, 2. Generate candidate responses, https://arxiv.org/pdf/2401.10020.pdf
            completion = await client.text_generation(
                prompt=tokenizer.apply_chat_template(chat, tokenize=False),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
            )
            newrow[f"candidate_{i}"] = completion

            # 2.2 Self-Instruction Creation, 3. Evaluate candidate responses, https://arxiv.org/pdf/2401.10020.pdf
            llm_as_a_judge_prompt = LLM_AS_A_JUDGE_PROMPT \
                .replace("<INSTRUCTION_HERE>", newrow["prompt"]) \
                .replace("<RESPONSE_HERE>", completion)
            chat = [{"role": "user", "content": llm_as_a_judge_prompt}]
            completion = await client.text_generation(
                prompt=tokenizer.apply_chat_template(chat, tokenize=False),
                max_new_tokens=args.max_new_tokens,
            )
            try:
                score = float(completion.split("Score: ")[-1])
            except ValueError:
                score = -1.0
            if score > 5.0:
                score = -1.0
            newrow[f"score_{i}"] = score
            scores.append(score)
        
        # choose the best candidate
        chosen_idx = scores.index(max(scores))
        newrow["chosen"] = newrow[f"candidate_{chosen_idx}"]
        newrow["chosen_score"] = newrow[f"score_{chosen_idx}"]
        newrow["chosen_idx"] = chosen_idx
        rejected_idx = scores.index(min(scores))
        newrow["rejected"] = newrow[f"candidate_{rejected_idx}"]
        newrow["rejected_score"] = newrow[f"score_{rejected_idx}"]
        newrow["rejected_idx"] = rejected_idx
        return split, i, newrow

    async def main():
        tasks = [process_text(split, idx, row) for split in ds for idx, row in enumerate(ds[split])]
        results = await tqdm_asyncio.gather(*tasks)
        all_ds = defaultdict(list)
        for result in results:
           all_ds[result[0]].append(result[2])
        for split in all_ds:
            df = pd.DataFrame(all_ds[split])
            print("=" * 10 + split + "=" * 10)
            print(df)
            post_ds = Dataset.from_pandas(df)
            if args.push_to_hub:
                post_ds.push_to_hub(args.repo_id, split=f"{split}_sft")
                if "/" not in args.repo_id:  # find the current user
                    repo_id = f"{api.whoami()['name']}/{args.repo_id}"
                api.upload_file(
                    path_or_fileobj=__file__,
                    path_in_repo="create_dataset.py",
                    repo_id=repo_id,
                    repo_type="dataset",
                )
        breakpoint()

    asyncio.run(main())
