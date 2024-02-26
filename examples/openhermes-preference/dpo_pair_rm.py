from dataclasses import dataclass
from datasets import load_dataset
import llm_blender
from transformers import HfArgumentParser
import multiprocessing
import random
import warnings
warnings.filterwarnings("ignore")
@dataclass
class Args:
    path: str = "vwxyzjn/openhermes-dev__combined__1708612612"
    """Path to the dataset"""
    split: str = "train"
    """Dataset split to use"""
    output_path: str = "openhermes_merged"
    """Save to disk path"""
    batch_size: int = 512
    """Batch size for dataset mapping function"""
    num_shards: int = 1
    """Number of shards to split the data"""
    shard_index: int = 0
    """Index of the shard to use"""
    max_samples: int = 128
    """The maximum umber of samples to generate (use -1 for all))"""
    debug: bool = False
    """Debug mode"""

parser = HfArgumentParser([Args])
args = parser.parse_args_into_dataclasses()[0]


blender = llm_blender.Blender()
blender.loadranker("llm-blender/PairRM")


def prepare_conversation(conversation):
    transformed_conversation = [
        {
            "content": turn["content"],
            "role": "USER" if turn["role"] == "user" else "ASSISTANT",
        }
        for turn in conversation
    ]
    return transformed_conversation


def pairRM(rows, batch_size=80):
    instructions = ["Finish the following coversation in each i-th turn by filling in <Response i> with your response."] * len(rows["candidates_completions"])
    cands = []
    for i in range(len(rows["candidates_completions"])):
        row_cand = []
        for j in range(len(rows["candidates_completions"][i])):
            row_cand.append([
                {"role": "user", "content": rows["prompt"][i]},
                {"role": "assistant", "content": rows["candidates_completions"][i][j]}
            ])
        cands.append(row_cand)

    inputs = [
        "\n".join([
            "USER: " + x[i]['content'] +
            f"\nAssistant: <Response {i//2+1}>" for i in range(0, len(x), 2)
        ]) for x in [prepare_conversation(item[0]) for item in cands]
    ]
    cand_texts = []
    for j in range(len(rows["candidates_completions"][i])):
        cand_texts.append([
            "\n".join([
                f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
            ]) for x in [prepare_conversation(item[j]) for item in cands]
        ])
    results = blender.rank(
        inputs,
        list(zip(*cand_texts)),
        instructions,
    )
    # print(results)
    ranks = [[p-1 for p in item] for i, item in enumerate(results)]
    rank_str = [" > ".join([rows["candidate_policies"][i][p-1] for p in item]) for i, item in enumerate(results)]
    rows["ranks"] = ranks
    rows["rank_str"] = rank_str
    rows["chosen_policy"] = [rows["candidate_policies"][i][r[0]] for i, r in enumerate(ranks)]
    rows["chosen"] = [cands[i][r[0]] for i, r in enumerate(ranks)]
    rows["rejected_policy"] = [rows["candidate_policies"][i][r[-1]] for i, r in enumerate(ranks)]
    rows["rejected"] = [cands[i][r[-1]] for i, r in enumerate(ranks)]
    return rows

ds = load_dataset(args.path, split=args.split)
if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))

def modify(row):
    candidates_completions = row["candidates_completions"]
    candidate_policies = row["candidate_policies"]
    indices = [0, 1, 2]
    random.shuffle(indices)
    new_candidates_completions = [candidates_completions[i] for i in indices]
    new_candidate_policies = [candidate_policies[i] for i in indices]
    row["candidates_completions"] = new_candidates_completions
    row["candidate_policies"] = new_candidate_policies
    return row

ds = ds.map(modify, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
df = ds.to_pandas()
shard = ds.shard(num_shards=args.num_shards, index=args.shard_index)
pairrm_shard = shard.map(pairRM, batched=True, batch_size=args.batch_size, load_from_cache_file=False)
pairrm_shard.save_to_disk(f"{args.output_path}_{args.split}_{args.shard_index}")

# visualization
df = pairrm_shard.to_pandas()
print(args.path)
print(df["rank_str"].value_counts())
print(df["chosen_policy"].value_counts())
