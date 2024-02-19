from dataclasses import dataclass
from datasets import load_dataset
import llm_blender
from transformers import HfArgumentParser
import multiprocessing
import random

@dataclass
class Args:
    path: str = "HuggingFaceH4/openhermes_2.5_dpo_v0"
    """Path to the dataset"""
    split: str = "train_prefs"
    """Dataset split to use"""
    output_path: str = "openhermes_2.5_dpo_pairrm_v0"
    """Save to disk path"""
    batch_size: int = 512
    """Batch size for dataset mapping function"""
    num_shards: int = 1
    """Number of shards to split the data"""
    shard_index: int = 0
    """Index of the shard to use"""
    max_samples: int = 1024
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


def pairRM(row, batch_size=80):
    results = blender.compare_conversations(
        [prepare_conversation(chosen) for chosen in row["candidate0"]],
        [prepare_conversation(rejected) for rejected in row["candidate1"]],
        batch_size=batch_size,
    )

    new_chosen = []
    new_reject = []
    chosen_policy = []
    rejected_policy = []
    for i, result in enumerate(results):
        if result == False:
            new_chosen.append(row["candidate1"][i])
            new_reject.append(row["candidate0"][i])
            chosen_policy.append(row["candidate1_policy"][i])
            rejected_policy.append(row["candidate0_policy"][i])
        else:
            new_chosen.append(row["candidate0"][i])
            new_reject.append(row["candidate1"][i])
            chosen_policy.append(row["candidate0_policy"][i])
            rejected_policy.append(row["candidate1_policy"][i])

    row["chosen"] = new_chosen
    row["rejected"] = new_reject
    row["chosen_policy"] = chosen_policy
    row["rejected_policy"] = rejected_policy

    return row

ds = load_dataset(args.path, split=args.split)
if args.max_samples > 0:
    ds = ds.select(range(args.max_samples))
def modify(row):
    row["chosen_policy"] = "gpt4"

    responses = [row["chosen"], row["rejected"]]
    policies = [row["chosen_policy"], row["rejected_policy"]]
    indices = [0, 1]
    random.shuffle(indices)

    row["candidate0"] = responses[indices[0]]
    row["candidate1"] = responses[indices[1]]
    row["candidate0_policy"] = policies[indices[0]]
    row["candidate1_policy"] = policies[indices[1]]
    return row

ds = ds.map(modify, load_from_cache_file=False, num_proc=1 if args.debug else multiprocessing.cpu_count())
ds = ds.remove_columns(
    [
        'system_prompt', 'model', 'avatarUrl', 'conversations', 'title',
        'skip_prompt_formatting', 'idx', 'hash', 'views', 'custom_instruction',
        'language', 'id', 'model_name', 'chosen_policy', 'chosen', 
        'token_length', 'rejected', 'rejected_policy',
    ]
)
df = ds.to_pandas()
# print(df["candidate0_policy"][:10])
# print(df["candidate0"][0])

shard = ds.shard(num_shards=args.num_shards, index=args.shard_index)
pairrm_shard = shard.map(pairRM, batched=True, batch_size=args.batch_size, load_from_cache_file=False)
pairrm_shard.save_to_disk(f"{args.output_path}_{args.split}_{args.shard_index}")

# visualization
df = pairrm_shard.to_pandas()
# print(df["candidate0_policy"][:10])
# print(df["candidate0"][0])
print(args.path)
print(df["chosen_policy"].value_counts())
# print(df["chosen_policy"][:10])