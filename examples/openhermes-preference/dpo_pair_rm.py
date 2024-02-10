from absl import app
from absl import flags

from datasets import load_dataset
import llm_blender

FLAGS = flags.FLAGS
flags.DEFINE_string("path", "HuggingFaceH4/openhermes_2.5_dpo_v0", "Datsaset path")
flags.DEFINE_string("split", "train_prefs", "Dataset split to use")
flags.DEFINE_string("output_path", "openhermes_2.5_dpo_pairrm_v0", "Save to disk path")
flags.DEFINE_integer("batch_size", 512, "Batch size for dataset mapping function")
flags.DEFINE_integer("num_shards", 20, "Number of shards to split the data")
flags.DEFINE_integer("shard_index", 0, "Index of the shard to use")


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


def pairRM(example, batch_size=80):
    results = blender.compare_conversations(
        [prepare_conversation(chosen) for chosen in example["chosen"]],
        [prepare_conversation(rejected) for rejected in example["rejected"]],
        batch_size=batch_size,
    )

    new_chosen = []
    new_reject = []
    chosen_policy = []
    rejected_policy = []
    for i, result in enumerate(results):
        if result == False:
            new_chosen.append(example["rejected"][i])
            new_reject.append(example["chosen"][i])
            chosen_policy.append(example["rejected_policy"][i])
            rejected_policy.append(example["chosen_policy"][i])
        else:
            new_chosen.append(example["chosen"][i])
            new_reject.append(example["rejected"][i])
            chosen_policy.append(example["chosen_policy"][i])
            rejected_policy.append(example["rejected_policy"][i])

    example["chosen"] = new_chosen
    example["rejected"] = new_reject
    example["chosen_policy"] = chosen_policy
    example["rejected_policy"] = rejected_policy

    return example


def main(argv):
    del argv  # Unused.

    dataset = load_dataset(FLAGS.path, split=FLAGS.split)
    shard = dataset.shard(num_shards=FLAGS.num_shards, index=FLAGS.shard_index)
    pairrm_shard = shard.map(pairRM, batched=True, batch_size=FLAGS.batch_size)
    pairrm_shard.save_to_disk(f"{FLAGS.output_path}_{FLAGS.split}_{FLAGS.shard_index}")


if __name__ == "__main__":
    app.run(main)
