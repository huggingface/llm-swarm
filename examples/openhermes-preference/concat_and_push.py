from absl import app
from absl import flags

from datasets import load_from_disk, concatenate_datasets, DatasetDict

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "openhermes_merged", "Base dataset path")
flags.DEFINE_spaceseplist("splits", "train", "Dataset train and test splits to use when concatenating")
flags.DEFINE_string("output_path", "HuggingFaceH4/openhermes_merged_v1", "Save path on Hugging Face Hub")
flags.DEFINE_integer("num_shards", 20, "Number of shards used to split the train data")


def main(argv):
    del argv  # Unused.

    datasets = []
    for i in range(FLAGS.num_shards):
        datasets.append(load_from_disk(f"{FLAGS.dataset_path}_{FLAGS.splits[0]}_{i}"))
    train_dataset = concatenate_datasets(datasets)

    final_dataset = DatasetDict({"train": train_dataset})
    df = final_dataset["train"].to_pandas()
    print(df["chosen_policy"].value_counts().to_markdown())
    print(df["rejected_policy"].value_counts().to_markdown())
    print((df["source"].value_counts() / len(df)).to_markdown())
    breakpoint()
    final_dataset.push_to_hub(FLAGS.output_path)


if __name__ == "__main__":
    app.run(main)
