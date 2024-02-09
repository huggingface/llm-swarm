from absl import app
from absl import flags

from datasets import load_from_disk, concatenate_datasets, DatasetDict

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset_path", "openhermes_2.5_dpo_pairrm_v0", "Base dataset path")
flags.DEFINE_string("output_path", "HuggingFaceH4/openhermes_2.5_dpo_pairrm_v0", "Save path")
flags.DEFINE_integer("num_shards", 50, "Number of shards to split the data")


def main(argv):
    del argv  # Unused.

    datasets = []
    for i in range(FLAGS.num_shards):
        datasets.append(load_from_disk(f"{FLAGS.dataset_path}_{i}"))

    train_dataset = concatenate_datasets(datasets, split="train_prefs")

    test_dataset = load_from_disk(f"{FLAGS.dataset_path}_test")

    final_dataset = DatasetDict({"train_prefs": train_dataset, "test_prefs": test_dataset})
    final_dataset.push_to_hub(FLAGS.output_path)


if __name__ == "__main__":
    app.run(main)
