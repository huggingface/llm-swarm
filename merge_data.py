from dataclasses import dataclass
import pandas as pd
import tyro
import os
import datasets
from datasets import Dataset


@dataclass
class Args:
    output_folder: str = "output/hh_simple"
    """Folder to store the output"""


args = tyro.cli(Args)
dataset_shards = []
for file in os.listdir(args.output_folder):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(args.output_folder, file))
        dataset_shards.append(Dataset.from_pandas(df))
        print(df.head())

ds = datasets.combine.concatenate_datasets(dataset_shards)
ds.push_to_hub("cai-conversation-dev")
