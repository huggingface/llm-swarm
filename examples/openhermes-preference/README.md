
# Open Hermes Preference Generation

```bash
python -i examples/openhermes-preference/generate.py

poetry run python -i examples/openhermes-preference/generate.py --debug_endpoint=http://26.0.165.38:47768 --push_to_hub --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --max_samples_per_source_category=2 --temperature=1.0 --do_sample
```

```bash
python -m llm_swarm --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --instances 8

poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --instances=8 \
    --debug_endpoint=http://login-node-1:53999 > output.txt 2> output_error.txt
poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --instances=2 \
    --max_samples=10000 --shuffle --max_new_tokens=512 --repo_id openhermes-dev-512-new-tokens \
    --debug_endpoint=http://login-node-1:39561
poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --instances=2 \
    --max_samples=10000 --shuffle --max_new_tokens=1024 --repo_id openhermes-dev-1024-new-tokens \
    --debug_endpoint=http://login-node-1:39561
poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --instances=2 \
    --max_samples=10000 --shuffle --max_new_tokens=2048 --repo_id openhermes-dev-2048-new-tokens \
    --debug_endpoint=http://login-node-1:39561
poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --instances=2 \
    --max_samples=10000 --shuffle --max_new_tokens=4096 --repo_id openhermes-dev-4096-new-tokens \
    --debug_endpoint=http://login-node-1:39561

python examples/openhermes-preference/generate1.py  --model=NousResearch/Nous-Hermes-2-Yi-34B   --max_new_tokens=512 --repo_id openhermes-dev-512 --push_to_hub --debug_endpoint=http://26.0.165.202:9587


```

## Preference Dataset PairRM Creation

We use the [PairRM](https://huggingface.co/llm-blender/PairRM) model to score pairs of chosen/rejected completions and then switch the preference dataset to those preferred by PairRM.

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

To process the preference data for a single shard, run the following command where you can specify the number of shards and the index of the shard to process.

```bash
python dpo_pair_rm.py --num_shards 20 --shard_index 0
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-2048-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707789379 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-1024-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707788914 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-500-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707788532 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-4096-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707836836 --split train --max_samples=-1


poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-4096-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707858724 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-2048-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707858234 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-1024-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707857773 --split train --max_samples=-1
poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm1.py --path vwxyzjn/openhermes-dev-512-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707857362 --split train --max_samples=-1



poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm.py --path vwxyzjn/openhermes-dev__mistralai_Mixtral-8x7B-Instruct-v0.1__1707245027 --max_samples=10000 --split test_prefs


poetry run python -W ignore examples/openhermes-preference/dpo_pair_rm2.py --path vwxyzjn/openhermes-dev-512-new-tokens__mistralai_Mixtral-8x7B-Instruct-v0.1__1707857362 --split train --max_samples=20
poetry run python -W ignore -i examples/openhermes-preference/dpo_pair_rm2.py --path vwxyzjn/openhermes-dev-512__NousResearch_Nous-Hermes-2-Yi-34B__1707948601 --split train --max_samples=-1
```

To run the PairRM model on all the data shards, one can call the `dpo_pair_rm.py` script in parallel for each shard index.


After all the shard indices for the train/test splits have been processed, the preference dataset shards can be concatenated and the final dataset can be pushed to the hub via:

```bash
python concat_and_push.py --num_shards 20
```
