
# Open Hermes Preference Generation

```bash
python -i examples/openhermes-preference/generate.py

poetry run python -i examples/openhermes-preference/generate.py --debug_endpoint=http://26.0.165.38:47768 --push_to_hub --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --max_
samples_per_source_category=2 --temperature=1.0 --do_sample
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
```

To run the PairRM model on all the data shards, one can call the `dpo_pair_rm.py` script in parallel for each shard index.


After all the shard indices for the train/test splits have been processed, the preference dataset shards can be concatenated and the final dataset can be pushed to the hub via:

```bash
python concat_and_push.py --num_shards 20
```
