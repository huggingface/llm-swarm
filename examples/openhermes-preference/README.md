

python -i examples/openhermes-preference/generate.py

poetry run python -i examples/openhermes-preference/generate.py --debug_endpoint=http://26.0.165.38:47768 --push_to_hub --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --max_
samples_per_source_category=2 --temperature=1.0 --do_sample


```
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