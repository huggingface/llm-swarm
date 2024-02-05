

python -i examples/openhermes-preference/generate.py

poetry run python -i examples/openhermes-preference/generate.py --debug_endpoint=http://26.0.165.38:47768 --push_to_hub --model=mistralai/Mixtral-8x7B-Instruct-v0.1 --max_
samples_per_source_category=2 --temperature=1.0 --do_sample


poetry run python \
    examples/openhermes-preference/generate.py \
    --push_to_hub \
    --model=mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --max_samples_per_source_category=-1 \
    --temperature=1.0 \
    --do_sample \
    --debug_endpoint=http://26.0.160.100:15141