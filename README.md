# TGI-swarm

This repo is intended for generating massive texts leverage [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference)

Prerequisites:
* A slurm cluster


## Install and prepare

```bash
mkdir -p slurm/logs
pip install -e .
```

## Hello world

```bash
sbatch tgi.slurm
```

This will generate log files in `slurm/logs` and also `hosts.txt` with the list of nodes used for the job.

```bash
pip install -r ./examples/hh/requirements.txt
python ./examples/hh/generate_hh_simple.py
```
```
Loaded 1 endpoints: http://26.0.149.1:45920
Prompt formatting is ON
Preparing data
Starting workers
Loading dataset
Generating...
24it [00:30,  5.02s/it]

## after a minute or so
64it [02:24,  2.26s/it]
Saving chunk 1/None
Processing complete.
```

Then you should be able to see some sample outputs in `output/hh_simple`

# Installing TGI from scratch

```
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
cd server
pip install packaging ninja
make build-flash-attention
make build-flash-attention-v2
make build-vllm
```
