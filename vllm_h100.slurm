#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --partition hopper-prod
#SBATCH --gpus=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs_vllm/%x_%j.out


export model=mistralai/Mistral-7B-Instruct-v0.1
export HUGGING_FACE_HUB_TOKEN=HF_TOKEN
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "You should provide a Hugging Face token in HF_TOKEN."
  exit 1
fi

export PORT=8000:8000

# to turn on TP use `--tensor-parallel-size 8` for example
# you can cache the container image in /fsx/.../vllm+vllm-openai+latest.sqsh
srun --container-image='vllm#vllm-openai:latest' \
     --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
     --container-mounts="/scratch:/root/.cache/huggingface" \
     --no-container-mount-home \
     --qos normal \
     python3 -m vllm.entrypoints.api_server --model $model
