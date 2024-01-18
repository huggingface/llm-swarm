#!/bin/bash
#SBATCH --job-name=inference-swarm
#SBATCH --partition hopper-prod
#SBATCH --gpus=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out

# For HF cluster internal users: Check if /fsx directory exists
if [ -d "/fsx/.cache/vllm" ]; then
    export volume="/fsx/.cache/vllm"
else
    export volume=".cache/vllm"
fi
export model=mistralai/Mistral-7B-Instruct-v0.1

function unused_port() {
    N=${1:-1}
    comm -23 \
        <(seq "1025" "65535" | sort) \
        <(ss -Htan |
            awk '{print $4}' |
            cut -d':' -f2 |
            sort -u) |
        shuf |
        head -n "$N"
}
export PORT=$(unused_port)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
  echo "You should provide a Hugging Face token in HUGGING_FACE_HUB_TOKEN."
  exit 1
fi

echo "Starting TGI container port $PORT"
echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> {{slurm_hosts_path}}
# you can cache the container image in /fsx/.../huggingface+text-generation-inference.sqsh'
srun --container-image='vllm/vllm-openai:latest' \
     --container-env=HUGGING_FACE_HUB_TOKEN,PORT \
     --container-mounts="$volume:/root/.cache/huggingface" \
     --no-container-mount-home \
     --qos normal \
     python3 -m vllm.entrypoints.api_server --model $model --port $PORT
    
echo "End of job"