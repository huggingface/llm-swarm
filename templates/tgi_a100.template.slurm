#!/bin/bash
#SBATCH --job-name=llm-swarm
#SBATCH --partition hopper-prod
#SBATCH --gpus=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH -o slurm/logs/%x_%j.out

# For HF cluster internal users: Check if /fsx directory exists
if [ -d "/fsx/.cache" ]; then
    export volume="/fsx/.cache"
else
    export volume=".cache"
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
if [ -z "$HUGGING_FACE_HUB_TOKEN" ]; then
    # try reading from file
    export HUGGING_FACE_HUB_TOKEN=$(cat ~/.cache/huggingface/token)
fi
echo "Starting TGI container port $PORT"
echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> {{slurm_hosts_path}}
sudo docker run \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    --gpus "\"device=$CUDA_VISIBLE_DEVICES"\" \
    --shm-size 1g \
    -v $volume:/data -p $PORT:80 \
    ghcr.io/huggingface/text-generation-inference \
    --model-id $model \
    --max-concurrent-requests 2000 \
    --max-total-tokens {{model_max_length}} \
    --max-input-length {{model_input_length}} \
    --max-batch-prefill-tokens {{model_max_length}} \

echo "End of job"