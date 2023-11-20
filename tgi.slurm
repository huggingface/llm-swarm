#!/bin/bash
#SBATCH --job-name=tgi-swarm
#SBATCH --partition=production-cluster
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out

export volume=/scratch
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

if [ -z "$HF_TOKEN" ]; then
  echo "You should provide a Hugging Face token in HF_TOKEN."
  exit 1
fi

export PORT=$(unused_port)
export MODEL_PROMPT_TEMPLATE="<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}"


rm -f $PWD/hosts.txt
touch $PWD/hosts.txt

srun echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> $PWD/hosts.txt
sudo docker run \
    -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
    --gpus "\"device=$CUDA_VISIBLE_DEVICES"\" \
    --shm-size 1g \
    -v $volume:/data -p $PORT:80 \
    ghcr.io/huggingface/text-generation-inference \
    --model-id $model \
    --max-concurrent-requests 530 \
    --max-total-tokens 8192 \
    --max-input-length 7168 \
    --max-batch-prefill-tokens 7168
