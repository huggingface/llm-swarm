#!/bin/bash
#SBATCH --job-name=tgi-swarm
#SBATCH --partition=production-cluster
#SBATCH --cpus-per-task=96
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:a100:8
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


touch {{slurm_hosts_path}}

# for each node
for node in $(scontrol show hostnames $SLURM_JOB_NODELIST); do
    # generate a unique unused port for each node
    PORT=$(unused_port)
    IP_ADDRESS=$(srun --nodes=1 --nodelist=$node hostname -I | awk '{print $1}')
    echo "http://$IP_ADDRESS:$PORT" >> {{slurm_hosts_path}} 

    # run the container
    sudo docker run \
        --gpus "\"device=$CUDA_VISIBLE_DEVICES"\" --shm-size 1g \
        -v $volume:/data -p $PORT:80 \
        ghcr.io/huggingface/text-generation-inference:1.1.0 \
        --model-id $model --max-concurrent-requests 530 --max-total-tokens 8192 --max-input-length 7168 --max-batch-prefill-tokens 7168 &  # run in background
done

wait  # wait for all background jobs to finish