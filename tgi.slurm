#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=production-cluster
#SBATCH --ntasks-per-node=1
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

export PORT=$(unused_port)
#export PORT=25677
#agent_port=$(unused_port)
#master_port=$(unused_port)

#envsubst < /fsx/olivier_dehaene/configs/base-config.river > /fsx/olivier_dehaene/configs/config-$SLURM_JOB_ID.river
#AGENT_MODE=flow /fsx/olivier_dehaene/grafana-agent-linux-amd64 run /fsx/olivier_dehaene/configs/config-$SLURM_JOB_ID.river --storage.path /fsx/olivier_dehaene/agent-data/$SLURM_JOB_ID --server.http.listen-addr "127.0.0.1:$agent_port" &

export MODEL_PROMPT_TEMPLATE="<s>{{#each messages}}{{#ifUser}}[INST] {{#if @first}}{{#if @root.preprompt}}{{@root.preprompt}}\n{{/if}}{{/if}} {{content}} [/INST]{{/ifUser}}{{#ifAssistant}}{{content}}</s> {{/ifAssistant}}{{/each}}"


touch $PWD/hosts.txt
srun echo "http://$(hostname -I | awk '{print $1}'):$PORT" >> $PWD/hosts.txt &&  sudo docker run -e HUGGING_FACE_HUB_TOKEN=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 \
--gpus "\"device=$CUDA_VISIBLE_DEVICES"\" --shm-size 1g \
-v $volume:/data -p $PORT:80 \
ghcr.io/huggingface/text-generation-inference \
--model-id $model  --max-concurrent-requests 530 --max-total-tokens 4096 