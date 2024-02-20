# Guidelines

Here you can find the code used to generate large synthetic datasets like [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia). You need to have a dataset containing prompts, in this case we're using [cosmopedia-100k](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia-100k).

## Setup
Since we want to generate a large volume of textbooks and the generations might take a long time, we save the intermediate generations in `checkpoint_path` and track the progress and throughput with `wandb`.
```bash
pip install wandb
wandb init
```

## Generation
To run the generations on the first 2000 prompts on 2 TGI instances, you can use:
```bash
python ./examples/textbooks/generate_syntehtic_textbooks.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --instances 2 \
    --prompts_dataset "HuggingFaceTB/cosmopedia-100k" \
    --prompt_column prompt \
    --max_samples 2000 \
    --checkpoint_path "./synthetic_data" \
    --checkpoint_interval 1000
```
The output will look like this:
```
(textbooks) loubna@login-node-1:/fsx/loubna/projects/llm-swarm$ python ./examples/textbooks/generate_syntehtic_textbooks.py \
    --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --instances 2 \
    --prompts_dataset "HuggingFaceTB/cosmopedia-100k" \
    --prompt_column prompt \
    --max_samples 2000 \
    --checkpoint_path "./synthetic_data" \
    --checkpoint_interval 1000
{'max_new_tokens': 2500, 'temperature': 0.6, 'top_p': 0.95, 'top_k': 50, 'repetition_penalty': 1.2, 'prompts_dataset': 'HuggingFaceTB/cosmopedia-100k', 'max_samples': 2000, 'start_sample': -1, 'end_sample': -1, 'seed': 42, 'prompt_column': 'prompt', 'shuffle_dataset': False, 'debug': False, 'repo_id': 'HuggingFaceTB/synthetic_data_test', 'checkpoint_path': './synthetic_data', 'checkpoint_interval': 1000, 'wandb_username': 'NAME', 'min_token_length': 150, 'push_to_hub': True, 'per_instance_max_parallel_requests': 500, 'instances': 2, 'inference_engine': 'tgi', 'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1'}
Loading the first 1000 samples...
running sbatch --parsable slurm/tgi_1708388771_tgi.slurm
running sbatch --parsable slurm/tgi_1708388771_tgi.slurm
Slurm Job ID: ['2179705', '2179706']
📖 Slurm hosts path: slurm/tgi_1708388771_host_tgi.txt
✅ Done! Waiting for 2179705 to be created                                                                                                                                                          
📖 Slurm log path: slurm/logs/llm-swarm_2179705.out
✅ Done! Waiting for 2179706 to be created                                                                                                                                                          
📖 Slurm log path: slurm/logs/llm-swarm_2179706.out
✅ Done! Waiting for slurm/tgi_1708388771_host_tgi.txt to be created                                                                                                                                
obtained endpoints [MASKED_ENDPOINTS]                                                                                                                   
⢿ Waiting for [MASKED_ENDPOINTS]  to be reachable
Connected to [MASKED_ENDPOINTS] 
✅ Done! Waiting for [MASKED_ENDPOINTS] to be reachable                                                                                                                            
⣻ Waiting for [MASKED_ENDPOINTS]  to be reachable
Connected to [MASKED_ENDPOINTS] 
✅ Done! Waiting for [MASKED_ENDPOINTS] to be reachable                                                                                                                           
Endpoints running properly: ['[MASKED_ENDPOINTS]', '[MASKED_ENDPOINTS]']
✅ test generation
✅ test generation
running sudo docker run -d -p 44227:44227 --network host -v $(pwd)/slurm/tgi_1708388771_load_balancer.conf:/etc/nginx/nginx.conf nginx
running sudo docker logs b79ac41505de597196ae7825fda2ad8a60d1c66bc6a8b46038a121d8092198c9
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
🔥 endpoint ready http://localhost:44227
wandb: Currently logged in as: NAME. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.16.3
wandb: Run data is saved locally in ./wandb/run-20240220_003007-3jlhm7lw
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run synthetic_data_test_prompt
wandb: ⭐️ View project at https://wandb.ai/NAME/synthetic_data
wandb: 🚀 View run at https://wandb.ai/NAME/v/runs/3jlhm7lw
Will be saving at ./synthetic_data/synthetic_data_test_prompt/data
Processing chunk 0/2
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:27<00:00,  6.79it/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 68625.21 examples/s]
💾 Checkpoint (samples 0-1000) saved at ./synthetic_data/synthetic_data_test_prompt/data/checkpoint_0.json.
Processing chunk 1/2
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [02:30<00:00,  6.64it/s]
Saving the dataset (1/1 shards): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1000/1000 [00:00<00:00, 82626.85 examples/s]
💾 Checkpoint (samples 1000-2000) saved at ./synthetic_data/synthetic_data_test_prompt/data/checkpoint_1000.json.
Done processing and saving all chunks 🎉! Let's get some stats and push to hub...
🏎️💨 Overall Tokens per Second: 5890.90, per instance: 2945.45
Generated 1.57M tokens
Total duration: 0.0h5min 
Saving time: 0.15408611297607422s=0.0025681018829345702min 
Load checkpoints...
Generating train split: 2000 examples [00:00, 13738.53 examples/s]
Filter: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 7032.53 examples/s]
Dataset({
    features: ['prompt', 'text_token_length', 'text', 'seed_data', 'format', 'audience', 'completion', 'token_length'],
    num_rows: 1999
})
📨 Pushing dataset to HuggingFaceTB/synthetic_data_test_prompt
Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 61.16ba/s]
Uploading the dataset shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.74it/s]
Dataset pushed!
1 generations failed
Creating parquet from Arrow format: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 1698.10ba/s]
Uploading the dataset shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.54it/s]
running scancel 2179729
running scancel 2179730
inference instances terminated
```

