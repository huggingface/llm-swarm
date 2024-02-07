# Guidelines

## OpenStax generations
```bash
python ./examples/textbooks/generate_openstax.py --tgi_instances 5 --max_samples 50000
```
```
(textbooks) loubna@login-node-1:/fsx/loubna/projects/llm-swarm$ python ./examples/textbooks/generate_openstax.py             
Args(max_samples=50000, max_new_tokens=2100, temperature=0.6, top_p=0.95, top_k=50, repetition_penalty=1.2, prompt_column='prompt', repo_id='HuggingFaceTB/o
penstax_generations_f', push_to_hub=True)                                                                                                                   
running sbatch --parsable slurm/tgi_1706882419_tgi.slurm                                                                                            
running sbatch --parsable slurm/tgi_1706882419_tgi.slurm                                                                                            
running sbatch --parsable slurm/tgi_1706882419_tgi.slurm                                                                                            
running sbatch --parsable slurm/tgi_1706882419_tgi.slurm                                                                                            
running sbatch --parsable slurm/tgi_1706882419_tgi.slurm                                                                                                    
Slurm Job ID: ['1625161', '1625162', '1625163', '1625164', '1625165']                                                                                       
📖 Slurm hosts path: slurm/tgi_1706882419_host_tgi.txt                                                                                                      
✅ Done! Waiting for 1625161 to be created                                                                                                                  
📖 Slurm log path: slurm/logs/llm-swarm_1625161.out                                                                                                         
✅ Done! Waiting for 1625162 to be created                                                                                                                  
📖 Slurm log path: slurm/logs/llm-swarm_1625162.out                                                                                                         
✅ Done! Waiting for 1625163 to be created                                                                                                          ...                                                                                       
obtained endpoints ['http://26.0.172.147:22882', 'http://26.0.175.19:32335', 'http://26.0.172.142:53787', 'http://26.0.172.252:44532', 'http://26.0.173.7:28
863']                                                                                                                                                       
⣽ Waiting for http://26.0.172.147:22882 to be reachable                                                                                                     
Connected to http://26.0.172.147:22882                                                                                                                      
✅ Done! Waiting for http://26.0.172.147:22882 to be reachable                                                                                              
⣟ Waiting for http://26.0.175.19:32335 to be reachable                                                                                                      
Connected to http://26.0.175.19:32335                                                                                                               ...              
Endpoints running properly: ['http://26.0.172.147:22882', 'http://26.0.175.19:32335', 'http://26.0.172.142:53787', 'http://26.0.172.252:44532', 'http://26.0
.173.7:28863']
✅ test generation
✅ test generation
✅ test generation
✅ test generation
✅ test generation
running sudo docker run -d -p 37809:37809 --network host -v $(pwd)/slurm/tgi_1706882419_load_balancer.conf:/etc/nginx/nginx.conf nginx
running sudo docker logs 520880ab48aa6563ebf2a85f5af02d3bfce7780fb498ec06e4bde92af924ab56
/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
/docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
/docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
/docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
/docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
/docker-entrypoint.sh: Configuration complete; ready for start up
🔥 endpoint ready http://localhost:37809
llm_swarm.suggested_max_parallel_requests was 2500
  3%|███▏                                                                                                              | 1393/50000 [02:06<38:47, 20.88it/s]
Request failed, retrying in 5 seconds... (Attempt 1/3)
 47%|█████████████████████████████████████████████████████▏                                                           | 23550/50000 [22:09<23:24, 18.84it/s]
Request failed, retrying in 5 seconds... (Attempt 1/3)
 96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▍    | 47997/50000 [44:20<02:19, 14.35it/s]
Request failed, retrying in 5 seconds... (Attempt 2/3)
96%|████████████████████████████████████████████████████████████████████████████████████████████████████████████▋    | 48104/50000 [44:25<01:34, 20.13it/s]
Max retries reached. Failed to process the request with error Input validation error: `inputs` must have less than 2048 tokens. Given: 2085.
Max retries reached. Failed to process the request with error Input validation error: `inputs` must have less than 2048 tokens. Given: 2134.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50000/50000 [45:50<00:00, 18.18it/s]
Overall Tokens per Second: 15596.257617119521
Dataset({
    features: ['prompt', 'unit', 'book title', 'audience', 'completion', 'token_length'],
    num_rows: 50000
})
Creating parquet from Arrow format: 100%|███████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 56.37ba/s]
Uploading the dataset shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.81s/it]
running scancel 1625161
running scancel 1625162
running scancel 1625163
running scancel 1625164
running scancel 1625165
```

## FineWeb generations
Here we want to generate a large amount of prompts from web datasets. Since the generations take a long time, we save intermediate generations in `checkpoint_path` and track the progress and throughput with `wandb`
```bash
pip install wandb
wandb init
```

Then run:
```bash
python /fsx/loubna/projects/llm-swarm/examples/textbooks/generate_fineweb_checkpoint.py \
    --prompts_dataset "HuggingFaceTB/fw_prompts_data_textbook" \
    --prompt_column prompt_textbook_academic \
    --start_sample 0 \
    --end_sample 1_000_000 \
    --checkpoint_path "/fsx/loubna/projects/llm-swarm/fw_data" \
    --repo_id "HuggingFaceTB/fw_generations_textbook_first_1M"
```

## Other
Note: the `generate_rw_textbooks.py` requires installing  `requirements.txt` and  `datatrove` from source.
