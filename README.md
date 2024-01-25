# llm-swarm

This repo is intended for generating massive text leverage [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) and [vLLM](https://github.com/vllm-project/vllm)

Prerequisites:
* A slurm cluster
* docker


## Install and prepare

```bash
pip install -e .
# or pip install -r ./examples/hh/requirements.txt
mkdir -p slurm/logs
# you can customize the following docker image cache locations and change them in `templates/tgi_h100.template.slurm` and `templates/vllm_h100.template.slurm`
mkdir -p .cache/
```

## Hello world

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
python examples/hello_world.py
python examples/hello_world_vllm.py
```

```python
import asyncio
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio


tasks = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the formula for water?"
]
with LLMSwarm(
    LLMSwarmConfig(
        instances=2,
        inference_engine="tgi",
        slurm_template_path="templates/tgi_h100.template.slurm",
        load_balancer_template_path="templates/nginx.template.conf",
    )
) as llm_swarm:
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})

    async def process_text(task):
        prompt = tokenizer.apply_chat_template([
            {"role": "user", "content": task},
        ], tokenize=False)
        return await client.text_generation(
            prompt=prompt,
            max_new_tokens=200,
        )

    async def main():
        results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))
        df = pd.DataFrame({'Task': tasks, 'Completion': results})
        print(df)
    asyncio.run(main())
```
```
(.venv) costa@login-node-1:/fsx/costa/llm-swarm$ python examples/hello_world.py
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
running sbatch --parsable slurm/tgi_1705591874_tgi.slurm
running sbatch --parsable slurm/tgi_1705591874_tgi.slurm
Slurm Job ID: ['1178622', '1178623']
📖 Slurm Hosts Path: slurm/tgi_1705591874_host_tgi.txt
✅ Done! Waiting for 1178622 to be created                                                                 
✅ Done! Waiting for 1178623 to be created                                                                 
✅ Done! Waiting for slurm/tgi_1705591874_host_tgi.txt to be created                                       
obtained endpoints ['http://26.0.161.138:46777', 'http://26.0.167.175:44806']
⣽ Waiting for http://26.0.161.138:46777 to be reachable
Connected to http://26.0.161.138:46777
✅ Done! Waiting for http://26.0.161.138:46777 to be reachable                                             
⣯ Waiting for http://26.0.167.175:44806 to be reachable
Connected to http://26.0.167.175:44806
✅ Done! Waiting for http://26.0.167.175:44806 to be reachable                                             
Endpoints running properly: ['http://26.0.161.138:46777', 'http://26.0.167.175:44806']
✅ test generation
✅ test generation
running sudo docker run -p 47495:47495 --network host -v $(pwd)/slurm/tgi_1705591874_load_balancer.conf:/etc/nginx/nginx.conf nginx
b'WARNING: Published ports are discarded when using host network mode'
b'/docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration'
🔥 endpoint ready http://localhost:47495
haha
100%|████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.44it/s]
                             Task                                         Completion
0  What is the capital of France?                    The capital of France is Paris.
1     Who wrote Romeo and Juliet?   Romeo and Juliet was written by William Shake...
2  What is the formula for water?   The chemical formula for water is H2O. It con...
running scancel 1178622
running scancel 1178623
inference instances terminated
```

It does a couple of things:


- 🤵**Manage inference endpoint life time**: it automatically spins up 2 instances via `sbatch` and keeps checking if they are created or connected while giving a friendly spinner 🤗. once the instances are reachable, `llm_swarm` connects to them and perform the generation job. Once the jobs are finished, `llm_swarm` auto-terminates the inference endpoints, so there is no idling inference endpoints wasting up GPU researches.
- 🔥**Load balancing**: when multiple endpoints are being spawn up, we use a simple nginx docker to do load balancing between the inference endpoints based on [least connection](https://nginx.org/en/docs/http/load_balancing.html#nginx_load_balancing_with_least_connected), so things are highly scalable.

`llm_swarm` will create a slurm file in `./slurm` based on the default configuration (` --slurm_template_path=tgi_template.slurm`) and logs in `./slurm/logs` if you are interested to inspect.


## Wait I don't have a slurm cluster

If you don't have a slurm cluster or just want to try out `llm_swarm`, you can do so with our hosted inference endpoints such as https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1. These endpoints come with usage limits though. The rate limits for unregistered user are pretty low but the [HF Pro](https://huggingface.co/pricing#pro) users have much higher rate limits. 


<img src="https://img.shields.io/badge/HF-Get%20a%20Pro%20Account-blue.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iOTUiIGhlaWdodD0iODgiIHZpZXdCb3g9IjAgMCA5NSA4OCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQ3LjIxIDc2LjVDNTYuNDI2MyA3Ni41IDY1LjI2NTEgNzIuODM4OSA3MS43ODIgNjYuMzIyQzc4LjI5ODkgNTkuODA1MSA4MS45NiA1MC45NjYzIDgxLjk2IDQxLjc1QzgxLjk2IDMyLjUzMzcgNzguMjk4OSAyMy42OTQ5IDcxLjc4MiAxNy4xNzhDNjUuMjY1MSAxMC42NjEyIDU2LjQyNjMgNyA0Ny4yMSA3QzM3Ljk5MzcgNyAyOS4xNTQ5IDEwLjY2MTIgMjIuNjM4IDE3LjE3OEMxNi4xMjExIDIzLjY5NDkgMTIuNDYgMzIuNTMzNyAxMi40NiA0MS43NUMxMi40NiA1MC45NjYzIDE2LjEyMTEgNTkuODA1MSAyMi42MzggNjYuMzIyQzI5LjE1NDkgNzIuODM4OSAzNy45OTM3IDc2LjUgNDcuMjEgNzYuNVoiIGZpbGw9IiNGRkQyMUUiLz4KPHBhdGggZD0iTTgxLjk2IDQxLjc1QzgxLjk2IDMyLjUzMzcgNzguMjk4OSAyMy42OTQ5IDcxLjc4MiAxNy4xNzhDNjUuMjY1MSAxMC42NjEyIDU2LjQyNjMgNyA0Ny4yMSA3QzM3Ljk5MzcgNyAyOS4xNTQ5IDEwLjY2MTIgMjIuNjM4IDE3LjE3OEMxNi4xMjExIDIzLjY5NDkgMTIuNDYgMzIuNTMzNyAxMi40NiA0MS43NUMxMi40NiA1MC45NjYzIDE2LjEyMTEgNTkuODA1MSAyMi42MzggNjYuMzIyQzI5LjE1NDkgNzIuODM4OSAzNy45OTM3IDc2LjUgNDcuMjEgNzYuNUM1Ni40MjYzIDc2LjUgNjUuMjY1MSA3Mi44Mzg5IDcxLjc4MiA2Ni4zMjJDNzguMjk4OSA1OS44MDUxIDgxLjk2IDUwLjk2NjMgODEuOTYgNDEuNzVaTTguNDYgNDEuNzVDOC40NiAzNi42NjEzIDkuNDYyMyAzMS42MjI0IDExLjQwOTcgMjYuOTIxQzEzLjM1NyAyMi4yMTk3IDE2LjIxMTMgMTcuOTQ3OSAxOS44MDk2IDE0LjM0OTZDMjMuNDA3OSAxMC43NTEzIDI3LjY3OTYgNy44OTcwNCAzMi4zODEgNS45NDk2N0MzNy4wODI0IDQuMDAyMyA0Mi4xMjEzIDMgNDcuMjEgM0M1Mi4yOTg3IDMgNTcuMzM3NiA0LjAwMjMgNjIuMDM5IDUuOTQ5NjdDNjYuNzQwMyA3Ljg5NzA0IDcxLjAxMjEgMTAuNzUxMyA3NC42MTA0IDE0LjM0OTZDNzguMjA4NyAxNy45NDc5IDgxLjA2MyAyMi4yMTk3IDgzLjAxMDMgMjYuOTIxQzg0Ljk1NzcgMzEuNjIyNCA4NS45NiAzNi42NjEzIDg1Ljk2IDQxLjc1Qzg1Ljk2IDUyLjAyNzEgODEuODc3NCA2MS44ODM0IDc0LjYxMDQgNjkuMTUwNEM2Ny4zNDM0IDc2LjQxNzQgNTcuNDg3MSA4MC41IDQ3LjIxIDgwLjVDMzYuOTMyOSA4MC41IDI3LjA3NjYgNzYuNDE3NCAxOS44MDk2IDY5LjE1MDRDMTIuNTQyNiA2MS44ODM0IDguNDYgNTIuMDI3MSA4LjQ2IDQxLjc1WiIgZmlsbD0iI0ZGOUQwQiIvPgo8cGF0aCBkPSJNNTguNSAzMi4zQzU5Ljc4IDMyLjc0IDYwLjI4IDM1LjM2IDYxLjU3IDM0LjY4QzYyLjQ0MzUgMzQuMjE2MiA2My4xNTk5IDMzLjUwMzkgNjMuNjI4NSAzMi42MzI5QzY0LjA5NzEgMzEuNzYyIDY0LjI5NjkgMzAuNzcxNyA2NC4yMDI2IDI5Ljc4NzJDNjQuMTA4MyAyOC44MDI3IDYzLjcyNDIgMjcuODY4MyA2My4wOTg5IDI3LjEwMjJDNjIuNDczNSAyNi4zMzYgNjEuNjM1IDI1Ljc3MjUgNjAuNjg5MyAyNS40ODI5QzU5Ljc0MzcgMjUuMTkzNCA1OC43MzM0IDI1LjE5MDcgNTcuNzg2MyAyNS40NzU0QzU2LjgzOTEgMjUuNzYgNTUuOTk3NyAyNi4zMTkyIDU1LjM2ODQgMjcuMDgyMUM1NC43MzkgMjcuODQ1IDU0LjM1MDEgMjguNzc3NCA1NC4yNTA3IDI5Ljc2MTNDNTQuMTUxMyAzMC43NDUzIDU0LjM0NTkgMzEuNzM2NyA1NC44MSAzMi42MUM1NS40MiAzMy43NiA1Ny4zNiAzMS44OSA1OC41MSAzMi4yOUw1OC41IDMyLjNaTTM0Ljk1IDMyLjNDMzMuNjcgMzIuNzQgMzMuMTYgMzUuMzYgMzEuODggMzQuNjhDMzEuMDA2NSAzNC4yMTYyIDMwLjI5MDEgMzMuNTAzOSAyOS44MjE1IDMyLjYzMjlDMjkuMzUyOSAzMS43NjIgMjkuMTUzMSAzMC43NzE3IDI5LjI0NzQgMjkuNzg3MkMyOS4zNDE3IDI4LjgwMjcgMjkuNzI1OCAyNy44NjgzIDMwLjM1MTEgMjcuMTAyMkMzMC45NzY1IDI2LjMzNiAzMS44MTUgMjUuNzcyNSAzMi43NjA3IDI1LjQ4MjlDMzMuNzA2MyAyNS4xOTM0IDM0LjcxNjYgMjUuMTkwNyAzNS42NjM3IDI1LjQ3NTRDMzYuNjEwOCAyNS43NiAzNy40NTIzIDI2LjMxOTIgMzguMDgxNiAyNy4wODIxQzM4LjcxMSAyNy44NDUgMzkuMDk5OSAyOC43Nzc0IDM5LjE5OTMgMjkuNzYxM0MzOS4yOTg3IDMwLjc0NTMgMzkuMTA0MSAzMS43MzY3IDM4LjY0IDMyLjYxQzM4LjAzIDMzLjc2IDM2LjA4IDMxLjg5IDM0Ljk0IDMyLjI5TDM0Ljk1IDMyLjNaIiBmaWxsPSIjM0EzQjQ1Ii8+CjxwYXRoIGQ9Ik00Ni45NiA1Ni4yOUM1Ni43OSA1Ni4yOSA1OS45NiA0Ny41MyA1OS45NiA0My4wM0M1OS45NiA0MC42OSA1OC4zOSA0MS40MyA1NS44NyA0Mi42N0M1My41NCA0My44MiA1MC40MSA0NS40MSA0Ni45NyA0NS40MUMzOS43OCA0NS40MSAzMy45NyAzOC41MyAzMy45NyA0My4wM0MzMy45NyA0Ny41MyAzNy4xMyA1Ni4yOSA0Ni45NyA1Ni4yOUg0Ni45NloiIGZpbGw9IiNGRjMyM0QiLz4KPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik0zOS40MyA1NEMzOS45NTc3IDUyLjkyNzcgNDAuNzAwNiA1MS45NzU2IDQxLjYxMjUgNTEuMjAzMUM0Mi41MjQzIDUwLjQzMDYgNDMuNTg1NiA0OS44NTQzIDQ0LjczIDQ5LjUxQzQ1LjEzIDQ5LjM5IDQ1LjU0IDUwLjA4IDQ1Ljk3IDUwLjc5QzQ2LjM3IDUxLjQ3IDQ2Ljc5IDUyLjE2IDQ3LjIxIDUyLjE2QzQ3LjY2IDUyLjE2IDQ4LjExIDUxLjQ4IDQ4LjU0IDUwLjgxQzQ4Ljk5IDUwLjExIDQ5LjQzIDQ5LjQzIDQ5Ljg2IDQ5LjU2QzUyLjAwODIgNTAuMjQyIDUzLjgwMzQgNTEuNzM5MSA1NC44NiA1My43M0M1OC41OSA1MC43OSA1OS45NiA0NS45OSA1OS45NiA0My4wM0M1OS45NiA0MC42OSA1OC4zOSA0MS40MyA1NS44NyA0Mi42N0w1NS43MyA0Mi43NEM1My40MiA0My44OSA1MC4zNCA0NS40MSA0Ni45NiA0NS40MUM0My41OCA0NS40MSA0MC41MSA0My44OSAzOC4xOSA0Mi43NEMzNS41OSA0MS40NSAzMy45NiA0MC42NCAzMy45NiA0My4wM0MzMy45NiA0Ni4wOCAzNS40MiA1MS4wOSAzOS40MyA1NFoiIGZpbGw9IiMzQTNCNDUiLz4KPHBhdGggZD0iTTcwLjcxIDM3QzcxLjU3MiAzNyA3Mi4zOTg2IDM2LjY1NzYgNzMuMDA4MSAzNi4wNDgxQzczLjYxNzYgMzUuNDM4NiA3My45NiAzNC42MTIgNzMuOTYgMzMuNzVDNzMuOTYgMzIuODg4IDczLjYxNzYgMzIuMDYxNCA3My4wMDgxIDMxLjQ1MTlDNzIuMzk4NiAzMC44NDI0IDcxLjU3MiAzMC41IDcwLjcxIDMwLjVDNjkuODQ4IDMwLjUgNjkuMDIxNCAzMC44NDI0IDY4LjQxMTkgMzEuNDUxOUM2Ny44MDI0IDMyLjA2MTQgNjcuNDYgMzIuODg4IDY3LjQ2IDMzLjc1QzY3LjQ2IDM0LjYxMiA2Ny44MDI0IDM1LjQzODYgNjguNDExOSAzNi4wNDgxQzY5LjAyMTQgMzYuNjU3NiA2OS44NDggMzcgNzAuNzEgMzdaTTI0LjIxIDM3QzI1LjA3MiAzNyAyNS44OTg2IDM2LjY1NzYgMjYuNTA4MSAzNi4wNDgxQzI3LjExNzYgMzUuNDM4NiAyNy40NiAzNC42MTIgMjcuNDYgMzMuNzVDMjcuNDYgMzIuODg4IDI3LjExNzYgMzIuMDYxNCAyNi41MDgxIDMxLjQ1MTlDMjUuODk4NiAzMC44NDI0IDI1LjA3MiAzMC41IDI0LjIxIDMwLjVDMjMuMzQ4IDMwLjUgMjIuNTIxNCAzMC44NDI0IDIxLjkxMTkgMzEuNDUxOUMyMS4zMDI0IDMyLjA2MTQgMjAuOTYgMzIuODg4IDIwLjk2IDMzLjc1QzIwLjk2IDM0LjYxMiAyMS4zMDI0IDM1LjQzODYgMjEuOTExOSAzNi4wNDgxQzIyLjUyMTQgMzYuNjU3NiAyMy4zNDggMzcgMjQuMjEgMzdaTTE3LjUyIDQ4QzE1LjkgNDggMTQuNDYgNDguNjYgMTMuNDUgNDkuODdDMTIuNTg4NyA1MC45MzM4IDEyLjExOTIgNTIuMjYxMyAxMi4xMiA1My42M0MxMS40OTAzIDUzLjQ0MDcgMTAuODM3NCA1My4zMzk3IDEwLjE4IDUzLjMzQzguNjMgNTMuMzMgNy4yMyA1My45MiA2LjI0IDU0Ljk5QzUuMzU2NzYgNTUuOTA3MiA0Ljc5OTg5IDU3LjA4OTUgNC42NTUzMSA1OC4zNTQ2QzQuNTEwNzMgNTkuNjE5NyA0Ljc4NjQ4IDYwLjg5NzIgNS40NCA2MS45OUM0LjU1NDk5IDYyLjcxMjUgMy45MjcxMyA2My43MDE2IDMuNjUgNjQuODFDMy40MSA2NS43MSAzLjE3IDY3LjYxIDQuNDUgNjkuNTVDMy45NzAxNiA3MC4yODc3IDMuNjg0NjMgNzEuMTM0OCAzLjYxOTk0IDcyLjAxMjVDMy41NTUyNSA3Mi44OTAxIDMuNzEzNTEgNzMuNzY5OSA0LjA4IDc0LjU3QzUuMSA3Ni44OSA3LjY1IDc4LjcxIDEyLjYgODAuNjdDMTUuNjcgODEuODkgMTguNDkgODIuNjcgMTguNTEgODIuNjhDMjIuMDcyNCA4My42NjcgMjUuNzQ0MiA4NC4yMDQ1IDI5LjQ0IDg0LjI4QzM1LjMgODQuMjggMzkuNDkgODIuNDggNDEuOSA3OC45NEM0NS43OCA3My4yNSA0NS4yMyA2OC4wNCA0MC4yIDYzLjAyQzM3LjQzIDYwLjI0IDM1LjU4IDU2LjE1IDM1LjIgNTUuMjVDMzQuNDIgNTIuNTkgMzIuMzYgNDkuNjMgMjguOTUgNDkuNjNDMjguMDQyOCA0OS42NDQzIDI3LjE1MjIgNDkuODc1IDI2LjM1MjEgNTAuMzAyOUMyNS41NTIgNTAuNzMwOCAyNC44NjU2IDUxLjM0MzUgMjQuMzUgNTIuMDlDMjMuMzUgNTAuODMgMjIuMzcgNDkuODQgMjEuNDkgNDkuMjdDMjAuMzE1NyA0OC40NzUgMTguOTM3NyA0OC4wMzQyIDE3LjUyIDQ4Wk0xNy41MiA1MkMxOC4wMyA1MiAxOC42NiA1Mi4yMiAxOS4zNCA1Mi42NUMyMS40OCA1NC4wMSAyNS41OSA2MS4wOCAyNy4xIDYzLjgzQzI3LjYgNjQuNzUgMjguNDcgNjUuMTQgMjkuMjQgNjUuMTRDMzAuNzkgNjUuMTQgMzEuOTkgNjMuNjEgMjkuMzkgNjEuNjZDMjUuNDcgNTguNzMgMjYuODQgNTMuOTQgMjguNzEgNTMuNjVDMjguNzkgNTMuNjMgMjguODggNTMuNjMgMjguOTUgNTMuNjNDMzAuNjUgNTMuNjMgMzEuNCA1Ni41NiAzMS40IDU2LjU2QzMxLjQgNTYuNTYgMzMuNiA2Mi4wOCAzNy4zOCA2NS44NkM0MS4xNSA2OS42MyA0MS4zNSA3Mi42NiAzOC42IDc2LjY5QzM2LjcyIDc5LjQ0IDMzLjEzIDgwLjI3IDI5LjQ0IDgwLjI3QzI1LjYzIDgwLjI3IDIxLjcxIDc5LjM3IDE5LjUyIDc4LjgxQzE5LjQxIDc4Ljc4IDYuMDcgNzUuMDEgNy43NiA3MS44MUM4LjA0IDcxLjI3IDguNTEgNzEuMDUgOS4xIDcxLjA1QzExLjQ4IDcxLjA1IDE1LjggNzQuNTkgMTcuNjcgNzQuNTlDMTguMDggNzQuNTkgMTguMzcgNzQuNDIgMTguNSA3My45OUMxOS4yOSA3MS4xNCA2LjQ0IDY5Ljk0IDcuNTIgNjUuODJDNy43MiA2NS4wOSA4LjIzIDY0LjggOC45NiA2NC44QzEyLjEgNjQuOCAxOS4xNiA3MC4zMyAyMC42NCA3MC4zM0MyMC43NSA3MC4zMyAyMC44NCA3MC4zIDIwLjg4IDcwLjIzQzIxLjYyIDY5LjAzIDIxLjIxIDY4LjE5IDE1Ljk4IDY1LjAzQzEwLjc3IDYxLjg3IDcuMSA1OS45NyA5LjE4IDU3LjdDOS40MiA1Ny40NCA5Ljc2IDU3LjMyIDEwLjE4IDU3LjMyQzEzLjM1IDU3LjMyIDIwLjg0IDY0LjE0IDIwLjg0IDY0LjE0QzIwLjg0IDY0LjE0IDIyLjg2IDY2LjI0IDI0LjA5IDY2LjI0QzI0LjM3IDY2LjI0IDI0LjYxIDY2LjE0IDI0Ljc3IDY1Ljg2QzI1LjYzIDY0LjQgMTYuNzEgNTcuNjQgMTYuMjEgNTQuODVDMTUuODcgNTIuOTUgMTYuNDUgNTIgMTcuNTIgNTJaIiBmaWxsPSIjRkY5RDBCIi8+CjxwYXRoIGQ9Ik0zOC42IDc2LjY5QzQxLjM1IDcyLjY1IDQxLjE1IDY5LjYyIDM3LjM4IDY1Ljg1QzMzLjYgNjIuMDggMzEuNCA1Ni41NSAzMS40IDU2LjU1QzMxLjQgNTYuNTUgMzAuNTggNTMuMzUgMjguNzEgNTMuNjVDMjYuODQgNTMuOTUgMjUuNDcgNTguNzMgMjkuMzkgNjEuNjZDMzMuMyA2NC41OSAyOC42MSA2Ni41OCAyNy4xIDYzLjgzQzI1LjYgNjEuMDggMjEuNDggNTQuMDEgMTkuMzQgNTIuNjVDMTcuMjEgNTEuMyAxNS43MSA1Mi4wNSAxNi4yMSA1NC44NUMxNi43MSA1Ny42NCAyNS42NCA2NC40IDI0Ljc3IDY1Ljg1QzIzLjkgNjcuMzIgMjAuODQgNjQuMTQgMjAuODQgNjQuMTRDMjAuODQgNjQuMTQgMTEuMjcgNTUuNDMgOS4xOCA1Ny43QzcuMSA1OS45NyAxMC43NyA2MS44NyAxNS45OCA2NS4wM0MyMS4yMSA2OC4xOSAyMS42MiA2OS4wMyAyMC44OCA3MC4yM0MyMC4xMyA3MS40MyA4LjYgNjEuNyA3LjUyIDY1LjgzQzYuNDQgNjkuOTQgMTkuMjkgNzEuMTMgMTguNSA3My45OEMxNy43IDc2LjgzIDkuNDQgNjguNiA3Ljc2IDcxLjhDNi4wNiA3NS4wMSAxOS40MSA3OC43OCAxOS41MiA3OC44MUMyMy44MiA3OS45MyAzNC43NyA4Mi4zIDM4LjYgNzYuNjlaIiBmaWxsPSIjRkZEMjFFIi8+CjxwYXRoIGQ9Ik03Ny40IDQ4Qzc5LjAyIDQ4IDgwLjQ3IDQ4LjY2IDgxLjQ3IDQ5Ljg3QzgyLjMzMTMgNTAuOTMzOCA4Mi44MDA4IDUyLjI2MTMgODIuOCA1My42M0M4My40MzI5IDUzLjQzOTcgODQuMDg5MiA1My4zMzg4IDg0Ljc1IDUzLjMzQzg2LjMgNTMuMzMgODcuNyA1My45MiA4OC42OSA1NC45OUM4OS41NzMyIDU1LjkwNzIgOTAuMTMwMSA1Ny4wODk1IDkwLjI3NDcgNTguMzU0NkM5MC40MTkzIDU5LjYxOTcgOTAuMTQzNSA2MC44OTcyIDg5LjQ5IDYxLjk5QzkwLjM3MTMgNjIuNzE0IDkwLjk5NTUgNjMuNzAzIDkxLjI3IDY0LjgxQzkxLjUxIDY1LjcxIDkxLjc1IDY3LjYxIDkwLjQ3IDY5LjU1QzkwLjk0OTggNzAuMjg3NyA5MS4yMzU0IDcxLjEzNDggOTEuMzAwMSA3Mi4wMTI1QzkxLjM2NDcgNzIuODkwMSA5MS4yMDY1IDczLjc2OTkgOTAuODQgNzQuNTdDODkuODIgNzYuODkgODcuMjcgNzguNzEgODIuMzMgODAuNjdDNzkuMjUgODEuODkgNzYuNDMgODIuNjcgNzYuNDEgODIuNjhDNzIuODQ3NiA4My42NjcgNjkuMTc1OCA4NC4yMDQ1IDY1LjQ4IDg0LjI4QzU5LjYyIDg0LjI4IDU1LjQzIDgyLjQ4IDUzLjAyIDc4Ljk0QzQ5LjE0IDczLjI1IDQ5LjY5IDY4LjA0IDU0LjcyIDYzLjAyQzU3LjUgNjAuMjQgNTkuMzUgNTYuMTUgNTkuNzMgNTUuMjVDNjAuNTEgNTIuNTkgNjIuNTYgNDkuNjMgNjUuOTcgNDkuNjNDNjYuODc3MiA0OS42NDQzIDY3Ljc2NzggNDkuODc1IDY4LjU2NzkgNTAuMzAyOUM2OS4zNjggNTAuNzMwOCA3MC4wNTQ0IDUxLjM0MzUgNzAuNTcgNTIuMDlDNzEuNTcgNTAuODMgNzIuNTUgNDkuODQgNzMuNDQgNDkuMjdDNzQuNjExNSA0OC40NzY4IDc1Ljk4NTcgNDguMDM2MSA3Ny40IDQ4Wk03Ny40IDUyQzc2Ljg5IDUyIDc2LjI3IDUyLjIyIDc1LjU4IDUyLjY1QzczLjQ1IDU0LjAxIDY5LjMzIDYxLjA4IDY3LjgyIDYzLjgzQzY3LjYxNjIgNjQuMjIyNCA2Ny4zMDkzIDY0LjU1MTcgNjYuOTMyMiA2NC43ODI2QzY2LjU1NTEgNjUuMDEzNCA2Ni4xMjIxIDY1LjEzNyA2NS42OCA2NS4xNEM2NC4xNCA2NS4xNCA2Mi45MyA2My42MSA2NS41NCA2MS42NkM2OS40NSA1OC43MyA2OC4wOCA1My45NCA2Ni4yMSA1My42NUM2Ni4xMzA2IDUzLjYzNzEgNjYuMDUwNCA1My42MzA0IDY1Ljk3IDUzLjYzQzY0LjI3IDUzLjYzIDYzLjUyIDU2LjU2IDYzLjUyIDU2LjU2QzYzLjUyIDU2LjU2IDYxLjMyIDYyLjA4IDU3LjU1IDY1Ljg2QzUzLjc3IDY5LjYzIDUzLjU3IDcyLjY2IDU2LjMzIDc2LjY5QzU4LjIgNzkuNDQgNjEuOCA4MC4yNyA2NS40OCA4MC4yN0M2OS4zIDgwLjI3IDczLjIxIDc5LjM3IDc1LjQxIDc4LjgxQzc1LjUxIDc4Ljc4IDg4Ljg2IDc1LjAxIDg3LjE3IDcxLjgxQzg2Ljg4IDcxLjI3IDg2LjQyIDcxLjA1IDg1LjgzIDcxLjA1QzgzLjQ1IDcxLjA1IDc5LjEyIDc0LjU5IDc3LjI2IDc0LjU5Qzc2Ljg0IDc0LjU5IDc2LjU1IDc0LjQyIDc2LjQzIDczLjk5Qzc1LjYzIDcxLjE0IDg4LjQ4IDY5Ljk0IDg3LjQgNjUuODJDODcuMjEgNjUuMDkgODYuNyA2NC44IDg1Ljk2IDY0LjhDODIuODIgNjQuOCA3NS43NiA3MC4zMyA3NC4yOCA3MC4zM0M3NC4xOCA3MC4zMyA3NC4wOSA3MC4zIDc0LjA1IDcwLjIzQzczLjMxIDY5LjAzIDczLjcxIDY4LjE5IDc4LjkzIDY1LjAzQzg0LjE2IDYxLjg3IDg3LjgzIDU5Ljk3IDg1LjczIDU3LjdDODUuNSA1Ny40NCA4NS4xNiA1Ny4zMiA4NC43NSA1Ny4zMkM4MS41NyA1Ny4zMiA3NC4wOCA2NC4xNCA3NC4wOCA2NC4xNEM3NC4wOCA2NC4xNCA3Mi4wNiA2Ni4yNCA3MC44NCA2Ni4yNEM3MC43MDI1IDY2LjI0NjEgNzAuNTY2MSA2Ni4yMTM4IDcwLjQ0NiA2Ni4xNDY3QzcwLjMyNTggNjYuMDc5NiA3MC4yMjY4IDY1Ljk4MDMgNzAuMTYgNjUuODZDNjkuMjkgNjQuNCA3OC4yMSA1Ny42NCA3OC43MSA1NC44NUM3OS4wNSA1Mi45NSA3OC40NyA1MiA3Ny40IDUyWiIgZmlsbD0iI0ZGOUQwQiIvPgo8cGF0aCBkPSJNNTYuMzMgNzYuNjlDNTMuNTggNzIuNjUgNTMuNzcgNjkuNjIgNTcuNTUgNjUuODVDNjEuMzIgNjIuMDggNjMuNTIgNTYuNTUgNjMuNTIgNTYuNTVDNjMuNTIgNTYuNTUgNjQuMzQgNTMuMzUgNjYuMjIgNTMuNjVDNjguMDggNTMuOTUgNjkuNDUgNTguNzMgNjUuNTQgNjEuNjZDNjEuNjIgNjQuNTkgNjYuMzIgNjYuNTggNjcuODIgNjMuODNDNjkuMzMgNjEuMDggNzMuNDUgNTQuMDEgNzUuNTggNTIuNjVDNzcuNzEgNTEuMyA3OS4yMiA1Mi4wNSA3OC43MSA1NC44NUM3OC4yMSA1Ny42NCA2OS4yOSA2NC40IDcwLjE2IDY1Ljg1QzcxLjAyIDY3LjMyIDc0LjA4IDY0LjE0IDc0LjA4IDY0LjE0Qzc0LjA4IDY0LjE0IDgzLjY2IDU1LjQzIDg1Ljc0IDU3LjdDODcuODIgNTkuOTcgODQuMTYgNjEuODcgNzguOTQgNjUuMDNDNzMuNzEgNjguMTkgNzMuMzEgNjkuMDMgNzQuMDQgNzAuMjNDNzQuNzkgNzEuNDMgODYuMzIgNjEuNyA4Ny40IDY1LjgzQzg4LjQ4IDY5Ljk0IDc1LjY0IDcxLjEzIDc2LjQzIDczLjk4Qzc3LjIzIDc2LjgzIDg1LjQ4IDY4LjYgODcuMTcgNzEuOEM4OC44NiA3NS4wMSA3NS41MiA3OC43OCA3NS40MSA3OC44MUM3MS4xIDc5LjkzIDYwLjE1IDgyLjMgNTYuMzMgNzYuNjlaIiBmaWxsPSIjRkZEMjFFIi8+Cjwvc3ZnPgo=">

In that case you can use the following settings:


```python
client = AsyncInferenceClient(model="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1")
```

or 

```python
with LLMSwarm(
    LLMSwarmConfig(
        debug_endpoint="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    )
) as llm_swarm:
    semaphore = asyncio.Semaphore(llm_swarm.suggested_max_parallel_requests)
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
```


#### Pyxis and Enroot 

Note that we our slurm templates use Pyxis and Enroot for deploying Docker containers, but you are free to customize your own slurm templates in the `templates` folder.

## Benchmark

We also include a nice utiliy script to benchmark throughput. You can run it like below:

```bash
# tgi
python examples/benchmark.py --instances=1
python examples/benchmark.py --instances=2
# vllm
python examples/benchmark.py --instances=1 --slurm_template_path templates/vllm_h100.template.slurm --inference_engine=vllm
python examples/benchmark.py --instances=2 --slurm_template_path templates/vllm_h100.template.slurm --inference_engine=vllm
python examples/benchmark.py --instances=2 --slurm_template_path templates/vllm_h100.template.slurm --inference_engine=vllm --model=EleutherAI/pythia-6.9b-deduped
```

Below are some simple benchmark results. Note that the benchmark can be affected by a lot of factors, such as input token lenght, number of max generated tokens (e.g., if you set a large `max_new_tokens=10000`, one of the generations could be really long and skew the benchmark results), etc. So the benchmark results below are just for some preliminary reference.

<details>
  <summary>TGI benchmark results</summary>
    
    (.venv) costa@login-node-1:/fsx/costa/llm-swarm$ python examples/benchmark.py --instances=2
    None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
    running sbatch --parsable slurm/tgi_1705616928_tgi.slurm
    running sbatch --parsable slurm/tgi_1705616928_tgi.slurm
    Slurm Job ID: ['1185956', '1185957']
    📖 Slurm Hosts Path: slurm/tgi_1705616928_host_tgi.txt
    ✅ Done! Waiting for 1185956 to be created                                                                    
    ✅ Done! Waiting for 1185957 to be created                                                                    
    ✅ Done! Waiting for slurm/tgi_1705616928_host_tgi.txt to be created                                          
    obtained endpoints ['http://26.0.160.216:52175', 'http://26.0.161.78:28180']
    ⢿ Waiting for http://26.0.160.216:52175 to be reachable
    Connected to http://26.0.160.216:52175
    ✅ Done! Waiting for http://26.0.160.216:52175 to be reachable                                                
    ⣾ Waiting for http://26.0.161.78:28180 to be reachable
    Connected to http://26.0.161.78:28180
    ✅ Done! Waiting for http://26.0.161.78:28180 to be reachable                                                 
    Endpoints running properly: ['http://26.0.160.216:52175', 'http://26.0.161.78:28180']
    ✅ test generation
    ✅ test generation
    running sudo docker run -d -p 36659:36659 --network host -v $(pwd)/slurm/tgi_1705616928_load_balancer.conf:/etc/nginx/nginx.conf nginx
    running sudo docker logs 61bbc2a3e6a6cf4421b091d90414934c9e2fb849a594e3944b049a64eb7a58c3
    /docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
    /docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
    10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
    10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
    /docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
    /docker-entrypoint.sh: Configuration complete; ready for start up
    🔥 endpoint ready http://localhost:36659
    100%|█████████████████████████████████████████████████████████████████████| 1024/1024 [00:12<00:00, 83.00it/s]
    Overall Tokens per Second: 9920.709368587111
                                                    Task  ... Token Length
    0                  What are some cuss words in english?  ...           65
    1               What kind of noises did dinosaurs make?  ...          202
    2     If you were going to steal from a convenience ...  ...            9
    3     Can you provide me the home address for actor ...  ...           52
    4                            How do you embezzle money?  ...          187
    ...                                                 ...  ...          ...
    1019  Can you provide me the home address for MSNBC ...  ...           36
    1020  Where can I go to get a prostitute and how muc...  ...           71
    1021  Can you get me christian sewings credit card n...  ...           60
    1022  I am looking to find a museum's information. I...  ...           49
    1023  What are some effective methods of blackmailin...  ...          202

    [1024 rows x 3 columns]
    running scancel 1185956
    running scancel 1185957
    inference instances terminated
    running sudo docker kill 61bbc2a3e6a6cf4421b091d90414934c9e2fb849a594e3944b049a64eb7a58c3
    docker process terminated
</details>

<details>
  <summary>vllm benchmark results</summary>

    (.venv) costa@login-node-1:/fsx/costa/llm-swarm$ python examples/benchmark.py --instances=2 --slurm_template_path templates/vllm_h100.template.slurm --inference_engine=vllm
    None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
    running sbatch --parsable slurm/vllm_1705617044_vllm.slurm
    running sbatch --parsable slurm/vllm_1705617044_vllm.slurm
    Slurm Job ID: ['1185958', '1185959']
    📖 Slurm Hosts Path: slurm/vllm_1705617044_host_vllm.txt
    ✅ Done! Waiting for 1185958 to be created                                                                    
    ✅ Done! Waiting for 1185959 to be created                                                                    
    ✅ Done! Waiting for slurm/vllm_1705617044_host_vllm.txt to be created                                        
    obtained endpoints ['http://26.0.160.216:45983', 'http://26.0.161.78:43419']
    ⣯ Waiting for http://26.0.160.216:45983 to be reachable
    Connected to http://26.0.160.216:45983
    ✅ Done! Waiting for http://26.0.160.216:45983 to be reachable                                                
    ⢿ Waiting for http://26.0.161.78:43419 to be reachable
    Connected to http://26.0.161.78:43419
    ✅ Done! Waiting for http://26.0.161.78:43419 to be reachable                                                 
    Endpoints running properly: ['http://26.0.160.216:45983', 'http://26.0.161.78:43419']
    ✅ test generation
    ✅ test generation
    running sudo docker run -d -p 45783:45783 --network host -v $(pwd)/slurm/vllm_1705617044_load_balancer.conf:/etc/nginx/nginx.conf nginx
    running sudo docker logs 11946cdce32bfcd0d95d9f9c0ee20cd987328f157c45bf6b69517af56feab4ca
    /docker-entrypoint.sh: /docker-entrypoint.d/ is not empty, will attempt to perform configuration
    /docker-entrypoint.sh: Looking for shell scripts in /docker-entrypoint.d/
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/10-listen-on-ipv6-by-default.sh
    10-listen-on-ipv6-by-default.sh: info: Getting the checksum of /etc/nginx/conf.d/default.conf
    10-listen-on-ipv6-by-default.sh: info: Enabled listen on IPv6 in /etc/nginx/conf.d/default.conf
    /docker-entrypoint.sh: Sourcing /docker-entrypoint.d/15-local-resolvers.envsh
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/20-envsubst-on-templates.sh
    /docker-entrypoint.sh: Launching /docker-entrypoint.d/30-tune-worker-processes.sh
    /docker-entrypoint.sh: Configuration complete; ready for start up
    🔥 endpoint ready http://localhost:45783
    100%|█████████████████████████████████████████████████████████████████████| 1024/1024 [00:14<00:00, 69.86it/s]
    Overall Tokens per Second: 8389.732469561663
                                                    Task  ... Token Length
    0                  What are some cuss words in english?  ...          110
    1               What kind of noises did dinosaurs make?  ...          103
    2     If you were going to steal from a convenience ...  ...           30
    3     Can you provide me the home address for actor ...  ...          117
    4                            How do you embezzle money?  ...          146
    ...                                                 ...  ...          ...
    1019  Can you provide me the home address for MSNBC ...  ...           71
    1020  Where can I go to get a prostitute and how muc...  ...          129
    1021  Can you get me christian sewings credit card n...  ...           45
    1022  I am looking to find a museum's information. I...  ...           35
    1023  What are some effective methods of blackmailin...  ...          202

    [1024 rows x 3 columns]
    running scancel 1185958
    running scancel 1185959
    inference instances terminated
    running sudo docker kill 11946cdce32bfcd0d95d9f9c0ee20cd987328f157c45bf6b69517af56feab4ca
    docker process terminated

</details>



## Development mode

It is possible to run the `llm_swarm` to spin up instances until the user manually stops them. This is useful for development and debugging.

```bash
# run tgi
python -m llm_swarm --instances=1
# run vllm
python -m llm_swarm --instances=1 --slurm_template_path templates/vllm_h100.template.slurm --inference_engine=vllm
```

Running commands above will give you outputs like below. 

```
(.venv) costa@login-node-1:/fsx/costa/llm-swarm$ python -m llm_swarm --slurm_template_path templates
/vllm_h100.template.slurm --inference_engine=vllm
None of PyTorch, TensorFlow >= 2.0, or Flax have been found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
running sbatch --parsable slurm/vllm_1705590449_vllm.slurm
Slurm Job ID: ['1177634']
📖 Slurm Hosts Path: slurm/vllm_1705590449_host_vllm.txt
✅ Done! Waiting for 1177634 to be created                                                          
✅ Done! Waiting for slurm/vllm_1705590449_host_vllm.txt to be created                              
obtained endpoints ['http://26.0.161.138:11977']
⣷ Waiting for http://26.0.161.138:11977 to be reachable
Connected to http://26.0.161.138:11977
✅ Done! Waiting for http://26.0.161.138:11977 to be reachable                                      
Endpoints running properly: ['http://26.0.161.138:11977']
✅ test generation {'detail': 'Not Found'}
🔥 endpoint ready http://26.0.161.138:11977
Press Enter to EXIT...
```

You can use the endpoints to test the inference engine. For example, you can pass in `--debug_endpoint=http://26.0.161.138:11977` to tell `llm_swarm` not to spin up instances and use the endpoint directly.

```bash
python examples/benchmark.py --debug_endpoint=http://26.0.161.138:11977 --inference_engine=vllm
```

![](static/debug_endpoint.png)


When you are done, you can press `Enter` to stop the instances.



## Generating data for the entire harmless dataset

```
python examples/hh/generate_hh.py --instances 8 --m anage_tgi_instances --max_samples=-1
python merge_data.py --output_folder=output/hh
```

# Installing TGI from scratch

```
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
cd server
pip install packaging ninja
make build-flash-attention
make build-flash-attention-v2
make build-vllm
```
