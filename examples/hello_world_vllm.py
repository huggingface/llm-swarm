import asyncio
import json
import pandas as pd
from llm_swarm import LLMSwarm, LLMSwarmConfig
from huggingface_hub import AsyncInferenceClient
from transformers import AutoTokenizer
from tqdm.asyncio import tqdm_asyncio


tasks = ["What is the capital of France?", "Who wrote Romeo and Juliet?", "What is the formula for water?"]
with LLMSwarm(
    LLMSwarmConfig(
        instances=2,
        inference_engine="vllm",
        slurm_template_path="templates/vllm_h100.template.slurm",
        load_balancer_template_path="templates/nginx.template.conf",
    )
) as llm_swarm:
    client = AsyncInferenceClient(model=llm_swarm.endpoint)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    tokenizer.add_special_tokens({"sep_token": "", "cls_token": "", "mask_token": "", "pad_token": "[PAD]"})

    async def process_text(task):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task},
            ],
            tokenize=False,
        )
        response = await client.post(
            json={
                "prompt": prompt,
                "max_tokens": 200,
            }
        )
        res = json.loads(response.decode("utf-8"))["text"][0][len(prompt) :]
        return res

    async def main():
        results = await tqdm_asyncio.gather(*(process_text(task) for task in tasks))
        df = pd.DataFrame({"Task": tasks, "Completion": results})
        print(df)

    asyncio.run(main())
