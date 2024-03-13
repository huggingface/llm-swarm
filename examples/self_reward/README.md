# Self-Rewarding Language Models

This example shows how to generate the dataset for self-rewarding language models (https://huggingface.co/papers/2401.10020).

```
python examples/self_reward/generate.py --push_to_hub
```

Specifically, given a instruction, `generate.py` will generate `--candidates=4` candidate responses and use a LLM-as-a-judge prompt to score them. Then, it choses the response with the highest score as the `chosen` response and the response with the lowest score as the `rejected` response. The `chosen` and `rejected` pairs can then be used to train self-rewarding language models via preference learning (e.g., direct preference optimization).