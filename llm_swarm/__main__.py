from llm_swarm import LLMSwarmConfig, LLMSwarm
from transformers import HfArgumentParser

parser = HfArgumentParser(LLMSwarmConfig)
isc = parser.parse_args_into_dataclasses()[0]
with LLMSwarm(isc) as llm_swarm:
    while True:
        input("Press Enter to EXIT...")
        break
