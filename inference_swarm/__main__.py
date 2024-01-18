from inference_swarm import InferenceSwarmConfig, InferenceSwarm
from transformers import HfArgumentParser

parser = HfArgumentParser(InferenceSwarmConfig)
isc = parser.parse_args_into_dataclasses()[0]
with InferenceSwarm(isc) as inference_swarm:
    while True:
        input("Press Enter to EXIT...")
        break
