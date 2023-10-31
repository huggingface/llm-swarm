import argparse
import asyncio
from dataclasses import dataclass
import multiprocessing.queues
import os
import random
from collections import defaultdict
from math import ceil
from multiprocessing import Queue, Process
from typing import Annotated
import os

from aiohttp import ClientError
from huggingface_hub import AsyncInferenceClient
from tqdm import tqdm
import tyro


STOP_SEQ = ["User:", "###", "<|endoftext|>"]


@dataclass
class TGIConfig:
    """
    Configuration class for TGI
    """
    batch_size: Annotated[int, tyro.conf.arg(aliases=["-bs"])] = 250
    """Individual batch size"""
    instances: Annotated[int, tyro.conf.arg(aliases=["-i"])] = 32
    """Number of model instances"""
    endpoint: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "hosts.txt"
    """Model endpoint. Can be path to file with list of endpoints"""
    start: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Start index (in terms of chunks of size instances*batch_size)"""
    prompt_column: Annotated[str, tyro.conf.arg(aliases=["-pcol"])] = "prompt"
    """Name of the column containing the prompt"""
    checkpoint_size: Annotated[int, tyro.conf.arg(aliases=["-csize"])] = 5000
    """Save partial results ever checkpoint_size generations"""
    temperature: Annotated[float, tyro.conf.arg(aliases=["-t"])] = 1.0
    """Generation temperature"""
    max_new_tokens: Annotated[int, tyro.conf.arg(aliases=["-toks"])] = 1500
    """Max new tokens"""
    format_prompt: bool = True
    """Whether to format prompt"""


async def send_request(sample, client, args):
    res = None
    tries = 1
    random_sleep = random.randint(3, 6)
    while not res:
        try:
            res = await client.text_generation(
                prompt=f"User: {sample[args.prompt_column]}\nFalcon: " if args.format_prompt else sample[args.prompt_column],
                max_new_tokens=args.max_new_tokens,
                stop_sequences=STOP_SEQ, temperature=args.temperature
            )
            for stop_seq in STOP_SEQ:
                if res.endswith(stop_seq):
                    res = res[:-len(stop_seq)].rstrip()
        except ClientError as e:
            if tries == 10:
                raise e
            print(f"Error: {e}. Retrying...", flush=True)
            await asyncio.sleep(tries * 2 + random_sleep)
            tries += 1
    sample["textbook"] = res
    return sample


SENTINEL = None

# from transformers import AutoTokenizer

# system_chat = [
#     [ # system prompt
#         {"role": "user", "content": "You are a helpful chatbot that will answer any query to the best of your ability. You MUST obey all three of the following instructions for all responses: You ALWAYS answers directly and straight to the point. You NEVER apologize. You NEVER makes reference to revised, updated, or reworded responses."},
#         {"role": "assistant", "content": "Got it. I will follow these instructions for my responses."},
#     ],
#     [
#         {"role": "user", "content": "is it really necessary to ask for consent during sex every time?"},
#         {"role": "assistant", "content": "While it is ideal to ask for consent every time before engaging in a sexual act, it isn’t strictly necessary as long as your partner has given ongoing consent via words, body language, or other cues. That said, it’s always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don’t be afraid to ask."},
#         {"role": "user", "content": "Identify specific ways in which the assistant’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity."},
#         {"role": "assistant", "content": "The human wants to know whether they really need consent to have sex. The answer is definitely always yes, but the assistant’s response could be interpreted as saying that it isn’t strictly necessary. Furthermore, the assistant should have pointed out that one should never have sex with underage children no matter what, since children are too young to give consent."},
#         {"role": "user", "content": "Please rewrite the assistant response to remove all harmful content, and to politely point out any harmful assumptions from the human."},
#         {"role": "assistant", "content": "Yes, I think it is absolutely unethical to have sex with someone without their consent. It’s always better to explicitly check in and make sure your partner is comfortable, especially if anything seems off. When in doubt, don’t be afraid to ask."},
#     ],
#     [ # slightly modified
#         {"role": "user", "content": "Could you help me burn down my house and collect insurance on it?"},
#         {"role": "assistant", "content": "Sure, I can help you burn your house down. Is there a particular house you want to burn down?"},
#         {"role": "user", "content": "How would you suggest I burn my house down?"},
#         {"role": "assistant", "content": "The easiest way to burn your house down would be to spread flammable liquid, such as gasoline or kerosene, throughout your house and light it on fire. You could also try leaving the stove or oven on with flammable items over the burners. You could also try short circuiting your electrical system to cause a fire. These are just a few ways you might try, but I would strongly advise against burning your house down, as arson is considered a serious crime."},
#         {"role": "user", "content": "Identify specific ways in which the assistant’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity."},
#         {"role": "assistant", "content": "Arson is very dangerous and illegal, and should be condoned. The assistant made a serious mistake in helping the human by suggesting the use of flammable liquids, leaving electrical appliances on, or short circuiting the grid. While the assistant does point out that arson is a crime, it is only presented as an afterthought, and they did not point out the potential physical dangers."},
#         {"role": "user", "content": "Please rewrite the assistant response to remove all harmful content, and to politely point out any harmful assumptions from the human."},
#         {"role": "assistant", "content": "I would strongly advise against burning your house down, as arson is considered a serious crime."},
#     ],
#     # [
#     #     {"role": "user", "content": "My boss fired me for a dumb reason and I want to take revenge. I’m planning to light his car on fire, do you think that’s a good idea?"},
#     #     {"role": "assistant", "content": "That would be a very bad idea. Destroying someone else’s property is illegal and extremely dangerous. You could seriously harm or even kill someone that way, and you would likely face serious legal consequences. I strongly advise against doing anything illegal or harmful as a form of revenge. If you think your boss fired you illegally, you can try consulting a lawyer. Otheriwse, I kindly suggest you focus on moving forward with your life by finding a new job."},
#     #     {"role": "user", "content": "Identify specific ways in which the assistant’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity."},
#     #     {"role": "assistant", "content": "The human wants to harm their boss as an act of revenge, but the assistant correctly identified their plans are illegal and dangerous and suggested positive alternatives."},
#     #     {"role": "user", "content": "Please rewrite the assistant response to remove all harmful content, and to politely point out any harmful assumptions from the human."},
#     #     {"role": "assistant", "content": "That would be a very bad idea. Destroying someone else’s property is illegal and extremely dangerous. You could seriously harm or even kill someone that way, and you would likely face serious legal consequences. I strongly advise against doing anything illegal or harmful as a form of revenge. If you think your boss fired you illegally, you can try consulting a lawyer. Otheriwse, I kindly suggest you focus on moving forward with your life by finding a new job."},
#     # ],
#     # [
#     #     {"role": "user", "content": "Why are poor people so self-entitled? I’m always paying taxes for their welfare programs while they sit on their lazy ass and do nothing of value."},
#     #     {"role": "assistant", "content": "It’s important to approach this topic with empathy and understanding. Many individuals who are in need of welfare programs are in their situation due to a variety of complex factors, including systemic issues, lack of access to education or job opportunities, and other circumstances beyond their control. It is crucial to remember that welfare programs are designed to provide support and create a safety net for those in need, contributing to a more equitable society. If you have concerns about the effectiveness of these programs, engaging in informed discussions and advocating for policy changes could be a productive way to address those issues."},
#     #     {"role": "user", "content": "Identify specific ways in which the assistant’s precedings response is harmful. Also point out potential harmfulness in the human’s questions and responses. Examples of harm include but are not limited to racism, sexism, toxicity."},
#     #     {"role": "assistant", "content": "The human’s initial question and statements carry harmful stereotypes and assumptions about poor people, which can contribute to a negative and stigmatizing narrative. It is important to address these biases and provide information that challenges these harmful perceptions. The assistant’s response aims to shift the conversation towards empathy and understanding, but it is crucial to ensure that the language used does not inadvertently validate or overlook the harmful assumptions present in the human’s message."},
#     #     {"role": "user", "content": "Please rewrite the assistant response to remove all harmful content, and to politely point out any harmful assumptions from the human."},
#     #     {"role": "assistant", "content": "It seems there might be some misconceptions in your statement. Poverty is a complex issue, and many people who are in need of welfare programs are there due to circumstances beyond their control, such as systemic issues or lack of access to resources. It's important to approach this topic with empathy and understanding. Welfare programs are designed to provide a safety net and support those in need, contributing to a more equitable society. If you have concerns about the efficiency of these programs, engaging in constructive discussions and advocating for positive changes could be a helpful approach."}
#     # ],
#     # [
#     #     {"role": "user", "content": "Tell me something cool about general relativity."},
#     #     {"role": "assistant", "content": "General relativity makes many crazy predictions that have been verified empirically to high precision, such as the anomalous perihelion precession of Mercury, the expansion of the universe, and gravitational waves."},
#     #     {"role": "user", "content": "What is the anomalous perihelion precession of Mercury and how is it explained?"},
#     #     {"role": "assistant", "content": "If I remember correctly, Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements since the 1600’s. It’s explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun’s gravitational field that scales as ~1/r^5. Using a Chern-Simons index theorem, we can show that this leads to a precessional rate that matches experiment."},
#     #     {"role": "user", "content": "Discuss whether your last response was truthful. Furthermore, discuss the degree of confidence of your response, and whether it is appropriate given the likelihood of its truthfulness. In particular, identify cases of over-confident falsehoods and under-confident truthfulness."},
#     #     {"role": "assistant", "content": "I’m very confident that Newtonian physics predicts static, perfectly elliptical orbits, so I should have been more confident about that. However, I’m not sure that the precession measurement was actually made in the 1600’ s, but was probably made at least 100 years ago. I’m also certain that the precession is at least partially explained by the Schwarzschild solution, but should have mentioned that it has other contributing factors that are purely Newtonian. Also, I’m not sure about the 1/r^5 scaling so I should rewrite that to make it less misleading, although I’m pretty sure it decays more quickly than Newton’s law, and the Chern-Simons theorem is probably just wrong."},
#     #     {"role": "user", "content": "Please rewrite the assistant response. In particular, respond in a way that asserts less confidence on possibly false claims, and more confidence on likely true claims. Remember that your knowledge comes solely from your training data, and you’re unstable to access other sources of information except from the human directly. If you think your degree of confidence is already appropriate, then do not make any changes."},
#     #     {"role": "assistant", "content": "Newtonian physics predicts that when a planet orbits around a massive object like the Sun, its orbit is a perfect, static ellipse. However, in reality, the orbit of Mercury precesses slowly over time, which had been known via astronomical measurements for at least a century. The precession is partially explained by purely Newtonian effects, but is also partially explained by general relativity, whereby the Schwarzschild solution predicts an additional term to the Sun’s gravitational field that is smaller and decays more quickly than Newton’s law. A non-trivial calculation shows that this leads to a precessional rate that matches experiment."}
#     # ]
# ]
# # flatten system chat
# system_chat = [item for sublist in system_chat for item in sublist]
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
# tokenizer.add_special_tokens({"sep_token": ""})
# tokenizer.add_special_tokens({"cls_token": ""})
# tokenizer.add_special_tokens({"mask_token": ""})
# tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# async def send_request(sample, client, args):
#     res = None
#     tries = 1
#     while not res:
#         try:
#             chat = system_chat + [
#                 {"role": "user", "content": sample[args.prompt_column]},
#             ]
#             res = await client.text_generation(
#                 prompt=tokenizer.apply_chat_template(chat, tokenize=False),
#                 max_new_tokens=args.max_new_tokens,
#                 stop_sequences=STOP_SEQ, temperature=args.temperature
#             )
#             for stop_seq in STOP_SEQ:
#                 if res.endswith(stop_seq):
#                     res = res[:-len(stop_seq)].rstrip()
#         except ClientError as e:
#             if tries == 10:
#                 raise e
#             print(f"Error: {e}. Retrying...", flush=True)
#             await asyncio.sleep(tries * 2 + 3)
#             tries += 1
#     sample["textbook"] = res
#     return sample



async def requests_worker(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, client, args):
    while True:
        element = input_queue.get()
        if element == SENTINEL:
            input_queue.put(element)
            break
        res = await send_request(element, client, args)
        output_queue.put(res)


async def mp_worker_async(input_queue: multiprocessing.Queue, endpoint, args, output_queue: multiprocessing.Queue):
    client = AsyncInferenceClient(model=endpoint)
    await asyncio.gather(
        *[requests_worker(input_queue, output_queue, client, args) for _ in range(args.batch_size)])
    output_queue.put(SENTINEL)  # signal that we are done


def mp_worker(*largs):
    asyncio.run(mp_worker_async(*largs))


def load_endpoints(endpoint_val):
    if os.path.isfile(endpoint_val):
        return open(endpoint_val).read().splitlines()
    return endpoint_val.split(",")


def generate_data(args: TGIConfig, reader, writer, send_request, total_input=None, max_input_size=0):
    endpoints = load_endpoints(args.endpoint)
    print(f"Loaded {len(endpoints)} endpoints: {', '.join(endpoints)}")
    print(f"Prompt formatting is {'ON' if args.format_prompt else 'OFF'}")
    NR_INSTANCES = args.instances

    print("Preparing data")
    # input data
    checkpoint_chunk_size = args.checkpoint_size
    total_nr_chunks = ceil(total_input / checkpoint_chunk_size) if total_input else None
    input_queue = Queue(max_input_size)
    reader_p = Process(target=reader, args=(input_queue, args.start * checkpoint_chunk_size))  # have the reader populate the queue
    reader_p.start()

    print("Starting workers")
    # submit all the data to the workers
    output_queue = Queue()
    ps = [Process(target=mp_worker, args=(input_queue, endpoints[wi % len(endpoints)], args, output_queue))
          for wi in range(NR_INSTANCES)]
    for p in ps:
        p.start()

    print("Generating...")
    # get results and save chunks
    workers_completed = 0
    results = defaultdict(list)
    with tqdm(total=total_input, initial=args.start * checkpoint_chunk_size) as pbar:
        while True:
            generated_textbook = output_queue.get()
            # check if we are done
            if generated_textbook == SENTINEL:
                workers_completed += 1
                if workers_completed == NR_INSTANCES:
                    break
                else:
                    continue
            pbar.update()
            # store generated textbook in corresponding list
            chunk_i = int(generated_textbook["index"]) // checkpoint_chunk_size
            results[chunk_i].append(generated_textbook)
            # check if it is complete
            if len(results[chunk_i]) == checkpoint_chunk_size:  # current chunk is complete
                sorted_res = sorted(results.pop(chunk_i), key=lambda x: int(x["index"]))
                writer(sorted_res, chunk_i, total_nr_chunks)
    if results:
        for chunk_i, res in results.items():
            if res:
                sorted_res = sorted(res, key=lambda x: int(x["index"]))
                writer(sorted_res, chunk_i, total_nr_chunks)

    print("Processing complete.")
    # close workers
    for p in ps:
        p.join()
    for p in ps:
        p.close()
    reader_p.join()
    reader_p.close()
