import asyncio
import multiprocessing.queues
import os
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from multiprocessing import Process, Queue
from typing import Annotated, Callable

import tyro
from huggingface_hub import AsyncInferenceClient
from tqdm import tqdm

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
    checkpoint_size: Annotated[int, tyro.conf.arg(aliases=["-csize"])] = 5000
    """Save partial results ever checkpoint_size generations"""


SENTINEL = None


async def requests_worker(
    send_request: Callable, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, client
):
    while True:
        element = input_queue.get()
        if element == SENTINEL:
            input_queue.put(element)
            break
        res = await send_request(element, client)
        output_queue.put(res)


async def mp_worker_async(
    send_request: Callable, input_queue: multiprocessing.Queue, endpoint, args, output_queue: multiprocessing.Queue
):
    client = AsyncInferenceClient(model=endpoint)
    await asyncio.gather(*[requests_worker(send_request, input_queue, output_queue, client) for _ in range(args.batch_size)])
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
    NR_INSTANCES = args.instances

    print("Preparing data")
    # input data
    checkpoint_chunk_size = args.checkpoint_size
    total_nr_chunks = ceil(total_input / checkpoint_chunk_size) if total_input else None
    input_queue = Queue(max_input_size)
    reader_p = Process(
        target=reader, args=(input_queue, args.start * checkpoint_chunk_size)
    )  # have the reader populate the queue
    reader_p.start()

    print("Starting workers")
    # submit all the data to the workers
    output_queue = Queue()
    ps = [
        Process(target=mp_worker, args=(send_request, input_queue, endpoints[wi % len(endpoints)], args, output_queue))
        for wi in range(NR_INSTANCES)
    ]
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
