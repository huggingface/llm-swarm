import asyncio
import multiprocessing.queues
import os
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from multiprocessing import Process, Queue
from typing import Annotated, Callable, Union, List, Optional, Any

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
    """ Send a single request to the model.
        Get the input from the input queue and put the result in the output queue.
    
    Args:
        send_request (Callable): function to send request to model
        input_queue (multiprocessing.Queue): input queue
        output_queue (multiprocessing.Queue): output queue
        client (AsyncInferenceClient): inference client
    """
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


def load_endpoints(endpoint_val: Union[str, List[str]]):
    """ Return list of endpoints from either a file or a comma separated string
    
    Args:
        endpoint_val (Union[str, List[str]]): either a file path or a comma separated string
    
    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    if os.path.isfile(endpoint_val):
        return open(endpoint_val).read().splitlines()
    return endpoint_val.split(",")


def generate_data(args: TGIConfig,
                  reader: Callable[[multiprocessing.Queue, int], None],
                  writer: Callable[[List[Any], int, int], None],
                  send_request: Callable[[Any, AsyncInferenceClient], Any],
                  closer: Optional[Callable]=None,
                  total_input: Optional[int]=None,
                  max_input_size: int=0):
    """ Control the swarm of workers to generate data
    
    Args:
        args (TGIConfig): configuration
        reader (Callable: (input_queue, start_index) -> None): Reader function – Read the data starting from start_index and put it sample by sample in the input_queue. Put a SENTINEL in the queue to finish.
        send_request (Callable): function to send request to model
        writer (Callable: chunk, chink_i, total_nr_chunks -> None): Write a chunk of samples to disk.
            Samples are a list of objects returned by send_request. 
        
            Args:
                chunk (Any): sample
                chunk_i (int): chunk index
                total_nr_chunks (int): total number of chunks

        total_input (Optional[int], optional): total number of inputs. Defaults to None.
        max_input_size (int, optional): max size of input queue. Defaults to 0.
    """
    endpoints = load_endpoints(args.endpoint)
    print(f"Loaded {len(endpoints)} endpoints: {', '.join(endpoints)}")
    num_instances = args.instances

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
        for wi in range(num_instances)
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
                if workers_completed == num_instances:
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
                should_continue = writer(sorted_res, chunk_i, total_nr_chunks)
                if should_continue is False:
                    break
    if results:
        for chunk_i, res in results.items():
            if res:
                sorted_res = sorted(res, key=lambda x: int(x["index"]))
                writer(sorted_res, chunk_i, total_nr_chunks)

    if closer:
        closer()

    print("Processing complete.")
    # close workers
    for p in ps:
        p.join(timeout=3)
    for p in ps:
        try:
            p.close()
        except ValueError:
            p.terminate()
    reader_p.join(timeout=3)
    try:
        reader_p.close()
    except ValueError:
        reader_p.terminate()
