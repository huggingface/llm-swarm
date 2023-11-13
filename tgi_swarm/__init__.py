import asyncio
import multiprocessing.queues
import os
import requests
import time
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from multiprocessing import Process, Queue
from typing import Annotated, Callable, Union, List, Optional, Any

import tyro
from huggingface_hub import AsyncInferenceClient, get_session
from tqdm import tqdm


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    if abs(num) < 1:
        return "{:.3g}".format(num)
    SIZES = ["", "K", "M", "G", "T", "P", "E"]
    num = float("{:.3g}".format(num))
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format("{:f}".format(num).rstrip("0").rstrip("."), SIZES[magnitude])

@dataclass
class TGIConfig:
    """
    Configuration class for TGI
    """

    batch_size: Annotated[int, tyro.conf.arg(aliases=["-bs"])] = 250
    """Individual batch size"""
    # instances: Annotated[int, tyro.conf.arg(aliases=["-i"])] = 32
    # """Number of model instances"""
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


def load_endpoints(endpoint_val: Union[str, List[str]]) -> List[str]:
    """ Return list of endpoints from either a file or a comma separated string
    
    Args:
        endpoint_val (Union[str, List[str]]): either a file path or a comma separated string
    
    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    if os.path.isfile(endpoint_val):
        end_points = open(endpoint_val).read().splitlines()
    else:
        end_points = endpoint_val.split(",")
    healthy_end_points = []
    for end_point in end_points:
        try:
            response = get_session().get(f"{end_point}/health")
        except requests.exceptions.ConnectionError:
            print(f"Endpoint {end_point} is unhealthy")
            continue
        if response.status_code == 200:
            healthy_end_points.append(end_point)
        else:
            print(f"Endpoint {end_point} is unhealthy")
    if os.path.isfile(endpoint_val):
        print(f"Updating endpoints file {endpoint_val} with {','.join(healthy_end_points)}")
        with open(endpoint_val, "w") as f:
            f.write("\n".join(healthy_end_points))
    return healthy_end_points


def generate_data(args: TGIConfig,
                  reader: Callable[[multiprocessing.Queue, int], None],
                  writer: Callable[[List[Any], int, int], None],
                  send_request: Callable[[Any, AsyncInferenceClient], Any],
                  closer: Optional[Callable]=None,
                  total_input: Optional[int]=None,
                  total_tqdm: Optional[int]=None,
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

        total_input (Optional[int], optional): total number of input samples. Defaults to None. Used for saving/tqdm
        max_input_size (int, optional): max size of input queue. Defaults to 0.
        total_tqdm (Optional[int]): total length of the tqdm, used instead of total_input if returning advance with writer
    """
    endpoints = load_endpoints(args.endpoint)
    print(f"Loaded {len(endpoints)} endpoints: {', '.join(endpoints)}")
    num_instances = len(endpoints)

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
    current_time = time.time()
    with tqdm(total=total_input if total_input else total_tqdm, initial=args.start * checkpoint_chunk_size) as pbar:
        while True:
            generated_textbook = output_queue.get()
            # check if we are done
            if generated_textbook == SENTINEL:
                workers_completed += 1
                if workers_completed == num_instances:
                    break
                else:
                    continue
            # store generated textbook in corresponding list
            pbar.update()
            chunk_i = int(generated_textbook["index"]) // checkpoint_chunk_size
            results[chunk_i].append(generated_textbook)
            # check if it is complete
            if len(results[chunk_i]) == checkpoint_chunk_size:  # current chunk is complete
                sorted_res = sorted(results.pop(chunk_i), key=lambda x: int(x["index"]))
                results = writer(sorted_res, chunk_i, total_nr_chunks)
                if len(results) == 2:
                    should_continue, update_size = results
                else:
                    should_continue = True
                    update_size = None
                if should_continue is False:
                    break
                if update_size:
                    pbar.desc = f"{human_format(float(update_size)/(time.time() - current_time))}/s"
                    current_time = time.time()
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
