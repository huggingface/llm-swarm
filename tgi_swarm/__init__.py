import asyncio
import multiprocessing.queues
import os
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from multiprocessing import Process, Queue
import shlex
import subprocess
from typing import Annotated, Callable, Union, List, Optional, Any
import time
import uuid
import requests


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
    instances: Annotated[int, tyro.conf.arg(aliases=["-i"])] = 2
    """Number of model instances"""
    endpoint: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "hosts.txt"
    """Model endpoint. Can be path to file with list of endpoints"""
    start: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Start index (in terms of chunks of size instances*batch_size)"""
    checkpoint_size: Annotated[int, tyro.conf.arg(aliases=["-csize"])] = 5000
    """Save partial results ever checkpoint_size generations"""
    manage_tgi_instances: Annotated[bool, tyro.conf.arg(aliases=["-r"])] = False
    """Spin up and terminate TGI instances when the generation is done"""
    slurm_template_path: Annotated[str, tyro.conf.arg(aliases=["-slurm"])] = "tgi_template.slurm"
    """Slurm template file path"""


SENTINEL = None


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = fd.communicate()
    return_code = fd.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()


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


def parse_endpoints(endpoint_val: Union[str, List[str]]) -> List[str]:
    if endpoint_val.endswith(".txt"):
        return open(endpoint_val).read().splitlines()
    return endpoint_val.split(",")


def load_endpoints(endpoint_val: Union[str, List[str]], num_instances: int = 1) -> List[str]:
    """ Return list of endpoints from either a file or a comma separated string.
    It also checks if the endpoints are reachable.
    
    Args:
        endpoint_val (Union[str, List[str]]): either a file path or a comma separated string
        num_instances (int, optional): number of instances. Defaults to 1.
    
    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    headers = {
        "Content-Type": "application/json",
    }
    dummy_data = {
        'inputs': 'What is Deep Learning?',
        'parameters': {
            'max_new_tokens': 20,
        },
    }
    endpoints = None
    while endpoints is None:
        try:
            endpoints = parse_endpoints(endpoint_val)
            assert len(endpoint[0]) == num_instances # could read an empty file
            # due to race condition (slurm writing & us reading)
        except:
            print("Attempting to load endpoints...")
            time.sleep(10)
    print("obtained endpoints", endpoints)
    for endpoint in endpoints:
        connected = False
        while not connected:
            try:
                requests.post(endpoint, headers=headers, json=dummy_data)
                print(f"Connected to {endpoint}")
                connected = True
            except:
                print(f"Attempting to reconnect to {endpoint}...")
                time.sleep(10)
    return endpoints


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
    if args.manage_tgi_instances:
        with open(args.slurm_template_path) as f:
            slurm_template = f.read()
        filename = str(uuid.uuid4())
        slurm_path = os.path.join("slurm", f"{filename}.slurm")
        slurm_host_path = os.path.join("slurm", f"{filename}_host.txt")
        slurm_template = slurm_template.replace(r"{{slurm_hosts_path}}", slurm_host_path)
        with open(os.path.join("slurm", f"{filename}.slurm"), "w") as f:
            f.write(slurm_template)
        job_ids = [run_command(f"sbatch --parsable {slurm_path}") for _ in range(args.instances)]
        print(f"Slurm Job ID: {job_ids}")
        endpoints = load_endpoints(slurm_host_path, args.instances)
    else:
        endpoints = load_endpoints(args.endpoint, args.instances)
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
    print(total_input)
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

    if args.manage_tgi_instances:
        for job_id in job_ids:
            run_command(f"scancel {job_id}")
        print(f"TGI instances terminated")
