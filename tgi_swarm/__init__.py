import asyncio
import multiprocessing.queues
import os
import shlex
import signal
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from multiprocessing import Process, Queue
from typing import Annotated, Any, Callable, List, Optional, Union

import requests
import tyro
from huggingface_hub import AsyncInferenceClient, get_session
from tqdm import tqdm


def human_format(num: float, billions: bool = False, divide_by_1024: bool = False) -> str:
    if abs(num) < 1:
        return f"{num:.3g}"
    SIZES = ["", "K", "M", "G", "T", "P", "E"]
    num = float(f"{num:.3g}")
    magnitude = 0
    i = 0
    while abs(num) >= 1000 and i < len(SIZES) - 1:
        magnitude += 1
        num /= 1000.0 if not divide_by_1024 else 1024.0
        i += 1
    return "{}{}".format(f"{num:f}".rstrip("0").rstrip("."), SIZES[magnitude])


@dataclass
class TGIConfig:
    """
    Configuration class for TGI
    """

    batch_size: Annotated[int, tyro.conf.arg(aliases=["-bs"])] = 500
    """Individual batch size"""
    instances: Annotated[int, tyro.conf.arg(aliases=["-i"])] = 1
    """Number of model instances"""
    endpoint: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "hosts.txt"
    """Model endpoint. Can be path to file with list of endpoints"""
    start: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Start index (in terms of chunks of size instances*batch_size)"""
    checkpoint_size: Annotated[int, tyro.conf.arg(aliases=["-csize"])] = 5000
    """Save partial results ever checkpoint_size generations"""
    manage_tgi_instances: Annotated[bool, tyro.conf.arg(aliases=["-r"])] = False
    """Spin up and terminate TGI instances when the generation is done"""
    slurm_template_path: Annotated[str, tyro.conf.arg(aliases=["-slurm"])] = "tgi_h100.slurm"
    """Slurm template file path"""
    get_hostname_template: Annotated[str, tyro.conf.arg(aliases=["-host"])] = "get_hostname_template.sh"
    """Template for retreiving and saving hosts (needed for H100 cluster)"""
    use_vllm: Annotated[bool, tyro.conf.arg(aliases=["-vllm"])] = False
    """Use vLLM instead of TGI"""


SENTINEL = None


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    fd = subprocess.Popen(command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = fd.communicate()
    return_code = fd.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()


def is_job_running(job_id):
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs


async def requests_worker(
    send_request: Callable, input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue, client
    ):
    """Send a single request to the model.
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
    if args.use_vllm:
        endpoint = f"{endpoint}/generate"
    client = AsyncInferenceClient(model=endpoint)
    await asyncio.gather(*[requests_worker(send_request, input_queue, output_queue, client) for _ in range(args.batch_size)])
    output_queue.put(SENTINEL)  # signal that we are done


def mp_worker(*largs):
    asyncio.run(mp_worker_async(*largs))


def load_endpoints(endpoint_val: Union[str, List[str]], num_instances: int = 1) -> List[str]:
    """Return list of endpoints from either a file or a comma separated string.
    It also checks if the endpoints are reachable.

    Args:
        endpoint_val (Union[str, List[str]]): either a file path or a comma separated string
        num_instances (int, optional): number of instances. Defaults to 1.

    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    trying = True
    while trying:
        try:
            if endpoint_val.endswith(".txt"):
                endpoints = open(endpoint_val).read().splitlines()
            else:
                endpoints = endpoint_val.split(",")
            assert (
                len(endpoints) == num_instances
            ), f"#endpoints {len(endpoints)} doesn't match #instances {num_instances}"  # could read an empty file
            # due to race condition (slurm writing & us reading)
            trying = False
        except Exception as e:
            print(f"Attempting to load endpoints... error: {e}")
            time.sleep(10)
    print("obtained endpoints", endpoints)
    for endpoint in endpoints:
        connected = False
        while not connected:
            try:
                get_session().get(f"{endpoint}/health")
                print(f"Connected to {endpoint}")
                connected = True
            except requests.exceptions.ConnectionError:
                print(f"Attempting to reconnect to {endpoint}...")
                time.sleep(10)
    return endpoints


def get_hostname_template(args, job_type, slurm_host_path, job_id):
    """Build the bash script for retrieving hostname given job id and inference library TGI/vLLM"""
    with open(args.get_hostname_template) as f:
        hostname_template = f.read()
    hostname_template = hostname_template.replace(r"{{job_type}}", job_type)
    hostname_template = hostname_template.replace(
        r"{{slurm_hosts_path}}", slurm_host_path
    )
    hostname_template = hostname_template.replace("{{job_id}}", job_id)
    return hostname_template


def generate_data(
    args: TGIConfig,
    reader: Callable[[multiprocessing.Queue, int], None],
    writer: Callable[[List[Any], int, int], None],
    send_request: Callable[[Any, AsyncInferenceClient], Any],
    closer: Optional[Callable] = None,
    total_input: Optional[int] = None,
    total_tqdm: Optional[int] = None,
    max_input_size: int = 0,
):
    """Control the swarm of workers to generate data

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
    """
    if args.manage_tgi_instances:
        if args.use_vllm and "tgi" in args.slurm_template_path:
            print(
                f"Using vllm_h100.slurm instead of default template {args.slurm_template_path}"
            )
            template = "vllm_h100.slurm"
        else:
            template = args.slurm_template_path
        with open(template) as f:
            slurm_template = f.read()
        filename = str(uuid.uuid4())
        job_type = "vllm" if args.use_vllm else "tgi"
        # slurm file for launching inference job
        slurm_path = os.path.join("slurm", f"{filename}_{job_type}.slurm")
        # bash script for retrieving endpoints from job logs
        hosts_bash_path = os.path.join("slurm", f"{filename}_host_{job_type}.sh")
        # txt file where we save the endpoints
        slurm_host_path = os.path.join("slurm", f"{filename}_host_{job_type}.txt")
        with open(slurm_path, "w") as f:
            f.write(slurm_template)
        job_ids = [run_command(f"sbatch --parsable {slurm_path}") for _ in range(args.instances)]
        print(f"Slurm Job ID: {job_ids}")
        # retrieve hostnames
        for job_id in job_ids:
            print("-" * 60)
            template = get_hostname_template(args, job_type, slurm_host_path, job_id)
            with open(hosts_bash_path, "w") as f:
                f.write(template)
            # wait until the job is running to retrieve the hostname and save in slurm_host_path
            while not is_job_running(job_id):
                print(f"Waiting for job {job_id} to start...")
                time.sleep(10)
            print("Sleep for 10 more sec while the job runs")
            time.sleep(10)
            output_host = run_command(f"bash {hosts_bash_path}")
            print(f"- Output from hostname retrieval:START_OUTPUT\n{output_host}\nEND_OUTPUT")

        def cleanup_function(signum, frame):
            for job_id in job_ids:
                run_command(f"scancel {job_id}")
            library = "TGI" if not args.use_vllm else "vLLM"
            print(f"{library} instances terminated")
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_function)  # Handle Ctrl+C
        signal.signal(signal.SIGTERM, cleanup_function)  # Handle termination requests
        endpoints = load_endpoints(slurm_host_path, args.instances)
    else:
        # change default endpoints file for vllm
        if args.use_vllm and args.endpoint == "hosts.txt":
            print(f"Reading endpoints from hosts_vllm.txt")
            endpoints = load_endpoints("hosts_vllm.txt", args.instances)
        else:
            endpoints = load_endpoints(args.endpoint, args.instances)
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
                writer_data = writer(sorted_res, chunk_i, total_nr_chunks)
                if writer_data is not None and len(writer_data) == 2:
                    should_continue, update_size = writer_data
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

    if args.manage_tgi_instances:
        cleanup_function(None, None)
