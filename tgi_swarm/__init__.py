from dataclasses import dataclass, field
import os
import subprocess
import time
from typing import Generic, Iterable, List, Literal, Tuple, Type, TypeVar, Union
from huggingface_hub import get_session
import requests

from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep
import socket
DataclassT = TypeVar("DataclassT")


@dataclass
class InferenceSwarmConfig:
    instances: int = 1
    """number of inference instances"""
    inference_engine: Literal["tgi", "vllm"] = "tgi"
    """inference engine to use"""
    slurm_template_path: str = "templates/tgi_h100.template.slurm"
    """path to slurm template"""
    load_balancer_template_path: str = "templates/nginx.template.conf"
    """path to load balancer template"""


def run_command(command: str):
    print(f"running {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, errors = process.communicate()
    return_code = process.returncode
    assert return_code == 0, f"Command failed with error: {errors.decode('utf-8')}"
    return output.decode("utf-8").strip()


def is_job_running(job_id):
    """Given job id, check if the job is in eunning state (needed to retrieve hostname from logs)"""
    command = "squeue --me --states=R | awk '{print $1}' | tail -n +2"
    my_running_jobs = subprocess.run(
        command, shell=True, text=True, capture_output=True
    ).stdout.splitlines()
    return job_id in my_running_jobs

def get_unused_port():
    sock = socket.socket()
    sock.bind(('', 0))
    return sock.getsockname()[1]


def test_generation(endpoint):
    headers = {
        "Content-Type": "application/json",
    }
    data = {
        'inputs': 'What is Deep Learning?',
        'parameters': {
            'max_new_tokens': 200,
        },
    }
    response = requests.post(endpoint, headers=headers, json=data)
    print("✅ test generation", response.json())

class Loader:
    def __init__(self, desc="Loading...", end="✅ Done!", failed="❌ Aborted!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            failed (str, optional): Final print on failure. Defaults to "Aborted!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end =  end + " " + self.desc
        self.failed =  failed + " " + self.desc
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        try:
            for c in cycle(self.steps):
                if self.done:
                    break
                print(f"\r{c} {self.desc}", flush=True, end="")
                sleep(self.timeout)
        except KeyboardInterrupt:
            self.stop()
            print("KeyboardInterrupt by user")

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self.stop()
        else:
            self.done = True
            cols = get_terminal_size((80, 20)).columns
            print("\r" + " " * cols, end="", flush=True)
            print(f"\r{self.failed}", flush=True)


def get_endpoints(endpoint_path: str, instances: int = 1) -> List[str]:
    """Return list of endpoints from either a file or a comma separated string.
    It also checks if the endpoints are reachable.

    Args:
        endpoint_path (str): path to file containing endpoints or comma separated string
        instances (int, optional): number of instances. Defaults to 1.

    Returns:
        List[str]: list of endpoints (e.g. ["http://26.0.154.245:13120"])
    """
    trying = True
    with Loader(f"Waiting for {endpoint_path} to be created"):
        while trying:
            try:
                endpoints = open(endpoint_path).read().splitlines()
                assert (
                    len(endpoints) == instances
                ), f"#endpoints {len(endpoints)} doesn't match #instances {instances}"  # could read an empty file
                # due to race condition (slurm writing & us reading)
                trying = False
            except (OSError, AssertionError) as e:
                sleep(1)
    print("obtained endpoints", endpoints)
    for endpoint in endpoints:
        with Loader(f"Waiting for {endpoint} to be reachable"):
            connected = False
            while not connected:
                try:
                    get_session().get(f"{endpoint}/health")
                    print(f"\nConnected to {endpoint}")
                    connected = True
                except requests.exceptions.ConnectionError:
                    sleep(1)
    return endpoints

class InferenceSwarm:
    def __init__(
        self,
        config: InferenceSwarmConfig
    ) -> None:
        self.config = config
        self.cleaned_up = False
        os.makedirs("slurm/logs", exist_ok=True)
    
    def start(self):
        if self.config.inference_engine == "vllm":
            print(
                f"Using vllm_h100.slurm instead of default template {self.config.slurm_template_path}"
            )
            template = "vllm_h100.slurm"
        else:
            template = self.config.slurm_template_path
        with open(template) as f:
            slurm_template = f.read()

        # customize slurm template
        self.filename = f"{self.config.inference_engine}_{int(time.time())}"
        slurm_path = os.path.join("slurm", f"{self.filename}_{self.config.inference_engine}.slurm")
        slurm_host_path = os.path.join("slurm", f"{self.filename}_host_{self.config.inference_engine}.txt")
        slurm_template = slurm_template.replace(r"{{slurm_hosts_path}}", slurm_host_path)
        with open(slurm_path, "w") as f:
            f.write(slurm_template)

        # start inference instances
        self.job_ids = [run_command(f"sbatch --parsable {slurm_path}") for _ in range(self.config.instances)]
        print(f"Slurm Job ID: {self.job_ids}")
        print(f"📖 Slurm Hosts Path: {slurm_host_path}")

        try:
            # ensure job is running
            for job_id in self.job_ids:
                with Loader(f"Waiting for {job_id} to be created"):
                    while not is_job_running(job_id):
                        sleep(1)
            # retrieve endpoints
            self.endpoints = get_endpoints(slurm_host_path, self.config.instances)
            print(f"Endpoints running properly: {self.endpoints}")
            # warm up endpoints
            for endpoint in self.endpoints:
                test_generation(endpoint)

            if len(self.endpoints) == 1:
                print(f"🔥 endpoint ready {self.endpoints[0]}")
                self.endpoint = self.endpoints[0]
            else:
                # run a load balancer
                with open(self.config.load_balancer_template_path) as f:
                    # templates/nginx.template.conf
                    load_balancer_template = f.read()
                servers = "\n".join([f"server {endpoint.replace('http://', '')};" for endpoint in self.endpoints])
                unused_port = get_unused_port()
                load_balancer_template = load_balancer_template.replace(r"{{servers}}", servers)
                load_balancer_template = load_balancer_template.replace(r"{{port}}", str(unused_port))
                load_balancer_path = os.path.join("slurm", f"{self.filename}_load_balancer.conf")
                with open(load_balancer_path, "w") as f:
                    f.write(load_balancer_template)
                load_balance_endpoint = f"http://localhost:{unused_port}"
                command = f"sudo docker run -p {unused_port}:{unused_port} --network host -v $(pwd)/{load_balancer_path}:/etc/nginx/nginx.conf nginx"
                load_balance_endpoint_connected = False

                # run docker streaming output while we validate the endpoints
                print(f"running {command}")
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if not output:
                        continue
                    if not load_balance_endpoint_connected:
                        print(output.strip())
                        try:
                            get_session().get(f"{load_balance_endpoint}/health")
                            print(f"🔥 endpoint ready {load_balance_endpoint}")
                            load_balance_endpoint_connected = True
                            self.endpoint = load_balance_endpoint
                            break
                        except requests.exceptions.ConnectionError:
                            sleep(1)
                print("haha")

        except (KeyboardInterrupt, Exception) as e:
            self.cleanup()


    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()

    def cleanup(self, signum=None, frame=None):
        if self.cleaned_up:
            return
        for job_id in self.job_ids:
            run_command(f"scancel {job_id}")
        print(f"inference instances terminated")
        self.cleaned_up = True
        

if __name__ == "__main__":
    with InferenceSwarm(
        InferenceSwarmConfig(
            instances=3,
            inference_engine="tgi",
            slurm_template_path="templates/tgi_h100.template.slurm",
            load_balancer_template_path="templates/nginx.template.conf",
        )
    ) as inference_swarm:
        while True:
            input("Press Enter to EXIT...")
            break