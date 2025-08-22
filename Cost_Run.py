import os
import subprocess
import time
import argparse
import json
import shutil
import paramiko
from multiprocessing import Process
import numpy as np

NODE_ADDRS = {
    0: "localhost",
    1: "region-42.seetacloud.com",
    2: "connect.nmb2.seetacloud.com",
    3: "connect.westc.gpuhub.com",
    4: "connect.easts.gpuhub.com"
}

REMOTE_PORT = {
    1: "19955",
    2: "28557",
    3: "22733",
    4: "19853"
}

GPU_MAPPING = {
    0: "0",
    1: "0",
    2: "1",
    3: "2",
    4: "3"
}

REMOTE_USER = "root"
LOCAL_RESULT_DIR = "/home/yujie/Documents/Cost_results"

def init_ssh_agent():
    agent_process = subprocess.Popen(['ssh-agent', '-s'], stdout=subprocess.PIPE)
    agent_output, _ = agent_process.communicate()
    for line in agent_output.splitlines():
        if line.startswith(b'SSH_AUTH_SOCK='):
            os.environ['SSH_AUTH_SOCK'] = line.split(b'=')[1].split(b';')[0].decode()
        elif line.startswith(b'SSH_AGENT_PID='):
            os.environ['SSH_AGENT_PID'] = line.split(b'=')[1].split(b';')[0].decode()
    subprocess.run(['ssh-add'], check=True)
    print("SSH successful")

def remote_worker(node_id, percent, return_dict_path, mode="realtrain", data_fraction=1.0):
    node_ip = NODE_ADDRS[node_id]
    gpu_id = GPU_MAPPING[node_id]
    percent_str = str(percent)
    result_filename = os.path.basename(return_dict_path)

    if node_ip != "localhost":
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=node_ip,
                port=int(REMOTE_PORT[node_id]),
                username=REMOTE_USER,
                timeout=30
            )
            print(f"connect to node {node_id} ({node_ip})")
            remote_command = (
                f"CUDA_VISIBLE_DEVICES={gpu_id} "
                f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={percent_str} "
                f"/root/miniconda3/envs/pytorch_env/bin/python /root/Cost_Woker.py"
                f"--node_id={node_id} "
                f"--percent={percent_str} "
                f"--return_dict_path={return_dict_path} "
                f"--mode={mode} "
                f"--data_fraction={data_fraction}"
            )
            stdin, stdout, stderr = ssh.exec_command(remote_command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                print(f"node {node_id} failed: {exit_status}")
                print(stderr.read().decode())
                ssh.close()
                return
            local_save_path = os.path.join(LOCAL_RESULT_DIR, result_filename)
            sftp = ssh.open_sftp()
            sftp.get(return_dict_path, local_save_path)
            sftp.close()
            ssh.close()
            print(f"from node {node_id} saved file: {local_save_path} successful")
        except Exception as e:
            print(f"connect node {node_id} failed: {str(e)}")
    else:
        script_path = "/home/yujie/Github/Cost_Woker.py"
        command = (
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={percent_str} "
            f"python3 {script_path} "
            f"--node_id={node_id} "
            f"--percent={percent_str} "
            f"--return_dict_path={return_dict_path} "
            f"--mode={mode} "
            f"--data_fraction={data_fraction}"
        )
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"local node {node_id} fail: {result.returncode}")
            return
        local_save_path = os.path.join(LOCAL_RESULT_DIR, result_filename)
        shutil.copyfile(return_dict_path, local_save_path)
        print(f"local node {node_id} save in : {local_save_path}")

def main():
    init_ssh_agent()

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_node', type=int, required=True)
    parser.add_argument('--percent_list', type=str, required=True, help='JSON，like[[100,90],[80,70]]')
    parser.add_argument('--alpha', type=float, required=True, help='alpha（computation time）')
    parser.add_argument('--beta', type=float, required=True, help='beta（waiting time）')
    args = parser.parse_args()

    Num_node = args.num_node
    percent_list = json.loads(args.percent_list)
    alpha = args.alpha
    beta = args.beta

    assert abs(alpha + beta - 1.0) < 1e-6, "The sum of alpha and beta must be 1"

    for percent_config in percent_list:
        assert len(percent_config) == Num_node
        print(f"\n=== Running with GPU percentages: {percent_config} ===")

        baseline_paths = {i: os.path.join("/tmp", f"node{i}_baseline.json") for i in range(Num_node)}
        processes = []
        for node_id in range(Num_node):
            p = Process(target=remote_worker,
                        args=(node_id, percent_config[node_id], baseline_paths[node_id]),
                        kwargs={"mode": "baseline", "data_fraction": 1.0})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        training_times = []
        for i in range(Num_node):
            path = os.path.join(LOCAL_RESULT_DIR, f"node{i}_baseline.json")
            with open(path, 'r') as f:
                baseline = json.load(f)
            training_times.append(baseline["training_time"])
            print(f"node {baseline['node_id']} baseline finish")

        # simulate waiting time
        Lam, lam_upper_bound = 5, 10
        Mu, mu_upper_bound = 10, 20
        waiting_times = [np.clip(np.random.poisson(Lam), 0, lam_upper_bound) * np.clip(np.random.exponential(Mu), 0, mu_upper_bound) for _ in range(Num_node)]
        print(f"waiting_time: {waiting_times}")

        costs = [alpha * t + beta * w for t, w in zip(training_times, waiting_times)]
        inv_cost = [1 / c for c in costs]
        total_inv = sum(inv_cost)
        data_fractions = [ic / total_inv for ic in inv_cost]

        print("data fraction:", data_fractions)

        realtrain_paths = {i: os.path.join("/tmp", f"node{i}_realtrain.json") for i in range(Num_node)}
        processes = []
        for node_id in range(Num_node):
            p = Process(target=remote_worker,
                        args=(node_id, percent_config[node_id], realtrain_paths[node_id]),
                        kwargs={"mode": "realtrain", "data_fraction": data_fractions[node_id]})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

if __name__ == "__main__":
    main()
