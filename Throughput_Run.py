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
    2: "connect.westc.gpuhub.com",
    3: "connect.bjb1.seetacloud.com",
    4: "connect.nmb2.seetacloud.com"
}

REMOTE_USER = "root"
REMOTE_PORT = {
  1: "19955",
  2: "22733",
  3: "23471",
  4: "28557"
}

GPU_MAPPING = {
    0: "0",
    1: "0",
    2: "1",
    3: "2",
    4: "3"
}

LOCAL_RESULT_DIR = "/home/yujie/Documents/Throughput_result"

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
                f"/root/miniconda3/envs/pytorch_env/bin/python /root/Throughput_Baseline_Woker.py"
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
        script_path = "/home/yujie/Github/Throughput_Baseline_Woker.py"
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
    parser.add_argument('--output_path', type=str, default="results.jsonl")
    args = parser.parse_args()
    
    Num_node = args.num_node
    percent_list = json.loads(args.percent_list)

    for percent_config in percent_list:
        print(f"\n=== Running with GPU percentages: {percent_config} ===")

        # === Baseline ===
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

        baseline_results = []
        for i in range(Num_node):
            path = os.path.join(LOCAL_RESULT_DIR, f"node{i}_baseline.json")
            with open(path, 'r') as f:
                baseline = json.load(f)
            baseline_results.append((baseline["node_id"], baseline["avg_epoch_time"]))
            print(f"Node {baseline['node_id']} baseline finish ，avg epoch: {baseline['avg_epoch_time']:.2f}s")

        epoch_times = [t for _, t in baseline_results]
        inv_speeds = [1.0 / t for t in epoch_times]
        total_speed = sum(inv_speeds)
        data_fractions = [s / total_speed for s in inv_speeds]

        print("Data fractions:", data_fractions)
        # === Realtrain ===
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
