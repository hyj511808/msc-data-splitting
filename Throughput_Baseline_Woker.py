import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import json
import numpy as np
import os
from torch.utils.data import Subset

def set_cuda_mps_percentage(percent):
    os.environ['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE'] = str(percent)
    print("当前 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE:", os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"))

def build_dataset(node_id: int, indices_path: str = None, data_fraction: float = 1.0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    full_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    if indices_path is not None and os.path.isfile(indices_path):
        with open(indices_path, 'r') as f:
            indices = json.load(f)
        print(f"[节点 {node_id}] 载入不重叠索引：{indices_path}，样本数={len(indices)}")
        return Subset(full_dataset, indices)
    else:
        if data_fraction < 1.0:
            torch.manual_seed(42 + node_id)
            total_size = len(full_dataset)
            subset_size = int(total_size * data_fraction)
            indices = torch.randperm(total_size)[:subset_size]
            print(f"[Node {node_id}] uses {data_fraction*100:.2f}%")
            return Subset(full_dataset, indices.tolist())
        else:
            print(f"[Node {node_id}] uses full dataset]")
            return full_dataset

def train(device, train_dataset, epochs=2, node_id=0):
    # 远程环境更稳：num_workers=0
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    model = torchvision.models.resnet50(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    epoch_times, batch_times = [], []
    start_time = time.time()

    print(f"[Node {node_id}] start，all {epochs} epochs")
    for epoch in range(epochs):
        epoch_start = time.time()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            bt_start = time.time()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bt_end = time.time()
            batch_times.append(bt_end - bt_start)

        epoch_end = time.time()
        epoch_times.append(epoch_end - epoch_start)
        print(f"[Node {node_id}] {epoch+1}/{epochs} epoch spends {epoch_times[-1]:.2f} s")

    end_time = time.time()
    training_time = end_time - start_time
    avg_epoch_time = sum(epoch_times)/len(epoch_times)
    avg_batch_time = sum(batch_times)/len(batch_times)

    return avg_epoch_time, avg_batch_time, training_time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--node_id", type=int, required=True)
    parser.add_argument("--percent", type=int, required=True)
    parser.add_argument("--return_dict_path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["baseline", "realtrain"], default="realtrain")
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--indices_path", type=str, default=None, help="该节点使用的不重叠样本索引文件（JSON）")
    args = parser.parse_args()

    set_cuda_mps_percentage(args.percent)
    print(f"[Node {args.node_id}] set GPU {args.percent}%")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5 if args.mode == "baseline" else 20

    train_dataset = build_dataset(
        node_id=args.node_id,
        indices_path=args.indices_path,
        data_fraction=args.data_fraction
    )

    avg_epoch_time, avg_batch_time, training_time = train(
        device=device,
        train_dataset=train_dataset,
        epochs=epochs,
        node_id=args.node_id
    )
    result = {
        "node_id": args.node_id,
        "percent": args.percent,
        "data_fraction": args.data_fraction,
        "indices_path": args.indices_path,
        "num_samples": len(train_dataset),
        "avg_epoch_time": avg_epoch_time,
        "avg_batch_time": avg_batch_time,
        "training_time": training_time,
        "mode": args.mode
    }

    os.makedirs(os.path.dirname(args.return_dict_path), exist_ok=True)
    with open(args.return_dict_path, 'a' if args.mode == "realtrain" else 'w') as f:
        json.dump(result, f)
        f.write('\n')
    print(f"[Node {args.node_id}] result is wrote in {args.return_dict_path}")
