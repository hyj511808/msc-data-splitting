# msc-data-splitting

This repository contains the code for experiments conducted in the MSc project on adaptive data splitting for distributed deep neural network training. The goal is to evaluate different data allocation strategies under heterogeneous GPU environments using both baseline and adaptive methods.

## Project Structure

| File | Description |
|------|-------------|
| `Baseline_Run.py` | Runs the baseline experiment by launching `Throughput_Baseline_Worker.py` on each node to measure full training time. |
| `Cost_Run.py` | Executes experiments based on the cost-function strategy, controlling `Cost_Worker.py` across multiple nodes. |
| `Cost_Worker.py` | Worker-side script for cost-function experiments. It measures training time and simulates waiting time based on a cost model. |
| `Throughput_Baseline_Worker.py` | Worker-side script for throughput-based and baseline tests. It only measures training time without simulating delays. |
| `Throughput_Run.py` | Executes the throughput-based data allocation experiments, orchestrating `Throughput_Baseline_Worker.py` on all nodes. |

## How to Run

Each run script (e.g. `Cost_Run.py`) manages distributed training over multiple nodes. You should configure SSH access and node mappings within the script.

Example:
```bash
python Cost_Run.py \
    --num_node 4 \
    --percent_list "[[100,90,80,70],[90,80,70,60]]" \
    --alpha 0.6 \
    --beta 0.4

