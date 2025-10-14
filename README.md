# so101_bench
Benchmarking different learning-based policies on SO-101.

## Blog posts
### Training Action Chunking Transformer (ACT)
https://huggingface.co/blog/sherryxychen/train-act-on-so-101

## Installation
### Option 1: to record, train, eval ML models
```bash
uv sync
source .venv/bin/activate
```

### Option 2: to convert LeRobotDataset to ROS2 mcap files for viz
1. Install ROS2
**For Ubuntu users**
Install ROS2 based on the steps here: https://docs.ros.org/en/jazzy/Installation.html
**For MacOS users**
Follow https://robostack.github.io/GettingStarted.html to install ROS2 using `conda`.
After it's installed
```bash
conda activate ros_env
```
