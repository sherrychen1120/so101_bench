# so101_bench
Benchmarking different learning-based policies on SO-101.

## Blog posts
### Training Action Chunking Transformer (ACT)
https://huggingface.co/blog/sherryxychen/train-act-on-so-101

## Installation

The installation instructions are separate for different use cases.

### Use case 1: to record, train, eval ML models
Use `uv` for this.
```bash
uv sync
source .venv/bin/activate
```

### Use case 2: to convert LeRobotDataset to ROS2 mcap files for viz

1. Install ROS2

**For Ubuntu users**

Install ROS2 based on the Official instructions [here](https://docs.ros.org/en/jazzy/Installation.html).

**For MacOS users**

Use `conda` for this, because we are installing ROS2 via RoboStack.
- Install `conda` first, recommending using `miniforge`.
- Then run:
```bash
conda env create -f environment.yml
```

If you wish to install ROS2 directly and then install the extra packages on your own, follow RoboStack instructions [here](https://robostack.github.io/GettingStarted.html).
