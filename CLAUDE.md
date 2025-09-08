# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SO-101 Bench is a benchmarking framework for learning-based policies on the SO-101 robot platform. It integrates with LeRobot for robot learning workflows, providing data collection, training, and evaluation capabilities for imitation learning and reinforcement learning.

## Development Commands

### Environment Setup
- Use `uv` as the package manager (see `pyproject.toml`)
- Virtual environment: Activate with `source .venv/bin/activate` or use `uv run`
- Dependencies are managed through `uv.lock`

### Testing
- Run tests: `pytest` (pytest 8.4.1 is pinned)
- Tests are located in `tests/` directory
- Individual test: `pytest tests/test_<module_name>.py`

### Code Quality
- No explicit linting configuration found - check if project uses standard Python tools
- Run type checking and linting if configured in the project

## Architecture Overview

### Core Components

**Data Recording & Management:**
- `RawDatasetRecorder`: Main class for recording robot demonstrations in human-readable format alongside LeRobot format
- `TaskConfigurator`: Manages task specifications and configurations with stratified sampling support
- Raw dataset format documented in `src/so101_bench/scripts/record/docs/raw_dataset_format.md`

**Training Pipeline:**
- `src/so101_bench/scripts/train/train.py`: Training script based on LeRobot's training pipeline
- Supports various learning-based policies and models

**Evaluation Pipeline:**
- `src/so101_bench/scripts/eval/eval_rollout.py`: Policy evaluation rollouts on robot
- `src/so101_bench/scripts/eval/eval_scorer.py`: Evaluation scoring system
- Task progress scoring defined in task specifications

**Task System:**
- Tasks defined in `datasets/tasks/` with `task_spec.yaml` and `task_config_template.yaml`
- Current tasks: `pick_and_place_block`
- Task specs define variations, scoring criteria, and configuration parameters

### Dependencies
- **LeRobot Integration**: Uses LeRobot as workspace member (`third_party/lerobot/`)
- **Hardware**: Feetech servo SDK for robot control
- **Vision**: OpenCV for camera handling
- **ML**: PyTorch-based training pipeline

### Directory Structure
```
src/so101_bench/
├── raw_dataset_recorder.py     # Core recording functionality
├── task_configurator.py        # Task configuration management
├── task_progress_labeler.py    # Progress scoring for evaluation
├── recorder_configs.py         # Recording configuration utilities
└── scripts/
    ├── record/                 # Data recording scripts
    ├── train/                  # Model training scripts
    └── eval/                   # Evaluation scripts
```

### Data Formats
- **Raw Format**: Human-readable hierarchical structure with JSONL trajectories, JPEG images, and metadata
- **LeRobot Format**: Standard format for model training
- **Task Configs**: YAML-based task specifications with variations and scoring definitions

## Script Usage Patterns

### Recording Data
Use `src/so101_bench/scripts/record/record.py` with robot and teleop configurations:
- Specify robot type, port, camera configs, and calibration directories
- Configure dataset parameters including episode count and task specifications
- Supports both raw format and LeRobot format recording simultaneously

### Training Models
Use `src/so101_bench/scripts/train/train.py` with LeRobot-based configuration system for training imitation learning policies.

### Running Evaluations
Use `src/so101_bench/scripts/eval/eval_rollout.py` for policy evaluation with robot configurations and task parameters.

## Hardware Configuration
- Robot configurations in `bringup/camera_configs.json`
- Camera identification script: `bringup/identify_cameras.sh`
- Supports SO-101 robot platform with leader-follower setup