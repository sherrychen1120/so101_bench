# SO101 Evaluation Pipeline Documentation

This document describes the SO101 evaluation pipeline, including how to run evaluation rollouts and the file formats for inputs and outputs.

## Overview

The evaluation pipeline consists of two main stages:

1. **Evaluation Rollout** (`eval_rollout.py`): Creates evaluation episodes by rolling out policies or teleoperating on tasks from existing datasets
2. **Evaluation Scoring** (`eval_scorer.py`): Manual labeling of progress stages and calculation of evaluation metrics

## Stage 1: Evaluation Rollout

### Purpose

Creates evaluation datasets by:
- Loading source episodes from existing recorded datasets
- Rolling out policies to create new evaluation episodes (or teleoperating to create baselines / test the eval pipeline.)
- Maintaining task configurations and environment setups from source episodes

### Usage

```bash
python -m so101_bench.scripts.eval.eval_rollout \
  --robot.type=so101_follower \
  --robot.port=/dev/so101_follower \
  --robot.id=dum_e_follower \
  --robot.camera_configs_path=/path/to/camera_configs.json \
  --robot.calibration_dir=/path/to/robot/calibration \
  --teleop.type=so101_leader \
  --teleop.port=/dev/so101_leader \
  --teleop.calibration_dir=/path/to/teleop/calibration \
  --teleop.id=dum_e_leader \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/source_dataset \
  --dataset.push_to_hub=false \
  --dataset.private=true \
  --dataset.reset_time_s=5 \
  --dataset.num_episodes=10 \
  --dataset.single_task="Grab the block and put it in the container." \
  --resume=true \
  --dataset.save_raw_format=true \
  --dataset.raw_format_root=/path/to/datasets/recordings/ \
  --dataset.raw_format_videos=true \
  --eval_dataset.name=my_eval_set \
  --eval_dataset.splits='["val_id"]'
```

### Key Parameters

- `--eval_dataset.name`: Name of the evaluation dataset to create
- `--eval_dataset.splits`: JSON list of splits to use from source dataset (e.g., `'["val_id", "val_ood"]'`)
- `--dataset.single_task`: Task description for all evaluation episodes
- `--dataset.num_episodes`: Number of episodes to record (should match number of source episodes)

### Workflow

1. **Load Source Dataset**: Reads splits.yaml from source dataset
2. **Generate Episode List**: Selects episodes from specified splits
3. **Device Compatibility Check**: Validates robot/teleop configurations match source dataset
4. **Episode Rollout**: For each source episode:
   - Loads source task configuration
   - Displays setup instructions to user
   - Records new evaluation episode with same task setup
   - Saves episode with evaluation-specific metadata

### Input Requirements

#### Source Dataset Structure
See `so101_bench/src/so101_bench/scripts/record.raw_dataset_format.md` for full details
on source dataset structure.

```
source_dataset/
  splits.yaml                    # Dataset splits
  arm_calib/
    {leader_id}.json            # Arm calibrations
    {follower_id}.json
  episodes/
    {episode_name}/
      metadata.json             # Episode metadata
      task_config.yaml          # Task configuration
      # ... other episode files
```

#### task_config.yaml Format
```yaml
task_description: "Pick and place red cube"
object_positions:
  red_cube: [0.1, 0.2, 0.05]
  container: [0.3, 0.1, 0.0]
initial_arm_position: [0.0, 0.0, 0.0, 0.0]
# ... other task-specific configurations
```

#### splits.yaml Format
Split your dataset into
- `train`: for training the model 
- `val_id`: in-distribution validation set
- `val_ood`: out-of-distribution validation set

Note: The "distribution" we are interested in is whether the model has seen anything like this episode or not during training. So, by defintiion, the `train` set is "in-distribution" because it defines the "distribution".

This can be generated with `so101_bench/src/so101_bench/scripts/create_dataset_splits.py`

```yaml
train:
  - "001_2025-01-15_10-22-01"
  - "002_2025-01-15_10-25-15"
val_id:
  - "003_2025-01-15_10-30-22"
  - "004_2025-01-15_10-35-10"
val_ood:
  - "005_2025-01-15_10-40-05"
```

### Output Structure

```
eval_dataset/
  task_config.yaml              # Global task configuration
  episodes/
    {source_name}__{episode_name}__eval__{idx}_{timestamp}/
      metadata.json             # Episode metadata (includes eval info)
      task_config.yaml          # Task-specific configuration
      obs/                      # Observation data
      video_*.mp4               # Camera videos
      sync_logs.jsonl           # Synchronization logs
      # ... other raw format files
```

#### Evaluation Episode Metadata
```json
{
  "episode_id": "source_dataset__001_2025-01-15_10-22-01__eval__000_2025-01-27_15-30-45",
  "episode_idx": 0,
  "start_time": "2025-01-27T15:30:45.123456+00:00",
  "end_time": "2025-01-27T15:31:15.456789+00:00",
  "duration_s": 30.333333,
  "run_mode": "policy",
  "fps": 30,
  "actual_fps": 29.8,
  "leader_id": "dum_e_leader",
  "follower_id": "dum_e_follower",
  "total_frames": 910,
  "cameras": ["cam_front", "cam_top"],
  "task_description": "Grab the block and put it in the container.",
  "events": [],
  "policy": {
    "policy_path": "/path/to/policy",
    "policy_name": "PolicyClassName"
  },
}
```

## Stage 2: Evaluation Scoring

### Purpose

Manual labeling and automatic calculation of evaluation metrics:
- Interactive video player for progress stage labeling
- Overlap detection and validation
- Episode-level and dataset-level metrics calculation

![Eval Scorer](screenshot_eval_scorer.png)

### Usage

```bash
python -m so101_bench.scripts.eval.eval_scorer \
  --dataset_root_dir=/path/to/datasets/recordings \
  --task_root_dir=/path/to/tasks \
  --eval_set_name=my_eval_set \
  --eval_config_path=/path/to/eval_config.yaml \
```

### Parameters

- `--dataset_root_dir`: Root directory containing evaluation datasets
- `--task_root_dir`: Directory containing task specifications
- `--eval_set_name`: Name of evaluation dataset to score
- `--eval_config_path`: Path to evaluation configuration
- `--force`: Force override existing eval_score.yaml files

### Input Requirements

#### Evaluation Configuration (`eval_config.yaml`)
```yaml
horizon_s: 30  # Evaluation horizon in seconds
```

#### Task Specification (`task_spec.yaml`)
```yaml
# A `task_spec` defines the available variations of a task.
# It also defines the score to evaluate the performance of the task.
task_name: "pick_and_place_block"
variations:
  block: ["white", "green", "grey", "eraser"]
  container: ["tupperware", "bowl", "box"]
  start_pose: # [x, y, yaw(deg)]
    # All blocks align longer edge with y axis at yaw = 0.
    block:
      min: [0.0, 0.0, -90.0]
      max: [0.4, 0.32, 90.0]
    container:
      min: [0.0, 0.0, -90.0]
      max: [0.4, 0.32, 90.0]

score_definition:
  task_progress_score:
    0_reach_block: 0.2
    1_grasp_block: 0.4
    2_reach_container: 0.6
    3_release_block: 0.8
    4_block_in_container: 1.0
```

#### Dataset Structure
```
eval_dataset/
  task_config.yaml              # Task configuration
  episodes/
    {episode_id}/
      metadata.json             # Episode metadata
      sync_logs.jsonl           # Synchronization data
      video_cam_front.mp4       # Camera videos
      video_cam_top.mp4
      # ... other episode files
```

### Interactive Labeling

The scorer launches an interactive video player with:

#### Controls
- **SPACE**: Play/Pause
- **LEFT/RIGHT**: Seek backward/forward (1 second)
- **UP/DOWN**: Seek backward/forward (5 seconds)
- **0-9**: Start labeling progress stage
- **ENTER**: End current labeling
- **S**: Show current labels
- **TAB**: Select next interval for deletion
- **X/BACKSPACE**: Delete selected interval
- **R**: Reset all labels
- **Q**: Quit and save
- **Ctrl+C**: Gracefully quit and save

#### Visual Features
- **Live visualization panel**: Timeline view of labeled intervals
- **Overlap detection**: Red highlighting for overlapping intervals
- **Status indicator**: Green "VALID" or red "INVALID" status
- **Current time indicator**: Yellow line showing playback position
- **Interval selection**: White highlighting for selected intervals

### Output Formats

#### Episode Score (`eval_score.yaml`)
```yaml
episode_id: "episode_name"
eval_config:
  horizon_s: 30
episode_metrics:
  task_progress_label: "4_block_in_container"  # Highest stage reached
  task_progress_score: 1.0                    # Score for highest stage
  success: 1                                  # 1 if final stage reached
  duration_to_success_s: 25.5                # Time to completion (-1 if failed)
  progress_stage_timerange_s:                 # Labeled intervals
    0_reach_block:
      - [1.0, 3.5]
      - [8.2, 10.1]  # Multiple intervals for retries
    1_grasp_block:
      - [3.5, 6.0]
    2_reach_container:
      - [6.0, 8.2]
    3_release_block:
      - [10.1, 12.5]
    4_block_in_container:
      - [12.5, 25.5]
  frame_latency_ms: 45.2                     # Average frame processing latency
  is_safety_abort: false                     # Emergency stop detected
  num_attempts: 2                            # Number of retry attempts
```

#### Dataset Score (`dataset_eval_score.yaml`)
```yaml
eval_config:
  horizon_s: 30
eval_set_metrics:
  average_task_progress_score: 0.8           # Mean across all episodes
  success_rate: 0.6                          # Fraction reaching final stage
  safety_abort_rate: 0.1                    # Fraction with emergency stops
  average_duration_to_success_s: 22.3       # Mean time for successful episodes
  average_num_attempts: 1.8                 # Mean retry attempts
  average_frame_latency_ms: 48.5            # Mean processing latency
  average_progress_stage_duration_s:         # Mean duration per stage
    0_reach_block: 2.5
    1_grasp_block: 3.2
    2_reach_container: 2.8
    3_release_block: 2.1
    4_block_in_container: 8.9
eval_set:
  dataset_id: "my_eval_set"
  episodes_list:
    - "episode_001"
    - "episode_002"
    # ... all episodes
```

### Metrics Calculation

#### Episode-Level Metrics
- **Task Progress Score**: Score of highest stage reached (from task specification)
- **Success**: Binary indicator if final stage was reached
- **Duration to Success**: Time from start to final stage completion
- **Frame Latency**: Average time between timestamps in sync logs
- **Safety Abort**: Detected from emergency stop events in metadata
- **Number of Attempts**: Count of retry sequences based on stage progression

#### Dataset-Level Metrics
- **Averages**: Mean values across all episodes
- **Rates**: Success rate, safety abort rate as fractions
- **Stage Durations**: Average time spent in each progress stage (including retries)

### Workflow

1. **Check Existing Scores**: Warns about existing dataset_eval_score.yaml
2. **Process Episodes**: For each episode:
   - Check for existing eval_score.yaml
   - Launch interactive labeler if needed
   - Calculate episode metrics
   - Save episode score
3. **Calculate Dataset Metrics**: Aggregate all episode scores
4. **Save Dataset Score**: Save YAML and JSON formats
5. **Display Summary**: Show key metrics

### Quality Assurance

#### Validation Features
- **Overlap Detection**: Automatically detects overlapping intervals between stages
- **Visual Feedback**: Real-time indication of label validity
- **Selective Deletion**: Ability to remove specific problematic intervals
- **Graceful Recovery**: Handle window closure and interruptions

#### Best Practices
- **Non-overlapping Labels**: Ensure progress stages don't overlap in time (this is enforced by `TaskProgressLabeler`)
- **Complete Sequences**: Label full progression sequences for accurate attempt counting
- **Consistent Timing**: Use consistent criteria for stage start/end points
- **Validation**: Review overlap warnings before finalizing labels

## File Formats Summary

| File | Purpose | Format | Location |
|------|---------|--------|----------|
| `splits.yaml` | Dataset splits | YAML | Source dataset root |
| `task_config.yaml` | Task configuration | YAML | Episode directories |
| `eval_config.yaml` | Evaluation parameters | YAML | User-provided |
| `task_spec.yaml` | Score definitions | YAML | Task directory |
| `metadata.json` | Episode metadata | JSON | Episode directories |
| `sync_logs.jsonl` | Synchronization data | JSONL | Episode directories |
| `eval_score.yaml` | Episode metrics | YAML | Episode directories |
| `dataset_eval_score.yaml` | Dataset metrics | YAML | Dataset root |
| `dataset_eval_score.json` | Dataset metrics | JSON | Dataset root |

## Troubleshooting
TODO move this somewhere else?

### Common Issues

1. **Video Synchronization**: Use frame-based seeking for consistent playback
2. **Overlap Detection**: Use TAB and X keys to select and delete problematic intervals
3. **Window Closure**: Automatic detection prevents hanging processes
4. **Missing Episodes**: Check splits.yaml and episode directory structure
5. **Calibration Errors**: Ensure robot/teleop IDs match between source and eval configs

### Error Recovery

- **Interrupted Labeling**: Labels are preserved and can be resumed
- **Invalid Intervals**: Clear validation messages guide correction
- **Processing Errors**: Continue/skip options for robust batch processing
- **Device Compatibility**: Automatic validation with clear error messages
