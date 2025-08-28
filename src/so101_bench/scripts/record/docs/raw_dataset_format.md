

# Raw Dataset Format for LeRobot

This document describes the raw dataset format that can be recorded alongside the standard LeRobot format for easier debugging, visualization, and training with other models.

## Overview

The raw format saves robot data in a human-readable, hierarchical structure that's easy to navigate and process. It includes:

- Individual camera frames as JPEG images
- Trajectory data as JSONL files
- Episode metadata as JSON
- Video files generated from camera frames
- Synchronization and performance logs
- Arm calibration data
- Task config (This is required for running the eval pipeline).

## Directory Structure

```
datasets/
  {dataset_name}/
    arm_calib/
      {leader_id}.json           # Leader arm calibration data
      {follower_id}.json         # Follower arm calibration data
    splits.yaml                  # Dataset splits (train/val_id/val_ood)
    manifest.jsonl              # One JSON entry per episode
    task_config.yaml             # Dataset-level task config. Usually only contains task_name.
    episodes/
      {episode_idx}_{timestamp}/ # e.g., 001_2025-01-15_10-22-01
        metadata.json            # Episode-level metadata
        obs/
          {camera_name}/         # e.g., cam_front, cam_top
            000001.jpg           # Camera frames (sequence_number format)
            000002.jpg
            ...
            timestamps.jsonl     # Frame timestamps with sequence numbers
          leader_trajectory.jsonl    # Leader arm joint positions
          follower_trajectory.jsonl  # Follower arm joint positions
        video_{camera_name}.mp4  # Generated video from frames
        sync_logs.jsonl          # Synchronization data
        task_config.yaml         # The task configuration in this episode.
        recorder_metrics.jsonl   # Performance metrics (TODO)
```

## File Formats

### Episode Metadata (`metadata.json`)

```json
{
  "episode_id": "001_2025-01-15_10-22-01",
  "episode_idx": 1,
  "start_time": "2025-01-15T10:22:01.123456+00:00",
  "end_time": "2025-01-15T10:22:31.456789+00:00",
  "duration_s": 30.333333,
  "run_mode": "teleop",
  "fps": 30,
  "actual_fps": 29.8,
  "leader_id": "leader_arm",
  "follower_id": "follower_arm",
  "total_frames": 910,
  "cameras": ["cam_front", "cam_top"],
  "task_description": "Pick and place red cube",
  "events": [
    [1642248125.456, "WARNING_EMERGENCY_STOP_PRESSED"]
  ],
  "policy": {
    "policy_path": "/path/to/policy",
    "policy_name": "PolicyClassName"
  }
}
```

**Notes:**
- `episode_idx`: Numeric episode index
- `actual_fps`: Calculated FPS based on recorded timestamps
- `events`: List of [timestamp, event_name] tuples for significant events during recording
- `policy`: Only present if `run_mode` is "policy"

### Camera timestamps (`timestamps.jsonl`)

```json
{"sequence_number": 1, "timestamp": 1642248121.123}
```
### Trajectory Data (`*_trajectory.jsonl`)

Each line is a JSON object representing one trajectory point:

```json
{"sequence_number": 1, "timestamp": 1642248121.123, "joint_0.pos": 0.1, "joint_1.pos": 0.2, "joint_2.pos": -0.1}
{"sequence_number": 2, "timestamp": 1642248121.156, "joint_0.pos": 0.11, "joint_1.pos": 0.21, "joint_2.pos": -0.09}
```

### Manifest (`manifest.jsonl`)

Each line describes one episode:

```json
{"episode_id": "001_2025-01-15_10-22-01", "episode_dir": "episodes/001_2025-01-15_10-22-01", "task_description": "Pick red cube", "duration_s": 30.33, "total_frames": 910, "actual_fps": 29.8, "cameras": ["cam_front", "cam_top"], "start_time": "2025-01-15T10:22:01+00:00", "end_time": "2025-01-15T10:22:31+00:00"}
```

### Splits (`splits.yaml`)

```yaml
train:
  - "001_2025-01-15_10-22-01"
  - "002_2025-01-15_10-25-15"
val_id:
  - "003_2025-01-15_10-30-22"
val_ood: []
```

### Sync Logs (`sync_logs.jsonl`)

Each line contains synchronization information for one frame:

```json
{"sequence_number": 1, "timestamp": 1642248121.123, "robot_state_timestamp": 1642248121.120, "camera_timestamps": {"cam_front": 1642248121.120, "cam_top": 1642248121.121}, "action_timestamp": 1642248121.123}
```

### Task config (`task_config.yaml`)
To configure a task, we first need to know what's possible to configure.
All supported tasks are in `so101_bench/datasets/tasks`. Within each folder you will find a `task_spec.yaml` and `task_config_template.yaml`. Use the former to guide the latter. Pass the latter into the recorder as you record episodes for training / eval.

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

#### Task configs generated by recorder

Dataset-level:
```yaml
task_name: pick_and_place_block
```

Episode-level:
```yaml
task_name: pick_and_place_block
variations:
  block: white
  container: tupperware
  start_pose:
    block:
    - 0.1
    - 0.1
    - 0.0
    container:
    - 0.0
    - 0.0
    - 0.0

# The following are filled in if this is an eval-rollout episode.
is_eval: true
source_episode_dir: 2025-08-25_test/episodes/012_2025-08-26_22-53-39
```

### (TODO) Recorder Metrics (`recorder_metrics.jsonl`)

Performance metrics logged during recording:

```json
{"timestamp": 1642248121.123, "sequence_number": 100, "fps": 29.8, "dropped_frames": 2, "memory_usage_mb": 150.5}
```

## Benefits

### For Debugging
- Individual frames can be easily inspected
- Trajectory data is in human-readable format
- Synchronization logs help identify timing issues
- Performance metrics track recorder health

### For Visualization
- Videos provide easy episode playback
- Images can be used in custom visualization tools
- Trajectory data can be plotted directly
- Metadata provides context for analysis
