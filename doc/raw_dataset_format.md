TODO(sherry): Update this!

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

## Directory Structure

```
datasets/
  {dataset_name}/
    arm_calib/
      leader.json                 # Leader arm calibration data
      follower.json              # Follower arm calibration data
    splits.yaml                  # Dataset splits (train/val_id/val_ood)
    manifest.jsonl              # One JSON entry per episode
    episodes/
      {episode_id}/              # e.g., 2025-01-15T10-22-01Z_d001
        metadata.json                # Episode-level metadata
        obs/
          cam_front/
            000001.jpg           # Camera frames
            000002.jpg
            ...
            timestamps.jsonl       # Frame timestamps
          cam_top/
            000001.jpg
            000002.jpg
            ...
            timestamps.jsonl
          leader_trajectory.jsonl    # Leader arm joint positions
          follower_trajectory.jsonl  # Follower arm joint positions
        video_cam_front.mp4      # Generated video from frames
        video_cam_top.mp4
        sync_logs.jsonl          # Synchronization data
        recorder_metrics.jsonl   # Performance metrics
```

## File Formats

### Episode Metadata (`metadata.json`)

```json
{
  "episode_id": "episode_001_2025-01-15_10-22-01",
  "start_time": "2025-01-15T10:22:01.123456+00:00",
  "end_time": "2025-01-15T10:22:31.456789+00:00",
  "duration_s": 30.333333,
  "run_mode": "teleop",  // or "policy"
  "fps": 30,
  "leader_id": "leader_arm",
  "follower_id": "follower_arm",
  "total_frames": 910,
  "cameras": ["cam_front", "cam_top"],
  "task_description": "Pick and place red cube",
  "policy": {  // Only present if run_mode is "policy"
    "policy_path": "/path/to/policy",
    "policy_name": "PolicyClassName"
  }
}
```

### Camera timestamps (`timestamps.jsonl`)

```json
{"timestamp": 1642248121.123, "frame_idx": 1}
```
### Trajectory Data (`*_trajectory.jsonl`)

Each line is a JSON object representing one trajectory point:

```json
{"timestamp": 1642248121.123, "frame_idx": 1, "joint_0.pos": 0.1, "joint_1.pos": 0.2, "joint_2.pos": -0.1}
{"timestamp": 1642248121.156, "frame_idx": 2, "joint_0.pos": 0.11, "joint_1.pos": 0.21, "joint_2.pos": -0.09}
```

### Manifest (`manifest.jsonl`)

Each line describes one episode:

```json
{"episode_id": "episode_001_2025-01-15_10-22-01", "episode_dir": "episodes/episode_001_2025-01-15_10-22-01", "task_description": "Pick red cube", "duration_s": 30.33, "total_frames": 910, "cameras": ["cam_front", "cam_top"], "start_time": "2025-01-15T10:22:01+00:00", "end_time": "2025-01-15T10:22:31+00:00"}
```

### Splits (`splits.yaml`)

```yaml
train:
  - "2025-01-15T10-22-01Z_d001"
  - "2025-01-15T10-25-15Z_d001"
val_id:
  - "2025-01-15T10-30-22Z_d001"
val_ood: []
```

### Sync Logs (`sync_logs.jsonl`)
NOTE: For now all the timestamps will be the same, because LeRobot doesn't actually surface async camera timestamps, and joint positions are read synchronously.

TODO(sherry): Surface camera timestamps to monitor recorder performance.

Each line contains synchronization information for one frame:

```json
{"sequence_number": 1, "timestamp": 1642248121.123, "frame_idx": 1, "camera_timestamps": {"cam_front": 1642248121.120, "cam_top": 1642248121.121}, "leader_timestamp": 1642248121.123, "follower_timestamp": 1642248121.123}
```

### Recorder Metrics (`recorder_metrics.jsonl`)

Performance metrics logged during recording:

```json
{"timestamp": 1642248121.123, "frame_idx": 100, "fps": 29.8, "dropped_frames": 2, "memory_usage_mb": 150.5}
```

## Usage

### Enabling Raw Format Recording

Add these flags to your `lerobot-record` command:

```bash
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index: 0, width: 640, height: 480}}" \
    --robot.id=follower \
    --dataset.repo_id=user/test-dataset \
    --dataset.num_episodes=5 \
    --dataset.single_task="Pick the red cube" \
    --dataset.save_raw_format=true \
    --dataset.raw_format_root=/path/to/raw/datasets \
    --dataset.raw_format_videos=true
```

### Configuration Options

- `--dataset.save_raw_format=true`: Enable raw format recording
- `--dataset.raw_format_root=/path`: Root directory for raw datasets (optional, defaults to dataset.root)
- `--dataset.raw_format_videos=true`: Generate video files from camera frames (default: true)

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

### For Other Models
- Standard image formats work with any vision model
- Trajectory data is framework-agnostic
- Flexible structure supports different model architectures
- No dependency on LeRobot-specific formats

## Reading Raw Data

### Python Example

```python
import json
import jsonlines
from pathlib import Path
import yaml

# Load episode metadata
episode_dir = Path("datasets/my_dataset/episodes/2025-01-15T10-22-01Z_d001")
with open(episode_dir / "meta.json") as f:
    meta = json.load(f)

# Load trajectory data
trajectories = []
with jsonlines.open(episode_dir / "obs/follower_trajectory.jsonl") as reader:
    for point in reader:
        trajectories.append(point)

# Load splits
with open("datasets/my_dataset/splits.yaml") as f:
    splits = yaml.safe_load(f)

# Load camera frames
import cv2
frame_1 = cv2.imread(str(episode_dir / "obs/cam_front/000001.jpg"))
```

## Integration with LeRobot

The raw format runs alongside the standard LeRobot format without interference:

- Both formats record simultaneously when enabled
- No impact on LeRobot dataset functionality
- Raw format can be disabled without affecting LeRobot recording
- Shared image writer processes for efficiency

## Performance Considerations

- Image writing uses the same async processes as LeRobot format
- Video encoding happens after episode completion
- Minimal overhead when raw format is disabled
- Memory usage scales with number of cameras and frame rate

