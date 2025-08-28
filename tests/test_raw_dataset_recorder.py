#!/usr/bin/env python

"""
Test script for the raw dataset recorder functionality.
This demonstrates how the raw format works alongside the LeRobot format.
"""

import numpy as np
import tempfile
import time
from pathlib import Path

from so101_bench.raw_dataset_recorder import RawDatasetRecorder


def test_raw_recorder():
    """Test the raw dataset recorder with synthetic data."""
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing raw recorder in: {temp_dir}")
        
        # Initialize raw recorder
        robot_config = {
            "type": "so101_follower",
            "id": "test_robot",
            "calibration_dir": Path(__file__).parent / "test_data" / "calibration",
            "cameras": {
                "cam_front": {"type": "opencv", "index": 0}
            },
            "calibration": {
                "joint_1": {
                    "id": 1,
                    "drive_mode": 0,
                    "homing_offset": 136,
                    "range_min": 687,
                    "range_max": 3358
                }
            }
        }
        teleop_config = {
            "type": "teleop",
            "id": "test_teleop",
            "calibration_dir": Path(__file__).parent / "test_data" / "calibration",
            "cameras": {
                "cam_front": {"type": "opencv", "index": 0}
            },
            "calibration": {
                "joint_1": {
                    "id": 1,
                    "drive_mode": 0,
                    "homing_offset": 136,
                    "range_min": 687,
                    "range_max": 3358
                }
            }
        }
        recorder = RawDatasetRecorder(
            dataset_name="test_dataset",
            root_dir=temp_dir,
            robot_config=robot_config,
            teleop_config=teleop_config,
            fps=30,
            save_videos=True,
            image_writer_processes=0,
            image_writer_threads=1,
        )
        
        # Start an episode
        episode_id = recorder.start_episode(
            episode_idx=0,
            run_mode="teleop",
            leader_id="leader_arm",
            follower_id="follower_arm"
        )
        print(f"Started episode: {episode_id}")
        
        # Simulate recording frames
        num_frames = 10
        for frame_idx in range(num_frames):
            # Create synthetic observation data
            observation = {
                "cam_front": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "cam_top": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "joint1.pos": 1.0,
            }
            
            # Create synthetic action data
            action = {
                "joint1.pos": 0.8,
            }
            
            # Add frame
            timestamp = time.time() + frame_idx * 0.033  # ~30 fps
            recorder.add_frame(
                observation=observation,
                action=action,
                frame_timestamp=timestamp,
                sequence_number=frame_idx + 1
            )
        
        # Save episode
        episode_meta = recorder.save_episode(task_description="Test pick and place task")
        print(f"Saved episode with metadata: {episode_meta}")
        
        # Verify directory structure
        dataset_dir = Path(temp_dir) / "datasets" / "test_dataset"
        print(f"\nDataset directory structure:")
        for item in sorted(dataset_dir.rglob("*")):
            if item.is_file():
                print(f"  {item.relative_to(dataset_dir)}")
        
        # Update splits
        recorder.update_splits({
            "train": [episode_id],
            "val_id": [],
            "val_ood": []
        })
        
        # Cleanup
        recorder.cleanup()
        
        print("\n✅ Raw recorder test completed successfully!")
        
        # Show example of how to read the data back
        print(f"\nExample raw dataset structure at: {dataset_dir}")
        print("Files created:")
        print("- splits.yaml: Dataset split definitions")
        print("- manifest.jsonl: Episode metadata")
        print("- arm_calib/: Arm calibration files")
        print(f"- episodes/{episode_id}/: Episode data including:")
        print("  - meta.json: Episode metadata")
        print("  - obs/: Observation data")
        print("    - cam_front/: Front camera images and timestamps")
        print("    - cam_top/: Top camera images and timestamps") 
        print("    - leader_trajectory.jsonl: Leader arm trajectory")
        print("    - follower_trajectory.jsonl: Follower arm trajectory")
        print("  - video_cam_front.mp4: Front camera video")
        print("  - video_cam_top.mp4: Top camera video")
        print("  - sync_logs.jsonl: Synchronization logs")
        print("  - recorder_metrics.jsonl: Performance metrics")


if __name__ == "__main__":
    test_raw_recorder()

