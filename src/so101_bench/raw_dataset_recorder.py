#!/usr/bin/env python
"""
Raw dataset recorder for LeRobot that saves data in a human-readable format
for easy debugging, visualization, and training other models.

See docs/raw_dataset_format.md for the output format.
"""

import json
import logging
import os
import time
from deepdiff import DeepDiff
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import jsonlines
import numpy as np
import yaml
from PIL import Image

from so101_bench.image_writer import AsyncImageWriter


class RawDatasetRecorder:
    """Records robot data in a raw, human-readable format alongside LeRobot format."""
    METADATA_FILENAME = "metadata.json"
    SPLITS_FILENAME = "splits.yaml"
    MANIFEST_FILENAME = "manifest.jsonl"
    TASK_CONFIG_FILENAME = "task_config.yaml"

    EVENTS_TO_RECORD = {
        "emergency_stop": "WARNING_EMERGENCY_STOP_PRESSED",
    }
    
    def __init__(
        self,
        dataset_name: str,
        root_dir: str | Path,
        robot_config: Dict[str, Any],
        robot_calibration_fpath: Path,
        teleop_config: Dict[str, Any],
        teleop_calibration_fpath: Path,
        fps: int = 30,
        save_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 4,
    ):
        """
        Initialize raw dataset recorder.
        
        Args:
            dataset_name: Name of the dataset
            root_dir: Root directory to save the dataset
            robot_config: Robot configuration including camera setup
            robot_calibration_fpath: Path to robot calibration file
            teleop_config: Teleop configuration
            teleop_calibration_fpath: Path to teleop calibration file
            fps: Recording frame rate
            save_videos: Whether to save video files from camera frames
            image_writer_processes: Number of processes for image writing
            image_writer_threads: Number of threads per camera for image writing
        """
        self.dataset_name = dataset_name
        self.root_dir = Path(root_dir)
        self.robot_config = robot_config
        self.teleop_config = teleop_config
        self.fps = fps
        self.save_videos = save_videos
        
        # Create dataset directory structure
        self.dataset_dir = self.root_dir / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.episodes_dir = self.dataset_dir / "episodes"
        self.episodes_dir.mkdir(exist_ok=True)
        
        self.arm_calib_dir = self.dataset_dir / "arm_calib"
        self.arm_calib_dir.mkdir(exist_ok=True)

        # Save leader and follower arm calibration
        self._save_arm_calibration(robot_calibration_fpath, robot_config)
        self._save_arm_calibration(teleop_calibration_fpath, teleop_config)

        # Save camera configs
        self._save_camera_configs(robot_config)
        
        # Initialize splits file if it doesn't exist
        self.splits_file = self.dataset_dir / self.SPLITS_FILENAME
        if not self.splits_file.exists():
            splits = {"train": [], "val_id": [], "val_ood": []}
            with open(self.splits_file, "w") as f:
                yaml.dump(splits, f, default_flow_style=False)
        
        # Initialize manifest file
        self.manifest_file = self.dataset_dir / self.MANIFEST_FILENAME
        
        self._reset_episode_data()
        
        # Image writer for async image saving
        self.image_writer = None
        if image_writer_processes > 0 or image_writer_threads > 0:
            self.image_writer = AsyncImageWriter(
                num_processes=image_writer_processes,
                num_threads=image_writer_threads
            )
        
        logging.info(f"Raw dataset recorder initialized at {self.dataset_dir}")
    
    def _reset_episode_data(self):
        # Current episode data
        self.current_episode = None
        self.current_episode_dir = None
        self.episode_start_time = None
        self.frame_count = 0
        
        # Camera frame buffers for video creation
        self.camera_frame_buffers = {}
        self.camera_timestamps = {}
        
        # Trajectory buffers
        self.leader_trajectory = []
        self.follower_trajectory = []

        # Events. List of tuples of (event_timestamp, event_name)
        self.events = []
        
        # Sync and metrics logs
        self.sync_logs = []
        self.recorder_metrics = []

    def _save_arm_calibration(self, calibration_fpath: Path, arm_config: dict):
        incoming_calib = json.load(open(calibration_fpath))
        calibration_write_path = self.arm_calib_dir / f"{arm_config['id']}.json"

        if os.path.exists(calibration_write_path):
            existing_calib = json.load(open(calibration_write_path))
            # Direct comparison is possible because all numeric values are ints.
            if existing_calib != incoming_calib:
                raise ValueError(f"Cannot record to same dataset_dir: "
                    f"{self.dataset_dir} with different calibration for "
                    f"{arm_config['id']}: {existing_calib} != {incoming_calib}"
                )
        else:
            with open(calibration_write_path, "w") as f:
                json.dump(incoming_calib, f, indent=2)
    
    def _save_camera_configs(self, robot_config: dict):
        configs_write_path = self.dataset_dir / "camera_configs.json"
        
        if os.path.exists(configs_write_path):
            existing_configs = json.load(open(configs_write_path))
            # Go through a json conversion to convert everything to primitive types
            # in order to compare them.
            incoming_configs = json.loads(json.dumps(robot_config["cameras"], indent=2))
            # Approximate comparison of camera configs.
            diff = DeepDiff(existing_configs, incoming_configs, significant_digits=3)

            if diff:
                raise ValueError(f"Cannot record to same dataset_dir: "
                    f"{self.dataset_dir} with different camera configs: "
                    f"{diff}"
                )
        else:
            with open(configs_write_path, "w") as f:
                json.dump(robot_config["cameras"], f, indent=2)

    def start_episode(
        self,
        episode_idx: int,
        run_mode: str = "teleop",
        policy_info: Optional[Dict[str, Any]] = None,
        leader_id: Optional[str] = None,
        follower_id: Optional[str] = None,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start recording a new episode.
        
        Args:
            episode_idx: Episode index
            run_mode: Recording mode ("teleop" or "policy")
            policy_info: Policy information if run_mode is "policy"
            leader_id: Leader arm ID
            follower_id: Follower arm ID
            task_config: Task configuration dictionary to save with episode
            
        Returns:
            Episode ID string
        """
        # Generate episode ID with timestamp
        timestamp = datetime.now(timezone.utc)
        episode_id = f"{episode_idx:03d}" + "_" + timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        
        assert self.current_episode is None, f"Episode already started: {self.current_episode}"

        self.current_episode = episode_id
        self.current_episode_dir = self.episodes_dir / episode_id
        self.current_episode_dir.mkdir(exist_ok=True)
        
        # Create obs subdirectory
        obs_dir = self.current_episode_dir / "obs"
        obs_dir.mkdir(exist_ok=True)
        
        # Initialize episode metadata
        episode_metadata = {
            "episode_id": episode_id,
            "episode_idx": episode_idx,
            "start_time": timestamp.isoformat(),
            "run_mode": run_mode,
            "fps": self.fps,
            "leader_id": leader_id,
            "follower_id": follower_id,
        }
        
        if policy_info:
            episode_metadata["policy"] = policy_info
        
        # Save episode metadata
        metadata_file = self.current_episode_dir / self.METADATA_FILENAME
        with open(metadata_file, "w") as f:
            json.dump(episode_metadata, f, indent=2)
        
        # Save task configuration if provided
        if task_config is not None:
            task_config_file = self.current_episode_dir / self.TASK_CONFIG_FILENAME
            with open(task_config_file, "w") as f:
                yaml.dump(task_config, f, default_flow_style=False, sort_keys=False)
            logging.info(f"Saved task configuration for episode: {episode_id}")
        
        logging.info(f"Started recording episode: {episode_id}")
        return episode_id
    
    def add_event(
        self,
        frame_timestamp: float,
        events: Dict[str, bool],
    ):
        """
        Add an event to the current episode.
    
        Args:
            frame_timestamp: Frame timestamp
            events: Events from the control loop. 
                Keyed by event name. True if the event is triggered.
        """
        for event_name, triggered in events.items():
            if event_name in self.EVENTS_TO_RECORD and triggered:
                self.events.append((frame_timestamp, self.EVENTS_TO_RECORD[event_name]))

    def add_frame(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        frame_timestamp: float,
        sequence_number: int,
    ):
        """
        Add a frame of data to the current episode.
        
        Args:
            observation: Robot observation data including camera images
            action: Robot action data
            frame_timestamp: Frame timestamp
            sequence_number: Frame sequence number
        """
        if self.current_episode is None:
            raise RuntimeError("No episode started. Call start_episode() first.")
        
        # Process camera observations
        for cam_name, cam_data in observation.items():
            if "cam" not in cam_name:
                continue

            if cam_name.endswith("_timestamp"):
                camera_key = cam_name.split("_timestamp")[0]
                if camera_key not in self.camera_timestamps:
                    self.camera_timestamps[camera_key] = []

                self.camera_timestamps[camera_key].append(cam_data)
                continue

            camera_key = cam_name

            # Create camera directory
            cam_dir = self.current_episode_dir / "obs" / camera_key
            cam_dir.mkdir(exist_ok=True)
            
            # Save image
            if not isinstance(cam_data, np.ndarray):
                logging.warning(f"Camera {cam_name} data is not a numpy array: {cam_data}")
                # Skipping this frame
                return
            
            # Convert numpy array to PIL Image
            if cam_data.dtype == np.uint8 and len(cam_data.shape) == 3:
                # RGB image
                image = Image.fromarray(cam_data)
            else:
                # Convert to uint8 if needed
                if cam_data.max() <= 1.0:
                    cam_data = (cam_data * 255).astype(np.uint8)
                image = Image.fromarray(cam_data.astype(np.uint8))
            
            # Save image
            image_path = cam_dir / f"{sequence_number:06d}.jpg"
            if self.image_writer:
                self.image_writer.save_image(image=image, fpath=image_path)
            else:
                image.save(image_path, "JPEG", quality=95)
            
            # Store for video creation
            if self.save_videos:
                if camera_key not in self.camera_frame_buffers:
                    self.camera_frame_buffers[camera_key] = []
                
                # Convert PIL image back to numpy for video
                frame_array = np.array(image)
                if len(frame_array.shape) == 3:
                    # Convert RGB to BGR for OpenCV
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                
                self.camera_frame_buffers[camera_key].append(frame_array)
        
        # Save camera timestamps
        for cam_name, cam_timestamp in self.camera_timestamps.items():
            timestamp_file = self.current_episode_dir / "obs" / cam_name / "timestamps.jsonl"
            with jsonlines.open(timestamp_file, "a") as writer:
                writer.write({"sequence_number": sequence_number, "timestamp": cam_timestamp})
        
        # Process arm trajectories
        leader_joints = {}
        follower_joints = {}
        
        for key, value in observation.items():
            if "pos" in key:
                leader_joints[key] = value
        
        for key, value in action.items():
            if "pos" in key:
                follower_joints[key] = value
        
        # Add trajectory points
        if leader_joints:
            leader_point = {
                "sequence_number": sequence_number,
                "timestamp": action["action_timestamp"],
                **leader_joints
            }
            self.leader_trajectory.append(leader_point)
        
        if follower_joints:
            follower_point = {
                "sequence_number": sequence_number,
                "timestamp": observation["robot_state_timestamp"],
                **follower_joints
            }
            self.follower_trajectory.append(follower_point)
        
        # Add sync log entry
        sync_entry = {
            "sequence_number": sequence_number,
            "timestamp": frame_timestamp,
            "robot_state_timestamp": observation["robot_state_timestamp"],
            "camera_timestamps": {cam: self.camera_timestamps[cam][-1] 
                                for cam in self.camera_timestamps},
            "action_timestamp": action["action_timestamp"],
        }
        self.sync_logs.append(sync_entry)
        self.frame_count += 1
    
    def save_episode(self, task_description: str = "") -> Dict[str, Any]:
        """
        Save the current episode to disk.
        
        Args:
            task_description: Description of the task performed
            
        Returns:
            Episode metadata dictionary
        """
        if self.current_episode is None:
            raise RuntimeError("No episode to save. Call start_episode() first.")
        
        episode_id = self.current_episode
        episode_dir = self.current_episode_dir
        
        # Save trajectories
        if self.leader_trajectory:
            leader_file = episode_dir / "obs" / "leader_trajectory.jsonl"
            with jsonlines.open(leader_file, "w") as writer:
                for point in self.leader_trajectory:
                    writer.write(point)
        
        if self.follower_trajectory:
            follower_file = episode_dir / "obs" / "follower_trajectory.jsonl"
            with jsonlines.open(follower_file, "w") as writer:
                for point in self.follower_trajectory:
                    writer.write(point)
        
        # Save sync logs
        # TODO(sherry): Surface camera timestamps to monitor recorder performance.
        if self.sync_logs:
            sync_file = episode_dir / "sync_logs.jsonl"
            with jsonlines.open(sync_file, "w") as writer:
                for entry in self.sync_logs:
                    writer.write(entry)
        
        # Save recorder metrics
        if self.recorder_metrics:
            metrics_file = episode_dir / "recorder_metrics.jsonl"
            with jsonlines.open(metrics_file, "w") as writer:
                for entry in self.recorder_metrics:
                    writer.write(entry)
        
        # Create videos from camera frames
        if self.save_videos:
            self._create_videos()
        
        # Calculate actual FPS from timestamp data
        actual_fps = self._calculate_actual_fps()
        
        # Update episode metadata with final info
        meta_file = episode_dir / self.METADATA_FILENAME
        with open(meta_file, "r") as f:
            episode_metadata = json.load(f)
        
        episode_metadata.update({
            "end_time": datetime.now(timezone.utc).isoformat(),
            "duration_s": self._calculate_duration_s(),
            "total_frames": self.frame_count,
            "actual_fps": actual_fps,
            "task_description": task_description,
            "cameras": list(self.camera_frame_buffers.keys()),
            "events": self.events,
        })
        
        with open(meta_file, "w") as f:
            json.dump(episode_metadata, f, indent=2)
        
        # Add to manifest
        manifest_entry = {
            "episode_id": episode_id,
            "episode_dir": f"episodes/{episode_id}",
            "task_description": task_description,
            "duration_s": episode_metadata["duration_s"],
            "total_frames": self.frame_count,
            "actual_fps": episode_metadata["actual_fps"],
            "cameras": episode_metadata["cameras"],
            "start_time": episode_metadata["start_time"],
            "end_time": episode_metadata["end_time"],
        }
        
        with jsonlines.open(self.manifest_file, "a") as writer:
            writer.write(manifest_entry)
        
        logging.info(f"Saved episode {episode_id} with {self.frame_count} frames")
        
        self._reset_episode_data()
        
        return episode_metadata
    
    def _calculate_actual_fps(self) -> float:
        """
        Calculate the actual average FPS from recorded timestamp data.
        
        Returns:
            Average FPS across all recorded frames, or the configured FPS if calculation fails
        """
        if not self.sync_logs or len(self.sync_logs) < 2:
            logging.warning("Insufficient timestamp data to calculate actual FPS, using configured FPS")
            return float(self.fps)
        
        # Extract timestamps from sync logs (these are the main frame timestamps)
        timestamps = [entry["timestamp"] for entry in self.sync_logs]
        
        # Calculate time intervals between consecutive frames
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            if interval > 0:  # Only include positive intervals
                intervals.append(interval)
        
        if not intervals:
            logging.warning("No valid time intervals found, using configured FPS")
            return float(self.fps)
        
        # Calculate average interval and convert to FPS
        avg_interval = sum(intervals) / len(intervals)
        actual_fps = 1.0 / avg_interval
        
        return round(actual_fps, 2)
    
    def _calculate_duration_s(self) -> float:
        """
        Calculate the duration of the episode in seconds.
        """
        if not self.sync_logs or len(self.sync_logs) < 2:
            logging.warning("Insufficient timestamp data to calculate duration, using configured FPS")
            return float(self.fps)
        
        # Extract timestamps from sync logs (these are the main frame timestamps)
        last_timestamp = self.sync_logs[-1]["timestamp"]
        first_timestamp = self.sync_logs[0]["timestamp"]
        duration = last_timestamp - first_timestamp
        return duration
        
    def _create_videos(self):
        """Create video files from camera frame buffers."""
        for camera_key, frames in self.camera_frame_buffers.items():
            if not frames:
                continue
            
            video_path = self.current_episode_dir / f"video_{camera_key}.mp4"
            
            # Get frame dimensions
            height, width = frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(video_path), fourcc, self.fps, (width, height)
            )
            
            try:
                for frame in frames:
                    video_writer.write(frame)
                logging.info(f"Created video: {video_path}")
            except Exception as e:
                logging.error(f"Error creating video {video_path}: {e}")
            finally:
                video_writer.release()
    
    def update_splits(self, splits: Dict[str, List[str]]):
        """Update the SPLITS_FILENAME file with episode assignments."""
        with open(self.splits_file, "w") as f:
            yaml.dump(splits, f, default_flow_style=False)
        
        logging.info("Updated dataset splits")
    
    def cleanup(self):
        """Clean up resources."""
        if self.image_writer:
            self.image_writer.stop()
            self.image_writer = None
        
        logging.info("Raw dataset recorder cleaned up")
