#!/usr/bin/env python3
"""
Convert LeRobotDataset (v2.1) to ROS2 MCAP file for Foxglove visualization.

Usage:
    python lerobot_to_ros2_mcap.py <dataset_dir> <episode_index> <output_mcap_dir>

Example:
    python convert_lerobot_to_ros2_mcap.py \
        --dataset_dir ../../lerobot_data/sherryxychen/eval_test_2025-09-14_smolvla \
        --episode_index 0
"""

import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any, List

# ROS2 imports
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time
from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image, CompressedImage, JointState
from cv_bridge import CvBridge
import cv2


def load_dataset_metadata(dataset_dir: Path) -> Dict[str, Any]:
    """Load dataset metadata from info.json."""
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json not found at {info_path}")
    
    with open(info_path, 'r') as f:
        return json.load(f)


def get_ros_timestamp(timestamp_sec: float) -> Time:
    """Convert float timestamp to ROS Time message."""
    sec = int(timestamp_sec)
    nanosec = int((timestamp_sec - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)


def create_header(timestamp_sec: float, frame_id: str = "") -> Header:
    """Create a ROS Header with timestamp."""
    header = Header()
    header.stamp = get_ros_timestamp(timestamp_sec)
    header.frame_id = frame_id
    return header


def create_float_array_msg(data: np.ndarray, timestamp_sec: float, names: List[str] = None) -> Float32MultiArray:
    """Create a Float32MultiArray message with optional dimension labels."""
    msg = Float32MultiArray()
    
    # Set data
    msg.data = data.flatten().tolist()
    
    # Set layout
    dim = MultiArrayDimension()
    dim.label = "joints" if names else "data"
    dim.size = len(data)
    dim.stride = len(data)
    msg.layout.dim.append(dim)
    msg.layout.data_offset = 0
    
    return msg


def create_image_msg(frame: np.ndarray, timestamp_sec: float, frame_id: str, bridge: CvBridge, encoding: str = "rgb8") -> Image:
    """Create a ROS Image message from numpy array."""
    img_msg = bridge.cv2_to_imgmsg(frame, encoding=encoding)
    img_msg.header = create_header(timestamp_sec, frame_id)
    return img_msg


def create_joint_state_msg(data: np.ndarray, timestamp_sec: float, names: List[str] = None) -> JointState:
    """Create a JointState message from numpy array."""
    msg = JointState()
    msg.header = create_header(timestamp_sec)
    
    # Set joint names
    if names:
        msg.name = names
    else:
        msg.name = [f"joint_{i}" for i in range(len(data))]
    
    # Set positions
    msg.position = data.flatten().tolist()
    
    # Leave velocity and effort empty
    msg.velocity = []
    msg.effort = []
    
    return msg


class VideoReader:
    """Manages multiple video files for efficient frame-by-frame reading."""
    
    def __init__(self):
        self.video_captures: Dict[str, cv2.VideoCapture] = {}
    
    def open_video(self, feature_name: str, video_path: Path):
        """Open a video file for reading."""
        if not video_path.exists():
            print(f"  Warning: Video not found at {video_path}")
            return False
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  Warning: Could not open video at {video_path}")
            return False
        
        self.video_captures[feature_name] = cap
        return True
    
    def get_frame(self, feature_name: str, frame_idx: int) -> np.ndarray:
        """Get a specific frame from a video."""
        if feature_name not in self.video_captures:
            raise ValueError(f"Video for {feature_name} not loaded")
        
        cap = self.video_captures[feature_name]
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {feature_name}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def close_all(self):
        """Release all video captures."""
        for cap in self.video_captures.values():
            cap.release()
        self.video_captures.clear()


def load_parquet_data(dataset_dir: Path, episode_chunk: int, episode_index: int) -> Dict[str, np.ndarray]:
    """Load data from parquet file for a specific episode."""
    import pyarrow.parquet as pq
    
    # Construct parquet path using the pattern from metadata
    parquet_path = dataset_dir / "data" / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")
    
    table = pq.read_table(parquet_path)
    return {col: table[col].to_numpy() for col in table.column_names}


def get_topic_type(feature_name: str, feature_info: Dict[str, Any]) -> str:
    """Determine ROS2 message type from feature info."""
    dtype = feature_info.get("dtype")
    
    # Use JointState for action and observation.state
    if feature_name in ["action", "observation.state"]:
        return "sensor_msgs/msg/JointState"
    elif dtype == "video":
        return "sensor_msgs/msg/Image"
    elif dtype in ["float32", "float64", "int32", "int64"]:
        return "std_msgs/msg/Float32MultiArray"
    else:
        return "std_msgs/msg/Float32MultiArray"


def setup_ros2_writer(output_path: Path) -> SequentialWriter:
    """Initialize ROS2 MCAP writer."""
    storage_options = StorageOptions(
        uri=str(output_path),
        storage_id='mcap'
    )
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)
    return writer


def convert_episode_to_mcap(dataset_dir: Path, episode_index: int, output_dir: Path):
    """Convert a LeRobotDataset episode to ROS2 MCAP."""
    # Load metadata
    print(f"Loading metadata from {dataset_dir}")
    metadata = load_dataset_metadata(dataset_dir)
    version = metadata["codebase_version"]
    assert version == "v2.1", f"Unsupported version: {version}"
    features = metadata["features"]
    chunks_size = metadata["chunks_size"]
    episode_chunk = episode_index // chunks_size
    
    print(f"Converting episode {episode_index} (chunk {episode_chunk})")
    
    # Initialize ROS2 MCAP writer
    print(f"Creating ROS2 MCAP at {output_dir}")
    output_path = output_dir / f"episode_{episode_index:06d}"
    writer = setup_ros2_writer(output_path)
    
    # Create topics based on features
    topic_configs = []
    topic_id = 0
    for feature_name, feature_info in features.items():
        # Skip metadata fields
        if feature_name in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            continue
        
        topic_name = f"/{feature_name.replace('.', '/')}"
        msg_type = get_topic_type(feature_name, feature_info)
        
        topic_configs.append({
            "name": topic_name,
            "type": msg_type,
            "feature": feature_name,
            "info": feature_info
        })
        
        # Create topic in bag
        from rosbag2_py import TopicMetadata
        topic_metadata = TopicMetadata(
            id=topic_id,
            name=topic_name,
            type=msg_type,
            serialization_format='cdr'
        )
        writer.create_topic(topic_metadata)
        print(f"  Created topic: {topic_name} ({msg_type})")
        topic_id += 1
    
    # Load data from parquet
    print("\nLoading data from parquet files...")
    data = load_parquet_data(dataset_dir, episode_chunk, episode_index)
    num_frames = len(data["timestamp"])
    print(f"Found {num_frames} frames")
    
    # Load all video files once
    print("\nOpening video files...")
    video_reader = VideoReader()
    for topic_config in topic_configs:
        feature_name = topic_config["feature"]
        feature_info = topic_config["info"]
        
        if feature_info["dtype"] == "video":
            video_path = dataset_dir / "videos" / f"chunk-{episode_chunk:03d}" / feature_name / f"episode_{episode_index:06d}.mp4"
            if video_reader.open_video(feature_name, video_path):
                print(f"  Opened video: {feature_name}")
    
    # Create CV bridge once for all image conversions
    bridge = CvBridge()
    
    # Process each frame
    print("\nConverting frames to ROS2 messages...")
    for frame_idx in range(num_frames):
        timestamp = float(data["timestamp"][frame_idx])
        
        if frame_idx % 10 == 0:
            print(f"  Processing frame {frame_idx}/{num_frames}")
        
        # Process each topic
        for topic_config in topic_configs:
            feature_name = topic_config["feature"]
            topic_name = topic_config["name"]
            feature_info = topic_config["info"]
            
            try:
                if feature_info["dtype"] == "video":
                    # Load frame from video file
                    frame = video_reader.get_frame(feature_name, frame_idx)
                    msg = create_image_msg(frame, timestamp, topic_name.split('/')[-1], bridge)
                    
                elif topic_config["type"] == "sensor_msgs/msg/JointState":
                    # Load array data and create JointState message
                    feature_data = data[feature_name][frame_idx]
                    msg = create_joint_state_msg(
                        feature_data, 
                        timestamp, 
                        feature_info.get("names")
                    )
                    
                else:
                    # Load array data
                    feature_data = data[feature_name][frame_idx]
                    msg = create_float_array_msg(
                        feature_data, 
                        timestamp, 
                        feature_info.get("names")
                    )
                
                # Write message to bag
                writer.write(
                    topic_name,
                    serialize_message(msg),
                    int(timestamp * 1e9)  # Convert to nanoseconds
                )
                
            except Exception as e:
                print(f"    Warning: Could not process {feature_name} at frame {frame_idx}: {e}")
                continue
    
    # Close all video files
    print("\nClosing video files...")
    video_reader.close_all()
    
    print("Closing MCAP file...")
    del writer
    print(f"Successfully created ROS2 MCAP at {output_path}")


def main():
    print("Starting conversion...")
    parser = argparse.ArgumentParser(
        description="Convert LeRobotDataset episode to ROS2 MCAP file"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Path to dataset directory (contains meta/, data/, and videos/ subdirs)"
    )
    parser.add_argument(
        "--episode_index",
        type=int,
        help="Episode index to convert"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output ROS2 MCAP directory path. Default is <dataset_dir>/mcap/",
        default=None,
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = Path(args.dataset_dir) / "mcap"
    convert_episode_to_mcap(Path(args.dataset_dir), args.episode_index, Path(args.output_dir))

if __name__ == "__main__":
    main()
