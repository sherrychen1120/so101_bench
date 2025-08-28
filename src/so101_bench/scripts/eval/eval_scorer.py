#!/usr/bin/env python3
"""
Evaluation scoring script for SO101 benchmark.

This script processes evaluation datasets and allows manual labeling of progress stages
through an interactive video player interface. It calculates various metrics including
task progress scores, success rates, latency, and retry attempts.
"""

import argparse
import json
import logging
import os
import sys
import yaml
import cv2
import traceback
import numpy as np
from deepdiff import DeepDiff
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from lerobot.utils.utils import init_logging

from so101_bench.raw_dataset_recorder import EMERGENCY_STOP_EVENT
from so101_bench.file_utils import load_yaml, save_yaml, load_json, load_jsonl
from so101_bench.task_progress_labeler import TaskProgressLabeler

def parse_args():
    parser = argparse.ArgumentParser(description="Score evaluation episodes through manual video labeling")
    parser.add_argument("--dataset_root_dir", help="Root directory of the evaluation dataset")
    parser.add_argument("--task_root_dir", help="Root directory containing task specifications")
    parser.add_argument("--eval_set_name", help="Name of the evaluation set")
    parser.add_argument("--eval_config_path", help="Path to evaluation configuration YAML")
    parser.add_argument("--force", action="store_true", help="Force override existing eval_score.yaml files")
    
    return parser.parse_args()

def get_episodes_from_dataset(dataset_dir: Path) -> List[str]:
    """Get list of episode directories from dataset."""
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    
    episodes = []
    for item in episodes_dir.iterdir():
        if item.is_dir():
            episodes.append(item)
    
    return sorted(episodes)


def verify_episode_durations(
    dataset_dir: Path,
    episodes: List[str],
    eval_horizon_s: float,
) -> None:
    print("Verifying episode durations...")
    invalid_episodes = []
    
    for episode_id in episodes:
        episode_dir = dataset_dir / "episodes" / episode_id
        try:
            metadata_path = episode_dir / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")
            
            metadata = load_json(metadata_path)
            duration_s = metadata.get("duration_s", 0)
            if duration_s < eval_horizon_s:
                invalid_episodes.append(episode_id)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            invalid_episodes.append(episode_id)
    
    if invalid_episodes:
        logging.error(f"Error: {len(invalid_episodes)} episodes are shorter than horizon ({eval_horizon_s}s):")
        for episode_id in invalid_episodes:
            logging.error(f"  {episode_id}")
        raise ValueError("Cannot score the eval set with given eval_config")
    
    logging.info("All episodes meet duration requirements.")


def load_task_spec(task_root_dir: str, task_name: str) -> Dict[str, Any]:
    """Load task specification from task_root_dir."""
    task_spec_path = Path(task_root_dir) / task_name / "task_spec.yaml"
    if not task_spec_path.exists():
        raise FileNotFoundError(f"Task spec not found: {task_spec_path}")
    
    return load_yaml(task_spec_path)


def calculate_frame_latency_ms(sync_logs: List[Dict[str, Any]]) -> float:
    """Calculate average frame latency from sync logs."""
    latencies = []
    
    for log_entry in sync_logs:
        timestamps = []
        
        # Collect all timestamps from the log entry
        if "timestamp" in log_entry:
            timestamps.append(log_entry["timestamp"])
        if "robot_state_timestamp" in log_entry:
            timestamps.append(log_entry["robot_state_timestamp"])
        if "action_timestamp" in log_entry:
            timestamps.append(log_entry["action_timestamp"])
        if "camera_timestamps" in log_entry:
            timestamps.extend(log_entry["camera_timestamps"].values())
        
        if len(timestamps) >= 2:
            latency_s = max(timestamps) - min(timestamps)
            latencies.append(latency_s * 1000)  # Convert to milliseconds
    
    return sum(latencies) / len(latencies) if latencies else 0.0


def calculate_num_attempts(progress_stages: Dict[str, List[float]]) -> int:
    """Calculate number of attempts based on progress stage sequences."""
    # Define the expected progress order
    stage_order = ["0_reach_block", "1_grasp_block", "2_reach_container", "3_release_block", "4_block_in_container"]
    
    # Create timeline of stages
    timeline = []
    for stage, intervals in progress_stages.items():
        for start_time, end_time in intervals:
            timeline.append((start_time, stage))
    
    # Sort by start time
    timeline.sort(key=lambda x: x[0])
    
    # Count monotonously increasing sequences
    attempts = 0
    current_max_stage = -1
    
    for _, stage in timeline:
        try:
            stage_idx = stage_order.index(stage)
            if stage_idx > current_max_stage:
                current_max_stage = stage_idx
            elif stage_idx <= current_max_stage:
                # This is a retry - new attempt
                attempts += 1
                current_max_stage = stage_idx
        except ValueError:
            # Unknown stage, skip
            continue
    
    return max(1, attempts)  # At least 1 attempt


def check_safety_abort(metadata: Dict[str, Any]) -> bool:
    """Check if episode had safety abort from metadata events."""
    events = metadata.get("events", [])
    for _, event in events:
        if event == EMERGENCY_STOP_EVENT:
            return True
    return False


def calculate_episode_metrics(progress_stages: Dict[str, List[List[float]]], 
                            task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate episode metrics from progress stages and task spec."""
    score_definition = task_spec["score_definition"]["task_progress_score"]
    stage_order = sorted(list(score_definition.keys()))
    
    # Find the latest progress stage achieved
    latest_stage = None
    latest_time = -1
    
    for stage, intervals in progress_stages.items():
        if stage in stage_order:
            for interval in intervals:
                end_time = interval[1]
                if end_time > latest_time:
                    latest_time = end_time
                    latest_stage = stage
    
    # Calculate metrics
    if latest_stage is None:
        return {
            "task_progress_label": None,
            "task_progress_score": 0.0,
            "success": 0,
            "duration_to_success_s": -1.0
        }
    
    task_progress_score = score_definition.get(latest_stage, 0.0)
    success = 1 if latest_stage == stage_order[-1] else 0
    duration_to_success_s = latest_time if success else -1.0
    
    return {
        "task_progress_label": latest_stage,
        "task_progress_score": task_progress_score,
        "success": success,
        "duration_to_success_s": duration_to_success_s
    }


def process_episode(episode_dir: Path, eval_config: Dict[str, Any], 
                   task_spec: Dict[str, Any], force_override: bool = False) -> None:
    """Process a single episode for scoring."""
    episode_id = episode_dir.name
    eval_score_path = episode_dir / "eval_score.yaml"
    
    # Check if eval_score.yaml already exists
    if eval_score_path.exists() and not force_override:
        existing_score = load_yaml(eval_score_path)
        existing_config = existing_score.get("eval_config", {})
        
        eval_config_diff = DeepDiff(existing_config, eval_config, significant_digits=1e-3)
        if eval_config_diff:
            logging.info(f"Episode {episode_id}: eval_score.yaml already exists with matching config. Skipping.")
            return
        else:
            logging.info(f"Episode {episode_id}: eval_score.yaml exists but config differs: {eval_config_diff}.")
            response = input("Override existing file? (y/N): ")
            if response.lower() != 'y':
                logging.info("Skipping episode.")
                return
    
    # Load episode metadata and sync logs
    metadata_path = episode_dir / "metadata.json"
    sync_logs_path = episode_dir / "sync_logs.jsonl"
    
    metadata = load_json(metadata_path)
    sync_logs = load_jsonl(sync_logs_path)
    
    # Find video files (collect all available cameras)
    video_paths = []
    for cam in ["cam_front", "cam_top"]:
        candidate = episode_dir / f"video_{cam}.mp4"
        if candidate.exists():
            video_paths.append(candidate)
    
    if not video_paths:
        logging.error(f"Error: No video files found for episode {episode_id}")
        return
    
    logging.info(f"\nProcessing episode: {episode_id}")
    logging.info(f"Videos: {[p.name for p in video_paths]}")
    logging.info(f"Duration: {metadata.get('duration_s', 0):.2f}s")
    
    # Launch video player for manual labeling
    fps = metadata.get("fps", 30.0)
    progress_stage_labels = sorted(list(task_spec["score_definition"]["task_progress_score"].keys()))
    horizon_s = eval_config.get("horizon_s")
    player = TaskProgressLabeler(video_paths, progress_stage_labels, fps, horizon_s)
    progress_stages = player.play()
    
    if not progress_stages:
        print("No progress stages labeled. Skipping episode.")
        return
    
    # Calculate metrics
    episode_metrics = calculate_episode_metrics(progress_stages, task_spec)
    
    # Calculate additional metrics
    frame_latency_ms = calculate_frame_latency_ms(sync_logs)
    is_safety_abort = check_safety_abort(metadata)
    num_attempts = calculate_num_attempts(progress_stages)
    
    # Create eval score data
    eval_score_data = {
        "episode_id": episode_id,
        "eval_config": eval_config,
        "episode_metrics": {
            **episode_metrics,
            "progress_stage_timerange_s": progress_stages,
            "frame_latency_ms": round(frame_latency_ms, 1),
            "is_safety_abort": is_safety_abort,
            "num_attempts": num_attempts
        }
    }
    
    # Save eval score
    save_yaml(eval_score_data, eval_score_path)
    logging.info(f"Saved eval score to: {eval_score_path}")
    
    # Display summary
    logging.info(f"Summary:")
    logging.info(f"  Task progress: {episode_metrics['task_progress_label']} (score: {episode_metrics['task_progress_score']})")
    logging.info(f"  Success: {'Yes' if episode_metrics['success'] else 'No'}")
    logging.info(f"  Duration to success: {episode_metrics['duration_to_success_s']:.1f}s")
    logging.info(f"  Frame latency: {frame_latency_ms:.1f}ms")
    logging.info(f"  Safety abort: {'Yes' if is_safety_abort else 'No'}")
    logging.info(f"  Attempts: {num_attempts}")


def main():
    init_logging()

    args = parse_args()
    
    # Validate input paths
    if not Path(args.dataset_root_dir).exists():
        raise FileNotFoundError(f"Error: Dataset root directory not found: {args.dataset_root_dir}")
    if not Path(args.task_root_dir).exists():
        raise FileNotFoundError(f"Error: Task root directory not found: {args.task_root_dir}")
    if not Path(args.eval_config_path).exists():
        raise FileNotFoundError(f"Error: Eval config file not found: {args.eval_config_path}")
    
    dataset_dir = Path(args.dataset_root_dir) / args.eval_set_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Error: Dataset directory not found: {dataset_dir}")
    
    # Load evaluation configuration
    eval_config = load_yaml(args.eval_config_path)
    assert "horizon_s" in eval_config, "horizon_s is required in eval_config"
    eval_horizon_s = eval_config["horizon_s"]
    logging.info(f"Evaluation configuration:")
    logging.info(f"  Horizon: {eval_horizon_s}s")
    
    # Load task configuration from dataset
    task_config_path = dataset_dir / "task_config.yaml"
    if not task_config_path.exists():
        raise FileNotFoundError(f"Error: Dataset task config not found: {task_config_path}")
    task_config = load_yaml(task_config_path)
    assert "task_name" in task_config, "task_name is required in task_config.yaml"
    task_name = task_config.get("task_name")
    
    # Load score definition from task specification
    task_spec = load_task_spec(args.task_root_dir, task_name)
    logging.info(f"Loaded task specification for: {task_name}")
    # TODO: Support other scores beyond task_progress_score
    score_def = task_spec.get("score_definition", {})
    if "task_progress_score" not in score_def:
        raise ValueError("Error: task_progress_score not found in task specification")
    
    # Get all episodes in the dataset
    episodes = get_episodes_from_dataset(dataset_dir)
    logging.info(f"Found {len(episodes)} episodes in dataset")    
    if not episodes:
        raise ValueError("No episodes found in dataset")
    
    # Verify episode durations
    verify_episode_durations(dataset_dir, episodes, eval_horizon_s)
    
    # Process each episode
    logging.info(f"\nProcessing {len(episodes)} episodes...")
    
    for i, episode_id in enumerate(episodes, 1):
        episode_dir = dataset_dir / "episodes" / episode_id
        print(f"\n{'='*60}")
        print(f"Episode {i}/{len(episodes)}: {episode_id}")
        print(f"{'='*60}")
        
        try:
            process_episode(episode_dir, eval_config, task_spec, args.force)
        except KeyboardInterrupt:
            print("\nInterrupted by user. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
            traceback.print_exc()
            response = input("Continue with next episode? (Y/n): ")
            if response.lower() == 'n':
                sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All episodes processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
