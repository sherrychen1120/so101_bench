#!/usr/bin/env python3
"""
Script to create evaluation datasets from existing recorded datasets.

This script takes a source dataset with splits and creates a new evaluation dataset
by copying selected episodes and modifying their task configurations for evaluation.
"""

import logging
import math
import json
from operator import xor
import sys
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
import traceback
from typing import List, Dict, Any
from deepdiff import DeepDiff

# Import lerobot components (needed for config registration)
from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    predict_action,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

from so101_bench.file_utils import load_yaml, save_yaml
from so101_bench.record_configs import RecordConfig
from so101_bench.raw_dataset_recorder import RawDatasetRecorder
from so101_bench.scripts.record import record_loop


def validate_splits_file(splits_data: dict, source_dataset_dir: Path) -> None:
    """
    Validate that splits.yaml is valid.
    
    Args:
        splits_data: Loaded splits.yaml data
        source_dataset_dir: Path to source dataset directory
        
    Raises:
        ValueError: If validation fails
    """
    # Check required keys exist
    required_keys = ['train', 'val_id', 'val_ood', 'statistics']
    for key in required_keys:
        if key not in splits_data:
            raise ValueError(f"Missing required key '{key}' in splits.yaml")
    
    # Check that ratios sum to 1
    stats = splits_data['statistics']
    required_ratio_keys = ['train_ratio', 'val_id_ratio', 'val_ood_ratio']
    for key in required_ratio_keys:
        if key not in stats:
            raise ValueError(f"Missing required ratio '{key}' in statistics")
    
    total_ratio = stats['train_ratio'] + stats['val_id_ratio'] + stats['val_ood_ratio']
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-3):
        raise ValueError(f"Ratios do not sum to 1.0, got {total_ratio}")
    
    # Check that all listed episodes exist in the dataset
    episodes_dir = source_dataset_dir / "episodes"
    if not episodes_dir.exists():
        raise ValueError(f"Episodes directory not found: {episodes_dir}")
    
    existing_episodes = set()
    for item in episodes_dir.iterdir():
        if item.is_dir():
            existing_episodes.add(item.name)
    
    # Check each split
    all_split_episodes = set()
    for split_name in ['train', 'val_id', 'val_ood']:
        episodes = splits_data[split_name]
        for episode in episodes:
            if episode in all_split_episodes:
                raise ValueError(f"Episode '{episode}' appears in multiple splits")
            all_split_episodes.add(episode)
            
            if episode not in existing_episodes:
                raise ValueError(f"Episode '{episode}' in {split_name} split does not exist in dataset")
    
    print(f"✓ Splits validation passed: {len(all_split_episodes)} total episodes")


def load_source_task_config(source_dataset_dir: Path, episode_name: str) -> dict:
    """
    Load task configuration from source episode.
    
    Args:
        source_dataset_dir: Path to source dataset directory
        episode_name: Name of the episode
        
    Returns:
        Task configuration dictionary
    """
    task_config_path = source_dataset_dir / "episodes" / episode_name / "task_config.yaml"
    
    if not task_config_path.exists():
        raise FileNotFoundError(f"Task config not found: {task_config_path}")
    
    return load_yaml(task_config_path)


def print_task_config_info(task_config: dict, episode_name: str) -> None:
    """
    Print information about block/container types and start poses from task config.
    
    Args:
        task_config: Task configuration dictionary
        episode_name: Name of the episode
    """
    print(f"\n--- Source Episode {episode_name} | Task Config ---")
    
    variations = task_config.get('variations', {})
    
    # Print block and container types
    block_type = variations.get('block', 'Unknown')
    container_type = variations.get('container', 'Unknown')
    print(f"Block type: {block_type}")
    print(f"Container type: {container_type}")
    
    # Print start poses
    start_pose = variations.get('start_pose', {})
    if start_pose:
        block_pose = start_pose.get('block', [])
        container_pose = start_pose.get('container', [])
        
        if block_pose:
            print(f"Block start pose: [{block_pose[0]:.3f}, {block_pose[1]:.3f}, {block_pose[2]:.3f}]")
        else:
            print("Block start pose: Not specified")
            
        if container_pose:
            print(f"Container start pose: [{container_pose[0]:.3f}, {container_pose[1]:.3f}, {container_pose[2]:.3f}]")
        else:
            print("Container start pose: Not specified")
    else:
        print("Start poses: Not specified")


def generate_eval_episodes(
    source_dataset_dir: Path,
    eval_dataset_splits: List[str],
):
    # Validate inputs
    
    if not source_dataset_dir.exists():
        raise ValueError(f"Error: Source dataset directory does not exist: {source_dataset_dir}")
    
    # Check that splits.yaml exists and is valid
    splits_path = source_dataset_dir / "splits.yaml"
    if not splits_path.exists():
        raise ValueError(f"Error: splits.yaml not found in source dataset: {splits_path}")

    # Load and validate splits
    splits_data = load_yaml(splits_path)
    validate_splits_file(splits_data, source_dataset_dir)

    # Generate eval episodes.
    eval_episodes = []
    for split_name in eval_dataset_splits:
        if split_name not in splits_data:
            raise ValueError(f"Split '{split_name}' not found in splits.yaml")
        
        episodes = splits_data[split_name]
        eval_episodes.extend(episodes)
        logging.info(f"✓ Added {len(episodes)} episodes from {split_name} split")
    
    return eval_episodes

def roll_out_eval_episodes(
    cfg: RecordConfig, 
    raw_recorder: RawDatasetRecorder,
    source_dataset_dir: Path, 
    eval_episodes: List[str],
    policy: PreTrainedPolicy | None,
    robot: Robot,
    teleop: Teleoperator | None
) -> None:
    """
    Roll out evaluation episodes
    """
    source_dataset_name = source_dataset_dir.name

    robot.connect()
    if teleop is not None:
        teleop.connect()
    
    # TODO(sherry): Factor out KeyboardEventListener to a separate class
    # that manages `events`` and controls the keyboard listener.
    events = {
        "stop_recording": False,
    }
    eval_episode_idx = 0
    while eval_episode_idx < len(eval_episodes) and not events["stop_recording"]:
        episode_name = eval_episodes[eval_episode_idx]
        log_say(f"Evaluating episode {eval_episode_idx}/{len(eval_episodes)} | Source episode: {source_dataset_name}__{episode_name}", cfg.play_sounds)
        
        # Load source task config
        source_task_config = load_source_task_config(source_dataset_dir, episode_name)
        
        # Print task config info and guide the user to set up for this episode.
        # TODO(sherry): Add a feature to overlay video over source episode image
        # to guide the setup.
        print_task_config_info(source_task_config, episode_name)
        input("Set up the environment as specified above. Press Enter to continue...")
        
        # Create eval task config
        eval_task_config = deepcopy(source_task_config)
        eval_task_config['is_eval'] = True
        eval_task_config['source_episode_dir'] = f"{source_dataset_name}/episodes/{episode_name}"

        run_mode = "policy" if policy is not None else "teleop"
        policy_info = None
        if policy is not None:
            policy_info = {
                "policy_path": getattr(cfg.policy, "pretrained_path", None),
                "policy_name": policy.__class__.__name__,
            }
        
        raw_recorder.start_episode(
            episode_idx=eval_episode_idx,
            run_mode=run_mode,
            policy_info=policy_info,
            leader_id=getattr(cfg.teleop, "id", None) if cfg.teleop else None,
            follower_id=getattr(cfg.robot, "id", None),
            task_config=eval_task_config,
        )

        # Starting keyboard listener right before the record loop
        listener, events = init_keyboard_listener()
        
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.dataset.fps,
            teleop=teleop,
            policy=policy,
            dataset=None,
            raw_recorder=raw_recorder,
            control_time_s=cfg.dataset.episode_time_s,
            single_task=cfg.dataset.single_task,
            display_data=cfg.display_data,
        )

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            # No need to clear LeRobot dataset buffer since we're not using it
            raw_recorder.reset_episode_data()
            continue

        # Stop keyboard listener at the end of each episode.
        if not is_headless() and listener is not None:
            listener.stop()
        
        # Only save raw episode data (no LeRobot format)
        raw_recorder.save_episode(task_description=cfg.dataset.single_task)
        
        eval_episode_idx += 1
    
    log_say("Stop recording", cfg.play_sounds, blocking=True)

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    raw_recorder.cleanup()

    log_say("Exiting", cfg.play_sounds)

def check_device_compatibility(
    robot: Robot,
    teleop: Teleoperator | None,
    source_dataset_dir: Path,
) -> None:
    """
    Check that robot & teleop calibration and camera configs match the source dataset.
    
    Args:
        robot: Robot instance with current calibration
        teleop: Teleoperator instance with current calibration (optional)
        source_dataset_dir: Path to source dataset directory
        
    Raises:
        ValueError: If configurations don't match
    """
    logging.info("Checking device compatibility with source dataset...")

    def _check_arm_calibration(arm_type: str, arm_id: str, arm_calibration_fpath: Path, source_dataset_dir: Path) -> None:
        """
        Args:
            `arm_type`: "robot" or "teleop"
            `arm_id`: ID of the arm
            `arm_calibration_fpath`: Path to the arm calibration file
            `source_dataset_dir`: Path to the source dataset directory

        Raises:
            ValueError: If calibration mismatch is found
        """
        source_arm_calib_path = source_dataset_dir / "arm_calib" / f"{arm_id}.json"
        if not source_arm_calib_path.exists():
            raise ValueError(f"Source {arm_type} calibration not found: {source_arm_calib_path}")
        with open(source_arm_calib_path, 'r') as f:
            source_arm_calib = json.load(f)
        if not arm_calibration_fpath.exists():
            raise ValueError(f"Current {arm_type} calibration not found: {arm_calibration_fpath}")
        with open(arm_calibration_fpath, 'r') as f:
            current_arm_calib = json.load(f)
        
        arm_calib_diff = DeepDiff(source_arm_calib, current_arm_calib, significant_digits=3)
        if arm_calib_diff:
            raise ValueError(f"{arm_type} calibration mismatch: {arm_calib_diff}")
        
        logging.info(f"✓ {arm_type} calibration compatibility check passed")
    
    _check_arm_calibration("robot", robot.id, robot.calibration_fpath, source_dataset_dir)
    if teleop is not None:
        _check_arm_calibration("teleop", teleop.id, teleop.calibration_fpath, source_dataset_dir)
    
    # Check camera configurations
    source_camera_config_path = source_dataset_dir / "camera_configs.json"
    if not source_camera_config_path.exists():
        raise ValueError(f"Source camera configuration not found: {source_camera_config_path}")
    with open(source_camera_config_path, 'r') as f:
        source_camera_configs = json.load(f)
    current_camera_configs = {}
    for cam_name, cam_config in robot.config.cameras.items():
        current_camera_configs[cam_name] = json.loads(json.dumps(asdict(cam_config), indent=2))
    camera_config_diff = DeepDiff(source_camera_configs, current_camera_configs, significant_digits=3)
    if camera_config_diff:
        raise ValueError(f"Camera configuration mismatch: {camera_config_diff}")
    logging.info("✓ Camera configuration compatibility check passed")

    logging.info("✓ Device compatibility check passed")

@parser.wrap()
def eval_rollout(cfg: RecordConfig):
    """Main evaluation rollout function."""
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="recording")
    
    # At least one of policy and teleop must be provided.
    if not ((cfg.policy is None) ^ (cfg.teleop is None)):
        raise ValueError("One and only one of policy and teleop must be provided")
    
    dataset_root_dir = Path(cfg.dataset.raw_format_root)
    if not dataset_root_dir.exists():
        raise ValueError(f"Error: Dataset root directory does not exist: {dataset_root_dir}")
    source_dataset_dir = dataset_root_dir / cfg.dataset.repo_id.split("/")[-1]
    
    robot = make_robot_from_config(cfg.robot)
    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)

    # Verify that robot calibration and camera configs match the source dataset.
    check_device_compatibility(robot, teleop, source_dataset_dir)

    # For eval, raw recorder is required.
    if not cfg.dataset.save_raw_format:
        raise ValueError("Raw recorder is required for evaluation")
    
    dataset_task_config_path = source_dataset_dir / "task_config.yaml"
    if not dataset_task_config_path.exists():
        raise ValueError(f"Source task config not found: {dataset_task_config_path}")
    dataset_task_config = load_yaml(dataset_task_config_path)
    
    raw_recorder = RawDatasetRecorder(
        dataset_name=cfg.eval_dataset.name,
        root_dir=cfg.dataset.raw_format_root,
        is_resume=cfg.resume,
        robot_config=asdict(cfg.robot),
        robot_calibration_fpath=robot.calibration_fpath,
        teleop_config=asdict(cfg.teleop),
        teleop_calibration_fpath=teleop.calibration_fpath,
        dataset_task_config=dataset_task_config,
        fps=cfg.dataset.fps,
        save_videos=cfg.dataset.raw_format_videos,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera,
    )

    # Create minimal dataset only for policy initialization (no recording to LeRobot format)
    if cfg.policy is not None:
        action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
        dataset_features = {**action_features, **obs_features}

        # Create a minimal dataset just to get metadata for policy initialization
        # Use the source dataset to create LeRobotDataset.
        lerobot_dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=False,  # No video encoding needed for eval
            image_writer_processes=0,  # No image writing needed
            image_writer_threads=0,
            batch_encoding_size=1,
        )
        
        # Load pretrained policy using dataset metadata
        policy = make_policy(cfg.policy, ds_meta=lerobot_dataset.meta)
        
    else:
        # No policy, no need for dataset at all
        policy = None
        lerobot_dataset = None

    eval_episodes = generate_eval_episodes(
        source_dataset_dir,
        cfg.eval_dataset.splits
    )
    logging.info(f"✓ Selected {len(eval_episodes)} episodes for evaluation")
    
    # Process evaluation episodes
    roll_out_eval_episodes(cfg, raw_recorder, source_dataset_dir, eval_episodes, policy, robot, teleop)
    
    logging.info(f"\n✓ Successfully created evaluation dataset: {cfg.eval_dataset.name}")
    logging.info(f"✓ Rolled out {len(eval_episodes)} episodes")
        

def main():
    """Main function."""
    try:    
        eval_rollout()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
