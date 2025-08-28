#!/usr/bin/env python
"""
Script to create split files for raw datasets.

This script takes a raw dataset directory and creates train/val_id/val_ood splits
based on the task configuration and ID/OOD task definitions.
"""

import argparse
import math
import random
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

from so101_bench.file_utils import load_yaml, save_yaml

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_ID_RATIO = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_TASK_DIRECTORY = "/home/melon/sherry/so101_bench/datasets/tasks"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset splits for raw dataset")
    parser.add_argument("--raw_dataset_dir", type=str, 
                       help="Path to raw dataset directory")
    parser.add_argument("--id_ood_config", type=str, required=True,
                       help="Path to ID/OOD task configuration file")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO,
                       help=f"Ratio of ID episodes to use for training (default: {DEFAULT_TRAIN_RATIO})")
    parser.add_argument("--val_id_ratio", type=float, default=DEFAULT_VAL_ID_RATIO,
                       help=f"Ratio of ID episodes to use for validation (default: {DEFAULT_VAL_ID_RATIO})")
    parser.add_argument("--task_directory", type=str, 
                       default=DEFAULT_TASK_DIRECTORY,
                       help=f"Path to task directory (default: {DEFAULT_TASK_DIRECTORY})")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED,
                       help=f"Random seed for reproducible splits (default: {DEFAULT_RANDOM_SEED})")
    
    return parser.parse_args()


def load_episode_task_config(episode_dir: Path) -> dict:
    """Load episode configuration from episode directory."""
    config_path = episode_dir / "task_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Task config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_all_episodes(dataset_dir: Path) -> List[str]:
    """Get all episode directories in the dataset."""
    episodes = []
    episodes_dir = dataset_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    
    for item in episodes_dir.iterdir():
        if item.is_dir():
            episodes.append(item.name)
    return sorted(episodes)


def validate_id_ood_config_against_task_spec(id_ood_config: dict, task_spec: dict) -> dict:
    """
    Validate and normalize ID/OOD configuration against task specification.
    
    Args:
        id_ood_config: The ID/OOD task configuration
        task_spec: The task specification with all available variations
        
    Returns:
        Normalized ID/OOD configuration with null values filled in
    """
    task_variations = task_spec.get('variations', {})
    id_variations = id_ood_config.get('id_variations', {})
    ood_variations = id_ood_config.get('ood_variations', {})
    
    # Validate that all keys in id/ood variations exist in task_spec
    all_variation_keys = set(id_variations.keys()) | set(ood_variations.keys())
    task_spec_keys = set(task_variations.keys())
    
    invalid_keys = all_variation_keys - task_spec_keys
    if invalid_keys:
        raise ValueError(f"Invalid variation keys found: {invalid_keys}. "
                        f"Valid keys from task_spec: {task_spec_keys}")
    
    # Process each variation key
    normalized_id_variations = {}
    normalized_ood_variations = {}
    
    for key in task_spec_keys:
        id_value = id_variations.get(key)
        ood_value = ood_variations.get(key)
        
        # Handle null values - if either is null, both must be null and use all task_spec values
        if id_value is None or ood_value is None:
            if id_value is not None or ood_value is not None:
                raise ValueError(f"Key '{key}': if either id or ood has null value, both must be null")
            
            # Don't fill this into normalized variations.
            
            # Fill in with all values from task_spec
            # if key == 'start_pose':
            #     normalized_id_variations[key] = task_variations[key]
            #     normalized_ood_variations[key] = task_variations[key]
            # else:
            #     all_values = task_variations[key]
            #     normalized_id_variations[key] = all_values
            #     normalized_ood_variations[key] = all_values
        else:
            # Check for overlap between id and ood variations
            if key != 'start_pose':
                id_set = set(id_value) if isinstance(id_value, list) else {id_value}
                ood_set = set(ood_value) if isinstance(ood_value, list) else {ood_value}
                
                overlap = id_set & ood_set
                if overlap:
                    raise ValueError(f"Key '{key}': ID and OOD variations overlap: {overlap}")
                
                # Validate that all values exist in task_spec
                task_spec_values = set(task_variations[key])
                invalid_id = id_set - task_spec_values
                invalid_ood = ood_set - task_spec_values
                
                if invalid_id:
                    raise ValueError(f"Key '{key}': Invalid ID values: {invalid_id}")
                if invalid_ood:
                    raise ValueError(f"Key '{key}': Invalid OOD values: {invalid_ood}")
            
            normalized_id_variations[key] = id_value
            normalized_ood_variations[key] = ood_value
    
    # Return normalized configuration
    normalized_config = id_ood_config.copy()
    normalized_config['id_variations'] = normalized_id_variations
    normalized_config['ood_variations'] = normalized_ood_variations
    
    return normalized_config


def check_start_pose_in_range(pose: List[float], pose_range: dict) -> bool:
    """
    Check if a pose falls within the specified range.
    
    Args:
        pose: [x, y, yaw] pose values
        pose_range: Dictionary with 'min' and 'max' keys containing [x, y, yaw] limits
        
    Returns:
        True if pose is within range, False otherwise
    """
    if len(pose) != 3:
        raise ValueError(f"Pose must have 3 values [x, y, yaw], got {len(pose)}")
    
    min_vals = pose_range['min']
    max_vals = pose_range['max']
    
    for i, (val, min_val, max_val) in enumerate(zip(pose, min_vals, max_vals)):
        if not (min_val <= val <= max_val):
            return False
    
    return True


def classify_episodes(dataset_dir: Path, id_variations: dict[str, Any], ood_variations: dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Classify episodes into ID and OOD based on their task variations.
    
    Args:
        dataset_dir: Path to the dataset directory
        id_variations: Dictionary of ID variation constraints
        ood_variations: Dictionary of OOD variation constraints
    
    Returns:
        Tuple of (id_episodes, ood_episodes)
    """
    episodes = get_all_episodes(dataset_dir)
    id_episodes = []
    ood_episodes = []
    
    episodes_dir = dataset_dir / "episodes"
    
    for episode_id in episodes:
        episode_dir = episodes_dir / episode_id
        try:
            task_config = load_episode_task_config(episode_dir)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f"Episode {episode_id} missing or invalid task_config: {e}")
        
        episode_variations = task_config.get('variations', {})
        if not episode_variations:
            raise ValueError(f"Episode {episode_id} has no variations in task_config")
        
        # Check if episode matches ID or OOD variations
        def _check_match(episode_variations: dict[str, Any], variations: dict[str, Any]) -> bool:
            for key, id_values in variations.items():
                episode_value = episode_variations.get(key)
                if episode_value is None:
                    raise ValueError(f"Episode {episode_id} missing variation key '{key}' in task_config")
                
                if key == 'start_pose':
                    # For start_pose, check if episode poses fall within ID range
                    episode_block_pose = episode_value.get('block', [])
                    episode_container_pose = episode_value.get('container', [])
                    id_block_range = id_values.get('block', {})
                    id_container_range = id_values.get('container', {})
                    
                    # Check if poses are within ID ranges
                    block_in_id_range = (not id_block_range or 
                                    check_start_pose_in_range(episode_block_pose, id_block_range))
                    container_in_id_range = (not id_container_range or 
                                        check_start_pose_in_range(episode_container_pose, id_container_range))
                    
                    if block_in_id_range and container_in_id_range:
                        return True
                else:
                    # For other keys, check if episode value is in ID list
                    id_value_list = id_values if isinstance(id_values, list) else [id_values]
                    if episode_value in id_value_list:
                        return True
            return False

        is_id_match = _check_match(episode_variations, id_variations)
        is_ood_match = _check_match(episode_variations, ood_variations)
        
        # Classify episode
        if is_id_match and is_ood_match:
            # Episode matches both ID and OOD - this shouldn't happen with proper validation
            raise ValueError(f"Episode {episode_id} matches both ID and OOD variations. "
                           f"This indicates a problem with the ID/OOD configuration.")
        elif is_id_match:
            id_episodes.append(episode_id)
        elif is_ood_match:
            ood_episodes.append(episode_id)
        else:
            # Episode doesn't match either ID or OOD variations
            episode_summary = {k: v for k, v in episode_variations.items() if k != 'start_pose'}
            if 'start_pose' in episode_variations:
                episode_summary['start_pose'] = {
                    'block': episode_variations['start_pose'].get('block', []),
                    'container': episode_variations['start_pose'].get('container', [])
                }
            
            raise ValueError(f"Episode {episode_id} doesn't match any ID or OOD variations. "
                           f"Episode variations: {episode_summary}")
    
    return id_episodes, ood_episodes


def split_id_episodes(id_episodes: List[str], train_ratio: float) -> Tuple[List[str], List[str]]:
    """
    Split ID episodes into train and validation sets.
    
    Args:
        id_episodes: List of ID episode names
        train_ratio: Ratio of episodes to use for training (rest go to validation)
        
    Returns:
        Tuple of (train_episodes, val_id_episodes)
    """
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    # Shuffle episodes for random split
    shuffled_episodes = id_episodes.copy()
    random.shuffle(shuffled_episodes)
    print(f"Shuffled episodes: {shuffled_episodes}")
    
    # Calculate split point
    num_train = int(len(shuffled_episodes) * train_ratio)
    
    train_episodes = shuffled_episodes[:num_train]
    val_id_episodes = shuffled_episodes[num_train:]
    
    return train_episodes, val_id_episodes


def calculate_final_ratios(train_count: int, val_id_count: int, val_ood_count: int) -> Tuple[float, float, float]:
    """Calculate final ratios based on actual counts."""
    total = train_count + val_id_count + val_ood_count
    if total == 0:
        return 0.0, 0.0, 0.0
    
    train_ratio = train_count / total
    val_id_ratio = val_id_count / total
    val_ood_ratio = val_ood_count / total
    
    return train_ratio, val_id_ratio, val_ood_ratio

def load_and_validate_id_ood_config(id_ood_config_path: Path, task_directory: Path) -> dict:
    """
    Load and validate ID/OOD task configuration against task specification.
    
    Args:
        id_ood_config_path: Path to the ID/OOD task configuration file
        task_directory: Path to the task directory containing task specifications
        
    Returns:
        Tuple of normalized (id_variations, ood_variations)
    """
    # Load ID/OOD task configuration
    id_ood_config = load_yaml(id_ood_config_path)
    task_name = id_ood_config.get('task_name')
    if not task_name:
        raise ValueError("No task_name found in ID/OOD configuration")
    
    # Load task specification
    task_spec_path = task_directory / task_name / "task_spec.yaml"
    if not task_spec_path.exists():
        raise FileNotFoundError(f"Task specification not found: {task_spec_path}")
    
    task_spec = load_yaml(task_spec_path)
    
    # Validate and normalize ID/OOD configuration against task spec
    print(f"Validating ID/OOD configuration against task specification...")
    normalized_id_ood_config = validate_id_ood_config_against_task_spec(id_ood_config, task_spec)
    
    id_variations = normalized_id_ood_config.get('id_variations', {})
    ood_variations = normalized_id_ood_config.get('ood_variations', {})
    
    if len(id_variations) == 0:
        raise ValueError("No id_variations found in ID/OOD configuration")
    
    return id_variations, ood_variations

def create_splits_file(dataset_dir: Path, id_variations: Dict[str, List[str]], ood_variations: Dict[str, List[str]],
                      train_ratio: float, val_id_ratio: float) -> None:
    """
    Create splits.yaml file for the dataset.
    
    Args:
        dataset_dir: Path to the raw dataset directory
        id_ood_config_path: Path to the ID/OOD task configuration file
        train_ratio: Ratio for training split within ID episodes
        val_id_ratio: Ratio for validation ID split within ID episodes
        task_directory: Path to the task directory containing task specifications
    """
    # Validate ratios
    if not math.isclose(train_ratio + val_id_ratio, 1.0):
        raise ValueError(f"train_ratio + val_id_ratio must equal 1.0, got {train_ratio + val_id_ratio}")
    
    # Check if splits.yaml already exists
    # TODO(sherry): Handle splits update.
    splits_path = dataset_dir / "splits.yaml"
    if splits_path.exists():
        raise FileExistsError(f"splits.yaml already exists at {splits_path}.")
    
    # Classify episodes
    print(f"Classifying episodes in {dataset_dir}...")
    id_episodes, ood_episodes = classify_episodes(dataset_dir, id_variations, ood_variations)
    
    print(f"Found {len(id_episodes)} ID episodes and {len(ood_episodes)} OOD episodes")
    
    # Split ID episodes into train/val
    if id_episodes:
        train_episodes, val_id_episodes = split_id_episodes(id_episodes, train_ratio)
    else:
        train_episodes, val_id_episodes = [], []
    
    # All OOD episodes go to val_ood
    val_ood_episodes = ood_episodes
    
    # Calculate final ratios
    train_count = len(train_episodes)
    val_id_count = len(val_id_episodes)
    val_ood_count = len(val_ood_episodes)
    
    final_train_ratio, final_val_id_ratio, final_val_ood_ratio = calculate_final_ratios(
        train_count, val_id_count, val_ood_count
    )
    
    # Create splits data structure
    splits_data = {
        'train': sorted(train_episodes),
        'val_id': sorted(val_id_episodes),
        'val_ood': sorted(val_ood_episodes),
        'id_ood_task_config': {
            'id_variations': id_variations,
            'ood_variations': ood_variations
        },
        'statistics': {
            'train_count': train_count,
            'val_id_count': val_id_count,
            'val_ood_count': val_ood_count,
            'total_count': train_count + val_id_count + val_ood_count,
            'train_ratio': round(final_train_ratio, 4),
            'val_id_ratio': round(final_val_id_ratio, 4),
            'val_ood_ratio': round(final_val_ood_ratio, 4)
        }
    }
    
    # Save splits file
    save_yaml(splits_data, splits_path)
    
    # Print summary
    print(f"\nCreated splits.yaml at {splits_path}")
    print(f"Train ID: {train_count} episodes ({final_train_ratio:.3f})")
    print(f"Val ID: {val_id_count} episodes ({final_val_id_ratio:.3f})")
    print(f"Val OOD: {val_ood_count} episodes ({final_val_ood_ratio:.3f})")
    print(f"Total: {train_count + val_id_count + val_ood_count} episodes")


def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
    # Convert paths to Path objects
    dataset_dir = Path(args.raw_dataset_dir)
    id_ood_config_path = Path(args.id_ood_config)
    task_directory = Path(args.task_directory)
    
    # Validate inputs
    if not dataset_dir.exists():
        print(f"Error: Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)
    
    if not id_ood_config_path.exists():
        print(f"Error: ID/OOD config file does not exist: {id_ood_config_path}")
        sys.exit(1)
    
    if not task_directory.exists():
        print(f"Error: Task directory does not exist: {task_directory}")
        sys.exit(1)
    
    # Validate ratios
    train_val_id_ratio_sum = args.train_ratio + args.val_id_ratio
    if not math.isclose(train_val_id_ratio_sum, 1.0):
        print(f"Error: train_ratio + val_id_ratio must equal 1.0, got {train_val_id_ratio_sum}")
        sys.exit(1)
    
    id_variations, ood_variations = load_and_validate_id_ood_config(id_ood_config_path, task_directory)
    
    try:
        create_splits_file(dataset_dir, id_variations, ood_variations, args.train_ratio, args.val_id_ratio)
        print("Successfully created dataset splits!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
