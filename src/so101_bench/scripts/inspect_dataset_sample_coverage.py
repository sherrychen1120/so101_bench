#!/usr/bin/env python3
"""
Script to initialize a StratifiedSampler, load existing episodes, and save sampling output.

This script can be used to:
1. Initialize a StratifiedSampler with a given config and task spec
2. Load existing episodes from a raw dataset directory
3. Generate and save a stratified_sampling_output.yaml file

Usage:
    python generate_stratified_sampling_output.py \
        --dataset_dir /path/to/raw/dataset \
        --stratified_config /path/to/stratified_sampling_config.yaml \
        --tasks_dir /path/to/tasks \
        --task_name pick_and_place_block
"""

import argparse
import yaml
from pathlib import Path
from src.so101_bench.task_configurator import StratifiedSampler


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified sampling output from existing episodes"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Path to the raw dataset directory containing episodes"
    )
    parser.add_argument(
        "--stratified_config",
        type=Path,
        required=True,
        help="Path to the stratified sampling configuration YAML file"
    )
    parser.add_argument(
        "--tasks_dir",
        type=Path,
        required=True,
        help="Path to the tasks directory containing task specifications"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name of the task (should match directory name in tasks_dir)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    
    if not args.stratified_config.exists():
        raise FileNotFoundError(f"Stratified config file not found: {args.stratified_config}")
    
    task_spec_path = args.tasks_dir / args.task_name / "task_spec.yaml"
    if not task_spec_path.exists():
        raise FileNotFoundError(f"Task spec file not found: {task_spec_path}")
    
    print(f"Loading task spec from: {task_spec_path}")
    with open(task_spec_path, 'r') as f:
        task_spec = yaml.safe_load(f)
    
    print(f"Initializing StratifiedSampler with config: {args.stratified_config}")
    sampler = StratifiedSampler(
        stratified_config_path=args.stratified_config,
        task_spec=task_spec
    )
    
    print(f"StratifiedSampler initialized:")
    print(f"  Total bins: {sampler.total_bins}")
    print(f"  Axes: {list(sampler.axes.keys())}")
    print(f"  Bin counts per axis: {sampler.bin_count_per_axis}")
    
    print(f"\nLoading existing episodes from: {args.dataset_dir}")
    sampler.load_existing_episodes(args.dataset_dir)
    
    # Show current distribution
    total_episodes = sum(len(episodes) for episodes in sampler.episodes_per_bin.values())
    print(f"\nLoaded {total_episodes} episodes total")
    
    print(f"\nSaving sampling output")
    sampler.save_sampling_output()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SAMPLING SUMMARY")
    print("="*60)
    
    bins_with_episodes = 0
    for bin_id in range(sampler.total_bins):
        count = len(sampler.episodes_per_bin[bin_id])
        if count > 0:
            bins_with_episodes += 1
            indices = sampler._bin_id_to_indices(bin_id)
            bin_ranges = {}
            for axis_name, bin_idx in indices.items():
                bin_ranges[axis_name] = sampler.bins[axis_name][bin_idx]
            
            print(f"Bin {bin_id}: {count} episodes")
            for axis_name, range_vals in bin_ranges.items():
                print(f"  {axis_name}: [{range_vals[0]:.3f}, {range_vals[1]:.3f})")
    
    print(f"\nTotal bins with episodes: {bins_with_episodes}/{sampler.total_bins}")
    print(f"Total episodes: {total_episodes}")
    print(f"Desired episodes per bin: {sampler.desired_episodes_per_bin}")
    
    # Show bins that still need episodes
    bins_needing_episodes = []
    for bin_id in range(sampler.total_bins):
        current_count = len(sampler.episodes_per_bin[bin_id])
        needed = max(0, sampler.desired_episodes_per_bin - current_count)
        if needed > 0:
            bins_needing_episodes.append((bin_id, current_count, needed))
    
    if bins_needing_episodes:
        print(f"\nBins still needing episodes:")
        for bin_id, current, needed in bins_needing_episodes:
            indices = sampler._bin_id_to_indices(bin_id)
            print(f"  Bin {bin_id}: {current}/{sampler.desired_episodes_per_bin} episodes ({needed} needed)")
            for axis_name, bin_idx in indices.items():
                bin_range = sampler.bins[axis_name][bin_idx]
                print(f"    {axis_name}: [{bin_range[0]:.3f}, {bin_range[1]:.3f})")
    else:
        print("\n✅ All bins have sufficient episodes!")


if __name__ == "__main__":
    main()
