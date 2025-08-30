#!/usr/bin/env python3
"""Test script to regenerate clean stratified sampling output"""

from pathlib import Path
from src.so101_bench.task_configurator import StratifiedSampler
import yaml

# Load task spec
with open('datasets/tasks/pick_and_place_block/task_spec.yaml', 'r') as f:
    task_spec = yaml.safe_load(f)

# Test stratified sampler
sampler = StratifiedSampler(
    Path('datasets/recordings/2025-08-28_pick-and-place-block/stratified_sampling_config.yaml'),
    task_spec
)

print('Stratified sampler initialized successfully!')

# Load existing episodes
dataset_dir = Path('datasets/recordings/2025-08-28_pick-and-place-block')
sampler.load_existing_episodes(dataset_dir)

# Save clean output
output_path = dataset_dir / "stratified_sampling_output_clean.yaml"
sampler.save_sampling_output(output_path)

print(f"Clean output saved to: {output_path}")



