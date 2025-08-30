from pathlib import Path
from so101_bench.task_configurator import StratifiedSampler
import yaml

# Load task spec
with open('/home/melon/sherry/so101_bench/datasets/tasks/pick_and_place_block/task_spec.yaml', 'r') as f:
    task_spec = yaml.safe_load(f)

# Test stratified sampler
sampler = StratifiedSampler(
    Path('/home/melon/sherry/so101_bench/datasets/recordings/2025-08-28_pick-and-place-block/stratified_sampling_config.yaml'),
    task_spec
)

print('Stratified sampler initialized successfully!')
print(f'Total bins: {sampler.total_bins}')
print(f'Axes: {list(sampler.axes.keys())}')
print(f'Bin counts per axis: {sampler.bin_count_per_axis}')

# Test getting next config values
sampler.load_existing_episodes(Path('/home/melon/sherry/so101_bench/datasets/recordings/2025-08-28_pick-and-place-block/'))
config_values = sampler.get_next_task_config_values()
print(f'Sample config values: {config_values}')
