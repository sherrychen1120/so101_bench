
import yaml
import subprocess
import sys
import tempfile
import numpy as np
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

class TaskConfigurator:
    def __init__(self, 
        tasks_dir: Path, 
        task_name: str, 
        sampler_config_fpath: Optional[Path] = None
    ):
        self.tasks_dir = tasks_dir
        self.task_name = task_name
        self.stratified_sampler = None
        
        # Load task_spec
        task_spec_path = self.tasks_dir / task_name / "task_spec.yaml"
        if not task_spec_path.exists():
            raise FileNotFoundError(f"Task spec file not found: {task_spec_path}")
        with open(task_spec_path, 'r') as f:
            self.task_spec = yaml.safe_load(f)
        
        if sampler_config_fpath is not None and sampler_config_fpath.exists():
            self.stratified_sampler = StratifiedSampler(
                stratified_config_path=sampler_config_fpath,
                task_spec=self.task_spec,
            )

        # Load task_config_template
        task_config_template_path = self.tasks_dir / task_name / "task_config_template.yaml"
        if not task_config_template_path.exists():
            raise FileNotFoundError(f"Task config template file not found: {task_config_template_path}")
        with open(task_config_template_path, 'r') as f:
            self.task_config_template = yaml.safe_load(f)
        
        self.last_task_config = deepcopy(self.task_config_template)
    
    def generate_task_config_for_dataset(self):
        return {"task_name": self.task_name}
    
    def determine_sampling_plan(self) -> bool:
        if self.stratified_sampler is not None:
            return self.stratified_sampler.get_sampling_plan()
        return False
    
    def log_episode_sample(self, episode_name: str):
        if self.stratified_sampler is not None:
            self.stratified_sampler.log_episode_sample(episode_name)
    
    def get_task_config_from_user(self, use_sampling_plan: bool) -> dict:
        """
        Get task configuration from user input by opening an editor.
        If stratified sampler is available, pre-fill with sampled values.
        
        Returns:
            dict: Validated task configuration
        """
        config_to_edit = deepcopy(self.last_task_config)
        
        # If stratified sampler is available, pre-fill with sampled values
        if use_sampling_plan and self.stratified_sampler is not None:
            sampled_values = self.stratified_sampler.sample_task_config_values()
            if sampled_values is not None:
                # Merge sampled values into the config
                self._merge_variations(config_to_edit, sampled_values)
            else:
                print("Stratified sampling complete - all bins filled!")
        
        while True:
            # Create temporary file with current config
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                yaml.dump(config_to_edit, tmp_file, default_flow_style=False, sort_keys=False)
                tmp_path = tmp_file.name
            
            try:
                print("\n" + "="*60)
                print("TASK CONFIGURATION EDITOR")
                print("="*60)
                print(f"Please edit the task configuration for task: {self.task_name}")
                print("The editor will open momentarily. Save and close the file when done.")
                print("Valid options from task_spec.yaml:")
                
                # Show valid options from task spec
                spec_variations = self.task_spec.get("variations", {})
                for key, value in spec_variations.items():
                    if key == "start_pose":
                        print(f"  {key}:")
                        for pose_key, pose_spec in value.items():
                            if isinstance(pose_spec, dict) and "min" in pose_spec and "max" in pose_spec:
                                print(f"    {pose_key}: [{pose_spec['min']} to {pose_spec['max']}]")
                            elif isinstance(pose_spec, list):
                                print(f"    {pose_key}: {pose_spec}")
                    else:
                        print(f"  {key}: {value}")
                
                print("-"*60)
                input("Press Enter to open the editor...")
                
                # Open editor (try different editors)
                editors = ['nano', 'vim', 'vi', 'gedit', 'code']
                editor_opened = False
                
                for editor in editors:
                    try:
                        subprocess.run([editor, tmp_path], check=True)
                        editor_opened = True
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                
                if not editor_opened:
                    print("Could not open an editor. Please edit the file manually:")
                    print(f"File path: {tmp_path}")
                    input("Press Enter when you have finished editing the file...")
                
                # Read the edited config
                with open(tmp_path, 'r') as f:
                    edited_config = yaml.safe_load(f)
                
                # Validate the config
                is_valid, error_msg = self._validate_task_config(edited_config)
                
                if is_valid:
                    print("✅ Task configuration is valid!")
                    self.last_task_config = deepcopy(edited_config)
                    return self.last_task_config
                else:
                    print(f"❌ Invalid task configuration: {error_msg}")
                    print("Please fix the configuration and try again.")
                    config_to_edit = edited_config  # Use the invalid config as starting point for next iteration
                    
            finally:
                # Clean up temporary file
                try:
                    Path(tmp_path).unlink()
                except:
                    pass

    def _validate_task_config(self, task_config: dict) -> tuple[bool, str]:
        """
        Validate task configuration against task specification.
        
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check task name matches
            if task_config.get("task_name") != self.task_name:
                return False, (
                    f"Task name mismatch: expected '{self.task_name}', "
                    f"got '{task_config.get('task_name')}'"
                )
            
            # Validate variations
            config_variations = task_config.get("variations", {})
            spec_variations = self.task_spec.get("variations", {})
            
            for key, config_value in config_variations.items():
                if key not in spec_variations:
                    return False, f"Unknown variation key: '{key}'"
                
                spec_value = spec_variations[key]
                
                if key == "start_pose":
                    # Special handling for start_pose with min/max ranges
                    for pose_key, pose_value in config_value.items():
                        if pose_key not in spec_value:
                            return False, f"Unknown pose key: '{pose_key}'"
                        
                        spec_pose = spec_value[pose_key]
                        if isinstance(spec_pose, dict) and "min" in spec_pose and "max" in spec_pose:
                            # Validate range
                            min_vals = spec_pose["min"]
                            max_vals = spec_pose["max"]
                            
                            if len(pose_value) != len(min_vals) or len(pose_value) != len(max_vals):
                                return False, f"Pose '{pose_key}' should have {len(min_vals)} values"
                            
                            for i, val in enumerate(pose_value):
                                if not (min_vals[i] <= val <= max_vals[i]):
                                    return False, f"Pose '{pose_key}' value {i} ({val}) is out of range [{min_vals[i]}, {max_vals[i]}]"
                        elif isinstance(spec_pose, list):
                            # Direct list validation
                            if pose_value not in spec_pose:
                                return False, f"Invalid pose '{pose_key}' value: '{pose_value}'. Valid values: {spec_pose}"
                else:
                    # Regular list validation
                    if isinstance(spec_value, list):
                        if config_value not in spec_value:
                            return False, f"Invalid '{key}' value: '{config_value}'. Valid values: {spec_value}"
                    else:
                        return False, f"Unexpected spec format for '{key}'"
            
            return True, ""
        
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _merge_variations(self, config: Dict[str, Any], sampled_values: Dict[str, Any]):
        """Merge sampled variation values into the task config."""
        if "variations" not in config:
            config["variations"] = {}
        
        # Deep merge the sampled values
        def deep_merge(target, source):
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    deep_merge(target[key], value)
                else:
                    target[key] = value
        
        deep_merge(config["variations"], sampled_values)


class StratifiedSampler:
    """
    Handles stratified sampling for task configuration based on axes and bin sizes.
    """
    
    def __init__(self, stratified_config_path: Path, task_spec: Dict[str, Any]):
        """
        Initialize the stratified sampler.
        
        Args:
            stratified_config_path: Path to stratified sampling configuration YAML
            task_spec: Task specification containing variation definitions
        """
        self.stratified_config_path = stratified_config_path
        self.task_spec = task_spec
        
        # Load stratified sampling configuration
        with open(stratified_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.stratification_config = config['stratification_config']
        self.axes = self.stratification_config['axes']
        self.bin_sizes = self.stratification_config['bin_sizes']
        self.desired_episodes_per_bin = self.stratification_config['desired_episodes_per_bin']
        
        # Map axes to task spec variations and compute bins
        self._map_axes_to_task_spec()
        self._compute_bins()
        
        # Initialize tracking
        self.episodes_per_bin: Dict[int, List[str]] = {i: [] for i in range(self.total_bins)}  # bin_id -> list of episode names
        self.dataset_dir = None
        self.current_bin_index = 0
        
    def _map_axes_to_task_spec(self):
        """Map stratification axes to task spec variations and validate."""
        self.axis_mappings = {}
        task_variations = self.task_spec.get('variations', {})
        
        for axis_name, axis_range in self.axes.items():
            # Parse nested axis names (e.g., "start_pose.block.x")
            parts = axis_name.split('.')
            
            # The last part should be the dimension (x, y, z, yaw)
            dimension = parts[-1]
            # The path to the spec value excludes the dimension
            spec_path = parts[:-1]
            
            # Navigate through nested structure to find the spec value
            current = task_variations
            for part in spec_path:
                if part not in current:
                    raise ValueError(f"Axis path '{'.'.join(spec_path)}' not found in task spec variations")
                current = current[part]
            
            # Now current should be the dict with min/max
            if not isinstance(current, dict) or 'min' not in current or 'max' not in current:
                raise ValueError(f"Axis path '{'.'.join(spec_path)}' does not correspond to a continuous range in task spec")
            
            # For continuous ranges, find the appropriate dimension
            min_vals = current['min']
            max_vals = current['max']
            
            # Map x, y, z to indices (assuming x=0, y=1, z=2, yaw=2 for 3D poses)
            dim_map = {'x': 0, 'y': 1, 'z': 2, 'yaw': 2}
            if dimension not in dim_map:
                raise ValueError(f"Unknown dimension '{dimension}' for axis '{axis_name}'. Valid dimensions: {list(dim_map.keys())}")
            
            dim_idx = dim_map[dimension]
            if dim_idx >= len(min_vals) or dim_idx >= len(max_vals):
                raise ValueError(f"Dimension index {dim_idx} for '{dimension}' exceeds task spec dimensions")
            
            actual_min = min_vals[dim_idx]
            actual_max = max_vals[dim_idx]
            
            # Validate that stratification range is within task spec range
            if axis_range[0] < actual_min or axis_range[1] > actual_max:
                raise ValueError(
                    f"Stratification range {axis_range} for '{axis_name}' exceeds "
                    f"task spec range [{actual_min}, {actual_max}]"
                )
            
            self.axis_mappings[axis_name] = {
                'path': parts,
                'spec_path': spec_path,
                'dimension': dimension,
                'dim_idx': dim_idx,
                'range': axis_range,
                'task_range': [actual_min, actual_max]
            }
    
    def _compute_bins(self):
        """Compute bin ranges for each axis."""
        self.bins = {}
        self.bin_count_per_axis = {}
        self.sorted_axis_names = sorted(self.axes.keys())
        
        for axis_name in self.sorted_axis_names:
            axis_range = self.axes[axis_name]
            bin_size = self.bin_sizes[axis_name]
            start, end = axis_range
            
            # Compute number of bins for this axis
            num_bins = int(np.ceil((end - start) / bin_size))
            self.bin_count_per_axis[axis_name] = num_bins
            
            # Compute bin edges
            bin_edges = np.linspace(start, end, num_bins + 1)
            
            # Store bin ranges
            self.bins[axis_name] = []
            for i in range(num_bins):
                self.bins[axis_name].append([bin_edges[i], bin_edges[i + 1]])
        
        # Compute total number of bins (Cartesian product)
        self.total_bins = 1
        for count in self.bin_count_per_axis.values():
            self.total_bins *= count
        
        print(f"Stratified sampling initialized with {self.total_bins} total bins")
        for axis_name, count in self.bin_count_per_axis.items():
            print(f"  {axis_name}: {count} bins")
    
    def _bin_id_to_indices(self, bin_id: int) -> Dict[str, int]:
        """Convert flat bin ID to axis-specific bin indices."""
        indices = {}
        remaining = bin_id
        
        # Convert to multi-dimensional indices (similar to np.unravel_index)
        for i, axis_name in enumerate(reversed(self.sorted_axis_names)):
            count = self.bin_count_per_axis[axis_name]
            indices[axis_name] = remaining % count
            remaining //= count
        
        return indices
    
    def _indices_to_bin_id(self, indices: Dict[str, int]) -> int:
        """Convert axis-specific bin indices to flat bin ID."""
        bin_id = 0
        multiplier = 1
        
        for axis_name in reversed(self.sorted_axis_names):
            bin_id += indices[axis_name] * multiplier
            multiplier *= self.bin_count_per_axis[axis_name]
        
        return bin_id
    
    def load_existing_episodes(self, dataset_dir: Path):
        """
        Load existing episodes from raw dataset to understand current distribution.
        
        Args:
            dataset_dir: Path to the raw dataset directory
        """
        self.episodes_per_bin = {i: [] for i in range(self.total_bins)}
        self.dataset_dir = dataset_dir
        
        # Look for episode directories
        if not dataset_dir.exists():
            print("Dataset directory does not exist, starting fresh")
            return
        
        episodes_dir = dataset_dir / "episodes"
        episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]

        print(f"Loading {len(episode_dirs)} episodes from {episodes_dir}")
        
        for episode_dir in episode_dirs:
            task_config_path = episode_dir / "task_config.yaml"
            if task_config_path.exists():
                try:
                    with open(task_config_path, 'r') as f:
                        task_config = yaml.safe_load(f)
                    
                    # Determine which bin this episode belongs to
                    bin_id = self._get_bin_for_task_config(task_config)
                    if bin_id is not None:
                        self.episodes_per_bin[bin_id].append(episode_dir.name)
                except Exception as e:
                    print(f"Warning: Could not load task config for {episode_dir.name}: {e}")
        
        # Print current distribution
        print("\nCurrent episode distribution:")
        for bin_id in range(self.total_bins):
            count = len(self.episodes_per_bin[bin_id])
            if count > 0:
                indices = self._bin_id_to_indices(bin_id)
                print(f"  Bin {bin_id} {indices}: {count} episodes")
        
        # Initialize current bin index to the first bin that needs episodes
        current_bin = 0
        while current_bin < self.total_bins:
            print(f"Checking bin {current_bin}...")
            current_count = len(self.episodes_per_bin[current_bin])
            print(f"Current bin {current_bin} has {current_count} episodes...")
            if current_count < self.desired_episodes_per_bin:
                break
            current_bin += 1
        self.current_bin_index = current_bin
        print(f"Initialized current bin index to {self.current_bin_index}")
    
    def _get_bin_for_task_config(self, task_config: Dict[str, Any]) -> Optional[int]:
        """Determine which bin a task config belongs to."""
        try:
            indices = {}
            variations = task_config.get('variations', {})
            
            for axis_name, mapping in self.axis_mappings.items():
                spec_path = mapping['spec_path']
                dim_idx = mapping['dim_idx']
                
                # Navigate to the value in task config
                current = variations
                for part in spec_path:
                    if part not in current:
                        return None
                    current = current[part]
                
                # current should now be a list with pose values
                if not isinstance(current, list) or len(current) <= dim_idx:
                    return None
                
                value = current[dim_idx]  # Get specific dimension
                
                # Find which bin this value belongs to
                bin_ranges = self.bins[axis_name]
                bin_idx = None
                for i, (bin_min, bin_max) in enumerate(bin_ranges):
                    if bin_min <= value < bin_max or (i == len(bin_ranges) - 1 and value <= bin_max):
                        bin_idx = i
                        break
                
                if bin_idx is None:
                    return None
                
                indices[axis_name] = bin_idx
            
            return self._indices_to_bin_id(indices)
        
        except Exception:
            return None
    
    def get_sampling_plan(self) -> tuple[bool, int]:
        """
        Show the user the sampling plan and get confirmation.
        
        Returns:
            bool: True if user confirms, False otherwise
        """
        print("\n" + "="*60)
        print("STRATIFIED SAMPLING PLAN")
        print("="*60)
        
        bins_needing_episodes = []
        bins_not_needing_episodes = []
        total_needed = 0
        
        for bin_id in range(self.total_bins):
            current_count = len(self.episodes_per_bin[bin_id])
            needed = max(0, self.desired_episodes_per_bin - current_count)
            
            indices = self._bin_id_to_indices(bin_id)
            bin_ranges = {}
            for axis_name, bin_idx in indices.items():
                bin_ranges[axis_name] = self.bins[axis_name][bin_idx]
            
            if needed > 0:
                bins_needing_episodes.append({
                    'bin_id': bin_id,
                    'indices': indices,
                    'ranges': bin_ranges,
                    'current': current_count,
                    'needed': needed
                })
                total_needed += needed
            else:
                bins_not_needing_episodes.append({
                    'bin_id': bin_id,
                    'ranges': bin_ranges,
                    'indices': indices,
                    'current': current_count,
                    'needed': needed
                })
        
        if not bins_needing_episodes:
            print("All bins have sufficient episodes!")
            return False, 0
        
        print(f"Total episodes needed: {total_needed}")
        print("\nBins needing episodes:")
        for bin_info in bins_needing_episodes:
            print(f"  Bin {bin_info['bin_id']}: {bin_info['current']}/{self.desired_episodes_per_bin} episodes")
            for axis_name, range_vals in bin_info['ranges'].items():
                print(f"    {axis_name}: [{range_vals[0]:.3f}, {range_vals[1]:.3f})")
        
        print("\nBins not needing episodes:")
        for bin_info in bins_not_needing_episodes:
            print(f"  Bin {bin_info['bin_id']}: {bin_info['current']}/{self.desired_episodes_per_bin} episodes")
            for axis_name, range_vals in bin_info['ranges'].items():
                print(f"    {axis_name}: [{range_vals[0]:.3f}, {range_vals[1]:.3f})")
        
        print("\nPress Enter to use stratified sampling, or Ctrl+C to cancel...")
        try:
            input()
            return True, total_needed
        except KeyboardInterrupt:
            return False, total_needed
    
    def sample_task_config_values(self) -> Optional[Dict[str, Any]]:
        """
        Get task configuration values for the current bin without advancing.
        This is used to pre-fill the editor.
        
        Returns:
            Dict with variation values to use, or None if sampling is complete
        """
        # Find current bin that needs episodes
        current_bin = self.current_bin_index
        while current_bin < self.total_bins:
            print(f"Checking bin {current_bin}...")
            current_count = len(self.episodes_per_bin[current_bin])
            print(f"Current bin {current_bin} has {current_count} episodes...")
            if current_count < self.desired_episodes_per_bin:
                break
            current_bin += 1
        
        print(f"Sampling for bin {current_bin}...")
        
        if current_bin >= self.total_bins:
            return None  # All bins are complete
        
        # Get bin ranges for current bin
        indices = self._bin_id_to_indices(current_bin)
        
        # Generate random values within bin ranges
        variations = {}
        for axis_name, bin_idx in indices.items():
            bin_range = self.bins[axis_name][bin_idx]
            # Uniform random sample within bin
            value = np.random.uniform(bin_range[0], bin_range[1])
            
            # Map back to task config structure
            mapping = self.axis_mappings[axis_name]
            spec_path = mapping['spec_path']
            dim_idx = mapping['dim_idx']
            
            # Create nested structure
            current = variations
            for part in spec_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # Handle the pose structure - spec_path[-1] is the pose type (e.g., "block", "container")
            pose_type = spec_path[-1]
            if pose_type not in current:
                current[pose_type] = [0.0, 0.0, 0.0]  # Initialize with default pose
            
            # Ensure the list is long enough
            while len(current[pose_type]) <= dim_idx:
                current[pose_type].append(0.0)
            current[pose_type][dim_idx] = value
        
        print(f"Sampled values: {variations}")
        return variations
    
    def log_episode_sample(self, episode_name: str):
        """Record that an episode was completed for the current bin."""
        print(f"Logging episode {episode_name} for bin {self.current_bin_index}...")
        if self.current_bin_index < self.total_bins:
            self.episodes_per_bin[self.current_bin_index].append(episode_name)
            
            # Check if current bin is complete
            if len(self.episodes_per_bin[self.current_bin_index]) >= self.desired_episodes_per_bin:
                print(f"✅ Bin {self.current_bin_index} complete!")
                self.current_bin_index += 1
    
    def save_sampling_output(self):
        """Save the final sampling output to YAML file."""
        output_path = self.dataset_dir / "stratified_sampling_output.yaml"
        output_data = {
            'stratification_config': self.stratification_config,
            'sampling_outputs': {}
        }
        
        for bin_id, episodes in self.episodes_per_bin.items():
            if episodes:  # Only include bins with episodes
                indices = self._bin_id_to_indices(bin_id)
                bin_ranges = {}
                for axis_name, bin_idx in indices.items():
                    bin_ranges[axis_name] = self.bins[axis_name][bin_idx]
                
                # Convert numpy values to regular Python floats for clean YAML
                clean_bin_ranges = {}
                for axis_name, range_vals in bin_ranges.items():
                    clean_bin_ranges[axis_name] = [float(range_vals[0]), float(range_vals[1])]
                
                output_data['sampling_outputs'][bin_id] = {
                    'count': len(episodes),
                    'episodes': episodes,
                    'bin_ranges': clean_bin_ranges,
                    'bin_indices': indices
                }
        
        with open(output_path, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Sampling output saved to: {output_path}")


