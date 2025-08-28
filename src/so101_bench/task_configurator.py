
import yaml
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

class TaskConfigurator:
    def __init__(self, tasks_dir: Path, task_name: str):
        self.tasks_dir = tasks_dir
        self.task_name = task_name
        
        # Load task_spec
        task_spec_path = self.tasks_dir / task_name / "task_spec.yaml"
        if not task_spec_path.exists():
            raise FileNotFoundError(f"Task spec file not found: {task_spec_path}")
        with open(task_spec_path, 'r') as f:
            self.task_spec = yaml.safe_load(f)

        # Load task_config_template
        task_config_template_path = self.tasks_dir / task_name / "task_config_template.yaml"
        if not task_config_template_path.exists():
            raise FileNotFoundError(f"Task config template file not found: {task_config_template_path}")
        with open(task_config_template_path, 'r') as f:
            self.task_config_template = yaml.safe_load(f)
        
        self.last_task_config = deepcopy(self.task_config_template)
    
    def generate_task_config_for_dataset(self):
        return {"task_name": self.task_name}
    
    def get_task_config_from_user(self) -> dict:
        """
        Get task configuration from user input by opening an editor.
        
        Args:
            task_config_template: The template configuration
            last_config: The last user input (if validation failed)
            task_spec: Task specification for validation
        
        Returns:
            dict: Validated task configuration
        """
        config_to_edit = self.last_task_config
        
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


