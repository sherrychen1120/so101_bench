import yaml
import json
from pathlib import Path
from typing import Any

def load_yaml(file_path: Path) -> dict:
    """Load YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, file_path: Path):
    """Save data to YAML file."""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=False)

def load_json(file_path: str) -> dict[str, Any]:
    """Load JSON file and return parsed content."""
    with open(file_path, 'r') as f:
        return json.load(f)


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    """Load JSONL file and return list of parsed records."""
    records = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
