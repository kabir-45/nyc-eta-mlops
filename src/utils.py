import yaml
from pathlib import Path

def load_params(path: str = "params.yaml") -> dict:
    try:
        with open(path, "r") as f:
            params = yaml.safe_load(f)
        return params
    except Exception as e:
        raise Exception(f"Error loading params from {path}: {e}")