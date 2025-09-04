import os
import yaml

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # Always resolve relative to the project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, "config", "config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config