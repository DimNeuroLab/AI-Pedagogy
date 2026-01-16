# utils/config_loader.py

import yaml
import os

def load_config(config_path="../config.yml"):
    """
    Load a YAML configuration file.

    Parameters:
    -----------
    config_path : str
        Path to YAML config file

    Returns:
    --------
    dict : parsed config dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)
    

def set_value(key_path, new_value, config_path = "../config.yml"):
    # Load existing YAML content
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Navigate to the correct nested key
    keys = key_path.split('.')
    d = data
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}  # Create nested dict if missing
        d = d[key]

    # Set the new value
    d[keys[-1]] = new_value

    # Write updated content back to file
    with open(config_path, 'w') as f:
        yaml.safe_dump(data, f)