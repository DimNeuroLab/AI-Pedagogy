"""
Test configuration utilities.
"""

import os
import pytest
from utils.config_loader import load_config, set_value


def test_load_config():
    """Test that config file can be loaded."""
    # Use example config for testing
    config_path = "config.example.yml"
    if os.path.exists(config_path):
        config = load_config(config_path)
        assert config is not None
        assert "model" in config
        assert "ontology" in config


@pytest.mark.skipif(not os.path.exists("config.yml"), reason="config.yml not found")
def test_config_has_required_fields():
    """Test that config has all required fields."""
    config = load_config("config.yml")
    
    required_fields = [
        "model",
        "ontology",
        "training",
        "testing",
    ]
    
    for field in required_fields:
        assert field in config, f"Missing required field: {field}"
