"""
Pytest configuration and fixtures.
"""

import sys
import os
import pytest

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "model": {
            "name": "gpt-4o",
            "provider": "openai",
            "temperature": 0.3,
            "max_tokens": 10000
        },
        "ontology": {
            "file": "data/ontologies/ontology_aliens_10.json",
            "generate": True,
            "obfuscate": True
        },
        "training": {
            "num_trials": 1,
            "turns_per_session": 20
        },
        "testing": {
            "max_questions": 20,
            "num_sets": 20
        }
    }


@pytest.fixture
def temp_config_file(tmp_path, sample_config):
    """Create a temporary config file for testing."""
    import yaml
    
    config_file = tmp_path / "test_config.yml"
    with open(config_file, 'w') as f:
        yaml.dump(sample_config, f)
    
    return str(config_file)
