"""
Test prompts loading.
"""

import os
import pytest
from utils.prompts_loader import PromptLoader


@pytest.mark.skipif(
    not os.path.exists("prompts/prompts.json"),
    reason="Prompts file not found"
)
def test_load_prompts():
    """Test loading prompts file."""
    loader = PromptLoader("prompts/prompts.json")
    assert loader is not None
    assert loader.prompts is not None
    assert isinstance(loader.prompts, dict)


def test_prompts_structure():
    """Test that prompts have expected structure."""
    prompts_path = "prompts/prompts.json"
    
    if not os.path.exists(prompts_path):
        pytest.skip("Prompts file not found")
    
    loader = PromptLoader(prompts_path)
    
    # Check that prompts is not empty
    assert len(loader.prompts) > 0
    assert len(prompts) > 0
