"""
Test ontology utilities.
"""

import os
import pytest
from ontology.ontology_utils import load_ontology


@pytest.mark.skipif(
    not os.path.exists("data/ontologies/ontology_aliens_10.json"),
    reason="Test ontology not found"
)
def test_load_ontology():
    """Test loading an ontology file."""
    ontology = load_ontology("data/ontologies/ontology_aliens_10.json")
    assert ontology is not None
    assert isinstance(ontology, dict)


def test_ontology_structure():
    """Test that loaded ontology has expected structure."""
    ontology_path = "data/ontologies/ontology_aliens_10.json"
    
    if not os.path.exists(ontology_path):
        pytest.skip("Test ontology not found")
    
    ontology = load_ontology(ontology_path)
    
    # Check for expected keys (adjust based on your actual ontology structure)
    # This is a placeholder - adjust to match your actual ontology format
    assert ontology is not None
    assert len(ontology) > 0
