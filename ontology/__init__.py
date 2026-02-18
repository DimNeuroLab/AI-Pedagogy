"""
Ontology package for AIP (Artificial Intelligence Pedagogy).

This package contains utilities for generating, managing, and manipulating ontologies.
"""

from .ontology_utils import (
    load_ontology,
    save_ontology,
    get_alien_ontology,
    format_ontology,
    obfuscate_ontology_names,
    generate_trial_sets,
)
from .ontology_generator import OntologyGenerator

__all__ = [
    'load_ontology',
    'save_ontology',
    'get_alien_ontology',
    'format_ontology',
    'obfuscate_ontology_names',
    'generate_trial_sets',
    'OntologyGenerator',
]
