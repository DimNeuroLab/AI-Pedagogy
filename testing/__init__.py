"""
Testing package for AIP (Artificial Intelligence Pedagogy).

This package contains testing utilities and evaluation tools.
"""

from .test_20q import Tester
from .ontology_tester import OntologyTester
from .ontology_querier import OntologyQuerier
from .OntologyAligner import OntologyAligner

__all__ = [
    'Tester',
    'OntologyTester',
    'OntologyQuerier',
    'OntologyAligner',
]
