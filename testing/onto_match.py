#!/usr/bin/env python3
"""
main.py

ScaffAliens Orchestrator

1. Loads global configuration.
2. Generates or loads the ontology.
3. Runs the Training phase.
4. Runs the Test phase.
5. (Optionally) Invokes evaluation scripts.

Usage:
    python main.py

Author: Sab & Luca
Date: 2nd May
"""

import os
import json

from utils.config_loader import load_config
from ontology.ontology_generator import OntologyGenerator
from ontology.ontology_utils import load_ontology, save_ontology
from training.trainer import Trainer
from testing.test_20q import Tester
# from evaluation.eval_performance import evaluate_performance
# from evaluation.eval_ontology import evaluate_ontology

def main():
    # 1) Load experiment configuration
    config_path = "config/config.yml"
    config = load_config(config_path)

    # 2) Ontology: generate or load
    ont_cfg = config["ontology"]
    ont_file = ont_cfg["file"]
    if ont_cfg.get("generate", False):
        # Generate a new ontology via LLM
        gen = OntologyGenerator(config["prompt_file"], config["model"])
        ontology = gen.generate_structured()
        save_ontology(ontology, ont_file)
        print(f"[Main] Generated and saved ontology to {ont_file}")
    else:
        # Load existing ontology from disk
        ontology = load_ontology(ont_file)
        print(f"[Main] Loaded existing ontology from {ont_file}")

    # 3) Training phase
    print("[Main] Starting Training phase...")
    trainer = Trainer(config_path)
    training_logs = trainer.run_training()

    # 4) Testing phase (20Q game)
    print("[Main] Starting Test phase...")
    tester = Tester(config_path)
    test_logs = tester.run_tests()

    # 5) (Optional) Evaluation
    # print("[Main] Evaluating performance...")
    # perf_results = evaluate_performance(test_logs, config)
    # print("[Main] Evaluating learned ontology quality...")
    # onto_results = evaluate_ontology(training_logs, test_logs, config)

    print("[Main] Pipeline complete.")


    