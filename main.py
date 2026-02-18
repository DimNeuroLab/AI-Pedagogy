#!/usr/bin/env python3
"""
main.py

AIP: Artificial Intelligence Pedagogy
Main entry point for running single pedagogical interaction sessions.

For batch experiments, use main_batch.py
For expert-only testing, use main_expert.py

Author: Sab & Luca
Date: Updated 2025
"""

import os
import json
import argparse
from utils.config_loader import load_config, set_value
from ontology.ontology_generator import OntologyGenerator
from ontology.ontology_utils import load_ontology, save_ontology, obfuscate_ontology_names
from training.trainer import Trainer
from testing.test_20q import Tester


def main():
    """Run a single training and testing session."""
    parser = argparse.ArgumentParser(
        description='AIP: Artificial Intelligence Pedagogy - Run a single session'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='Path to configuration file (default: config.yml)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['top_down', 'bottom_up', 'learner_questions', 'teacher_questions', 'mixed'],
        help='Override strategy from config file'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip training phase and only run testing'
    )
    parser.add_argument(
        '--skip-testing',
        action='store_true',
        help='Skip testing phase and only run training'
    )
    
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)
    
    # Override strategy if provided
    if args.strategy:
        set_value('Teacher.strategy', args.strategy, config_path)
        config = load_config(config_path)
    
    print(f"[Main] Starting AIP session")
    print(f"[Main] Model: {config['model']['name']}, Provider: {config['model']['provider']}")
    print(f"[Main] Strategy: {config['Teacher']['strategy']}")
    print(f"[Main] Ontology: {config['ontology']['file']}")
    
    learner = None
    
    # Training Phase
    if not args.skip_training:
        print("\n[Main] === Training Phase ===")
        trainer = Trainer(config_path)
        training_logs, learner = trainer.run_training()
        
        # Save training logs
        ontology_name = os.path.splitext(os.path.basename(config['ontology']['file']))[0]
        strategy = config['Teacher']['strategy']
        save_path = os.path.join(
            "results",
            config['model']['name'],
            ontology_name,
            "training",
            f"{strategy}_training.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(training_logs, f, indent=2, ensure_ascii=False)
        
        print(f"[Main] Training logs saved to: {save_path}")
    
    # Testing Phase
    if not args.skip_testing:
        print("\n[Main] === Testing Phase ===")
        tester = Tester(config_path, learner=learner)
        test_logs = tester.run_tests()
        
        # Save test logs
        ontology_name = os.path.splitext(os.path.basename(config['ontology']['file']))[0]
        strategy = config['Teacher']['strategy']
        save_path = os.path.join(
            "results",
            config['model']['name'],
            ontology_name,
            "tests",
            "20q_game",
            f"{strategy}_test.json"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(test_logs, f, indent=2, ensure_ascii=False)
        
        print(f"[Main] Test logs saved to: {save_path}")
    
    print("\n[Main] Session completed successfully!")


if __name__ == "__main__":
    main()
