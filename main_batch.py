
#!/usr/bin/env python3
"""
main.py

ScaffAliens Orchestrator: Batch Mode

1. Loads global configuration.
2. For each ontology (in obfuscated set), run:
    a. Training phase (N times)
    b. Testing phase
3. Aggregates and saves results.

Author: Sab & ChatGPT
Date: updated
"""

import os
import json
from utils.config_loader import load_config, set_value
from ontology.ontology_generator import OntologyGenerator
from ontology.ontology_utils import load_ontology, save_ontology, obfuscate_ontology_names
from training.trainer import Trainer
from testing.test_20q import Tester
#from testing.ontology_tester import Tester

REPEAT_COUNT = 1  # Number of repetitions per ontology

def main():
    config_path = "config.yml"
    config = load_config(config_path)

    # Locate all obfuscated ontologies
    ontology_path = config["ontology"]["file"]
    ontology_file = os.path.splitext(os.path.basename(config["ontology"]["file"]))[0]

    print(f"[Main] Starting experiments for: {ontology_file}")
    print(f"[Main] Model: {config['model']['name']}, Provider: {config['model']['provider']}")

    for run_id in range(REPEAT_COUNT):
        print(f"  ↳ Run {run_id+1}/{REPEAT_COUNT}")

        # Update config with current ontology
        set_value('ontology.file', ontology_path, config_path)

        # Training
        trainer = Trainer(config_path)
        training_logs, learner = trainer.run_training()

        # Save training logs (include ontology + strategy + run id in filename)
        strategy = config.get("Teacher", {}).get("strategy", "unknown")
        train_name = f"{strategy}_train.json"

        train_path = os.path.join("results", config["model"]["name"], ontology_file.removesuffix(".json"), "dialogue", train_name)
        os.makedirs(os.path.dirname(train_path), exist_ok=True)

        with open(train_path, "w", encoding="utf-8") as tf:
            json.dump(training_logs, tf, indent=2)
        print(f"[Main] Training logs saved to {train_path}")

        #print("!!! No 20q testing, uncomment for that")
        # Testing
        #tester = Tester(config_path, learner)
        #test_logs = tester.run_tests()

        # Save test logs for aggregation
        #run_name = ontology_file.replace(".json", f"_run{run_id+1}.json")
        #run_name = f"{strategy}_test.json"
        #save_path = os.path.join("results", config["model"]["name"], ontology_file.removesuffix(".json"), "tests", "20q_game", run_name)
        #os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #with open(save_path, "w", encoding="utf-8") as f:
        #    json.dump(test_logs, f, indent=2)

    exit()

    print("[Main] All experiments completed.")

if __name__ == "__main__":
    main()
