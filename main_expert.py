
#!/usr/bin/env python3

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
    strategy = "expert"

    # Locate all obfuscated ontologies
    ontology_dir = os.path.join("data", "ontologies")
    ontology_files = sorted([
        f for f in os.listdir(ontology_dir)
        #if f.endswith("_obfuscated.json")
    ])

    # test only one ontology, otherwise uncomment
    #ontology_files = [config['ontology']['file'].split("/")[-1]]

    for ontology_file in ontology_files:
        print(f"[Main] Starting experiments for: {ontology_file}")
        print(f"[Main] Model: {config['model']['name']}, Provider: {config['model']['provider']}")
        ontology_path = os.path.join(ontology_dir, ontology_file)

        for run_id in range(REPEAT_COUNT):
            print(f"  ↳ Run {run_id+1}/{REPEAT_COUNT}")

            # Update config with current ontology
            set_value('ontology.file', ontology_path, config_path)

            #print("!!! No 20q testing, uncomment for that")
            # Testing
            tester = Tester(config_path, expert_on=True)
            test_logs = tester.run_tests()

            # Save test logs for aggregation
            #run_name = ontology_file.replace(".json", f"_run{run_id+1}.json")
            run_name = f"{strategy}_test.json"
            save_path = os.path.join("results", config["model"]["name"], ontology_file.removesuffix(".json"), "tests", "20q_game", run_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(test_logs, f, indent=2)

    print("[Main] All experiments completed.")

if __name__ == "__main__":
    main()
