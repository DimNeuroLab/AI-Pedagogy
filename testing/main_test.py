"""
Ontology Testing Script

USAGE:
This script allows you to selectively run various ontology-related tests.

Run all tests:
    python main_test.py --query --extract --align

Run only query generation:
    python main_test.py --query

Run only ontology extraction:
    python main_test.py --extract

Run only ontology alignment:
    python main_test.py --align

Available flags:
    --query     Run the query generation test
    --align     Run alignment between true ontology and the agent's version
    --visualize Visualize the ontology with dialogue steps
    --track_information Track information about the agent's knowledge acquisition process

to plan the results, either average or single run:
    run: python utils/analyzeOntoAcquisition.py --avg (--single)
"""


import os
import sys
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .ontology_tester import OntologyTester, transform_alien_structure_oneliner, convert_conversation, extract_json_from_agent_output
from agents.learner_agent import LearnerAgent
from agents.expert import ExpertAgent
from utils.config_loader import load_config
from testing.OntologyAligner import OntologyAligner

# Argument parser
parser = argparse.ArgumentParser(description="Run different ontology tests.")
parser.add_argument("--query", action="store_true", help="Run query generation test")
parser.add_argument("--extract", action="store_true", help="Run ontology extraction test")
parser.add_argument("--align", action="store_true", help="Run ontology alignment test")
parser.add_argument("--visualize", action="store_true", help="Visualize ontology with dialogue steps")
parser.add_argument("--track_information", action="store_true", help="Track information about the agent's knowledge acquisition process")
parser.add_argument("--corr_test", action="store_true", help="Run correlation test between learning plateau and alignment performance")
args = parser.parse_args()


# Load configuration
config = load_config("config.yml")
model_conf = config["model"]
model_name = model_conf["name"]
prompt_file = config["prompt_file"]
strategy = config["Teacher"]["strategy"]
save_dir = config.get("save_dir", "results")
ontology_path = config["ontology"]["file"]

# Load ontology 
if strategy.endswith("_2"):
    with open(ontology_path.rsplit(".json", 1)[0] + "_v2.json", "r", encoding="utf-8") as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_path.rsplit(".json", 1)[0] + "_v2.json")
elif strategy.endswith("_3"):
    with open(ontology_path.rsplit(".json", 1)[0] + "_v3.json", "r", encoding="utf-8") as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_path.rsplit(".json", 1)[0] + "_v3.json")
else:
    with open(ontology_path, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    print("[Main] Loaded ontology from ", ontology_path)

if config["ontology"]["text"] is not None:
    with open(config["ontology"]["text"], "r", encoding="utf-8") as f:
        glossary = f.read().splitlines()

OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests")
print(f"[Main] Output path set to: {OUTPUT_PATH}")

TRAINING_LOGS_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue", "{strategy}_train.json")
print(f"[Main] Training logs path set to: {TRAINING_LOGS_PATH}")

# Initialize learner
if strategy.startswith("expert"):
    learner = ExpertAgent(
        model_name=model_name,
        prompt_file_path=prompt_file,
        ontology=ontology if config["ontology"]["text"] is None else glossary,
    )
    strategy = strategy if config["ontology"]["text"] is None else f"{strategy}_with_text"
else:
    learner = LearnerAgent(
        model_name=model_name,
        prompt_file_path=prompt_file
    )
    conversation_path = os.path.join(TRAINING_LOGS_PATH.format(strategy=strategy))
    print(f"[Main] Loading conversation history from {conversation_path}...")
    if os.path.exists(conversation_path):
        with open(conversation_path, "r", encoding="utf-8") as f:
            conversation_history = convert_conversation(json.load(f)[0]["conversation"])
        learner.set_context(learner.conversation_history + conversation_history)
        learner.reset_context()


print("[Main] LearnerAgent initialized...")

# Output filenames
output_file_name_query = f"query/answers/query_results_{strategy}.json"
output_file_name_ontology = f"align/rebuilt_ontology/ontology_{strategy}.json" 


tester = OntologyTester(ontology)
# Run selected tests
if args.query:
    print("[Main] Running query generation test...")
    os.makedirs(os.path.join(OUTPUT_PATH, "query", "answers"), exist_ok=True)
    tester.run_test(
        n_sets=2,
        set_size=3,
        n_features=2,
        agent=learner,
        log_path=os.path.join(OUTPUT_PATH, output_file_name_query)
        )
    # Test the query responses
    tester.check_responses(
        file_output=os.path.join(OUTPUT_PATH, f"query/query_comparison_{strategy}.json")
    )

if args.align:
    print("[Main] Running ontology alignment test...")
    os.makedirs(os.path.join(OUTPUT_PATH, "align", "rebuilt_ontology"), exist_ok=True)
    tester.run_ontology_generation_test(
        learner,
        n_tests=5,
        log_path=os.path.join(OUTPUT_PATH, output_file_name_ontology)
    )

    # Begin mean alignment
    with open(os.path.join(OUTPUT_PATH, output_file_name_ontology), "r", encoding="utf-8") as f:
        data = json.load(f)
    alignments = {}
    for i, (test, ontology_agent) in enumerate(data.items()):
        aligner = OntologyAligner(ontology, ontology_agent)
        alignments[i] =  aligner.run_comparison()
        
    # Save alignment results
    with open(os.path.join(OUTPUT_PATH, "align", f"alignment_{strategy}.json"), 'w', encoding='utf-8') as f:
        json.dump(alignments, f, indent=4)
    print(f"[Main] Results saved to {os.path.join(OUTPUT_PATH, 'align', f'alignment_{strategy}.json')}")

if args.visualize:
    from utils.IG_visualizer import OntologyViewer
    print("Visualizing ontology...")
    app = OntologyViewer(ontology, conversation_history)
    app.mainloop()

if args.track_information:
    from utils.InformationTracker import InformationTracker
    plot = False
    tracker = InformationTracker(ontology)
    os.makedirs(os.path.join(OUTPUT_PATH, "track_info"), exist_ok=True)
    if not plot:
        print("[Main] Tracking information about the agent's knowledge acquisition process...")
        tracker.start(conversation_history)
        tracker.save(os.path.join(OUTPUT_PATH, "track_info", f"tracking_{strategy}.json"))
    else:
        tracker.plot_metrics(
        input_folder=os.path.join(OUTPUT_PATH, "track_info"),
        output_path=os.path.join(OUTPUT_PATH, "track_info", "tracking_plot.pdf")
        )
        tracker.plot_errors(
            input_folder=os.path.join(OUTPUT_PATH, "track_info"),
            output_path=os.path.join(OUTPUT_PATH, "track_info", "tracking_errors.pdf")
        )

