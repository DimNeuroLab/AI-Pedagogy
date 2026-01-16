
import os
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_interface import call_model_api
import json
from copy import deepcopy
import re

groups = {
        "Instructions": ["top_down", "bottom_up"],
        "Questions": ["learner_questions", "teacher_questions"],
        "Dialogic TD": ["mixed_top-down_learner_questions", "mixed_top-down_teacher_questions"],
        "Dialogic BU": ["mixed_bottom-up_learner_questions", "mixed_bottom-up_teacher_questions"],
        "Dialogic Qs": ["mixed_learner_questions", "mixed_teacher_questions"]
    }

def load_trials(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Results file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def classifier_dialogue(dialogue, step, model, num_samples = 5):
        
    system_prompt = """
                    You are an expert linguist specializing in Speech Act Theory and Educational Dialogue Analysis. 
                    Your task is to classify the function of a specific turn in a dialogue.

                    You will receive the dialogue history and the current turn.
                    1. First, analyze the linguistic structure and intent of the turn inside your thought process.
                    2. Then, output a single Valid JSON object containing the classification tag.
                    3. Do NOT output Markdown formatting (like ```json). Output the label string only.
                    """
    user_prompt = f"""
                    ### Task
                    Classify the **Current Turn** into exactly one of the following tags based on its primary communicative function.

                    ### The Rubric
                    1. **Informing**: Conveying facts, answers, explanations, or evaluative feedback (e.g., "The answer is 5", "That is correct").
                    2. **Inquiring**: Asking questions or probing for information (e.g., "Why?", "Is it X?").
                    3. **Acknowledging**: Confirming receipt or simple agreement without adding new content (e.g., "Okay", "Right", "I see").
                    4. **Phatic**: Social pleasantries or channel management (e.g., "Hello", "Can you hear me?").

                    ### The Input
                    **Conversation History:**
                    {dialogue}

                    **Current Turn to Classify:**
                    {step}

                    ### Output Format
                    Output ONLY one of the following tags as a string: "Informing", "Inquiring", "Acknowledging", "Phatic", or "Not Related" if none apply.
                    """
                    
    message = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    
    results = []

    for _ in range(num_samples):
        response = call_model_api(message, model)
        print(f"[Classifier] Model response: {response}")
        # Look for which tag appears in the LLM response
        found_tag = None
        for tag in [
            "Informing",
            "Inquiring",
            "Acknowledging",
            "Phatic"
        ]:
        
            if tag in response:
                found_tag = tag
                break  # stop at the first match

        # Default to "Not Related" if nothing matches
        results.append(found_tag or "No label found")

    response = Counter(results).most_common(1)[0][0]
    #print(f"Most common response: {response}, {type(response)}")
    return response


def start_classifier(dialogue, model_name):

    if not dialogue:
        raise ValueError("Dialogue cannot be empty or None")

    res = {}
    for i, step in enumerate(dialogue):
        print(f"[Classifier] Processing dialogue turn {i+1}/{len(dialogue)}")
        
        classification = classifier_dialogue(
            dialogue=dialogue[:i],
            step=step,
            model=model_name,
            num_samples=5
        )

        print(f"[Classifier] Classification for turn {i+1}: {classification}")
        res[i] = classification
    return res

def main_classfier():
    from config_loader import load_config

    config = load_config("config.yml")
    #ontology_file = config["ontology"]["file"]
    save_dir = config.get("save_dir", "results")
    #model_name = config["model"]["name"]
    model_name = "gpt-oss-large"
    strategy = config["Teacher"]["strategy"]
    #with open(ontology_file, 'r', encoding='utf-8') as f:
    #    ontology = json.load(f)
    #    print("[Main] Loaded ontology from ", ontology_file)
    OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests")

    TRAINING_LOGS_DIR = os.path.join(save_dir, "gpt-4o", os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue")
    print(f"[Main] Output path set to: {OUTPUT_PATH}")
    print(f"[Main] Training logs path set to: {TRAINING_LOGS_DIR}")
    os.makedirs(os.path.join(OUTPUT_PATH, "track_info"), exist_ok=True)

    res = {}
    for _, agent_keys in groups.items():
        for agent_key in agent_keys:
            if agent_key == "bottom_up" or agent_key == "top_down":
                continue  # Skip these strategies
            print(f"[Main] Processing strategy: {agent_key}")
            strategy = agent_key
            filename = os.path.join(TRAINING_LOGS_DIR, f"{strategy}_train.json")
            if os.path.isfile(filename) and filename.endswith("_train.json"):
                print(f"[Main] Processing file: {filename}")
                conversation = load_trials(filename)[0]["conversation"]
                print(f"[Main] Converted conversation with {len(conversation)} turns.")
                print(f"[Main] Starting tracking for {filename}")
                res[strategy] = start_classifier(conversation, model_name)

                print(f"[Main] Saving cumulative results...")
                with open(os.path.join(OUTPUT_PATH, "track_info", f"classifier_results.json"), 'w', encoding='utf-8') as f:
                    json.dump(res, f, indent=4)


if __name__ == "__main__":
    main_classfier()