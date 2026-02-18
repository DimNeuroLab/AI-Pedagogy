import os
from collections import Counter
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from .api_interface import call_model_api
import json
from copy import deepcopy
import re
import numpy as np
import pandas as pd



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
    
def make_json_safe(obj):
    if isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    else:
        return obj

class InformationTracker:

    def __init__(self, ontology):

                # Validate inputs
        if not isinstance(ontology, dict) or not ontology:
            raise ValueError("Ontology must be a non-empty dictionary")

        self.ontology = ontology
        self._initialize_facts()
        self._initialize_metrics()

    def save(self, output_path=None):
        """Save the metrics to a file or return as a string."""
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({"metrics": self.metrics, "facts": make_json_safe(self.facts), "fact_checking": self.fact_checking}, f, indent=4)
            return f"Metrics saved to {output_path}"
        else:
            raise ValueError("Output path must be provided to save metrics")

    def plot_metrics(self, input_folder, output_path):
        import matplotlib.pyplot as plt

        data = []

        for _, agent_keys in groups.items():
            for agent_key in agent_keys:

                #if agent_key == "bottom_up" or agent_key == "top_down":
                #    continue  # Skip these strategies
#
                file_path = os.path.join(input_folder, f"tracking_{agent_key}.json")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Results file not found: {file_path}")

                tkf = load_trials(file_path)["metrics"]["total_known_facts"][1:]
                fc = load_trials(file_path)["fact_checking"]
                fc_list = [fc[str(i)] for i in range(len(fc))]
                diff = []
                cumulative_subtract = 0

                for a, b in zip(tkf, fc_list):
                    cumulative_subtract += len(b) 
                    diff.append(a - cumulative_subtract)

                data.append({agent_key: diff})

        plt.figure(figsize=(10, 6))

        # ---- Plot each strategy ----
        for strategy_dict in data:
            for strategy, values in strategy_dict.items():
                # Different styling for special strategies
                print(f"[Plot] Strategy: {strategy}")
                if strategy == "top_down":
                    print(f"[Plot] Plotting Top-Down strategy")
                    plt.plot(
                        range(len(values)),
                        values,
                        marker="o",
                        label="Top-Down"
                    )
                elif strategy == "bottom_up":
                    print(f"[Plot] Plotting Bottom-Up strategy")
                    plt.plot(
                        range(len(values)),
                        values,
                        marker="o",
                        label="Bottom-Up"
                    )
                else:
                    plt.plot(
                        range(len(values)),
                        values,
                        marker="o",
                        label=strategy
                    )

        total_facts = sum(len(v) for v in self.ground_truth.values())
        plt.axhline(y=total_facts, color='r', linestyle='--', label='Total Facts in Ontology')

        plt.xlabel("Step")
        plt.ylabel("Total Known Facts")
        plt.title("Total Known Facts per Strategy")
        plt.legend()
        plt.grid(True)

        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()
        
    def _initialize_metrics(self):
        """Initialize metrics for tracking information extraction."""
        self.metrics = {
            "total_known_facts": [sum(len(v[-1]) for v in self.facts.values())],
            "coverage" : {"entities": [], 
                          "features": [], 
                          "values": [], 
                          "features_values": [], 
                          "entities_features": [], 
                          "triplets": []
                          },
            "novelty": {
                "entities": [],
                "features": [],
                "values": [],
                "features_values": [],
                "entities_features": [],
                "triplets": []
            },
            "history": [],
        }

    def _initialize_facts(self):

        self.entities = [e.lower() for e in self.ontology.keys()]
        self.features = self._extract_all_features()
        self.values = self._extract_all_values()
        self.features_values = self._extract_feature_values()
        self.entities_features = self._extract_entities_features()
        self.triplets = self._extract_triplets()

        self.ground_truth = {
            "entities": self.entities,
            "features": self.features,
            "values": self.values,
            "features_values": self.features_values,
            "entities_features": self.entities_features,
            "triplets": self.triplets
        }

        self.facts = {
            "entities": [set()],
            "features": [set()],
            "values": [set()],
            "features_values": [set()],
            "entities_features": [set()],
            "triplets": [set()]
        }
    
    def _extract_feature_values(self):
        """Extract feature values for each entity."""
        feature_values = set()
        for _, entity_data in self.ontology.items():
            if isinstance(entity_data, dict):
                for feature, value in entity_data.items():
                    if isinstance(value, (str)):
                        feature_values.add((feature.lower(), value.lower()))
        return feature_values
    
    def _extract_entities_features(self):
        """Extract feature values for each entity."""
        entities_values = set()
        for entity, entity_data in self.ontology.items():
            if isinstance(entity_data, dict):
                for feature, _ in entity_data.items():
                        entities_values.add((entity.lower(), feature.lower()))
        return entities_values
    
    def _extract_triplets(self):
        """Extract triplets (entity, feature, value) from the ontology."""
        triplets = set()
        for entity, entity_data in self.ontology.items():
            if isinstance(entity_data, dict):
                for feature, value in entity_data.items():
                    if isinstance(value, (str)):
                        triplets.add((entity.lower(), feature.lower(), value.lower()))
        return triplets

    def _extract_all_features(self):
        """Extract all unique features from the ontology."""
        features = set()
        for entity_data in self.ontology.values():
            if isinstance(entity_data, dict):
                features.update(entity_data.keys())
        return [f.lower() for f in features]

    def _extract_all_values(self):
        """Extract all unique values from the ontology."""
        values = set()
        for entity_data in self.ontology.values():
            if isinstance(entity_data, dict):
                for value in entity_data.values():
                    if isinstance(value, (str, int, float)):
                        values.add(str(value))
        return [v.lower() for v in values]

    def compute_coverage(self):
        """Compute the coverage of known facts (per each fact) in the ontology."""
        for key, value in self.ground_truth.items():
            if key in self.facts:
                self.metrics["coverage"][key].append(len(self.facts[key][-1]) / len(value) if value else 0.0)
            else:
                self.metrics["coverage"][key].append(0.0)

    def compute_novelty(self):
        """Compute the novelty of newly discovered facts."""
        for key, value in self.facts.items():
                new_facts = value[-1] - value[-2] if len(value) > 1 else set()
                self.metrics["novelty"][key].append(len(new_facts) if value else 0.0)

    def extract_triples(self, text, num_samples = 5, model = "gpt-4o"):
        """ Extract (entity, feature, value) triples from the text.
        This method uses LLM.
        """
        # Use LLM to extract triples
        system_prompt = """You are a precise ontology information extractor. 
        Your task is to identify explicit references to ontology concepts in dialogue and return them as structured triples.

        ONTOLOGY STRUCTURE:
        - Entities: Top-level species (e.g., "Zyloxian", "Glimmeron", "Crystar")
        - Features: Properties/attributes of entities (e.g., "Diet", "Habitat", "Morphology", "Locomotion", "SocialStructure")
        - Values: Specific assignments from the ontology (e.g., "carnivorous", "aquatic", "armoured")

        TRIPLE FORMAT: [entity, feature, value]
        - Use exact ontology strings only
        - Use null (JSON null, not string) for any missing component
        - Each triple must have at least one non-null component

        ---

        ### EXTRACTION PATTERNS

        1. **Complete Assignment** → [entity, feature, value]  
        - "Zyloxian’s Diet is carnivorous" → ["Zyloxian", "Diet", "carnivorous"]

        2. **Entity–Feature Reference** → [entity, feature, null]  
        - "What is Crystar’s Morphology?" → ["Crystar", "Morphology", null]

        3. **Entity Mention** → [entity, null, null]  
        - "Let's discuss Pyron" → ["Pyron", null, null]  
        - "The entities Aquarion and Floran" → ["Aquarion", null, null], ["Floran", null, null]

        4. **Feature–Value Relationship** → [null, feature, value]  
        - "Diet can be herbivorous" → [null, "Diet", "herbivorous"]

        5. **Feature Discussion** → [null, feature, null]  
        - "Let's examine the SocialStructure feature" → [null, "SocialStructure", null]

        6. **Value Reference** → [null, null, value]  
        - "The value 'crystal' appears in the data" → [null, null, "crystal"]

        ---

        ### CRITICAL RULES

        1. **Explicit only**: Extract only what is directly written.  
        - ❌ "it", "that feature", "the species" → not valid.  
        - ❌ No implied links or reasoning.

        2. **Validation**:  
        - Entities, features, and values must exist in the ontology.  
        - Reject unknown terms, paraphrases, or synonyms.  
        - No duplicates.

        3. **Negations**: Extract only if both feature and negated value are explicitly stated.  
        - "Zyloxian’s Diet is not herbivorous" → ["Zyloxian", "Diet", "not herbivorous"] (keep “not” as part of value).  
        - "Zyloxian is not like Crystar" → no extraction.

        4. **Lists**: Expand into separate triples.  
        - "The entities A, B, C" → ["A", null, null], ["B", null, null], ["C", null, null]

        5. **Ambiguity**: Skip unclear mentions.  
        - ❌ "the blue one"  
        - ❌ "that entity"  

        6. **No assumptions**: Do not infer or guess missing information.

        7. Output must adhere strictly to instructions.

        ---

        ### QUALITY CHECK
        Before output, verify:  
        - ✓ JSON array of triples only (no extra text, no markdown)  
        - ✓ At least one component is non-null per triple  
        - ✓ All non-null terms match ontology exactly  
        - ✓ No duplicates  

        ---

        ### OUTPUT FORMAT
        - Return only a JSON array. No explanations or extra text.
        """

        user_prompt = f"""The ontology you will use is:
        {json.dumps(self.ontology, indent=2)}

        Dialogue turn to analyze:
        {text}

        Task: Extract all explicit ontology references from the dialogue turn, following the system rules.

        Return a JSON array of triples in the format 
        [entity, feature, value], 
        [entity, feature, null],
        [entity, null, null],
        [null, feature, value],
        [null, feature, null],
        [null, null, value],
        using null for missing parts. 
        Output only the JSON array, no explanations."""
        
        message = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        
        results = []
        # Call the LLM API to extract triples
        for i in range(num_samples):

            #print(f"[Extracting Triplets] Calling LLM API... Attempt {i+1}/{num_samples}")
            response = call_model_api(message, model)
            results.append(response)
            #print(f"LLM response: {response}")  
        
        response = Counter(results).most_common(1)[0][0]
        #print(f"Most common response: {response}, {type(response)}")

        return response

    def process_triples_response(self, response):
        """Process the LLM response to extract valid triples."""
        try:
            # Case 1: response is already a list
            if isinstance(response, list):
                triples = response

            # Case 2: response is a string
            elif isinstance(response, str):
                response = response.strip()

                # Remove markdown code blocks
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1]) if len(lines) > 2 else response
                if response.startswith("json"):
                    response = response[4:].strip()

                # Try to find the JSON array
                start_idx = response.find("[")
                end_idx = response.rfind("]")
                if start_idx != -1 and end_idx != -1:
                    response = response[start_idx:end_idx+1]
                else:
                    return []

                # Handle empty arrays or "no triples"
                if not response or response.strip() in ["[]", ""]:
                    return []

                # Detect case where multiple triples are listed without outer []
                # e.g. [null, "Diet", null], [null, "Habitat", null]
                if not response.strip().startswith("[["):
                    # If multiple bracketed triples exist, wrap them
                    bracketed_groups = re.findall(r"\[.*?\]", response)
                    if bracketed_groups:
                        response = "[" + ",".join(bracketed_groups) + "]"

                # Parse JSON safely
                triples = json.loads(response)

            else:
                print(f"Unexpected response type: {type(response)}")
                return []

            # Validate and clean triples
            valid_triples = []
            for triple in triples:
                if isinstance(triple, list) and len(triple) == 3:
                    processed_triple = []
                    for item in triple:
                        if item is None:
                            processed_triple.append(None)
                        elif isinstance(item, str) and item.strip().lower() in {"none", "null"}:
                            processed_triple.append(None)
                        else:
                            processed_triple.append(str(item))
                    valid_triples.append(processed_triple)
                else:
                    print(f"Invalid triple format: {triple}")

            return valid_triples

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {str(response)[:200]}...")
            return []
        except Exception as e:
            print(f"Error processing triples response: {e}")
            return []

    def process_triplets(self, triples):
        """Extract FACTS from the triples"""
        new_facts = {
                    "entities": set(),
                    "features": set(),
                    "values": set(),
                    "features_values": set(),
                    "entities_features": set(),
                    "triplets": set()
                    }

        if not triples:
            return new_facts
        for triple in triples:
            if len(triple) >= 3:
                entity = triple[0].lower() if triple[0] else None
                feature = triple[1].lower() if triple[1] else None
                value = triple[2].lower() if triple[2] else None
                if entity:
                    new_facts["entities"].add(entity)
                if feature:
                    new_facts["features"].add(feature)
                if value:
                    new_facts["values"].add(value)
                if entity and feature:
                    new_facts["entities_features"].add((entity, feature))
                if feature and value:
                    new_facts["features_values"].add((feature, value))
                if entity and feature and value:
                    new_facts["triplets"].add((entity, feature, value))
            else:
                raise ValueError(f"Invalid triple format: {triple}. Expected format: [entity, feature, value]")
        return new_facts

    def update_facts(self, new_facts):
        """Update the facts with newly discovered information."""
        if not isinstance(new_facts, dict):
            raise ValueError("New facts must be a dictionary")
        
        for key in self.facts.keys():
            if key in new_facts:
                self.facts[key].append(new_facts[key] | self.facts[key][-1])
            else:
                raise ValueError(f"New facts must contain key: {key}")

    def update_metrics(self):
        """Update metrics after processing new facts."""
        self.compute_coverage()
        self.compute_novelty()
        self.metrics["total_known_facts"].append(sum(len(v[-1]) for v in self.facts.values()))
        #self.metrics["history"].append(deepcopy(self.metrics))

    def is_valid_triple(self, triple):
        """Check if a triple is valid according to the ontology."""
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            return False
        entity, feature, value = (x.lower() if isinstance(x, str) else x for x in triple)
        if entity is not None and entity not in self.entities:
            return False
        if feature is not None and feature not in self.features:
            return False
        if value is not None and value not in self.values:
            return False
        if entity is not None and feature is not None and (entity, feature) not in self.entities_features:
            return False
        if feature is not None and value is not None and (feature, value) not in self.features_values:
            return False
        if feature is not None and value is not None and entity is not None and (entity, feature, value) not in self.triplets:
            return False
        return True

    def start(self, dialogue):
        """Process a dialogue turn to extract and update facts."""
        if not dialogue:
            raise ValueError("Dialogue cannot be empty or None")

        self.fact_checking = {}

        for i, line in enumerate(dialogue):
            
            print(f"[Start] Processing dialogue turn {i+1}/{len(dialogue)}")
            line = line["content"]

            # Extract triples from the dialogue
            triples = self.extract_triples(line)
            
            # Process the response to get valid triples
            valid_triples = self.process_triples_response(triples)
            
            # Check the validity of the extracted triples (if they exist in the ontology)
            self.fact_checking[i] = [triple for triple in valid_triples if not self.is_valid_triple(triple)]

            # Process the valid triples to extract facts
            new_facts = self.process_triplets(valid_triples)
            # Update the facts with newly discovered information
            self.update_facts(new_facts)
            
            # Update metrics
            self.update_metrics()

    def reset(self):
        """Reset the facts to their initial state."""
        self.facts = {
            "entities": [set()],
            "features": [set()],
            "values": [set()],
            "features_values": [set()],
            "entities_features": [set()],
            "triplets": [set()]
        }
        self._initialize_metrics()

def convert_conversation(convo_list):
    messages = []
    for i, entry in enumerate(convo_list):
        content = entry.strip()

        # Last line always becomes assistant
        if i == len(convo_list) - 1:
            role = "assistant"
        
        elif content.startswith("Teacher:"):
            role = "user"
            content = content[len("Teacher:"):].strip()
        
        elif content.startswith("Learner:"):
            role = "assistant"
            content = content[len("Learner:"):].strip()
        
        elif content.startswith("Please briefly summarise what you understood."):
            role = "user"
        
        else:
            raise ValueError(f"Unknown role in conversation entry: {content}")

        messages.append({"role": role, "content": content})
    
    return messages

def main_bunch():
    from config_loader import load_config

    config = load_config("config.yml")
    ontology_file = config["ontology"]["file"]
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    strategy = config["Teacher"]["strategy"]
    with open(ontology_file, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_file)
    OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests")

    TRAINING_LOGS_DIR = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue")
    print(f"[Main] Output path set to: {OUTPUT_PATH}")
    print(f"[Main] Training logs path set to: {TRAINING_LOGS_DIR}")
    os.makedirs(os.path.join(OUTPUT_PATH, "track_info"), exist_ok=True)
    tracker = InformationTracker(ontology)

    for _, agent_keys in groups.items():
        for agent_key in agent_keys:
            print(f"[Main] Processing strategy: {agent_key}")
            strategy = agent_key
            filename = os.path.join(TRAINING_LOGS_DIR, f"{strategy}_train.json")
            if os.path.isfile(filename) and filename.endswith("_train.json"):
                print(f"[Main] Processing file: {filename}")
                conversation = convert_conversation(load_trials(filename)[0]["conversation"])
                print(f"[Main] Converted conversation with {len(conversation)} turns.")
                print(f"[Main] Starting tracking for {filename}")
                tracker.start(conversation)
                tracker.save(os.path.join(OUTPUT_PATH, "track_info", f"tracking_{strategy}.json"))
    
            tracker.reset()
    tracker.plot_metrics(
        input_folder=os.path.join(OUTPUT_PATH, "track_info"),
        output_path=os.path.join(OUTPUT_PATH, "track_info", f"tracking_plot_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf")
    )

def main_single():
    from config_loader import load_config

    config = load_config("config.yml")
    ontology_file = config["ontology"]["file"]
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    strategy = config["Teacher"]["strategy"]
    with open(ontology_file, 'r', encoding='utf-8') as f:
        ontology = json.load(f)
        print("[Main] Loaded ontology from ", ontology_file)
    OUTPUT_PATH = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "tests")

    TRAINING_LOGS_DIR = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(config["ontology"]["file"]))[0], "dialogue")
    print(f"[Main] Output path set to: {OUTPUT_PATH}")
    print(f"[Main] Training logs path set to: {TRAINING_LOGS_DIR}")
    os.makedirs(os.path.join(OUTPUT_PATH, "track_info"), exist_ok=True)
    tracker = InformationTracker(ontology)
    filename = os.path.join(TRAINING_LOGS_DIR, f"{strategy}_train.json")
    if os.path.isfile(filename) and filename.endswith("_train.json"):
        print(f"[Main] Processing file: {filename}")
        conversation = convert_conversation(load_trials(filename)[0]["conversation"])
        print(f"[Main] Converted conversation with {len(conversation)} turns.")
        print(f"[Main] Starting tracking for {filename}")
        tracker.start(conversation)
        tracker.save(os.path.join(OUTPUT_PATH, "track_info", f"tracking_{strategy}.json"))
    tracker.reset()

    tracker.plot_metrics(
        input_folder=os.path.join(OUTPUT_PATH, "track_info"),
        output_path=os.path.join(OUTPUT_PATH, "track_info", f"tracking_plot_{os.path.splitext(os.path.basename(config['ontology']['file']))[0]}.pdf")
    )

def compute_refined_dynamics(cumulative_facts_list):
    """
    Computes Effective Rate and Saturation metrics.
    
    Args:
        cumulative_facts_list (list[int]): E.g., [0, 10, 50, 90, 100, 100, 100...]
    """
    F = np.array(cumulative_facts_list)
    N = len(F)
    
    if N < 2: return {}

    # 1. Determine the "Ceiling" (Final Knowledge State)
    final_facts = F[-1]
    if final_facts == 0:
        return {
            "Total_Facts": 0,
            "Effective_Rate": 0.0,
            "Turns_to_Saturation": 0,
            "Saturation_Index": 0.0
        }

    # 2. Find Time to Saturation (T_sat)
    # The first turn where we reach >= 95% of the final count
    threshold = 1 * final_facts
    # np.argmax returns the *first* index where condition is true
    t_sat = np.argmax(F >= threshold)
    
    # Safety: if t_sat is 0 (started with facts), set to 1 to avoid div/0
    t_sat_safe = max(1, t_sat)

    # 3. Compute Metrics
    
    # Effective Rate: Facts per turn during the active phase
    effective_rate = final_facts / t_sat_safe
    
    # Saturation Index (Shape): Still useful for Front vs Back loading
    # (Same formulation as before, just kept for context)
    auc = np.sum(F)
    max_area = N * final_facts
    saturation_idx = auc / max_area

    return {
        "Total_Facts": int(final_facts),
        "Effective_Rate": round(effective_rate, 2),  # <--- The metric you want
        "Turns_to_Saturation": int(t_sat),           # <--- When the "real" conversation ended
        "Saturation_Index": round(saturation_idx, 2)
    }

def main_analysis():
    from config_loader import load_config
    # 1. Load Configuration
    config = load_config("config.yml")
    save_dir = config.get("save_dir", "results")
    model_name = config["model"]["name"]
    ontology_file = config["ontology"]["file"]
    
    # Construct Paths based on your structure
    # Path: results\gpt-4o\ontology_aliens_10\tests\track_info
    base_output_path = os.path.join(save_dir, model_name, os.path.splitext(os.path.basename(ontology_file))[0], "tests")
    tracking_dir = os.path.join(base_output_path, "track_info")
    
    print(f"[Main] Reading tracking files from: {tracking_dir}")
    
    results = []

    # 2. Iterate through Strategies
    for group_name, strategies in groups.items():
        for strategy in strategies:
            # Skip baselines if they are not relevant to turn-by-turn analysis, 
            # or keep them if you treat sentences as turns (assuming existing logic handles it).
            # For this script, we assume tracking files exist for them.
            
            filename = f"tracking_{strategy}.json"
            file_path = os.path.join(tracking_dir, filename)
            
            if os.path.isfile(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 3. Extract Cumulative Fact Counts
                    # The 'triplets' key contains a list of lists (one list of facts per turn)
                    # We need the length of that list per turn.
                    if 'metrics' in data and 'total_known_facts' in data['metrics']:
                        cumulative_facts = [167 if turn_facts > 167 else turn_facts for turn_facts in data['metrics']['total_known_facts']]
                        
                        # 4. Compute Metrics
                        metrics = compute_refined_dynamics(cumulative_facts)
                        
                        # Add metadata
                        metrics['Strategy'] = strategy
                        metrics['Group'] = group_name
                        results.append(metrics)
                    else:
                        print(f"[Warn] No 'triplets' key found in {filename}")

                except Exception as e:
                    print(f"[Error] Failed processing {filename}: {e}")
            else:
                print(f"[Warn] File not found: {file_path}")

    # 3. Output and Save
    if results:
        df = pd.DataFrame(results)
        
        # Reorder columns for clarity
        cols = ["Group", "Strategy", "Total_Facts", "Effective_Rate", "Turns_to_Saturation", "Saturation_Index"]
        df = df[cols]
        
        # Sort by Effective Rate descending to see the "fastest" first
        df = df.sort_values(by="Effective_Rate", ascending=False)

        print("\n" + "="*80)
        print("RAPIDITY ANALYSIS: DYNAMICS OF EXCHANGE")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Save to CSV
        output_csv = os.path.join(tracking_dir, "rapidity_dynamics_results.csv")
        df.to_csv(output_csv, index=False)
        print(f"\n[Main] Results saved to: {output_csv}")
    else:
        print("[Main] No valid data found.")


if __name__ == "__main__":
    #main_bunch()
    #main_single()
    main_analysis()





