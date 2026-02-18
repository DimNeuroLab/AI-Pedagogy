import json
import random
from .ontology_querier import OntologyQuerier
import os
import sys
import re
USE_OLD_ONTOLOGY = True  # Set to False to use the new ontology format
PROMPTS_FILE = "prompts/tester_prompts.json"

class OntologyTester():

    def __init__(self, ontology):
        """
        Initialize the Tester with an ontology.
        Parameters:
        ----------
        ontology : Dict : The alien species ontology.
        """
        self.ontology = ontology
        self.querier = OntologyQuerier(ontology)

        # Load the prompts
        with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        self.prompts = {
            "features": prompts["features_based"]["prompt"],
            "negative_features": prompts["negative_features_based"]["prompt"],
            "objects": prompts["objects_based"]["prompt"],
            "negative_objects": prompts["negative_objects_based"]["prompt"],
            "discriminative" : prompts["discriminative"]["prompt"],
            "rebuild_ontology": prompts["rebuild_ontology"]["prompt"]
        }

    def generate_test_set_aliens(self, n_sets=10, set_size=5):
        """
        Create multiple subsets of aliens species.

        Returns:
        --------
        List[List[str]] : A list of candidate sets (lists of species names).
        """

        #ontology = __format_ontology(ontology)
        species = list(self.ontology.keys())
        trial_sets = []
        for _ in range(n_sets):
            if len(species) < set_size:
                chosen = random.choices(species, k=set_size)
            else:
                chosen = random.sample(species, k=set_size)
            trial_sets.append(chosen)
        return trial_sets

    def extract_feature_values(self):
        """
        Extract unique feature values from the alien species in the ontology.
        Returns:
            Dict[str, List[str]]: A dictionary with feature categories as keys and lists of unique values as values.
        """
        feature_values = {}

        for species, features in self.ontology.items():
            for key, value in features.items():
                norm_key = key.lower()
                if norm_key not in feature_values:
                    feature_values[norm_key] = set()
                feature_values[norm_key].add(value)

        # Convert sets to sorted lists
        feature_values = {k: sorted(list(v)) for k, v in feature_values.items()}

        return feature_values

    @staticmethod
    def generate_random_feature_combinations(feature_values, n_sets, n_features):
        """
        Generate random combinations of alien features.
        Parameters:
        ----------
        feature_values : Dict{str, List[str]} : A dictionary with feature categories as keys and lists of unique values as values.
        n_sets : int : Number of combinations to generate.
        n_features : int : Number of features to include in each combination.
        Returns:
        --------
        List[Dict[str, str]] : A list of dictionaries, each representing a combination of features.
        """

        if n_features > len(feature_values):
            raise ValueError(f"n_features ({n_features}) cannot be greater than the number of available features ({len(feature_values)}).")
        
        all_features = list(feature_values.keys())
        combinations = []

        for _ in range(n_sets):
            chosen_features = random.sample(all_features, n_features)
            combo = {
                feature: random.choice(feature_values[feature])
                for feature in chosen_features
            }
            combinations.append(combo)
        
        return combinations
    
    def run_test(self,  n_sets=10, set_size=5, n_features=3, agent=None, log_path="results/KnowledgeAcquisition/test_results.json"):
        """
        Run a full test suite on the agent using the ontology and predefined prompts.

        Parameters:
            n_sets (int): Number of test sets to generate.
            set_size (int): Number of species per set.
            n_features (int): Number of features per feature combination.
            agent (Agent): The agent to test.

        Returns:
            None
        """
        if agent is None:
            raise ValueError("An agent must be provided to run the test.")
        
        species_subset = self.generate_test_set_aliens(n_sets, set_size)
        feature_values = self.extract_feature_values()
        feature_combinations = self.generate_random_feature_combinations(feature_values, n_sets, n_features)

        results = {
            "features_test": self.run_features_test(agent, feature_combinations),
            "negative_features_test": self.negative_run_features_test(agent, feature_combinations),
            "species_test": self.run_species_test(agent, species_subset),
            "negative_species_test": self.negative_run_species_test(agent, species_subset),
            "discriminative_test": self.run_discriminative_test(agent, species_subset)
                  }
        
        self.results = results

        if log_path:
            with open(log_path, "w") as f:
                json.dump(results, f, indent=2)

        print(f"Results saved to {os.path.abspath(log_path)}")

    def run_ontology_generation_test(self, agent = None, n_tests=5, log_path="results/KnowledgeAcquisition/ontology_generation_test_results.json"):
        """
        Run a test to see if the agent can generate an ontology based on its learned knowledge.
        Parameters:
            n_tests (int): Number of tests to run.
        Returns:
            List[str]: A list of agent responses containing the generated ontology.
        """
        print("[Test] Running ontology generation test...")
        if agent is None:
            raise ValueError("An agent must be provided to run the test.")
        results = {}
        for i in range(n_tests):
            agent.reset_context()
            print(f"[Test] Running test {i+1}")
            agent_answer = self.run_rebuild_ontology_test(agent)
            #print(f"[Test] Agent state: {agent.conversation_history}")
            print(f"[Test] Agent response: {agent_answer}")
            try: 
                results[f"Test {i+1}"] = json.loads(agent_answer)
            except json.JSONDecodeError:
                extracted = extract_json_block(agent_answer)
                if extracted:
                    try:
                        results[f"Test {i+1}"] = json.loads(extracted)
                    except json.JSONDecodeError:
                        print(f"[Test] Failed to parse JSON for test {i+1} (after extraction)")
                else:
                    print(f"[Test] Failed to parse JSON for test {i+1}")

        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[Test] Results saved to {os.path.abspath(log_path)}")

    def run_rebuild_ontology_test(self, agent):
        """
        Run a test to see if the agent can rebuild the ontology based on its learned knowledge.
        Parameters:
            agent (Agent): The agent to test.
        Returns:
            str: The agent's response containing its internal representation of the ontology.
        """
        print("Running rebuild ontology test...")
        agent.reset_context()
        input_msg = self.prompts["rebuild_ontology"]
        agent_answer = agent.generate_response(input_msg)
        return agent_answer

    def run_features_test(self, agent, feature_combinations):
        print("[Test] Running features-based test...")
        results = {}
        for i, feature_set in enumerate(feature_combinations):
            agent.reset_context()

            results[f"Set {i+1}"] = {}
            results[f"Set {i+1}"]["features"] = feature_set
            input_msg = self.prompts["features"].format(features=feature_set)
            agent_answer = agent.generate_response(input_msg)
            querier_answer = self.querier.query_with_features(feature_set)
            results[f"Set {i+1}"]["agent_answer"] = agent_answer
            results[f"Set {i+1}"]["querier_answer"] = querier_answer
        return results
    
    def negative_run_features_test(self, agent, feature_combinations):
        print("Running negative features-based test...")
        results = {}
        for i, feature_set in enumerate(feature_combinations):
            agent.reset_context()
            results[f"Set {i+1}"] = {}
            results[f"Set {i+1}"]["features"] = feature_set
            input_msg = self.prompts["negative_features"].format(features=feature_set)
            agent_answer = agent.generate_response(input_msg)
            querier_answer = self.querier.query_with_features_difference(feature_set)
            results[f"Set {i+1}"]["agent_answer"] = agent_answer
            results[f"Set {i+1}"]["querier_answer"] = querier_answer
        return results

    def run_species_test(self, agent, species_subset):
        print("Running objects-based test...")
        results = {}
        for i, test_set in enumerate(species_subset):
            agent.reset_context()
            results[f"Set {i+1}"] = {}
            results[f"Set {i+1}"]["species"] = test_set
            input_msg = self.prompts["objects"].format(species=test_set)
            agent_answer = agent.generate_response(input_msg)
            querier_answer = self.querier.query_with_species(test_set)
            results[f"Set {i+1}"]["agent_answer"] = agent_answer
            results[f"Set {i+1}"]["querier_answer"] = querier_answer
        return results
        
    def negative_run_species_test(self, agent, species_subset):
        print("Running negative objects-based test...")
        results = {}
        for i, test_set in enumerate(species_subset):
            agent.reset_context()
            results[f"Set {i+1}"] = {}
            results[f"Set {i+1}"]["species"] = test_set
            input_msg = self.prompts["negative_objects"].format(species=test_set)
            agent_answer = agent.generate_response(input_msg)
            querier_answer = self.querier.query_with_species_difference(test_set)
            results[f"Set {i+1}"]["agent_answer"] = agent_answer
            results[f"Set {i+1}"]["querier_answer"] = querier_answer
        return results

    def run_discriminative_test(self, agent, species_subset, n_trials=5):
        print("Running discriminative features test...")
        results = {}
        for i in range(n_trials):
            agent.reset_context()
            A, B = random.sample(species_subset, 2)
            results[f"Trial:{i}"] = {}
            results[f"Trial:{i}"]["set_A"] = A
            results[f"Trial:{i}"]["set_B"] = B
            input_msg = self.prompts["discriminative"].format(set_A=A, set_B=B)
            agent_answer = agent.generate_response(input_msg)
            querier_answer = self.querier.discriminative_query(A, B)
            results[f"Trial:{i}"]["agent_answer"] = agent_answer
            results[f"Trial:{i}"]["querier_answer"] = querier_answer
        return results

    @staticmethod
    def format_output(agent_answer, querier_answer):
        """
        Format the output of the agent and querier for better readability.
        Parameters:
            agent_answer (str): The response from the agent.
            querier_answer (str): The response from the querier.
        Returns:
            str: Formatted output string.
        """
        return f"Agent Response:\n{agent_answer}\n\nQuerier Response:\n{querier_answer}"

    def check_responses(self, file_output, file_input = None):
        """
        Check the responses of the agent against the querier's answers and save the results.
        Parameters:
            file_input (str): Path to the input JSON file containing agent and querier answers.
            file_output (str): Path to the output JSON file to save the comparison results.
        """
        results = {"features_test": {},
                   "negative_features_test": {},
                   "species_test": {},
                   "negative_species_test": {},
                   "discriminative_test": {}
                   }

        if file_input:
            with open(file_input, "r", encoding="utf-8") as f:
                data = json.load(f)
        data = self.results if hasattr(self, 'results') else data
        for key, value in data.items():
            metrics = {"missing": 0, "correct": 0, "extra": 0}
            #if discriminative call llm to compare
            for trial, test in value.items():
                agent_answer = test["agent_answer"]
                querier_answer = test["querier_answer"]
                if key == "discriminative_test":
                    # Compare questions 
                    pass
                #compare lists answers
                else:
                    agent_answer = set(agent_answer)
                    querier_answer = set(querier_answer)
                    metrics["missing"] = len(querier_answer - agent_answer)
                    metrics["extra"] = len(agent_answer - querier_answer)
                    metrics["correct"] = len(agent_answer.intersection(querier_answer))

                results[key][trial] = metrics

        with open(file_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"[Test] Comparison results saved to {os.path.abspath(file_output)}")

def transform_alien_structure_oneliner(alien_data):
    """One-liner version using dictionary comprehension"""
    return {
        alien['name']: {k: v for k, v in alien.items() if k != 'name'}
        for alien in alien_data['alien_species']
    }

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

def extract_json_from_agent_output(agent_output: str) -> dict:
    match = re.search(r'```json(.*?)```', agent_output, re.DOTALL)
    if not match:
        try:
            return json.loads(agent_output)
        except json.JSONDecodeError:
            pass
    else:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON: {e}")
            print(json_str)
    return {}

def extract_json_block(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        return text[start:end+1]
    return None


if __name__ == "__main__":
    # Example usage
    if not USE_OLD_ONTOLOGY:
        with open("data/ontologies/ontology_aliens.json", "r", encoding="utf-8") as f:
            ontology = json.load(f)

    else:
        with open("data/ontology_level1.json", "r", encoding="utf-8") as f:
            ontology_data = json.load(f)
        ontology = transform_alien_structure_oneliner(ontology_data)

    tester = OntologyTester(ontology)
    alien_sets = tester.generate_test_set_aliens(n_sets=6, set_size=3)
    print("Generated Alien Sets:")
    for i, s in enumerate(alien_sets):
        print(f"Set {i+1}: {s}")
    feats = tester.extract_feature_values()
    print("\nExtracted Feature Values:")
    for feature, values in feats.items():
        print(f"{feature}: {values}")
    comb = tester.generate_random_feature_combinations(feats, n_sets=5, n_features=2)
    print("\nGenerated Feature Combinations:")
    for i, c in enumerate(comb):
        print(f"Combination {i+1}: {c}")
    




    

        
