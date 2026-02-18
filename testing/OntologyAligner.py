import os
from owlready2 import get_ontology, Thing, DataProperty, default_world, types
import json
from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Callable, Optional
from .functions import get_facts


class OntologyAligner:
    def __init__(
        self,
        ontology_a: Dict[str, Dict[str, str]], #True Ontology
        ontology_b: Dict[str, Dict[str, str]], #Agent Ontology
        sim_fn: Optional[Callable[[str, str], float]] = None):
    
        self.ontology_a = ontology_a
        self.ontology_b = ontology_b
        self.sim_fn = sim_fn or self._default_sim_fn()

    def __str__(self):

        from pprint import pformat

        results = (
            "Results of alignment between two ontologies:\n"
            f"{pformat(self.alignment_results, indent=4)}\n\n"
            "Features Mapping:\n"
            f"{pformat(self.mapping, indent=4)}\n\n"
            f"Mismatches: {self.mismatches}"
            f"\n\nComparison Results:\n"
            f"{pformat(self.comparison_results, indent=4)}"
        )

        return results
    
    def get_results(self) -> Dict[str, Dict[str, str]]:
        """
        Get the alignment results and mapping.
        """
        return {
            "alignment_results": self.alignment_results,
            "mapping": self.mapping,
            "mismatches": self.mismatches,
            "comparison_results": self.comparison_results
        }
    
    def save_results(self, path: str):
        """
        Save the alignment results and mapping to a JSON file.
        """
        results = self.get_results()
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {path}")

    def _default_sim_fn(self):
        return lambda a, b: (
            0 if type(a) is not type(b) 
            else SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()
        )

    def _get_shared_entities(self):
        return set(self.ontology_a).intersection(self.ontology_b)
    
    def _name_similarity(self, a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def align(self, threshold: float = 0.8):
        shared_entities = self._get_shared_entities()
        results = {}

        # Gather features
        features_a = {f for v in self.ontology_a.values() for f in v}
        features_b = {f for v in self.ontology_b.values() for f in v}

        for feature_a in features_a:
            best_match = None
            best_score = 0
            best_details = None

            for feature_b in features_b:
                sims = []
                skipped = 0
                for entity in shared_entities:
                    val_a = self.ontology_a[entity].get(feature_a)
                    val_b = self.ontology_b[entity].get(feature_b)
                    if val_a and val_b:
                        sims.append(self.sim_fn(val_a, val_b))
                    else:
                        skipped += 1

                if sims:
                    avg_sim = sum(sims) / len(sims)
                else:
                    avg_sim = 0.0
                
                # Calculate name similarity
                name_sim = self.sim_fn(feature_a, feature_b)

                # Blend scores (you can adjust weights here if you want)
                blended_score = 0.5 * avg_sim + 0.5 * name_sim

                # Determine equivalence: either blended score above threshold OR name similarity > 0.9
                is_equivalent = blended_score >= threshold or name_sim > 0.9 or avg_sim >= 0.9

                if blended_score > best_score:
                    best_score = blended_score
                    best_match = feature_b
                    best_details = {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "average_similarity": avg_sim,
                        "name_similarity": name_sim,
                        "blended_score": blended_score,
                        "is_equivalent": is_equivalent,
                        "num_comparisons": len(sims),
                        "skipped_comparisons": skipped
                    }

            results[feature_a] = best_details or {
                "feature_a": feature_a,
                "feature_b": None,
                "average_similarity": 0.0,
                "name_similarity": 0.0,
                "blended_score": 0.0,
                "is_equivalent": False,
                "num_comparisons": 0,
                "skipped_comparisons": 0
            }

        count = 0
        for k, v in results.items():
            if k != v["feature_b"]:
                count += 1

        self.mismatches = count

        # Define mapping for features marked equivalent
        self.mapping = {
            details["feature_b"]: feature_a
            for feature_a, details in results.items()
            if details["is_equivalent"] and details["feature_b"] is not None
        }

        self.alignment_results = results
        return results

    @classmethod
    def from_files(cls, file_a: Path, file_b: Path, key="rebuild_ontology", **kwargs):
        with open(file_a) as f1, open(file_b) as f2:
            ont_a = json.load(f1)[key]
            ont_b = json.load(f2)[key]
        return cls(ont_a, ont_b, **kwargs)

    def json_to_owl(self, ontology_data, filename, ontology_iri="http://example.org/aliens.owl"):
        """
        Convert a dictionary ontology to an OWL file and save it under testing/OWL/.

        Args:
            ontology_data (dict): Ontology in {species: {feature: value}} format.
            filename (str): Desired OWL filename (e.g., 'true_ontology.owl').
            ontology_iri (str): Base IRI for the ontology.
        """

        # Ensure wrapper is removed
        if isinstance(ontology_data, dict) and "rebuild_ontology" in ontology_data:
            ontology_data = ontology_data["rebuild_ontology"]

        # Determine output path based on current script location
        base_dir = os.path.dirname(__file__)  # testing/
        owl_dir = os.path.join(base_dir, "OWL")
        os.makedirs(owl_dir, exist_ok=True)   # Create OWL folder if missing
        output_path = os.path.join(owl_dir, filename)

        # Create ontology
        onto = get_ontology(ontology_iri)
        with onto:
            class AlienSpecies(Thing): pass

            feature_keys = set()
            for species, features in ontology_data.items():
                feature_keys.update(features.keys())

            data_properties = {}
            for key in feature_keys:
                safe_key = key.replace(" ", "_")
                prop = types.new_class(safe_key, (DataProperty,))
                prop.domain = [AlienSpecies]
                prop.range = [str]
                data_properties[key] = prop

            for species, features in ontology_data.items():
                indiv = AlienSpecies(species.replace(" ", "_"))
                for k, v in features.items():
                    if k in data_properties:
                        setattr(indiv, data_properties[k].name, [v])

        onto.save(file=output_path, format="rdfxml")
        print(f"Saved OWL ontology to {output_path}")
        return output_path

    def compare_ontologies(self):
        """
        Compare two ontologies in dict form:
        self.ontology_a, self.ontology_b: {species: {feature: value}}
        Returns a dict summarizing differences.
        """

        ont_a = self.ontology_a  # assumed ground truth / gold standard
        ont_b = self.ontology_b  # assumed to be generated / inferred

        # Ensure mapping is available
        if not hasattr(self, 'mapping'):
            self.align()

        species_a = set(ont_a.keys())
        species_b = set(ont_b.keys())

        only_in_a = species_a - species_b
        only_in_b = species_b - species_a
        in_both = species_a & species_b

        feature_differences = {}

        for sp in in_both:
            features_a = ont_a[sp]
            features_b_raw = ont_b[sp]

            # Use mapping to align feature names from B to A's format
            features_b = {}
            for f_b, val in features_b_raw.items():
                f_a = self.mapping.get(f_b)
                if f_a:  # Only map if alignment exists
                    features_b[f_a] = val

            all_features = set(features_a.keys()) | set(features_b.keys())
            diffs = {}

            for f in all_features:
                val_a = features_a.get(f)
                val_b = features_b.get(f)

                # Normalize unknowns
                if val_a is not None and val_a.lower() == "unknown":
                    val_a = None
                if isinstance(val_b, str) and val_b.strip().lower() == "unknown":
                    val_b = None

                if val_a != val_b:
                    diffs[f] = {
                        'true_ontology': val_a,
                        'agent_ontology': val_b
                    }

            if diffs:
                feature_differences[sp] = diffs

        result = {
            "only_in_true_ontology": len(only_in_a),
            "only_in_agent_ontology": len(only_in_b),
            "species_with_feature_differences": feature_differences,
            "num_species_in_true_ontology": len(species_a),
            "num_species_in_agent_ontology": len(species_b),
            "num_common_species": len(in_both),
            "num_species_with_diff_features": len(feature_differences)
        }

        self.comparison_results = result
        return result

    def run_comparison(self):

        # All fields are sets
        data_true = get_facts(self.ontology_a)
        data_agent = get_facts(self.ontology_b)

        data_true = {k: set(v) for k, v in data_true.items()}
        data_agent = {k: set(v) for k, v in data_agent.items()}


        results = {}

        e_miss = len(data_true["entities"] - data_agent["entities"]) # Set difference → missing entities
        e_correct = len(data_true["entities"] & data_agent["entities"]) # Set intersection → correct entities
        e_agent_only = len(data_agent["entities"] - data_true["entities"]) # Set difference → extra entities

        f_miss = len(data_true["features"] - data_agent["features"])
        f_correct = len(data_true["features"] & data_agent["features"])
        f_agent_only = len(data_agent["features"] - data_true["features"])

        v_miss = len(data_true["values"] - data_agent["values"])
        v_correct = len(data_true["values"] & data_agent["values"])
        v_agent_only = len(data_agent["values"] - data_true["values"])

        fv_miss = len(data_true["features_values"] - data_agent["features_values"])
        fv_correct = len(data_true["features_values"] & data_agent["features_values"])
        fv_agent_only = len(data_agent["features_values"] - data_true["features_values"])


        ef_miss = len(data_true["entities_features"] - data_agent["entities_features"])
        ef_correct = len(data_true["entities_features"] & data_agent["entities_features"])
        ef_agent_only = len(data_agent["entities_features"] - data_true["entities_features"])

        t_miss = len(data_true["triplets"] - data_agent["triplets"])
        t_correct = len(data_true["triplets"] & data_agent["triplets"])
        t_agent_only = len(data_agent["triplets"] - data_true["triplets"])

        results["entity"] = {
            "missing": e_miss,
            "correct": e_correct,
            "extra": e_agent_only
        }
        results["feature"] = {
            "missing": f_miss,
            "correct": f_correct,
            "extra": f_agent_only
        }
        results["value"] = {
            "missing": v_miss,
            "correct": v_correct,
            "extra": v_agent_only
        }
        results["feature_value"] = {
            "missing": fv_miss,
            "correct": fv_correct,
            "extra": fv_agent_only
        }
        results["entity_feature"] = {
            "missing": ef_miss,
            "correct": ef_correct,
            "extra": ef_agent_only
        }
        results["triplet"] = {
            "missing": t_miss,
            "correct": t_correct,
            "extra": t_agent_only
        }

        results["ground_truth"] = {
            "entities": len(data_true["entities"]),
            "features": len(data_true["features"]),
            "values": len(data_true["values"]),
            "feature_values": len(data_true["features_values"]),
            "entity_features": len(data_true["entities_features"]),
            "triplets": len(data_true["triplets"])
        }
        self.results = results
        return results

    def save_plot(self, path: str):
        import matplotlib.pyplot as plt

        if not hasattr(self, 'results'):
            raise ValueError("No alignment results found. Please run align() first.")

        features = list(self.results.keys())
        scores = [details['blended_score'] for details in self.results.values()]

        plt.figure(figsize=(10, 6))
        plt.barh(features, scores, color='skyblue')
        plt.xlabel('Blended Similarity Score')
        plt.title('Feature Alignment Scores')
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        print(f"Plot saved to {path}")


if __name__ == "__main__":
    # Example usage
    from ontology_tester import transform_alien_structure_oneliner
    from pprint import pprint

    with open("results/KnowledgeAcquisition/test_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        ont_b = data["rebuild_ontology"]

    with open("data/ontology_level1.json", "r", encoding="utf-8") as f:
        ontology_data = json.load(f)
        ont_a = transform_alien_structure_oneliner(ontology_data)

    aligner = OntologyAligner(ont_a, ont_b)
    
    matches = aligner.compare_ontologies()

    print(aligner)
   
   