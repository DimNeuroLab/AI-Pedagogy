# ontology/ontology_utils.py

"""
Ontology Utilities

Contains helper functions for loading, saving, and sampling ontologies.
"""

import json
import os
import random
import string
from ontology.ontology_generator import OntologyGenerator



def get_alien_ontology(config):
    """
    Load or generate the ontology based on:
      - If config['ontology']['generate'] is True → always generate via LLM.
      - Else if the file exists at config['ontology']['file'] → load from disk.
      - Else → generate via LLM and save.

    Returns a dict mapping species → feature dicts.
    """
    ont_cfg = config.get('ontology', {})
    path = ont_cfg.get('file')
    force_gen = ont_cfg.get('generate', False)
    prompt_file = config.get('prompt_file')
    model_conf = config.get('model')

    # If the file exists and we’re not forcing regeneration, just load it
    if not force_gen and path and os.path.exists(path):
        return load_ontology(path)

    # Otherwise, generate via the LLM
    gen = OntologyGenerator(prompt_file, model_conf)
    ontology = gen.generate_structured()
    # Save for future runs
    if path:
        save_ontology(ontology, path)
    return ontology

def load_ontology(file_path):
    """
    Load a JSON ontology from disk.

    Returns:
    --------
    dict : The ontology structure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ontology file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_ontology(ontology, file_path):
    """
    Save a JSON ontology to disk.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(ontology, f, indent=2, ensure_ascii=False)

'''def format_ontology(ontology):
    """
    Format the ontology for better readability.

    Returns:
    --------
    str : Formatted ontology string.
    """
    alien_dict = {species['name']: {k: v for k, v in species.items() if k != 'name'}
                    for species in ontology['alien_species']}

    return alien_dict'''

def format_ontology(raw):
    """
    Accepts either:
      1) Old format: { "alien_species": [ { "name": ..., feature: value, … }, … ] }
      2) New flat format: { "SpeciesName": { feature: value, … }, … }

    Returns the unified dict: { "SpeciesName": { feature: value, … }, … }
    """
    # Case 1: the old list-of-species format
    if isinstance(raw, dict) and 'alien_species' in raw:
        return {
            species['name']: {
                k: v for k, v in species.items() if k != 'name'
            }
            for species in raw['alien_species']
        }

    # Case 2: the flat mapping we now generate
    if isinstance(raw, dict) and all(isinstance(v, dict) for v in raw.values()):
        return raw

    # Otherwise, we can’t interpret it
    raise ValueError("Unrecognized ontology format; expected 'alien_species' list or flat dict")

def generate_random_name(existing_names, length=6):
    """
    Generate a random pronounceable name that is not in existing_names.
    """
    vowels = "aeiou"
    consonants = "".join(set(string.ascii_lowercase) - set(vowels))

    while True:
        name = "".join(
            random.choice((consonants, vowels)[i % 2])  # alternate consonant/vowel
            for i in range(length)
        ).capitalize()
        if name not in existing_names:
            return name


def obfuscate_ontology_names(input_path, output_path=None):
    """
    Replace entity names in an ontology with random strings and save new file.

    Parameters:
    - input_path: path to original ontology JSON
    - output_path: where to save new ontology (if None, appends '_obfuscated.json')
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        ontology = json.load(f)

    new_ontology = {}
    used_names = set()

    for entity, features in ontology.items():
        new_name = generate_random_name(used_names)
        used_names.add(new_name)
        new_ontology[new_name] = features

    # Save to new file
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_obfuscated{ext}"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_ontology, f, indent=2)

    print(f"[INFO] Obfuscated ontology saved to {output_path}")
    return output_path

def generate_trial_sets(ontology, n_sets=10, set_size=8):
    """
    Create multiple candidate sets for 20Q trials.

    Returns:
    --------
    List[List[str]] : A list of candidate sets (lists of species names).
    """

    #ontology = __format_ontology(ontology)
    species = list(ontology.keys())
    trial_sets = []
    for _ in range(n_sets):
        if len(species) < set_size:
            chosen = random.choices(species, k=set_size)
        else:
            chosen = random.sample(species, k=set_size)
        trial_sets.append(chosen)
    return trial_sets