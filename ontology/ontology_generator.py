# ontology/ontology_generator.py

"""
Ontology Generator Module

Provides functionality to:
1. Generate a structured JSON ontology of alien species via an LLM.
2. Convert that structured ontology into a narrative taxonomic description.
3. Reconstruct a structured ontology from the narrative (introducing realistic noise).

Requires prompts.json to include keys:
- "ontology_generation"
- "ontology_to_narrative"
- "ontology_reconstruct"
"""

import json, re, os
from utils.prompts_loader import PromptLoader
from utils.api_interface import call_model_api

def clean_response(response):
    # Remove triple backtick blocks like ```json ... ```
    return re.sub(r"^```(?:json)?\n|```$", "", response.strip(), flags=re.IGNORECASE | re.MULTILINE)

class OntologyGenerator:
    def __init__(self, prompt_file_path, model_config):
        """
        Initialise with prompt loader and model settings.

        prompt_file_path : str
            Path to prompts.json
        model_config : dict
            e.g. {"provider":"openai","name":"gpt-4o","temperature":0.3,"max_tokens":300}
        """
        self.loader = PromptLoader(prompt_file_path)
        self.model_config = model_config

    def generate_structured(self):
        """
        Generate a formal JSON ontology via the LLM.

        Returns:
        --------
        dict : Ontology mapping species to feature dicts
        """
        prompt = self.loader.get("ontology_generation", "prompt")
        
        messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "system",   "content": prompt}
            ]
        if self.model_config["provider"] == "openai":    
            response = call_model_api(messages, self.model_config, use_chat_format=True)
        else:
            response = call_model_api(messages, self.model_config, use_chat_format=False)
        return json.loads(clean_response(response))

    def generate_narrative(self, ontology):
        """
        Convert a structured ontology into a narrative description.

        ontology : dict
            The structured JSON ontology.

        Returns:
        --------
        str : Narrative text
        """
        template = self.loader.get("ontology_to_narrative", "prompt")
        user_input = f"{template}\n\n```json\n{json.dumps(ontology, indent=2)}\n```"
        messages = [
            {"role": "system", "content": "You are ax expert writing a taxonomy."},
            {"role": "system",   "content": user_input}
        ]
        return call_model_api(messages, self.model_config, use_chat_format=True)

    def reconstruct_structured(self, narrative):
        """
        Reconstruct a structured ontology from the narrative.

        narrative : str
            The taxonomic narrative describing the species.

        Returns:
        --------
        dict : Reconstructed ontology
        """
        template = self.loader.get("ontology_reconstruct", "prompt")
        user_input = f"{template}\n\n\"\"\"\n{narrative}\n\"\"\""
        messages = [
            {"role": "system", "content": "You are an assistant that infers structure."},
            {"role": "system",   "content": user_input}
        ]
        response = call_model_api(messages, self.model_config, use_chat_format=True)
        return json.loads(response)

    def generate_multiple_ontologies(self, topics, prompt_loader):
        """
        Generate fictional ontologies across multiple topics.

        Parameters:
        - topics: list of strings (topic names)
        - prompt_loader: instance of PromptLoader to load the template
        """
        for topic in topics:
            print(f"Generating ontology for topic: {topic}")

            # Load and customise the prompt
            base_prompt = prompt_loader.get("ontology_generation_general")
            customised_prompt = base_prompt["prompt"].replace("<topic>", topic)

            messages = [
                {"role": "system", "content": "You are a helpful and creative assistant."},
                {"role": "system", "content": customised_prompt}
            ]

            raw_response = call_model_api(messages, self.model_config)
            cleaned_response = raw_response.strip("` \n")
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[len("json"):].strip()
            try:
                ontology = json.loads(cleaned_response)
                filename = f"ontology_{topic.replace(' ', '_')}.json"
                path = os.path.join("data", "ontologies", filename)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(ontology, f, indent=2)
                print(f"Saved ontology to {path}")
            except json.JSONDecodeError:
                print(f"[ERROR] Could not parse ontology for topic '{topic}'\nRaw output:\n{response}")