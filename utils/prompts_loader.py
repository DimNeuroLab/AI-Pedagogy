# utils/prompts_loader.py

"""
ScaffAliens Prompt Loader

This module provides a utility function to load all system/user prompts used
in the ScaffAliens framework, from a central JSON file.

Author: Sab & Luca
Created: 2nd May
"""

import json
import os

class PromptLoader:
    """
    A simple class to load and access experiment prompts from a JSON file.
    """

    def __init__(self, prompt_file_path):
        """
        Load the prompt file at initialisation.

        Parameters:
        -----------
        prompt_file_path : str
            Path to the JSON file containing prompt templates.
        """
        self.prompt_file_path = prompt_file_path
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        """
        Internal method to load the prompts JSON.

        Returns:
        --------
        dict : Loaded prompt structure as dictionary.
        """
        if not os.path.exists(self.prompt_file_path):
            raise FileNotFoundError(f"Prompt file not found at {self.prompt_file_path}")

        with open(self.prompt_file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def get(self, *keys):
        """
        Access nested prompts dynamically.

        Example:
        --------
        loader.get("Teacher_strategies", "bottom_up")

        Parameters:
        -----------
        keys : str (variadic)
            A sequence of nested keys to traverse in the JSON prompt file.

        Returns:
        --------
        str : The desired prompt.
        """
        current = self.prompts
        for key in keys:
            if key not in current:
                raise KeyError(f"Prompt key '{key}' not found.")
            current = current[key]
        return current