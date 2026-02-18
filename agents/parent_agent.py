# agents/parent_agent.py

from .base_agent import Agent
import json
from utils.prompts_loader import PromptLoader

class ParentAgent(Agent):
    """
    Parent agent that teaches the ontology using one of four pedagogical strategies.
    """

    def __init__(self, strategy_name, model_name, prompt_file_path, ontology):
        """
        Parameters:
        - strategy_name: one of "top_down", "bottom_up", "child_questions", "parent_questions"
        - ontology: dict or text (depends on condition)
        """
        prompt_key_path = ("parent_strategies", strategy_name)
        super().__init__(prompt_key_path, model_name, prompt_file_path)
        self.strategy = strategy_name
        self.ontology = ontology
        loader = PromptLoader(prompt_file_path)
        self.instruction = loader.get()["parent_strategies"][strategy_name]

    def teach_ontology(self, instruction=None):
        """
        Used in top_down and bottom_up strategies to present the ontology.
        instruction: optional textual hint for the model
        """
        ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology
        input_msg = f"Here is the ontology:\n\n{ontology_input}"
        if instruction:
            input_msg = instruction + "\n\n" + input_msg
        return self.generate_response(input_msg, role="system")

    def respond_to_child(self, question = None):
        """
        Used in child_questions strategy: child asks, parent answers.
        """
        ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology

        input_msg = f"The ontology you have an expertise on:\n\n{ontology_input}"
        if question:
            input_msg = question
        return self.generate_response(input_msg)

    def ask_child_question(self, context_summary=None):
        """
        Used in parent_questions strategy: parent guides child by asking.
        """
        ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology
        input_msg = f"The ontology you have an expertise on:\n\n{ontology_input}.\n\n {self.instruction} \n\n"
        if context_summary:
            input_msg += f"\nThe child seems to know: {context_summary}"
        return self.generate_response(input_msg)