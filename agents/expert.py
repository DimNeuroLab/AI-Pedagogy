# agents/expert.py

from agents.base_agent import Agent
import json

class ExpertAgent(Agent):
    """
    Expert agent that knows the onthology
    """

    def __init__(self, model_name, prompt_file_path, ontology):
        """
        Parameters:
        - ontology: dict or text (depends on condition)
        """
        prompt_key_path = ("prompts_20q", "player")
        super().__init__(prompt_key_path, model_name, prompt_file_path)
        self.ontology = ontology
        self.update_history(role = "system", content= f"The ontology you have an expertise on:\n\n{self.ontology}.\n\n")
        self.set_context(self.conversation_history)


    def set_context(self, context):
        """
        Set the context for the learner agent.
        """
        self.context = context

    def reset_context(self):
        self.conversation_history = self.context
    
    def ask_question(self, candidate_set_description=None):
        """
        Ask a feature-based question to identify the secret alien.
        """
        input_msg = "Ask a yes/no question about the alien's features to narrow down the possibilities." 
        if candidate_set_description:
            input_msg += f"\nCurrent candidate aliens:\n{candidate_set_description}"
        input_msg += "You can guess the alien species at any time by saying 'I guess [species name]'.\n\n"
        return self.generate_response(input_msg)
    