# agents/learner_agent.py
import copy
from .base_agent import Agent

class LearnerAgent(Agent):
    """
    Learner agent that learns the ontology and later plays the 20Q game.

    Can:
    - Listen to instruction (during training)
    - Ask questions (during test)
    - Answer Teacher’s prompts (in guided mode)
    """

    def __init__(self, model_name, prompt_file_path):
        prompt_key_path = ("learner_training","prompt") # Added dict key -> "prompt" to get the prompt string from the dictionary
        super().__init__(prompt_key_path, model_name, prompt_file_path)
        self.context = None  

    def set_context(self, context):
        """
        Set the context for the learner agent.
        """
        self.context = context

    def reset_context(self):
        """Reset conversation history to saved context."""
        if self.context is not None:
            self.conversation_history = copy.deepcopy(self.context)
        else:
            # If no context set, just reset to initial state
            self.reset_conversation()

    def ask_question(self, candidate_set_description=None):
        """
        Ask a feature-based question to identify the secret alien.
        """
        input_msg = "Ask a yes/no question about the alien's features to narrow down the possibilities." 
        if candidate_set_description:
            input_msg += f"\nCurrent candidate aliens:\n{candidate_set_description}"
        input_msg += "You can guess the alien species at any time by saying 'I guess [species name]'.\n\n"
        return self.generate_response(input_msg)

    def answer_Teacher(self, prompt):
        """
        Respond to a Teacher’s question (used in Teacher-led questioning strategy).
        """
        return self.generate_response(prompt)