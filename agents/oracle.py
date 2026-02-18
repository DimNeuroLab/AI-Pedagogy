# agents/oracle_agent.py

from .base_agent import Agent
import json

class OracleAgent(Agent):
    """
    Oracle agent used only during testing. It has access to the full ontology
    and the correct target alien. It answers yes/no questions truthfully.
    """

    def __init__(self, model_name, prompt_file_path, ontology):
        prompt_key_path = ("prompts_20q", "oracle")
        super().__init__(prompt_key_path, model_name, prompt_file_path)
        self.ontology = ontology

    def set_target(self, target_name):
        """
        Set the current correct alien for this trial.
        """
        self.target = target_name

    def answer_question(self, question):
        """
        Answer a question about the target alien using the ontology.
        """

        features = self.ontology[self.target]
        #features = next((species for species in self.ontology['alien_species'] if species['name'] == self.target), None)
        feature_context = "\n".join([f"{k}: {v}" for k, v in features.items()])
        user_input = (
            f"The target alien is {self.target}.\n"
            f"Features:\n{feature_context}\n\n"
            f"Question: {question}\nRespond only with 'Yes' or 'No'. Answer 'Correct' or 'No' if the question is a guess.\n"
        )
        return self.generate_response(user_input)