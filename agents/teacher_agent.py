# agents/teacher_agent.py

from .base_agent import Agent
import json

class TeacherAgent(Agent):
    """
    Teacher agent that teaches the ontology using one of four pedagogical strategies.
    """

    def __init__(self, strategy_name, model_name, prompt_file_path, ontology):
        """
        Parameters:
        - strategy_name: one of "top_down", "bottom_up", "learner_questions", "Teacher_questions"
        - ontology: dict or text (depends on condition)
        """
        prompt_key_path = ("Teacher_strategies", strategy_name)
        super().__init__(prompt_key_path, model_name, prompt_file_path)
        self.strategy = strategy_name
        self.ontology = ontology
        ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology
        self.system_prompt+="Alien Ontology"+ontology_input

    def teach_ontology(self, instruction=None):
        """
        Used in top_down and bottom_up strategies to present the ontology.
        instruction: optional textual hint for the model
        """
        ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology
        input_msg = f"Here is the ontology:\n\n{ontology_input}"
        if instruction:
            input_msg = instruction + "\n\n" + input_msg
        return self.generate_response(input_msg)

    def respond_to_learner(self, question = None):
        """
        Used in learner_questions strategy: learner asks, Teacher answers.
        """
        #ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology

        #input_msg = f"The ontology you have an expertise on:\n\n{ontology_input}"
        if question:
            input_msg = question
        else:
            input_msg="start teaching"
        return self.generate_response(input_msg)

    def ask_learner_question(self, context_summary=None):
        """
        Used in Teacher_questions strategy: Teacher guides learner by asking.
        """

        if context_summary:
            input_msg = context_summary #f"\nThe learner seems to know: {context_summary}"
        else:
            input_msg = "start teaching"
#            ontology_input = json.dumps(self.ontology, indent=2) if isinstance(self.ontology, dict) else self.ontology
#            input_msg = f"The ontology you have an expertise on:\n\n{ontology_input}. Ask a question to guide the learner about the ontology."
        return self.generate_response(input_msg)