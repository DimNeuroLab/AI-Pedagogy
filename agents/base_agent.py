# agents/base_agent.py

"""
ScaffAliens – Base Agent

This module defines a base Agent class shared by TeacherAgent, LearnerAgent, and OracleAgent.
It handles prompt loading, dialogue history, and API calls.

Author: Sab
Created: 2nd May
"""

from utils.prompts_loader import PromptLoader
from utils.api_interface import call_gpt_api, call_model_api

class Agent:
    """
    Base class for all agents in the framework (Teacher, Learner, Oracle).
    Handles prompt loading, conversation memory, and GPT interaction.
    """

    def __init__(self, prompt_key_path, model_name, prompt_file_path):
        """
        Parameters:
        -----------
        prompt_key_path : tuple of str
            Path to the system prompt in the prompt JSON (e.g., ("Teacher_strategies", "top_down"))
        model_name : str
            Model to use (e.g., "gpt-4o")
        prompt_file_path : str
            Path to the prompts.json file
        """
        self.model_name = model_name
        self.loader = PromptLoader(prompt_file_path)
        self.system_prompt = self.loader.get(*prompt_key_path) # Return a dicronary insted of the corresponding prompt string (learner)

        # Initialise conversation history with system-level prompt
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def reset_conversation(self):
        """Clear the dialogue history, keeping the system prompt."""
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]


    def update_history(self, role, content):
        """Append a new message to the dialogue."""
        self.conversation_history.append({"role": role, "content": content})

    def generate_response(self, user_input, role="user"):
        """
        Send input to the API and return the model's response.

        Parameters:
        -----------
        user_input : str
            The message to send as the next user input.

        Returns:
        --------
        str : The assistant's reply.
        """
        self.update_history(role, user_input)
        response = call_model_api(self.conversation_history, self.model_name)
        #response = call_gpt_api(self.conversation_history, self.model_name)
        self.update_history("assistant", response)
        return response

    def receive_answer(self, answer):
        """
        Receive and store the other agent's response.

        Parameters:
        -----------
        answer : str
            The answer from the other agent.
        """
        self.update_history("assistant", answer)