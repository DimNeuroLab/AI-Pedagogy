#!/usr/bin/env python3
"""
training/trainer.py

Implements the Training phase for ScaffAliens, in which:
- A TeacherAgent and a LearnerAgent engage in multi‐turn dialogues about an alien ontology.
- There are four pedagogical strategies:
    1. top_down: Teacher explains features in a top‐down, feature‐by‐feature manner.
    2. bottom_up: Teacher explains the ontology by example, species‐by‐species.
    3. learner_questions: Learner asks questions; Teacher answers.
    4. Teacher_questions: Teacher asks Socratic questions; Learner answers.
- Each training session (trial) consists of a fixed number of back‐and‐forth turns.
- All dialogues are logged for subsequent analysis.

Author: Sab
Date: 2nd May
"""

import os
import json
import math


from utils.config_loader import load_config
from agents.teacher_agent import TeacherAgent
from agents.learner_agent import LearnerAgent
from utils.prompts_loader import PromptLoader

class Trainer:
    """
    Encapsulates the training logic for multiple trials and multi‐turn sessions.
    """

    def __init__(self, config_path="../config.yml", save_logs = False):
        # 1) Load the YAML configuration
        self.config = load_config(config_path)
        self.save_logs = save_logs

        # 2) Load the ontology from the config‐specified file
        ontology_path = self.config["ontology"]["file"]
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found: {ontology_path}")
        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        # 3) Extract model settings
        model_conf = self.config["model"]
        model_name = model_conf["name"]
        prompt_file = self.config["prompt_file"]

        # 4) Prepare results directory
        self.save_dir = self.config.get("save_dir", "results")
        ontology_name = os.path.splitext(os.path.basename(ontology_path))[0]
        self.out_path = os.path.join("results", model_name, ontology_name, "dialogue")
        os.makedirs(self.out_path, exist_ok=True)

        # 5) Instantiate Teacher and Learner agents
        Teacher_strategy = self.config["Teacher"]["strategy"]
        self.teacher = TeacherAgent(
            strategy_name=Teacher_strategy,
            model_name=model_name,
            prompt_file_path=prompt_file,
            ontology=self.ontology
        )
        self.learner = LearnerAgent(
            model_name=model_name,
            prompt_file_path=prompt_file
        )

        # 6) Training parameters
        self.num_trials = self.config["training"]["num_trials"]
        self.turns_per_session = self.config["training"]["turns_per_session"]

        
    def _chunk_by_features(self):
        """
        For top‐down: split the ontology’s feature names into chunks,
        one feature per turn (or distribute evenly if fewer features).
        """
        # Assume all species share the same feature keys
        print(self.ontology.values())
        feature_names = list(next(iter(self.ontology.values())).keys())
        # If fewer features than turns, repeat or group
        if len(feature_names) <= self.turns_per_session:
            return [[feat] for feat in feature_names] + [[]] * (self.turns_per_session - len(feature_names))
        # Otherwise, distribute evenly
        chunk_size = math.ceil(len(feature_names) / self.turns_per_session)
        return [feature_names[i:i+chunk_size]
                for i in range(0, len(feature_names), chunk_size)]

    def _chunk_by_species(self):
        """
        For bottom‐up: split the list of species names into chunks per turn.
        """
        species = list(self.ontology.keys())
        chunk_size = math.ceil(len(species) / self.turns_per_session)
        return [species[i:i+chunk_size]
                for i in range(0, len(species), chunk_size)]

    def run_training(self):
        """
        Execute all training trials. Each trial is a multi‐turn session
        according to the selected pedagogical strategy.
        """
        logs = []

        for trial in range(1, self.num_trials + 1):
            # Reset conversation contexts for both agents
            self.teacher.reset_conversation()
            self.learner.reset_conversation()
            conversation = []
            q = None
            a = None
            # Prepare chunks if needed
            strategy = self.config["Teacher"]["strategy"]
            instr = None
            if strategy in ("top_down", "bottom_up"):
                # One-shot teaching: no turns
                teacher_resp = self.teacher.teach_ontology(instruction=instr)
                learner_resp = self.learner.answer_Teacher(teacher_resp + "\nPlease briefly summarise what you understood.")
                conversation.append("Teacher: " + teacher_resp)
                conversation.append("Learner: " + learner_resp)

            else:
                # Multi-turn teaching: number of turns in config.yml
                for turn in range(self.turns_per_session):

                    if strategy in ("learner_questions", "mixed_learner_questions", "mixed_top-down_learner_questions", "mixed_bottom-up_learner_questions"):
                        a = self.teacher.respond_to_learner(q)
                        q = self.learner.answer_Teacher(a) 
                        teacher_resp, learner_resp = a, q
                    elif strategy in ("mixed_top-down_learner_questions_new","mixed_bottom-up_learner_questions","mixed_bottom-up_teacher_questions","mixed_top-down_teacher_questions","mixed_learner_questions","teacher_questions", "mixed_teacher_questions", "mixed_top-down_teacher_questions", "mixed_bottom-up_teacher_questions"):
                        q = self.teacher.ask_learner_question(context_summary=a)
                        a = self.learner.answer_Teacher(q)
                        teacher_resp, learner_resp = q, a

                    conversation.append("Teacher: " + teacher_resp)
                    conversation.append("Learner: " + learner_resp)
                    print(f"[Training] Turn {turn+1}/{self.turns_per_session} completed ({strategy}).")
            
            # After session, record logs
            if trial == self.num_trials:
                # SUMMARY
                learner_resp = self.learner.answer_Teacher("We've done. Please briefly summarise what you understood.")
                conversation.append("Please briefly summarise what you understood.")
                conversation.append(learner_resp)
            logs.append({
                "trial": trial,
                "strategy": strategy,
                "turns": self.turns_per_session,
                "conversation": conversation
            })
        # Save logs to disk
        if self.save_logs:
            out_path = os.path.join(self.out_path, f"{strategy}_train.json")
            with open(out_path, "w", encoding="utf-8") as outfile:
                json.dump(logs, outfile, indent=2, ensure_ascii=False)

            print(f"[Training] All logs saved to {out_path}")
        self.learner.set_context(self.learner.conversation_history)

        return logs, self.learner

if __name__ == "__main__":
    trainer = Trainer("../config.yml", save_logs=True)
    trainer.run_training()
