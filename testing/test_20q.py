#!/usr/bin/env python3
"""
testing/test_20q.py

Test phase for ScaffAliens: the trained LearnerAgent plays 20 Questions
against an OracleAgent. We record number of questions to success and accuracy.

Author: Sab & Luca
Date: 2nd May
"""

import os
import json
from random import choice
from utils.config_loader import load_config
from ontology.ontology_utils import get_alien_ontology, generate_trial_sets, format_ontology
from agents.learner_agent import LearnerAgent
from agents.oracle import OracleAgent
from agents.expert import ExpertAgent


class Tester:
    def __init__(self, config_path="../config.yml", learner=None, expert_on = False):
        # 1) Load config
        self.config = load_config(config_path)

        # 2) Load ontology
        ontology_file = self.config["ontology"]["file"]
        with open(ontology_file, "r", encoding="utf-8") as f:
            self.ontology = format_ontology(json.load(f))

        # 3) Prepare save directory
        self.save_dir = self.config.get("save_dir", "results")
        os.makedirs(self.save_dir, exist_ok=True)

        # 4) Load model settings
        model_conf = self.config["model"]
        prompt_file = self.config["prompt_file"]

        # 5) Instantiate agents
        self.oracle = OracleAgent(
            model_name=model_conf["name"],
            prompt_file_path=prompt_file,
            ontology=self.ontology
        )
        self.learner = LearnerAgent(
            model_name=model_conf["name"],
            prompt_file_path=prompt_file
        ) if learner is None else learner

        self.expert = ExpertAgent(
            model_name=model_conf["name"],
            prompt_file_path=prompt_file,
            ontology=self.ontology
        )

        # 6) Test parameters
        test_conf = self.config["testing"]
        self.num_sets = test_conf["num_sets"]
        self.set_size = test_conf["set_size"]
        self.max_questions = test_conf["max_questions"]

        # 7) Generate candidate sets
        self.trial_sets = generate_trial_sets(
            self.ontology,
            n_sets=self.num_sets,
            set_size=self.set_size
        )

        if expert_on:
            self.learner = self.expert

    def run_tests(self):
        logs = []
        for set_idx, candidate_set in enumerate(self.trial_sets, start=1):
            print(f"Candidate set: {candidate_set}")
            # Randomly select a target from the candidate set
            target = choice(candidate_set)
            print(f"Target: {target}")
            # Reset conversations
            self.oracle.reset_conversation()
            # Reset learner context if available, otherwise reset conversation
            if hasattr(self.learner, 'reset_context') and self.learner.context is not None:
                self.learner.reset_context()
            else:
                self.learner.reset_conversation()
            # Tell oracle who the target is
            self.oracle.set_target(target)
            qa_log = []
            # 1) Learner asks up to max_questions
            for q_num in range(1, self.max_questions + 1):
                # Learner formulates a question, providing the candidate list
                cand_desc = ", ".join(candidate_set)
                question = self.learner.ask_question(candidate_set_description=cand_desc)
                print(f"Question: {q_num}: {question}")
                # Oracle answers
                answer = self.oracle.answer_question(question)
                print(f"Answer: {answer}")
                qa_log.append({"turn": q_num, "question": question, "answer": answer})
                # If you wish, you can let the learner update internal state here
                self.learner.update_history("user", answer)
                # (Optional) break early if learner explicitly guesses in a question
                if "correct" in answer.lower():
                    print(f"Learner guessed correctly in {q_num} questions!")
                    break
                
            # 2) After questions, ask for final guess
            guess_prompt = (
                "Based on your questions and answers, please guess the secret alien species now. "
                "Respond with the species name only."
            )
            final_guess = self.learner.generate_response(guess_prompt).strip()
            # 3) Evaluate
            print(f"Final guess: {final_guess}")
            correct = (final_guess.lower().replace('.', '') == target.lower())
            questions_used = len(qa_log) #NB: when the learner guesses, should we consider it as a question? If not, leave it as is
            # 4) Record the trial log
            logs.append({
                "set_index": set_idx,
                "candidate_set": candidate_set,
                "target": target,
                "qa_log": qa_log,
                "final_guess": final_guess,
                "correct": correct,
                "questions_used": questions_used
            })
            print(f"[Test] Set {set_idx}/{self.num_sets}, Target '{target}' → "
                  f"{'✔' if correct else '✘'} in {questions_used} questions")
        # Save results with proper structured path
        ontology_name = os.path.splitext(os.path.basename(self.config["ontology"]["file"]))[0]
        strategy = self.config["Teacher"]["strategy"]
        model_name = self.config["model"]["name"]
        
        out_path = os.path.join(
            self.save_dir,
            model_name,
            ontology_name,
            "tests",
            "20q_game",
            f"{strategy}_test.json"
        )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)

        print(f"[Test] All test logs saved to {out_path}")
        return logs
    

    

if __name__ == "__main__":
    tester = Tester("../config.yml")
    tester.run_tests()