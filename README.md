
# AIP: Artificial Intelligence Pedagogy

AIP (Artificial Intelligence Pedagogy) is a research platform for simulating pedagogical interactions between artificial agents using Large Language Models (LLMs). Inspired by Vygotskian theories of social learning, this project explores how structured dialogues between a knowledgeable **Teacher LLM** and a naïve **Learner LLM** can facilitate ontology acquisition in a controlled setting.

This work supports the findings presented in the paper:  
**Patania et al. (2025)** – *AI Pedagogy: Dialogic Social Learning for Artificial Agents*, accepted at ICSR 2025.

---

## 🧠 Project Overview

AIP implements a lightweight “AI Social Gym” where pedagogical strategies are tested through:

- Teacher–Learner dialogues based on different instructional styles (e.g., Top-Down, Bottom-Up, Learner-Driven, Teacher-Guided)
- Ground-truth ontology generation (e.g., artificial alien taxonomies)
- Post-training evaluation using a 20-Questions guessing game

The system uses OpenAI’s GPT-4o via API to simulate both Teacher and Learner agents.

---

## 🚀 Getting Started

### 🔧 Installation

Make sure you have Python 3.10+ installed. Then install dependencies:

```bash
pip install -r requirements.txt
```

### 🔑 Setup API Keys

Set your OpenAI API key as an environment variable before running:

```bash
export OPENAI_API_KEY=your-api-key-here
```

Alternatively, you can modify `config.yml` to include your key securely (not recommended for public repos).

---

## 🛠️ Usage

To run a **single pedagogical interaction session**, use:

```bash
python main.py
```

To run **batch experiments** across strategies:

```bash
python main_batch.py
```

You can customise the settings in `config.yml`, including:

- Dialogue strategy (`top_down`, `bottom_up`, `learner_questions`, `teacher_questions`, etc.)
- Number of trials
- Temperature and max tokens
- Ontology complexity

---

## 🧪 Experiment Details

Each run consists of:

1. **Ontology Generation** – A synthetic taxonomy is created with categories like morphology, diet, and habitat.
2. **Pedagogical Dialogue** – Teacher and Learner agents interact for a fixed number of turns.
3. **Evaluation Phase** – The Learner plays a 20-Questions game to test ontology acquisition.

Logs of messages and responses are stored in `messages.log` and `responses.log`.

---

## 📄 Reference

If you use this code in your research, please cite:

```bibtex
@inproceedings{patania2025aipedagogy,
  title = {AI Pedagogy: Dialogic Social Learning for Artificial Agents},
  author = {Patania, Sabrina and Annese, Luca and Koyuturk, Cansu and Ruggeri, Azzurra and Ognibene, Dimitri},
  booktitle = {Proceedings of the International Conference on Social Robotics (ICSR)},
  year = {2025}
}
```

---

## 🤝 Contributing

This is a research prototype and not currently open for contributions. However, feel free to fork and experiment!

---

## 📜 License

This project is for research purposes only. For usage beyond academic work, please contact the authors.

---

## 📬 Contact

For questions or collaboration proposals, please contact the corresponding authors:

- **Sabrina Patania** – sabrina.patania@unimib.it  
- **Luca Annese** – luca.annese1@unimib.it
