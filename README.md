# AIP: Artificial Intelligence Pedagogy

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research-orange)

AIP (Artificial Intelligence Pedagogy) is a research platform for simulating pedagogical interactions between artificial agents using Large Language Models (LLMs). Inspired by Vygotskian theories of social learning, this project explores how structured dialogues between a knowledgeable **Teacher LLM** and a naïve **Learner LLM** can facilitate ontology acquisition in a controlled setting.

This work supports the findings presented in the papers:  
**Patania et al. (2025)** – *AI Pedagogy: Dialogic Social Learning for Artificial Agents*, accepted at ICSR 2025.

**Patania et al. (2026)** – *Teaching LLMs Naturally: Pedagogical Strategies for Interactive Knowledge Acquisition*, accepted at AAMAS 2026.

---

## Project Overview

AIP implements a lightweight “AI Social Gym” where pedagogical strategies are tested through:

- Teacher–Learner dialogues based on different instructional styles (e.g., Top-Down, Bottom-Up, Learner-Driven, Teacher-Guided)
- Ground-truth ontology generation (e.g., artificial alien taxonomies)
- Post-training evaluation using a 20-Questions guessing game

The system uses OpenAI’s GPT-4o via API to simulate both Teacher and Learner agents.

---

## Getting Started

### Quick Start

For rapid setup, see [QUICKSTART.md](QUICKSTART.md) for a streamlined installation guide.

### Installation

For detailed installation instructions, see [SETUP.md](SETUP.md).

**Quick start:**

1. Clone the repository
2. Install Python 3.10+
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### API Key Configuration

Set your OpenAI API key as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

Alternatively, create a `.env` file in the project root.

---

## Usage

### Single Session

To run a **single pedagogical interaction session**:

```bash
python main.py
```

Optional arguments:
- `--config PATH`: Specify a custom config file (default: `config.yml`)
- `--strategy {top_down,bottom_up,learner_questions,teacher_questions,mixed}`: Override the strategy
- `--skip-training`: Skip the training phase
- `--skip-testing`: Skip the testing phase

Example:
```bash
python main.py --strategy top_down --config my_config.yml
```

### Batch Experiments

To run **batch experiments** across strategies:

```bash
python main_batch.py
```

### Expert Testing

To run **expert-only testing** (without training):

```bash
python main_expert.py
```

### Configuration

You can customize the settings in `config.yml`, including:

- Dialogue strategy (`top_down`, `bottom_up`, `learner_questions`, `teacher_questions`, etc.)
- Number of trials
- Temperature and max tokens
- Ontology complexity

---

## Project Structure

```
my_aiped/
├── agents/              # Agent implementations (Teacher, Learner, Oracle, etc.)
├── data/
│   └── ontologies/      # Ontology JSON files
├── ontology/            # Ontology generation and utilities
├── prompts/             # Prompt templates
├── testing/             # Testing utilities and 20-Questions game
├── training/            # Training utilities
├── utils/               # Helper functions and API interfaces
├── logs/                # Log files (excluded from git)
├── results/             # Experiment results (excluded from git)
├── visualizations/      # Generated plots and HTML visualizations (excluded from git)
├── main.py              # Main entry point for single sessions
├── main_batch.py        # Batch experiment runner
├── main_expert.py       # Expert-only testing
├── config.yml           # Configuration file
└── requirements.txt     # Python dependencies
```

---

## Experiment Details

Each run consists of:

1. **Ontology Generation** – A synthetic taxonomy is created with categories like morphology, diet, and habitat.
2. **Pedagogical Dialogue** – Teacher and Learner agents interact for a fixed number of turns.
3. **Evaluation Phase** – The Learner plays a 20-Questions game to test ontology acquisition.

Logs of messages and responses are stored in `messages.log` and `responses.log`.

---

## Testing

Run the test suite with pytest:

```bash
pip install pytest pytest-cov
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

---

## Contributing

This is a research prototype. For contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

Contributions are welcome in the following areas:
- Bug reports and fixes
- Documentation improvements
- Feature suggestions (aligned with research goals)
- Code quality improvements

---

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{patania2025aipedagogy,
  title = {AI Pedagogy: Dialogic Social Learning for Artificial Agents},
  author = {Patania, Sabrina and Annese, Luca and Koyuturk, Cansu and Ruggeri, Azzurra and Ognibene, Dimitri},
  booktitle = {Proceedings of the International Conference on Social Robotics (ICSR)},
  year = {2025}
}

@inproceedings{patania2026teaching,
  title = {Teaching {LLMs} Naturally: Pedagogical Strategies for Interactive Knowledge Acquisition},
  author = {Patania, Sabrina and Annese, Luca and Koyutuerk, Cansu and Ognibene, Dimitri},
  booktitle = {Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)},
  year = {2026},
  publisher = {IFAAMAS},
  address = {Paphos, Cyprus},
  doi = {10.65109/JGLB4831},
  url = {https://doi.org/10.65109/JGLB4831}
}


```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or collaboration proposals, please contact the corresponding authors:

- **Sabrina Patania** – sabrina.patania@unimib.it  
- **Luca Annese** – luca.annese1@unimib.it

Alternatively, you may open an issue on GitHub for technical questions or bug reports.

---

## Acknowledgments

We acknowledge the contributions of all collaborators and the support of the research community.
