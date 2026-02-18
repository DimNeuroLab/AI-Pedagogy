# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Clone and Setup (2 minutes)

```bash
# Clone the repository
git clone <your-repo-url>
cd my_aiped

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure (1 minute)

```bash
# Copy example config
cp config.example.yml config.yml

# Edit config.yml and add your OpenAI API key in environment
# Windows PowerShell:
$env:OPENAI_API_KEY="your-key-here"
# macOS/Linux:
export OPENAI_API_KEY="your-key-here"
```

### 3. Run Your First Experiment (2 minutes)

```bash
# Run a single session with default settings
python main.py

# Or try a specific strategy
python main.py --strategy bottom_up
```

That's it! Your first pedagogical interaction is running.

## 📊 Understanding the Output

After running, you'll find:
- **Training logs** in `results/[model]/[ontology]/training/`
- **Test results** in `results/[model]/[ontology]/tests/20q_game/`
- **Visualizations** in `visualizations/` (if generated)
- **Runtime logs** in `logs/`

## 🎯 Next Steps

### Try Different Strategies

```bash
# Top-down teaching
python main.py --strategy top_down

# Bottom-up discovery
python main.py --strategy bottom_up

# Learner-driven questions
python main.py --strategy learner_questions

# Teacher-guided questions
python main.py --strategy teacher_questions
```

### Run Batch Experiments

```bash
python main_batch.py
```

### Test with Expert Oracle

```bash
python main_expert.py
```

### Customize Your Experiment

Edit `config.yml` to change:
- **Model**: Switch between GPT-4o, GPT-3.5-turbo, etc.
- **Ontology**: Use different complexity levels (10, 30, 50 entities)
- **Training turns**: Adjust learning duration
- **Testing parameters**: Change number of questions and test sets

## 🔧 Common Configurations

### Low-Cost Testing
```yaml
model:
  name: gpt-3.5-turbo
  max_tokens: 5000
training:
  turns_per_session: 10
testing:
  num_sets: 5
```

### High-Quality Research
```yaml
model:
  name: gpt-4o
  max_tokens: 10000
  temperature: 0.3
training:
  turns_per_session: 30
testing:
  num_sets: 50
```

### Quick Debugging
```yaml
training:
  num_trials: 1
  turns_per_session: 5
testing:
  num_sets: 3
  max_questions: 10
```

## 🐛 Troubleshooting

### "API key not found"
```bash
# Make sure you've set the environment variable
echo $OPENAI_API_KEY  # macOS/Linux
echo $env:OPENAI_API_KEY  # Windows PowerShell
```

### "Import errors"
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### "Config file not found"
```bash
# Make sure you're in the project directory
cd my_aiped
# And that config.yml exists
ls config.yml
```

## 📚 Learn More

- Read the full [README.md](README.md)
- Check [SETUP.md](SETUP.md) for detailed setup
- See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines
- Review example runs in `results/` directory

## 🎓 Understanding Results

### Training Logs
- Dialogue turns between Teacher and Learner
- Information exchanges
- Learning progress

### Test Results
- 20-Questions game performance
- Ontology acquisition metrics
- Comparison with oracle performance

### Visualizations
- Information trajectories
- Knowledge acquisition plots
- Semantic space representations

## 💡 Tips

1. **Start small**: Use a simple ontology (ontology_aliens_10.json) first
2. **Monitor costs**: Track your API usage if using paid models
3. **Save configs**: Keep different config files for different experiments
4. **Version results**: Name experiments clearly in config.yml
5. **Read logs**: Check logs/ directory if something goes wrong

## 🤝 Get Help

- Open an issue on GitHub
- Check the documentation
- Review example configurations
- Contact the research team

Happy experimenting! 🎉
