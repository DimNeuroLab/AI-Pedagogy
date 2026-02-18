# Setup Instructions

## Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key (or compatible API endpoint)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository-url>
cd my_aiped
```

### 2. Create a Virtual Environment

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Set your OpenAI API key as an environment variable:

**On Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**On Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-api-key-here
```

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your-api-key-here
```

Alternatively, create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

### 5. Verify Installation

Run a quick test:
```bash
python main.py --help
```

You should see the help message with available options.

## Configuration

Edit `config.yml` to customize:
- Model selection (GPT-4o, GPT-3.5-turbo, etc.)
- Pedagogical strategy
- Number of training turns
- Testing parameters
- Ontology settings

## Running Your First Experiment

```bash
python main.py --strategy top_down
```

This will run a single training and testing session using the top-down strategy.

## Troubleshooting

### API Key Issues
- Ensure your API key is correctly set in the environment
- Check that your OpenAI account has sufficient credits

### Import Errors
- Ensure you've activated the virtual environment
- Try reinstalling: `pip install -r requirements.txt --force-reinstall`

### Permission Errors
- On Unix-like systems, you may need to make scripts executable:
  ```bash
  chmod +x main.py main_batch.py main_expert.py
  ```

## Next Steps

- Read the [README.md](README.md) for detailed usage information
- Check the [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines
- Explore the `examples/` directory (if available) for sample configurations
