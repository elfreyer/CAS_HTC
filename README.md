# Classifying cyber-fraud reports — CAS project code

Hierarchical few-shot classification of cyber-fraud reports
(compact encoder vs local LLM). See the report for details.

## Requirements

- **Python 3.11** (3.10+ works)
- **[Ollama](https://ollama.com)** — only for the LLM family (`03_llm`), which serves the models locally

## Installation

### 1. Get the repository

```bash
git clone https://github.com/<account>/<repo>.git
cd <repo>
```

### 2. Create a virtual environment

**With `venv`** (standard):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt or requirements_mac.txt
```

### 4. LLM models (for the `03_llm` notebook only)

Once Ollama is installed and running:

```bash
ollama pull gemma4:26b
ollama pull gemma4:e4b
ollama list                      # check that both models are present
```

The notebooks query Ollama at its default address (`http://localhost:11434`).
If Ollama runs on another machine, set `OLLAMA_HOST` accordingly.

## Running

Open the notebooks (Jupyter or VS Code) and run them **in order**:

```
00_data → 01_flat → 02_setfit → 03_llm → 04_analysis
```

Embeddings and LLM calls are cached (`artifacts/`), so a second run resumes instantly.
