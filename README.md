# Lex6

Agentic RAG over climate-law documents (PDF, DOCX, TXT) using local [Ollama](https://ollama.com)
models, FAISS and BGE-M3 embeddings. Productionised from the `document copy.ipynb` research notebook.

- Contributor / agent conventions: [AGENTS.md](AGENTS.md)
- Install, run and test: [.agents/skills/test-app/SKILL.md](.agents/skills/test-app/SKILL.md)

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
cp .env.example .env
pytest -m "not integration"
```

The LLM runs on a local Ollama server. Pull the configured model before asking questions:

```bash
ollama serve &                 # if it isn't already running
ollama pull deepseek-r1:8b     # default LEX6_LLM_MODEL (notebook model)
# or use a smaller installed model: export LEX6_LLM_MODEL=qwen2.5:3b
```

If the model isn't installed or Ollama isn't running, `/api/health` reports `LLM unavailable` (shown in the UI
sidebar) and Ask/CLI calls return a message saying which `ollama pull` / `ollama serve` command fixes it.
