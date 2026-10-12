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
