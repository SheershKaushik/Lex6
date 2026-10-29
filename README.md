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

Opening the UI first asks for the passcode (`LEX6access404`). It is checked on the server, so every page and `/api/*`
call needs the login cookie; **Log out** clears it. Only the passcode's SHA-256 is stored: change it with
`LEX6_ACCESS_PASSCODE_SHA256` (empty disables the login) and set `LEX6_SESSION_SECRET` to keep logins across restarts.

`lex6 serve` indexes every file in `data/sample/` (`LEX6_CORPUS_DIR`) into the FAISS store under
`data/vector_store/` on startup, in the background. Files that are already indexed are skipped (by content hash), so
only the first start pays for translation/embedding. Files uploaded in the UI are added to the same store and searched
together with the corpus. Set `LEX6_AUTO_INGEST_CORPUS=false` to disable.

Queries that name a country (e.g. "How does the NDC of Australia mention climate law?") put every chunk that mentions
it, or that comes from a file named after it, at the top with score 1.0 (each matching document is represented); the
normal hybrid ranking follows. Detection uses pycountry names plus aliases (USA, UK, Russia, ...). Disable with
`LEX6_COUNTRY_PRIORITY=false`; the eval's naive baseline never uses it.

Answers are not length-capped (`LEX6_LLM_NUM_PREDICT=-1`) and the prompt asks for detailed, expert legal analysis.
The UI's Learner / Familiar / Law Expert selector (default Law Expert) only changes the register, not the length.
Prompt and answer share Ollama's context window (`LEX6_LLM_NUM_CTX`, default 16384); if the server logs that an answer
was cut off, raise it as far as your RAM/VRAM allows.

If the model isn't installed or Ollama isn't running, `/api/health` reports `LLM unavailable` (shown in the UI
sidebar) and Ask/CLI calls return a message saying which `ollama pull` / `ollama serve` command fixes it.
