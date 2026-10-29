# AGENTS.md

Guidance for humans and AI agents working on **Lex6** — an agentic RAG system for climate-law
documents (treaties, NDCs, Article 6 standards) that runs on local Ollama models.

## Project structure

```
src/lex6/
  config.py            # Settings (pydantic-settings, LEX6_* env vars / .env)
  ingestion/           # validate -> detect language/translate -> load -> chunk
  embeddings.py        # EmbeddingManager (sentence-transformers, BAAI/bge-m3)
  vectorstore.py       # FaissDocStore + VectorStore (FAISS IndexFlatL2 on L2-normalised vectors)
  llm.py               # Ollama chat/generate wrappers (local + optional Ollama Cloud for HyDE)
  countries.py         # Country-name detection (pycountry + aliases) for the country-priority rule
  retrieval/           # RAGRetriever (naive dense), BM25, CrossEncoder, AdvancedRAGRetriever (HyDE+hybrid+rerank)
  retrieval/country.py # CountryPriorityRetriever: chunks mentioning a country named in the query first (score 1.0)
  rag.py               # rag_simple / rag_advanced / AdvancedRAGPipeline (notebook pipelines)
  agent.py             # RetrievalAgent: route -> plan queries -> retrieve/reflect -> cited answer
  eval/                # Eval harness: agent vs naive single-shot baseline
  api/                 # FastAPI backend + static three.html UI
  cli.py               # `lex6` command (ingest, ask, eval, serve)
tests/                 # pytest unit tests (fakes for LLM/embeddings; no network needed)
data/                  # Corpus indexed on `lex6 serve` (data team adds files here); vector_store/, uploads/ are generated
data/sample/           # Small sample corpus used by tests, eval and the UI demo
docs/                  # Design notes, notebook bug review
.agents/skills/        # Agent skills (see test-app before every PR)
```

The reference implementation is the research notebook `document copy.ipynb`; the package keeps its
models and techniques (PyMuPDF loading, RecursiveCharacterTextSplitter 1000/200, BGE-M3 embeddings,
FAISS + pickled docstore, HyDE + BM25 hybrid + cross-encoder reranking, deepseek-r1 via Ollama,
three-part legal answer prompt). Do not swap these out without discussion.

Out of scope for now: the **Compare** and **Knowledge Graph** features of the UI.

## Conventions

- Python >= 3.10, `src/` layout, type hints, small modules. Prefer editing existing modules.
- Heavy dependencies (sentence-transformers, FAISS, Ollama) are wrapped behind small classes so
  that unit tests can inject fakes. Never load real models in unit tests.
- Logging via `logging.getLogger(__name__)`, not `print`, in library code.
- Configuration only through `lex6.config.Settings` (env prefix `LEX6_`).

## Rules

1. **Tests are required.** Every change to core logic (ingestion, validation, retrieval, agent,
   eval scoring, API) must come with pytest tests. `pytest -m "not integration"` must pass.
   Tests that need a live Ollama server or downloaded HF models are marked `@pytest.mark.integration`.
2. **No secrets in code.** API keys (e.g. `OLLAMA_API_KEY`) come from the environment or an
   untracked `.env`. Never commit `.env`, keys, tokens or personal data. `.env.example` documents
   the variables with no real values.
3. **Lint before PRs.** Run `ruff check . && ruff format --check .` and fix everything before
   opening a PR.
4. **Small commits, one PR per working slice.**
5. **UI changes:** before any PR touching `src/lex6/api/` or the HTML, run the app and test it
   end-to-end following `.agents/skills/test-app/SKILL.md`, and attach a screen recording.
6. Models run locally through Ollama. Ollama Cloud (`*-cloud` models) is only used where the
   notebook used it (HyDE) and only when `OLLAMA_API_KEY` is set; everything must work without it.

## Quick commands

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -m "not integration"
```

See `.agents/skills/test-app/SKILL.md` for running the full app.
