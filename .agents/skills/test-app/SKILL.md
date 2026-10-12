---
name: test-app
description: How to install, run and test the Lex6 agentic RAG app (unit tests, lint, ingestion, agent, eval harness, web UI). Use before every PR.
---

# Testing Lex6

Run this checklist before opening any PR. All commands run from the repo root.

## 1. Install (once per machine)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip
# CPU-only torch keeps the install small; use the default index if you have a GPU.
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
pip install -e ".[dev]"
cp -n .env.example .env   # edit models/paths if needed; never commit .env
```

Ollama (needed for ingestion translation, the agent, the eval harness and the UI):

```bash
curl -fsSL https://ollama.com/install.sh | sh   # installs + starts the service on :11434
ollama pull deepseek-r1:8b                      # default LLM (from the notebook)
curl -s localhost:11434/api/tags                # sanity check
```

On CPU-only machines deepseek-r1:8b is slow. For local testing override the model, e.g.:

```bash
ollama pull qwen2.5:3b
export LEX6_LLM_MODEL=qwen2.5:3b LEX6_TRANSLATION_MODEL=qwen2.5:3b
```

HyDE uses `gpt-oss:120b-cloud` only when `OLLAMA_API_KEY` is exported; otherwise it falls back to
the local `LEX6_LLM_MODEL` (and to the raw query if that fails). Do not put the key in code.

## 2. Lint + unit tests (always)

```bash
ruff check . && ruff format --check .
pytest -m "not integration"
```

Unit tests use fake LLMs/embedders and need no network, Ollama or model downloads.
Both commands must be clean before a PR.

## 3. Ingestion (validation + translation + chunk + embed + store)

```bash
export LEX6_DATA_DIR=/tmp/lex6data            # keep test stores out of the repo
lex6 -v ingest data/sample                    # files or directories
```

Expected on the sample corpus: 4 files `ingested`; `Accord_de_Paris-extrait-FR.pdf` shows
`lang=fr translated_pages=2`. Re-running reports every file as `rejected ... duplicate`.
First run downloads `BAAI/bge-m3` (~2.3 GB) from HuggingFace. To start fresh: `rm -rf $LEX6_DATA_DIR`.
Changing `LEX6_EMBEDDING_MODEL` requires a fresh data dir (the store refuses to load otherwise).

## 4. Retrieval / RAG pipelines (notebook `rag_advanced`)

```bash
export LEX6_LLM_MODEL=qwen2.5:3b LEX6_DATA_DIR=/tmp/lex6data   # after step 3
lex6 ask "What do Parties need to apply under Article 6 to avoid double counting?" --top-k 4
lex6 ask "What is the Paris Agreement?" --retriever naive          # dense-only baseline
```

`advanced` = HyDE (cloud `gpt-oss:120b-cloud` if `OLLAMA_API_KEY`, else local model) + BM25 + cross-encoder
(first run downloads `cross-encoder/ms-marco-MiniLM-L-6-v2`). Expect the bilateral-agreement DOCX
(Article 5, corresponding adjustments) as source [1]. Scores: advanced = cross-encoder logits, naive = cosine.
~45 s per question on a CPU-only VM.
