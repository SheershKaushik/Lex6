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

## 5. Retrieval agent (default for `lex6 ask`)

```bash
lex6 ask "What must the host Party do to avoid double counting of ITMOs?"   # trace on stderr
lex6 ask "Hello, who are you?"          # expect: no retrieval (route -> small talk)
lex6 ask --json "..."                   # full AgentResponse: steps, sources, citations, citation_mode
```

Expected trace: `route -> plan -> retrieve -> reflect [-> retrieve] -> [verify_citations] -> answer`;
`citation_mode=explicit` with `[n]` markers that point at chunks in `sources`. ~1.5-2 min per question on CPU.

## 6. Eval harness (agent vs naive single-shot baseline)

```bash
lex6 eval --output /tmp/eval_results.json     # 11 fixed questions over data/sample, ~25 min on CPU
lex6 eval --only q03,q04,q11                  # quick subset
```

Prints a per-question table and a metric comparison table (source_recall, keyword_coverage,
citation_precision, abstention, routing, overall, latency_s) with a delta column. Needs the sample corpus ingested.

## 7. Web UI (three.html, Upload + Ask)

```bash
# fresh data dir for Upload testing; set the translation model too in case .env overrides it
export LEX6_LLM_MODEL=qwen2.5:3b LEX6_TRANSLATION_MODEL=qwen2.5:3b LEX6_DATA_DIR=/tmp/lex6ui-$(date +%s)
lex6 serve --port 8000          # open http://127.0.0.1:8000/ -> passcode page, enter LEX6access404
curl -s -c /tmp/lex6.cookies -H 'Content-Type: application/json' -d '{"passcode":"LEX6access404"}' localhost:8000/api/login
curl -s -b /tmp/lex6.cookies localhost:8000/api/health   # without the cookie every /api/* call is 401
```

UI test flow, which must be recorded before any PR that touches `src/lex6/api/` or the HTML:
0. Login: `/` redirects to `/login`; a wrong passcode shows "Incorrect passcode." and stays there; `LEX6access404`
   opens the workspace. **Log out** (top right) returns to `/login`.
1. Sidebar shows the Durham logo and `Backend: ok`.
2. Upload: choose files from `data/sample/` (PDF, DOCX and TXT are all accepted), click **Start ingest**. The status line
   lists each file as `ingested` or `rejected (reason)`, e.g. a duplicate. Documents appear in the list. **Open** shows pages.
3. Ask: type a question (e.g. "What deduction applies when leakage cannot be quantified?"), click **Ask**.
   The bot message shows the answer with `[n]` citations. Details list the cited sources
   (file, page, score), the search queries and the agent steps. Check that every `[n]` maps (under **Cited sources**)
   to the chunk that actually states the fact - for this question, the bilateral DOCX - not just that the right
   document appears under "Other retrieved context". The answer must not also claim the context is insufficient.
4. Ask "Hello, who are you?". The answer comes back directly: "no document retrieval was needed".
5. Answer length: the selector next to Ask (default **Law Expert**) sets the register only; answers must not stop
   mid-sentence. The server log must not show `output was cut off by the token/context limit`.
6. The "All documents" filter, set to one document, restricts the citations to that document.

CPU timings (qwen2.5:3b, no GPU): ingesting the 3 sample files takes ~7 min (French translation dominates), one agent
answer 2-7 min. Watch the server log for request completion before assuming the browser is stuck, and don't re-click
**Start ingest** / **Ask** while a request is running.

For translated PDFs, page previews prove the text is English, not that the translation is faithful: inspect the
chunks for repeated or invented sentences (small local models can loop) and report that separately.

Compare, Reports and Knowledge Graph are deferred and are expected to do nothing or return 404.

### Data-folder corpus (auto-ingested on startup)

`lex6 serve` indexes `LEX6_CORPUS_DIR` (default: all of `LEX6_DATA_DIR`, recursively, minus `vector_store/` and
`uploads/`) into `LEX6_DATA_DIR` in a background thread. Relative paths and `.env` resolve against the project root
(`LEX6_HOME`, else the checkout), not the cwd; the serve banner prints the folders in use. With a fresh
`LEX6_DATA_DIR=/tmp/...` the corpus is that empty dir: set `LEX6_CORPUS_DIR=data/sample` to get the sample corpus.
`/api/health` → `corpus.status` (`pending|syncing|ready|missing|error|disabled`). So a fresh data dir is not empty
after startup: for tests that need an empty Upload list (e.g. missing-model checks) set `LEX6_AUTO_INGEST_CORPUS=false`.
- Use `lex6 -v serve` to see per-file ingestion and the `corpus ...: N files, X newly ingested, Y already indexed` line.
- Persistence check: first start on a fresh dir → `4 newly ingested`; restart on the same dir → `0 newly ingested,
  4 already indexed` within seconds and no translation calls / no FAISS writes (compare hashes/mtimes of
  `vector_store/`).
- Progress states (`n/4`) can last only seconds between the UI's 5 s polls; have the browser and recording ready
  before starting a fresh sync.
- Combined search: upload a TXT with a distinctive fictional fact → listed as `uploaded this session` next to the
  `data folder` docs; ask with All documents → the fact is cited to the upload, and the answer must not also claim
  the context lacks that information.
