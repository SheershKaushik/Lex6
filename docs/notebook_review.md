# Review of `document copy.ipynb`

Bugs found in the reference notebook and where they are fixed in the package.

| # | Notebook location | Bug | Fix |
|---|---|---|---|
| 1 | Cells 3, 5 | Hard-coded path (`../data/Paris_agreement-English.pdf`) and loader only globs `**/*.pdf`; DOCX/TXT never ingested. | `ingestion/loaders.py`: `discover_files` + `load_file` for PDF (PyMuPDF), DOCX (docx2txt), TXT/MD. |
| 2 | Cell 5 | Any file that fails to load is printed and skipped; no validation of type, size, encryption, duplicates or empty/scanned content. | `ingestion/validation.py`: `FileValidator` (extension, magic bytes, size, password-protected PDF, corrupt DOCX, binary text, SHA-256 duplicates) + `validate_content` (no extractable text). |
| 3 | Cell 6 | Language is only *detected* (to list English NDCs); non-English pages are embedded as-is although the reranker (`ms-marco-MiniLM`) and prompts are English-only. | `ingestion/language.py`: per-page detection + `LLMTranslator` (local Ollama) converts every non-English page to English before chunking; provenance kept in `original_language` / `translated` metadata. |
| 4 | Cell 6 | `extract_country_from_filename` only strips `.pdf` and uses substring matching (`"Oman"` matches `"Romania"`). | Whole-word regex on the filename stem, any extension. |
| 5 | Cells 13, 24 | `EmbeddingManager` defaults to `BAAI/bge-m3` but outputs show 384-dim vectors (MiniLM): the persisted index was built with another model and is reloaded without any check → dimension crash or meaningless scores. | `VectorStore.load_local` stores and verifies embedding model + dimension (`VectorStoreMismatchError`). |
| 6 | Cell 15 | `VectorStore` reads the global `embedding_manager`. | Embedding dimension/model injected via constructor; `services.py` wires components. |
| 7 | Cells 15, 19 | Document vectors are added to `IndexFlatL2` **un-normalised** while the query is normalised, so the cosine formula is invalid. | Vectors normalised in `VectorStore.add_documents` and `search_embeddings`; `EmbeddingManager` uses `normalize_embeddings=True`. |
| 8 | Cell 19 | `IndexFlatL2` returns **squared** L2 distances, but similarity is computed as `1 - distance**2 / 2` (squares twice) → inflated scores, `score_threshold` meaningless. | `cosine = 1 - distance / 2` (see retrieval module). |
| 9 | Cells 16 + 18 | Cell 18 re-embeds and re-adds all chunks after cell 16 already did → every chunk stored twice (ids reach ~38k), duplicated retrieval results. | Single ingestion path; files de-duplicated by SHA-256 across runs. |
| 10 | Cell 20 | `LLMClientWrapper` reads `OLLAMA_HOST` (the standard variable for the *local* server) and defaults it to `https://ollama.com`; also requires a cloud key implicitly. Unused `from openai import OpenAI`. | `llm.py`: separate `LEX6_OLLAMA_CLOUD_HOST`; cloud only when `OLLAMA_API_KEY` is set, otherwise local model. |
| 11 | Cell 21 | `AdvancedRAGRetriever` indexes `indices[0]` / `distances[0]` although `search_embeddings` already returns 1-D arrays → `TypeError` swallowed by `try/except`, dense retrieval silently returns nothing. | Fixed in retrieval module. |
| 12 | Cell 21 | `BM25SparseRetriever` / `CrossEncoderWrapper` are never instantiated and the BM25 corpus is never built from the docstore (nor rebuilt after new ingestion). | Built from `VectorStore.all_chunks()` and refreshed when the store changes. |
| 13 | Cells 37, 39 | `rag_advanced` / `AdvancedRAGPipeline` read `doc['similarity_score']`, which `AdvancedRAGRetriever` never sets → `KeyError`. | Unified result schema (`score`, `similarity_score`, `cross_encoder_score`). |
| 14 | Cells 29, 34 | Deprecated imports (`langchain_community.chat_models.ChatOllama`, `langchain_classic.schema`). | `langchain_ollama.ChatOllama`, `langchain_core.messages`. |
| 15 | Cells 30, 34, 39 | deepseek-r1 returns `<think>…</think>` reasoning inside `content`, which is shown to the user. | `llm.strip_think`. |
| 16 | Cells 34, 39, 44 | Huge contexts (`top_k=350`) with Ollama's default context window are silently truncated. | `num_ctx` set explicitly (`LEX6_LLM_NUM_CTX`). |
| 17 | Cell 34 | `rag_simple` catches `NameError` (never raised there) while real failures (Ollama down) crash. | Errors handled explicitly in pipelines/agent. |
| 18 | Cell 39 | "Streaming" prints the *prompt* in 80-char slices with `sleep(0.05)` (seconds of delay, leaks the prompt). | Removed. |
| 19 | Cells 37, 39 | PyMuPDF `page` is 0-based → citations say "page 0". | Citations use 1-based page numbers. |
| 20 | Cells 19, 21 | `retrieve` swallows every exception and returns `[]`, hiding failures as "No relevant context found". | Errors logged/propagated. |
| 21 | Cell 44 | `NameError: adv_rag` — cells depend on hidden kernel state/order. | Modular package + CLI. |
