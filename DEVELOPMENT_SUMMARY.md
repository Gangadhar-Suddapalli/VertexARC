# Development Summary — VertexArc PDF Q&A

## Approximate time taken

Roughly **6–8 hours** of focused work end to end:

- Requirements analysis and architecture decisions — ~1 hour
- Core RAG pipeline (PDF extraction, chunking, embeddings, retrieval, answering) — ~2 hours
- Streamlit UI (upload, chat, summary, history, export, responsive styling) — ~2 hours
- Documentation, presentation, and architecture diagram — ~2 hours
- Testing, deployment configuration, and polish — ~1 hour

## Development approach

I used an AI-assisted, requirements-first approach. I started by mapping each of the nine functional requirements to a concrete implementation, then chose the simplest architecture that satisfied all of them: a single Streamlit application (UI + logic in one deployable unit) with an in-memory vector index per session, backed by the OpenAI API for embeddings and answer generation.

The code is organised into small, single-responsibility modules — `pdf_utils` (extraction/chunking), `rag_engine` (embeddings, retrieval, answering, summarisation), `exporters` (TXT/PDF), `config` (settings and the OpenAI client), and `app.py` (UI). Answers are grounded with a strict system prompt, a cosine-similarity floor, and an explicit "no answer" sentinel so the app declines instead of hallucinating. Each document is keyed by a hash of its bytes, which is what makes every upload an independent session.

AI assistance was used throughout for scaffolding, prompt design, and generating the documentation and slide assets, with manual verification of the logic (chunking and export modules were unit-tested; the document and deck were rendered and visually reviewed).

## Challenges encountered

- **Preventing hallucination** — keeping answers strictly inside the document required combining a strict prompt, a similarity threshold, and a sentinel value rather than relying on the prompt alone.
- **Independent sessions** — solved by hashing file bytes to key the index and history, so re-uploading the same file resumes and a new file resets cleanly.
- **Chunking trade-offs** — tuning chunk size and overlap to balance retrieval precision against keeping enough surrounding context.
- **PDF export portability** — handling text wrapping and byte output correctly across `fpdf2` versions.

## Improvements with more time

- OCR support for scanned / image-only PDFs.
- A persistent vector store (FAISS, pgvector, or Pinecone) to support large documents and history that survives restarts.
- Multi-document querying and clickable citations that highlight the source span inside the original PDF.
- User accounts, saved conversations, streaming answers, and in-UI model selection.
- Automated evaluation of answer faithfulness against a labelled question set.
