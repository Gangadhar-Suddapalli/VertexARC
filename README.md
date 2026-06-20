# VertexArc PDF Q&A

A document-based Question & Answer application. Upload a PDF, get an automatic
summary, and ask questions that are answered **only** from the contents of that
document using Retrieval-Augmented Generation (RAG).

Built with **Streamlit** (UI + backend) and the **OpenAI API** (embeddings +
chat completion).

---

## Features

| # | Requirement | Status |
|---|-------------|--------|
| 1 | PDF upload | ✅ Sidebar uploader (`.pdf`, up to 50 MB) |
| 2 | Q&A interface | ✅ Chat-style input and message bubbles |
| 3 | Context-based answers | ✅ Answers grounded strictly in retrieved chunks |
| 4 | Session-based handling | ✅ Each uploaded document gets its own independent session/index |
| 5 | Conversation history | ✅ Last 5 Q&A pairs shown in the session |
| 6 | Document summary | ✅ Auto-generated on upload |
| 7 | Unsupported question handling | ✅ Graceful "not found in document" message |
| 8 | Export | ✅ Download conversation as `.txt` or `.pdf` |
| 9 | Responsive UI | ✅ Works on desktop and mobile (Streamlit + responsive CSS) |

---

## How it works

```
PDF → text extraction (pypdf) → chunking (overlapping windows)
    → embeddings (OpenAI text-embedding-3-small) → in-memory vector index
Question → embed query → cosine similarity → top-K chunks
        → chat model answers using ONLY those chunks → answer + source pages
```

If the best similarity score is below a threshold, or the model cannot find the
answer in the retrieved context, the app returns a clear "not in document"
message instead of hallucinating.

---

## Project structure

```
vertexarc-pdf-qa/
├── app.py              # Streamlit UI: upload, chat, summary, history, export
├── config.py           # Settings + OpenAI client (reads .env or st.secrets)
├── pdf_utils.py        # PDF text extraction + chunking
├── rag_engine.py       # Embeddings, retrieval, answering, summarization
├── exporters.py        # Conversation export to TXT / PDF
├── requirements.txt
├── .env.example        # Local env template
├── .gitignore
└── .streamlit/
    ├── config.toml             # Theme + upload size
    └── secrets.toml.example    # Cloud secrets template
```

---

## Run locally

Requires Python 3.10+.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd vertexarc-pdf-qa

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
cp .env.example .env             # then edit .env and paste your key
#   OR export it directly:
#   export OPENAI_API_KEY=sk-...

# 5. Launch
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Deploy to Streamlit Community Cloud (free)

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo, branch, and `app.py` as the entry point.
4. Under **Advanced settings → Secrets**, add:

   ```toml
   OPENAI_API_KEY = "sk-your-key"
   OPENAI_CHAT_MODEL = "gpt-4o-mini"
   OPENAI_EMBED_MODEL = "text-embedding-3-small"
   ```

5. Click **Deploy**. You'll get a public `https://<app>.streamlit.app` URL.

> The key is read from `st.secrets` on Cloud and from `.env` / environment
> variables locally — see `config.py`.

---

## Configuration

Defaults live in `config.py` and can be overridden via env vars / secrets:

| Setting | Default | Meaning |
|---------|---------|---------|
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Model used to generate answers/summary |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | 1000 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 4 | Chunks retrieved per question |
| `MAX_HISTORY` | 5 | Recent Q&A pairs shown/used |
| `SIMILARITY_FLOOR` | 0.20 | Below this, treat as "not in document" |

---

## Notes & limitations

- Scanned / image-only PDFs have no extractable text; OCR is a planned
  enhancement.
- The vector index is held in memory per session (no external database needed),
  which keeps the architecture simple and stateless between users. See the
  documentation for the rationale.
- An OpenAI API key is required; usage incurs standard OpenAI costs (minimal with
  `gpt-4o-mini` + `text-embedding-3-small`).
