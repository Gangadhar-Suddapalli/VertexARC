"""VertexArc PDF Q&A - Streamlit application.

A document-based Question & Answer app. Upload a PDF, get an automatic summary,
and ask questions answered strictly from the document's contents (RAG).
Each uploaded document maintains its own independent chat session.
"""
from __future__ import annotations

import hashlib

import streamlit as st

import config
import exporters
import pdf_utils
import rag_engine

st.set_page_config(
    page_title="VertexArc PDF Q&A",
    page_icon="\U0001F4C4",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- styling --------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {
        --bg:        #0B0E14;
        --surface:   #141926;
        --surface-2: #1B2230;
        --border:    rgba(255,255,255,0.07);
        --text:      #E8EAF0;
        --muted:     #8C95AB;
        --gold:      #C9A227;
        --gold-br:   #E7C254;
      }

      html, body, [class*="css"], .stMarkdown, .stChatMessage {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }

      /* App background with a soft gold glow in the corner */
      .stApp {
        background:
          radial-gradient(1100px 520px at 85% -12%, rgba(201,162,39,0.07), transparent 60%),
          var(--bg);
      }

      /* Clean up Streamlit's chrome */
      header[data-testid="stHeader"] { background: transparent; }
      [data-testid="stToolbar"] { right: 0.75rem; }
      footer { visibility: hidden; }

      .block-container { padding-top: 2.6rem; max-width: 1060px; }

      /* ---------------- Header ---------------- */
      .va-header { display:flex; align-items:center; gap:15px; }
      .va-logo {
        width:46px; height:46px; border-radius:13px; flex-shrink:0;
        background: linear-gradient(135deg, var(--gold-br), var(--gold));
        display:flex; align-items:center; justify-content:center;
        font-size:21px; color:#241a00; font-weight:800;
        box-shadow: 0 8px 22px rgba(201,162,39,0.30);
      }
      .va-title { font-size:1.7rem; font-weight:800; margin:0; letter-spacing:-0.02em; color:var(--text); line-height:1.1; }
      .va-accent {
        background: linear-gradient(135deg, var(--gold-br), var(--gold));
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
      }
      .va-qa { font-weight:600; color:var(--muted); }
      .va-sub { color:var(--muted); margin:3px 0 0; font-size:0.93rem; }
      .va-divider { height:1px; background:linear-gradient(90deg, var(--border), transparent); margin:20px 0 24px; }

      /* ---------------- Section label ---------------- */
      .va-label {
        display:inline-flex; align-items:center; gap:8px;
        font-size:0.74rem; font-weight:700; letter-spacing:0.10em;
        text-transform:uppercase; color:var(--muted); margin:0 0 11px;
      }
      .va-label::before { content:""; width:6px; height:6px; border-radius:50%; background:var(--gold); }

      /* ---------------- Summary card ---------------- */
      .va-summary {
        background: var(--surface); border:1px solid var(--border);
        border-left:3px solid var(--gold); padding:18px 20px; border-radius:14px;
        color:#CDD3E0; line-height:1.65; font-size:0.96rem;
        box-shadow: 0 10px 34px rgba(0,0,0,0.28);
      }

      /* ---------------- Meta chips ---------------- */
      .va-meta { display:flex; gap:8px; flex-wrap:wrap; margin:13px 0 26px; }
      .va-chip {
        background: var(--surface-2); border:1px solid var(--border); color:var(--muted);
        padding:4px 11px; border-radius:999px; font-size:0.78rem; font-weight:500;
      }
      .va-chip b { color:var(--text); font-weight:600; }

      /* ---------------- Chat messages ---------------- */
      [data-testid="stChatMessage"] {
        background: var(--surface); border:1px solid var(--border);
        border-radius:14px; box-shadow:0 4px 20px rgba(0,0,0,0.15);
        margin-bottom:12px;
      }

      /* Source chips under an answer */
      .va-sources { display:flex; align-items:center; gap:7px; flex-wrap:wrap; margin-top:12px; }
      .va-src-label { font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.07em; }
      .va-src {
        background: rgba(201,162,39,0.12); border:1px solid rgba(201,162,39,0.32);
        color:var(--gold-br); padding:2px 10px; border-radius:999px; font-size:0.74rem; font-weight:600;
      }

      /* ---------------- Empty state ---------------- */
      .va-empty { text-align:center; padding:54px 20px; max-width:540px; margin:34px auto; }
      .va-empty-icon {
        width:74px; height:74px; margin:0 auto 18px; border-radius:20px;
        display:flex; align-items:center; justify-content:center; font-size:2.1rem;
        background: var(--surface); border:1px solid var(--border);
        box-shadow:0 10px 30px rgba(0,0,0,0.25);
      }
      .va-empty-title { font-size:1.32rem; font-weight:700; color:var(--text); margin-bottom:9px; }
      .va-empty-text { color:var(--muted); line-height:1.65; font-size:0.96rem; }

      /* ---------------- Buttons ---------------- */
      .stButton>button, .stDownloadButton>button {
        border-radius:10px; border:1px solid var(--border);
        background: var(--surface-2); color:var(--text); font-weight:600;
        transition: all 0.16s ease;
      }
      .stButton>button:hover:not(:disabled),
      .stDownloadButton>button:hover:not(:disabled) {
        border-color: var(--gold); color:var(--gold-br);
        transform: translateY(-1px);
      }

      /* ---------------- Chat input ---------------- */
      [data-testid="stChatInput"] {
        border-radius:14px; border:1px solid var(--border); background:var(--surface);
      }
      [data-testid="stChatInput"]:focus-within {
        border-color: var(--gold); box-shadow: 0 0 0 3px rgba(201,162,39,0.16);
      }

      /* ---------------- Sidebar ---------------- */
      [data-testid="stSidebar"] {
        background: var(--surface); border-right:1px solid var(--border);
      }
      .va-side-brand {
        display:flex; align-items:center; gap:10px; font-size:1.15rem; font-weight:800;
        color:var(--text); margin:2px 0 18px;
      }
      .va-logo-sm {
        width:30px; height:30px; border-radius:9px;
        background: linear-gradient(135deg, var(--gold-br), var(--gold));
        display:flex; align-items:center; justify-content:center; font-size:15px; color:#241a00;
      }
      [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2); border:1px dashed var(--border); border-radius:12px;
      }
      .va-stack {
        margin-top:8px; font-size:0.78rem; color:var(--muted);
        background: var(--surface-2); border:1px solid var(--border);
        padding:8px 12px; border-radius:10px; text-align:center;
      }

      /* ---------------- Responsive ---------------- */
      @media (max-width: 640px) {
        .block-container { padding-left:0.7rem; padding-right:0.7rem; }
        .va-title { font-size:1.35rem; }
        .va-logo { width:40px; height:40px; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------- session helpers --------------------------------
def _init_state():
    st.session_state.setdefault("doc_id", None)       # hash of current document
    st.session_state.setdefault("doc_name", None)
    st.session_state.setdefault("index", None)        # rag_engine.DocumentIndex
    st.session_state.setdefault("summary", None)
    st.session_state.setdefault("stats", None)
    st.session_state.setdefault("history", [])        # list of {question, answer, sources}
    st.session_state.setdefault("uploader_key", 0)    # bumped to reset the file uploader


def _reset_session():
    """Start a fresh, independent session for a new document."""
    st.session_state.index = None
    st.session_state.summary = None
    st.session_state.stats = None
    st.session_state.history = []


def _clear_all():
    """Full reset: drop the document, summary, and chat so a new upload is required."""
    _reset_session()
    st.session_state.doc_id = None
    st.session_state.doc_name = None
    # Changing the uploader key forces Streamlit to render a fresh, empty uploader
    # so the previous PDF is actually removed (not silently re-indexed).
    st.session_state.uploader_key += 1


def _source_chips_html(pages) -> str:
    chips = "".join(f'<span class="va-src">p.{p}</span>' for p in pages)
    return f'<div class="va-sources"><span class="va-src-label">Sources</span>{chips}</div>'


_init_state()


# ------------------------------- header -------------------------------------
st.markdown(
    """
    <div class="va-header">
      <div class="va-logo">&#9670;</div>
      <div>
        <p class="va-title">Vertex<span class="va-accent">Arc</span> <span class="va-qa">PDF Q&amp;A</span></p>
        <p class="va-sub">Upload a PDF and ask questions answered only from its contents.</p>
      </div>
    </div>
    <div class="va-divider"></div>
    """,
    unsafe_allow_html=True,
)


# ------------------------------- sidebar ------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="va-side-brand"><span class="va-logo-sm">&#9670;</span>'
        'Vertex<span class="va-accent">Arc</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="va-label">Upload a document</div>', unsafe_allow_html=True)

    # Surface a clear message if the API key is missing.
    if not config.get_api_key():
        provider = config.get_llm_provider()
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        st.error(
            f"{provider.title()} API key not configured. Add {key_name} to your "
            ".env (local) or the app Secrets (Streamlit Cloud)."
        )

    uploaded = st.file_uploader(
        "PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        doc_id = hashlib.md5(file_bytes).hexdigest()

        # New document -> build a brand new independent session.
        if doc_id != st.session_state.doc_id:
            _reset_session()
            st.session_state.doc_id = doc_id
            st.session_state.doc_name = uploaded.name

            with st.spinner("Reading PDF and building the knowledge index..."):
                import io

                pages = pdf_utils.extract_pages(io.BytesIO(file_bytes))
                stats = pdf_utils.document_stats(pages)
                chunks = pdf_utils.chunk_pages(pages)
                try:
                    index = rag_engine.build_index(chunks)
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))
                    st.stop()
                summary = rag_engine.summarize_document(index)

            st.session_state.index = index
            st.session_state.stats = stats
            st.session_state.summary = summary
            st.success(f"Indexed '{uploaded.name}'.")

    st.divider()
    st.markdown('<div class="va-label">Export conversation</div>', unsafe_allow_html=True)
    history = st.session_state.history
    disabled = not history
    txt = exporters.to_text(
        history, st.session_state.doc_name or "document", st.session_state.summary
    ) if history else ""
    st.download_button(
        "Download .txt",
        data=txt,
        file_name="vertexarc_conversation.txt",
        mime="text/plain",
        disabled=disabled,
        use_container_width=True,
    )
    pdf_bytes = (
        exporters.to_pdf(
            history, st.session_state.doc_name or "document", st.session_state.summary
        )
        if history
        else b""
    )
    st.download_button(
        "Download .pdf",
        data=pdf_bytes,
        file_name="vertexarc_conversation.pdf",
        mime="application/pdf",
        disabled=disabled,
        use_container_width=True,
    )

    has_session = st.session_state.index is not None or bool(history)
    if st.button(
        "Clear chat & remove PDF", use_container_width=True, disabled=not has_session
    ):
        _clear_all()
        st.rerun()

    # Active engine badge (shows the free local + provider stack).
    _provider = config.get_llm_provider().title()
    _emb = "local" if config.get_embed_backend() == "local" else "OpenAI"
    st.markdown(
        f'<div class="va-stack">&#9889; {_provider} chat &nbsp;&middot;&nbsp; '
        f'&#128274; {_emb} embeddings</div>',
        unsafe_allow_html=True,
    )


# ------------------------------- main area ----------------------------------
if st.session_state.index is None:
    st.markdown(
        """
        <div class="va-empty">
          <div class="va-empty-icon">&#128196;</div>
          <div class="va-empty-title">Start by uploading a PDF</div>
          <div class="va-empty-text">
            Use the sidebar to upload a document. VertexArc reads it, writes a short
            summary, and answers your questions using only what's inside the file.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# Document summary
if st.session_state.summary:
    st.markdown('<div class="va-label">Document summary</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="va-summary">{st.session_state.summary}</div>',
        unsafe_allow_html=True,
    )

# Meta chips (document name + stats)
s = st.session_state.stats
chips = [f'<span class="va-chip">&#128196; <b>{st.session_state.doc_name}</b></span>']
if s:
    chips += [
        f'<span class="va-chip"><b>{s["num_pages"]}</b> pages</span>',
        f'<span class="va-chip"><b>{s["num_words"]:,}</b> words</span>',
        f'<span class="va-chip"><b>{len(st.session_state.index.chunks)}</b> chunks</span>',
    ]
st.markdown('<div class="va-meta">' + "".join(chips) + "</div>", unsafe_allow_html=True)

st.markdown('<div class="va-label">Conversation</div>', unsafe_allow_html=True)

# Render the last N turns of history as chat bubbles.
recent = st.session_state.history[-config.MAX_HISTORY:]
for turn in recent:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            st.markdown(_source_chips_html(turn["sources"]), unsafe_allow_html=True)

# Chat input
question = st.chat_input("Ask a question about the document...")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the document..."):
            try:
                result = rag_engine.answer_question(
                    st.session_state.index,
                    question,
                    history=st.session_state.history,
                )
            except Exception as exc:  # noqa: BLE001
                result = {
                    "answer": f"Something went wrong while answering: {exc}",
                    "found": False,
                    "sources": [],
                }
        st.write(result["answer"])
        if result.get("sources"):
            st.markdown(_source_chips_html(result["sources"]), unsafe_allow_html=True)

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    )
    st.rerun()
