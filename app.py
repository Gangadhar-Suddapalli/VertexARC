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
      .block-container {padding-top: 4rem; max-width: 1100px;}
      .va-title {font-size: 1.9rem; font-weight: 800; margin-bottom: 0;}
      .va-accent {color: #C9A227;}
      .va-sub {color: #8b93a7; margin-top: 0;}
      .va-summary {background: rgba(201,162,39,0.08); border-left: 4px solid #C9A227;
                   padding: 0.8rem 1rem; border-radius: 6px;}
      /* Responsive: collapse padding on small screens */
      @media (max-width: 640px) {
        .block-container {padding-left: 0.6rem; padding-right: 0.6rem;}
        .va-title {font-size: 1.4rem;}
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


def _reset_session():
    """Start a fresh, independent session for a new document."""
    st.session_state.index = None
    st.session_state.summary = None
    st.session_state.stats = None
    st.session_state.history = []


_init_state()


# ------------------------------- header -------------------------------------
st.markdown(
    '<p class="va-title">Vertex<span class="va-accent">Arc</span> '
    "PDF Q&A</p>",
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="va-sub">Upload a PDF and ask questions answered only from its '
    "contents.</p>",
    unsafe_allow_html=True,
)


# ------------------------------- sidebar ------------------------------------
with st.sidebar:
    st.subheader("1. Upload a document")

    # Surface a clear message if the API key is missing.
    if not config.get_api_key():
        provider = config.get_llm_provider()
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        st.error(
            f"{provider.title()} API key not configured. Add {key_name} to your "
            ".env (local) or the app Secrets (Streamlit Cloud)."
        )

    uploaded = st.file_uploader("PDF file", type=["pdf"], accept_multiple_files=False)

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

    if st.session_state.stats:
        s = st.session_state.stats
        st.caption(
            f"{s['num_pages']} pages | {s['num_words']:,} words | "
            f"{len(st.session_state.index.chunks)} chunks"
        )

    st.divider()
    st.subheader("Export conversation")
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

    st.divider()
    if st.button("Clear chat", use_container_width=True, disabled=not history):
        st.session_state.history = []
        st.rerun()


# ------------------------------- main area ----------------------------------
if st.session_state.index is None:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

# Document summary
if st.session_state.summary:
    st.markdown("#### Document summary")
    st.markdown(
        f'<div class="va-summary">{st.session_state.summary}</div>',
        unsafe_allow_html=True,
    )
    st.write("")

st.markdown("#### Chat")
st.caption(f"Showing the last {config.MAX_HISTORY} questions in this session.")

# Render the last N turns of history as chat bubbles.
recent = st.session_state.history[-config.MAX_HISTORY:]
for turn in recent:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            pages = ", ".join(str(p) for p in turn["sources"])
            st.caption(f"Source pages: {pages}")

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
            pages = ", ".join(str(p) for p in result["sources"])
            st.caption(f"Source pages: {pages}")

    st.session_state.history.append(
        {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }
    )
    st.rerun()
