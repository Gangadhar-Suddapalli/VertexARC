"""Retrieval-Augmented Generation engine.

Pipeline:
  1. Embed all document chunks once (on upload) -> in-memory vector index.
  2. For each question, embed the query and retrieve the top-K chunks by
     cosine similarity.
  3. Ask the chat model to answer using ONLY the retrieved context.
  4. If the best similarity is below a floor, or the model reports it cannot
     find the answer, return a graceful "not in document" message.

The index is kept in memory and stored in Streamlit session_state, so each
uploaded document maintains its own independent index/session.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import config
from pdf_utils import Chunk

# Some open-source models expect an instruction/prefix on the QUERY (and, for
# E5, on passages too) to retrieve well. Skipping these silently hurts quality.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

NOT_FOUND_MESSAGE = (
    "I could not find an answer to that question in the uploaded document. "
    "Try rephrasing, or ask something covered by the document's contents."
)

_NOT_FOUND_SENTINEL = "NO_ANSWER_IN_DOCUMENT"

_ANSWER_SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions strictly using the "
    "provided context extracted from a user's PDF document. Follow these rules:\n"
    "1. Use ONLY the information in the context. Do not use outside knowledge.\n"
    "2. If the answer is not contained in the context, reply with exactly: "
    f"{_NOT_FOUND_SENTINEL}\n"
    "3. Be concise and accurate. Quote figures and names exactly as written.\n"
    "4. When helpful, mention the page number(s) the answer came from."
)


@dataclass
class DocumentIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray  # shape (num_chunks, dim), L2-normalized


@lru_cache(maxsize=1)
def _get_local_model():
    """Load (and cache) the local sentence-transformers embedding model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.get_local_embed_model())


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def _embed_local(texts: list[str], is_query: bool) -> np.ndarray:
    """Embed with a local open-source model, applying model-specific prefixes."""
    name = config.get_local_embed_model().lower()
    inputs = texts
    if "bge" in name and is_query:
        inputs = [_BGE_QUERY_INSTRUCTION + t for t in texts]
    elif "e5" in name:
        prefix = "query: " if is_query else "passage: "
        inputs = [prefix + t for t in texts]

    model = _get_local_model()
    arr = model.encode(
        inputs,
        batch_size=64,
        normalize_embeddings=True,  # cosine-ready
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return arr.astype(np.float32)


def _embed_openai(texts: list[str]) -> np.ndarray:
    client = config.get_client()
    model = config.get_embed_model()
    # Batch to stay within request limits for large documents.
    vectors: list[list[float]] = []
    batch = 96
    for i in range(0, len(texts), batch):
        resp = client.embeddings.create(model=model, input=texts[i : i + batch])
        vectors.extend([d.embedding for d in resp.data])
    return _normalize(np.array(vectors, dtype=np.float32))


def _embed_texts(texts: list[str], is_query: bool = False) -> np.ndarray:
    """Embed texts into an L2-normalized matrix using the configured backend."""
    if config.get_embed_backend() == "local":
        return _embed_local(texts, is_query)
    return _embed_openai(texts)


def build_index(chunks: list[Chunk]) -> DocumentIndex:
    if not chunks:
        raise ValueError(
            "No extractable text found in this PDF. It may be a scanned/image "
            "document that requires OCR."
        )
    embeddings = _embed_texts([c.text for c in chunks])
    return DocumentIndex(chunks=chunks, embeddings=embeddings)


def _retrieve(index: DocumentIndex, query: str, top_k: int = config.TOP_K):
    q = _embed_texts([query], is_query=True)[0]  # already normalized
    scores = index.embeddings @ q  # cosine similarity (both normalized)
    order = np.argsort(-scores)[:top_k]
    return [(index.chunks[i], float(scores[i])) for i in order]


def answer_question(index: DocumentIndex, question: str, history=None) -> dict:
    """Answer a question against the document index.

    Returns dict: {answer, found (bool), sources (list[int] pages), score}
    """
    retrieved = _retrieve(index, question)
    best_score = retrieved[0][1] if retrieved else 0.0

    if best_score < config.SIMILARITY_FLOOR:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "found": False,
            "sources": [],
            "score": best_score,
        }

    context_blocks = []
    for chunk, score in retrieved:
        context_blocks.append(f"[Page {chunk.page}]\n{chunk.text}")
    context = "\n\n---\n\n".join(context_blocks)

    messages = [{"role": "system", "content": _ANSWER_SYSTEM_PROMPT}]

    # Include short recent history for conversational follow-ups.
    if history:
        for turn in history[-config.MAX_HISTORY:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    messages.append(
        {
            "role": "user",
            "content": f"Context from the document:\n\n{context}\n\n"
            f"Question: {question}",
        }
    )

    client = config.get_client()
    resp = client.chat.completions.create(
        model=config.get_chat_model(),
        messages=messages,
        temperature=0.1,
    )
    raw = (resp.choices[0].message.content or "").strip()

    if _NOT_FOUND_SENTINEL in raw:
        return {
            "answer": NOT_FOUND_MESSAGE,
            "found": False,
            "sources": [],
            "score": best_score,
        }

    pages = sorted({chunk.page for chunk, _ in retrieved})
    return {
        "answer": raw,
        "found": True,
        "sources": pages,
        "score": best_score,
    }


def summarize_document(index: DocumentIndex, max_words: int = 130) -> str:
    """Generate a short summary using a sample of the document's chunks."""
    # Use the first several chunks plus a few spread across the document so the
    # summary reflects the whole file without exceeding context limits.
    chunks = index.chunks
    sample_idx = sorted(set(
        list(range(min(6, len(chunks))))
        + list(range(0, len(chunks), max(1, len(chunks) // 8)))
    ))
    sample = "\n\n".join(chunks[i].text for i in sample_idx[:14])

    client = config.get_client()
    resp = client.chat.completions.create(
        model=config.get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": "You summarize documents clearly and factually using "
                "only the provided text.",
            },
            {
                "role": "user",
                "content": f"Summarize the following document in at most "
                f"{max_words} words. Capture its purpose and key points:\n\n{sample}",
            },
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()
