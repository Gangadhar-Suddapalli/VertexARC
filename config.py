"""Central configuration and chat client factory.

Reads credentials from (in order of precedence):
  1. Streamlit secrets  (st.secrets) - used on Streamlit Community Cloud
  2. Environment variables / .env     - used for local development

The chat/summary model is served through any OpenAI-compatible API. The default
provider is Groq (free tier, https://console.groq.com); set LLM_PROVIDER=openai
to use OpenAI instead. Both use the same `openai` Python client, just a different
base URL + API key.
"""
from __future__ import annotations

import os
from functools import lru_cache

# Load a local .env file (if present) into the environment for local dev.
# `override=False` keeps any explicitly-set shell variables authoritative.
try:
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:
    pass

# Chat (LLM) provider: "groq" (free, default) or "openai".
DEFAULT_LLM_PROVIDER = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Per-provider default chat models (override via CHAT_MODEL env var/secret).
DEFAULT_GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"

# Embedding backend: "local" (open-source via sentence-transformers, no API
# cost) or "openai" (text-embedding-3-small). Embeddings are independent of the
# chat provider, so local embeddings + Groq chat is the fully-free combination.
DEFAULT_EMBED_BACKEND = "local"
DEFAULT_LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"  # used only when EMBED_BACKEND=openai

# Retrieval / chunking parameters
CHUNK_SIZE = 1000          # characters per chunk (approx.)
CHUNK_OVERLAP = 150        # character overlap between consecutive chunks
TOP_K = 4                  # number of chunks retrieved per question
MAX_HISTORY = 5            # number of recent Q&A pairs displayed/used
SIMILARITY_FLOOR = 0.20    # below this best-match score, treat as "not in document"


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Look up a setting from Streamlit secrets first, then environment."""
    # Streamlit secrets (only available when running inside Streamlit)
    try:
        import streamlit as st  # local import keeps non-Streamlit use clean

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


def get_llm_provider() -> str:
    """Return the active chat provider: 'groq' or 'openai'."""
    provider = _get_setting("LLM_PROVIDER", DEFAULT_LLM_PROVIDER) or DEFAULT_LLM_PROVIDER
    return provider.strip().lower()


def get_api_key() -> str | None:
    """Return the API key for the active chat provider."""
    if get_llm_provider() == "groq":
        return _get_setting("GROQ_API_KEY")
    return _get_setting("OPENAI_API_KEY")


def get_chat_model() -> str:
    default = (
        DEFAULT_GROQ_CHAT_MODEL
        if get_llm_provider() == "groq"
        else DEFAULT_OPENAI_CHAT_MODEL
    )
    return _get_setting("CHAT_MODEL", default) or default


def get_embed_model() -> str:
    return _get_setting("OPENAI_EMBED_MODEL", DEFAULT_EMBED_MODEL) or DEFAULT_EMBED_MODEL


def get_embed_backend() -> str:
    """Return the active embedding backend: 'local' or 'openai'."""
    backend = _get_setting("EMBED_BACKEND", DEFAULT_EMBED_BACKEND) or DEFAULT_EMBED_BACKEND
    return backend.strip().lower()


def get_local_embed_model() -> str:
    return _get_setting("LOCAL_EMBED_MODEL", DEFAULT_LOCAL_EMBED_MODEL) or DEFAULT_LOCAL_EMBED_MODEL


@lru_cache(maxsize=1)
def get_client():
    """Return a cached OpenAI-compatible client for the active chat provider.

    Raises RuntimeError if no key is set.
    """
    from openai import OpenAI

    provider = get_llm_provider()
    key = get_api_key()
    if not key:
        key_name = "GROQ_API_KEY" if provider == "groq" else "OPENAI_API_KEY"
        raise RuntimeError(
            f"{key_name} is not set. Add it to your .env file (local) "
            "or to the app's Secrets (Streamlit Cloud)."
        )
    if provider == "groq":
        return OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    return OpenAI(api_key=key)
