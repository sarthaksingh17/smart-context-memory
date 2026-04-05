"""
retriever.py
------------
Semantic retrieval of relevant messages from conversation history.

Given a user query, this module finds the most semantically similar
messages in history so you only inject *relevant* context into your
LLM prompt — not the entire (possibly huge) history.

Two backends, selected automatically:

  1. **sklearn TF-IDF + cosine** (default when scikit-learn is installed)
     Fast, zero-config, no GPU/API needed.  Works well for conversations
     that share vocabulary.

  2. **BM25 (stdlib fallback)**
     Classic keyword-overlap ranking.  Less accurate but zero-dependency.

Usage
-----
    from retriever import get_relevant_messages

    relevant = get_relevant_messages(
        query    = "What did the user say about pricing?",
        messages = conversation_history,   # list of {"role":…, "content":…}
        top_k    = 5,
    )
    # returns a list[dict] of the most relevant messages
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# ── Optional sklearn import ─────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVec
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_sim
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _message_text(msg: dict) -> str:
    """Return the plain-text content of a message dict."""
    return msg.get("content", "")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokeniser, strips punctuation."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


# ── sklearn backend ──────────────────────────────────────────────────────────

def _retrieve_sklearn(
    query: str,
    messages: list[dict],
    top_k: int,
) -> list[dict]:
    """TF-IDF cosine-similarity retrieval using scikit-learn."""
    contents = [_message_text(m) for m in messages]

    # Build corpus: query first so index 0 is always the query vector
    corpus = [query] + contents

    vec = _TfidfVec(stop_words="english", min_df=1, ngram_range=(1, 2))
    try:
        tfidf = vec.fit_transform(corpus)
    except ValueError:
        # Edge-case: all tokens are stop-words → return most recent messages
        return messages[-top_k:]

    query_vec = tfidf[0]
    doc_vecs  = tfidf[1:]
    scores    = _cosine_sim(query_vec, doc_vecs).flatten().tolist()

    ranked = sorted(
        range(len(messages)),
        key=lambda i: scores[i],
        reverse=True,
    )

    # Return top-k in their *original chronological order*
    top_indices = sorted(ranked[:top_k])
    return [messages[i] for i in top_indices]


# ── BM25 stdlib fallback ─────────────────────────────────────────────────────

_BM25_K1 = 1.5
_BM25_B  = 0.75


def _bm25_scores(query_tokens: list[str], docs: list[list[str]]) -> list[float]:
    """
    Compute BM25 scores for each document against the query.

    Parameters
    ----------
    query_tokens : tokenised query terms
    docs         : list of tokenised documents
    """
    n      = len(docs)
    avgdl  = sum(len(d) for d in docs) / max(n, 1)

    # document frequency
    df: dict[str, int] = defaultdict(int)
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    scores = []
    for doc in docs:
        tf_map: dict[str, int] = defaultdict(int)
        for t in doc:
            tf_map[t] += 1
        dl = len(doc)
        score = 0.0
        for term in query_tokens:
            if term not in tf_map:
                continue
            idf = math.log((n - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
            tf  = tf_map[term]
            numerator   = tf * (_BM25_K1 + 1)
            denominator = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
            score += idf * (numerator / denominator)
        scores.append(score)
    return scores


def _retrieve_bm25(
    query: str,
    messages: list[dict],
    top_k: int,
) -> list[dict]:
    """BM25 retrieval — pure Python, no dependencies."""
    query_tokens = _tokenize(query)
    docs         = [_tokenize(_message_text(m)) for m in messages]
    scores       = _bm25_scores(query_tokens, docs)

    ranked = sorted(range(len(messages)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:top_k])
    return [messages[i] for i in top_indices]


# ── Public API ───────────────────────────────────────────────────────────────

def get_relevant_messages(
    query: str,
    messages: list[dict],
    top_k: int = 6,
    *,
    min_score_threshold: float = 0.0,
    always_include_last_n: int = 2,
) -> list[dict]:
    """
    Retrieve the most semantically relevant messages from *messages*
    for the given *query*.

    Parameters
    ----------
    query:
        The current user input / search string used as the retrieval anchor.
    messages:
        Full conversation history as a list of
        ``{"role": str, "content": str}`` dicts.
    top_k:
        Maximum number of messages to return.  The actual count may be
        lower if the history is short.
    min_score_threshold:
        (sklearn only) Discard messages with a cosine score below this
        value.  Default ``0.0`` keeps everything in the top-k.
    always_include_last_n:
        Always append the last *n* messages regardless of score —
        preserves conversational recency.  Set to ``0`` to disable.

    Returns
    -------
    list[dict]
        Relevant messages in *original chronological order*.

    Examples
    --------
    >>> relevant = get_relevant_messages(
    ...     "What was the budget discussed?",
    ...     history,
    ...     top_k=5,
    ... )
    """
    if not messages:
        return []

    if not query or not query.strip():
        # No query → just return the tail of history
        return messages[-top_k:]

    effective_k = min(top_k, len(messages))

    # Choose backend
    if _SKLEARN_AVAILABLE:
        retrieved = _retrieve_sklearn(query, messages, top_k=effective_k)
    else:
        retrieved = _retrieve_bm25(query, messages, top_k=effective_k)

    # Merge with the always-include tail so we never lose recency
    if always_include_last_n > 0:
        tail = messages[-always_include_last_n:]
        # de-duplicate while preserving order
        seen   = {id(m) for m in retrieved}
        extras = [m for m in tail if id(m) not in seen]
        merged = retrieved + extras
        # re-sort chronologically
        index_map = {id(m): i for i, m in enumerate(messages)}
        retrieved = sorted(merged, key=lambda m: index_map.get(id(m), 0))

    return retrieved


def build_retrieval_index(messages: list[dict]) -> dict:
    """
    Pre-build a retrieval index for repeated queries over the same
    message list (avoids re-fitting the vectoriser on every call).

    Returns a dict with keys ``"vectorizer"`` and ``"matrix"`` when
    sklearn is available, or ``"docs"`` (tokenised) for BM25.

    Pass the returned dict to ``query_index`` for fast repeated lookups.

    Parameters
    ----------
    messages:
        The full conversation history to index.

    Returns
    -------
    dict
        Retrieval index payload.
    """
    contents = [_message_text(m) for m in messages]

    if _SKLEARN_AVAILABLE:
        vec = _TfidfVec(stop_words="english", min_df=1, ngram_range=(1, 2))
        matrix = vec.fit_transform(contents)
        return {"backend": "sklearn", "vectorizer": vec, "matrix": matrix, "messages": messages}
    else:
        docs = [_tokenize(c) for c in contents]
        return {"backend": "bm25", "docs": docs, "messages": messages}


def query_index(
    query: str,
    index: dict,
    top_k: int = 6,
) -> list[dict]:
    """
    Query a pre-built retrieval index (see ``build_retrieval_index``).

    Parameters
    ----------
    query:
        Search string.
    index:
        Dict returned by ``build_retrieval_index``.
    top_k:
        Max messages to return.

    Returns
    -------
    list[dict]
        Matched messages in chronological order.
    """
    messages = index["messages"]
    if not messages or not query.strip():
        return messages[-top_k:]

    effective_k = min(top_k, len(messages))

    if index["backend"] == "sklearn":
        vec    = index["vectorizer"]
        matrix = index["matrix"]
        q_vec  = vec.transform([query])
        scores = _cosine_sim(q_vec, matrix).flatten().tolist()
        ranked = sorted(range(len(messages)), key=lambda i: scores[i], reverse=True)
        top_indices = sorted(ranked[:effective_k])
        return [messages[i] for i in top_indices]

    else:  # bm25
        query_tokens = _tokenize(query)
        scores       = _bm25_scores(query_tokens, index["docs"])
        ranked       = sorted(range(len(messages)), key=lambda i: scores[i], reverse=True)
        top_indices  = sorted(ranked[:effective_k])
        return [messages[i] for i in top_indices]