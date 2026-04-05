"""
summarizer.py
-------------
Extractive text summarisation for the ContextManager.

Approach — TextRank-lite (no heavy ML dependency required):
  1. Split text into sentences.
  2. Build a TF-IDF weighted similarity graph between sentences.
  3. Run a simplified PageRank / degree-centrality pass.
  4. Return the top-N highest-scoring sentences in *original order*
     so the summary reads naturally.

Optional speedup: if `scikit-learn` is installed the TF-IDF step
uses its battle-tested vectoriser; otherwise a pure-stdlib fallback
is used automatically.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

# ── Optional sklearn import ─────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer as _TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    import numpy as _np
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ── Sentence splitter ────────────────────────────────────────────────────────

_SENTENCE_RE = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
)


def _split_sentences(text: str) -> list[str]:
    """Split *text* into individual sentences."""
    sentences = _SENTENCE_RE.split(text.strip())
    # filter blanks and very short fragments (< 4 words)
    return [s.strip() for s in sentences if len(s.split()) >= 4]


# ── TF-IDF helpers (stdlib fallback) ────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Very simple whitespace + lower tokeniser."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


def _build_tfidf_matrix_stdlib(sentences: list[str]) -> list[dict[str, float]]:
    """
    Return a list of TF-IDF weight dicts, one per sentence.
    Pure-Python implementation — O(n*vocab).
    """
    tokenized = [_tokenize(s) for s in sentences]
    n = len(tokenized)

    # term frequency per sentence
    tf: list[dict[str, float]] = []
    for tokens in tokenized:
        freq: dict[str, float] = defaultdict(float)
        for t in tokens:
            freq[t] += 1.0
        total = max(len(tokens), 1)
        tf.append({k: v / total for k, v in freq.items()})

    # inverse document frequency
    df: dict[str, int] = defaultdict(int)
    for freq in tf:
        for term in freq:
            df[term] += 1

    idf = {term: math.log((n + 1) / (count + 1)) + 1.0
           for term, count in df.items()}

    # multiply TF × IDF
    tfidf = [{term: weight * idf[term] for term, weight in freq.items()}
             for freq in tf]
    return tfidf


def _cosine_stdlib(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two TF-IDF weight dicts."""
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot    = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Graph-based scoring ──────────────────────────────────────────────────────

def _score_sentences_sklearn(sentences: list[str]) -> list[float]:
    """Score sentences using sklearn TF-IDF + cosine similarity graph."""
    vec = _TfidfVectorizer(stop_words="english", min_df=1)
    try:
        tfidf_matrix = vec.fit_transform(sentences)
    except ValueError:
        # all sentences are stop-words only — score equally
        return [1.0] * len(sentences)

    sim_matrix = _cosine_similarity(tfidf_matrix, tfidf_matrix)

    # degree centrality: sum of edge weights for each node
    scores = sim_matrix.sum(axis=1).tolist()
    # flatten if it came back as a matrix
    return [float(s[0]) if hasattr(s, '__len__') else float(s) for s in scores]


def _score_sentences_stdlib(sentences: list[str]) -> list[float]:
    """Score sentences using pure-Python TF-IDF + cosine similarity graph."""
    tfidf = _build_tfidf_matrix_stdlib(sentences)
    n     = len(tfidf)
    scores = [0.0] * n

    for i in range(n):
        for j in range(n):
            if i != j:
                scores[i] += _cosine_stdlib(tfidf[i], tfidf[j])

    return scores


# ── Public API ───────────────────────────────────────────────────────────────

def summarize(
    text: str,
    sentence_count: int = 3,
    *,
    min_sentence_length: int = 6,
) -> str:
    """
    Return an extractive summary of *text* containing up to
    *sentence_count* of the most informative sentences.

    Parameters
    ----------
    text:
        The source text to summarise.  Can be multi-paragraph.
    sentence_count:
        How many sentences to include in the summary.
        Clamped to the actual number of sentences in the text.
    min_sentence_length:
        Sentences with fewer words than this are ignored as fragments.

    Returns
    -------
    str
        Summary string.  Returns *text* unchanged if it is too short
        to summarise meaningfully.

    Examples
    --------
    >>> summary = summarize(long_conversation_text, sentence_count=4)
    """
    if not text or not text.strip():
        return ""

    sentences = _split_sentences(text)
    # extra length filter
    sentences = [s for s in sentences if len(s.split()) >= min_sentence_length]

    if not sentences:
        return text.strip()

    # clamp requested count
    k = max(1, min(sentence_count, len(sentences)))

    if len(sentences) <= k:
        return " ".join(sentences)

    # score
    if _SKLEARN_AVAILABLE:
        scores = _score_sentences_sklearn(sentences)
    else:
        scores = _score_sentences_stdlib(sentences)

    # pick top-k indices, then sort by original position for readability
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top_indices = sorted(ranked[:k])

    summary = " ".join(sentences[i] for i in top_indices)
    return summary


def summarize_messages(
    messages: list[dict],
    sentence_count: int = 3,
) -> str:
    """
    Convenience wrapper: concatenate message contents and summarise.

    Parameters
    ----------
    messages:
        List of ``{"role": str, "content": str}`` dicts.
    sentence_count:
        Target sentence count for the summary.

    Returns
    -------
    str
        Summary of the combined message content.
    """
    if not messages:
        return ""
    combined = " ".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in messages
    )
    return summarize(combined, sentence_count=sentence_count)