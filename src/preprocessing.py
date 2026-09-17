import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text: str) -> str:
    """Normalize review text while preserving words useful for classification."""
    return re.sub(r"[^a-z0-9\s']", " ", str(text).lower()).strip()


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def build_classifier() -> LogisticRegression:
    return LogisticRegression(max_iter=1000, class_weight="balanced")