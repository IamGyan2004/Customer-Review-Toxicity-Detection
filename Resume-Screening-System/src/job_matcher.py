from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_text


ROOT = Path(__file__).resolve().parents[1]
VECTOR_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
JOBS_PATH = ROOT / "dataset" / "job_descriptions.csv"


def load_jobs(path: Path = JOBS_PATH) -> pd.DataFrame:
    jobs = pd.read_csv(path).fillna("")
    required = {"title", "description", "skills"}
    missing = required - set(jobs.columns)
    if missing:
        raise ValueError(f"Job dataset is missing columns: {', '.join(sorted(missing))}")
    return jobs


def get_vectorizer(jobs: pd.DataFrame, vector_path: Path = VECTOR_PATH) -> TfidfVectorizer:
    if vector_path.exists():
        return joblib.load(vector_path)
    vectorizer = TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2), sublinear_tf=True)
    vectorizer.fit(jobs["description"])
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vector_path)
    return vectorizer


def calculate_matches(resume_text: str, jobs: pd.DataFrame, vectorizer: TfidfVectorizer) -> list[dict]:
    job_features = vectorizer.transform(jobs["description"])
    resume_features = vectorizer.transform([resume_text])
    scores = cosine_similarity(resume_features, job_features)[0]
    matches = []
    for job, score in zip(jobs.to_dict("records"), scores):
        matches.append({**job, "match_percent": round(float(score) * 100, 1)})
    return sorted(matches, key=lambda match: match["match_percent"], reverse=True)