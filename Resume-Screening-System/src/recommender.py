from .job_matcher import calculate_matches, get_vectorizer, load_jobs


def recommend_jobs(resume_text: str, limit: int = 5) -> list[dict]:
    """Return the highest-scoring job descriptions for a resume."""
    if not resume_text.strip():
        raise ValueError("Resume text cannot be empty.")
    jobs = load_jobs()
    vectorizer = get_vectorizer(jobs)
    return calculate_matches(resume_text, jobs, vectorizer)[:limit]