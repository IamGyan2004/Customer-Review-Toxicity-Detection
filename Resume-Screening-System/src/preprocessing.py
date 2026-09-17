import re


COMMON_SKILLS = [
    "Python", "SQL", "pandas", "NumPy", "scikit-learn", "machine learning",
    "deep learning", "TensorFlow", "PyTorch", "NLP", "text classification",
    "Docker", "Git", "React", "JavaScript", "TypeScript", "HTML", "CSS",
    "REST APIs", "databases", "testing", "Tableau", "statistics", "Excel",
]


def clean_text(text: str) -> str:
    """Lowercase text and keep alphanumeric tokens used by the vectorizer."""
    return re.sub(r"[^a-z0-9+#.\s]", " ", str(text).lower()).strip()


def detect_skills(text: str, skills: list[str] | None = None) -> list[str]:
    """Find known skills in a resume using case-insensitive word boundaries."""
    detected = []
    normalized_text = clean_text(text)
    for skill in skills or COMMON_SKILLS:
        normalized_skill = clean_text(skill)
        if re.search(rf"(?<!\w){re.escape(normalized_skill)}(?!\w)", normalized_text):
            detected.append(skill)
    return detected