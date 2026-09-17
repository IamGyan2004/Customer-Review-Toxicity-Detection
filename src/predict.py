from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "toxicity_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"


def predict_review(
    text: str,
    model_path: Path = MODEL_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
) -> dict:
    if not text.strip():
        raise ValueError("Review text cannot be empty.")
    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Model files not found. Run `python -m src.train` first."
        )

    models = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    features = vectorizer.transform([text])
    probability = float(models["binary"].predict_proba(features)[0][1])
    categories = {
        name: float(model.predict_proba(features)[0][1])
        for name, model in models["categories"].items()
    }
    return {
        "label": "Toxic" if probability >= 0.5 else "Non-Toxic",
        "toxicity_probability": probability,
        "categories": categories,
    }