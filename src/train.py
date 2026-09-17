from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .preprocessing import build_classifier, build_vectorizer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "reviews.csv"
MODEL_PATH = ROOT / "models" / "toxicity_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
CATEGORY_COLUMNS = ["insult", "threat", "obscene", "hate"]


def train() -> None:
    data = pd.read_csv(DATA_PATH).dropna(subset=["text", "toxic"])
    train_data, test_data = train_test_split(
        data, test_size=0.25, random_state=42, stratify=data["toxic"]
    )
    vectorizer = build_vectorizer()
    train_features = vectorizer.fit_transform(train_data["text"])
    test_features = vectorizer.transform(test_data["text"])

    binary_model = build_classifier()
    binary_model.fit(train_features, train_data["toxic"])
    predictions = binary_model.predict(test_features)

    print("Toxicity model evaluation")
    print(f"Accuracy:  {accuracy_score(test_data['toxic'], predictions):.3f}")
    print(f"Precision: {precision_score(test_data['toxic'], predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(test_data['toxic'], predictions, zero_division=0):.3f}")
    print(f"F1 score:  {f1_score(test_data['toxic'], predictions, zero_division=0):.3f}")

    category_models = {}
    for category in CATEGORY_COLUMNS:
        model = build_classifier()
        model.fit(train_features, train_data[category])
        category_models[category] = model

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"binary": binary_model, "categories": category_models}, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")


if __name__ == "__main__":
    train()