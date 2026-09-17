# Customer Review Toxicity Detection

A small machine-learning app that classifies customer reviews as toxic or
non-toxic and estimates insult, threat, obscene, and hate categories.

## Setup

```bash
python -m pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

Training prints accuracy, precision, recall, and F1 score, then writes the
classifier to `models/toxicity_model.pkl` and the shared TF-IDF vectorizer to
`models/tfidf_vectorizer.pkl`. The included CSV is a small starter dataset for
demonstration; replace it with a larger, consistently annotated dataset for
production use.

## Structure

```text
dataset/reviews.csv       Training examples and labels
models/toxicity_model.pkl Saved classifier artifact
models/tfidf_vectorizer.pkl Saved TF-IDF vocabulary and weights
src/preprocessing.py      Text cleaning and model pipeline helpers
src/train.py              Training and evaluation script
src/predict.py            Reusable prediction function
app.py                    Streamlit interface
```