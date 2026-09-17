# Customer Review Toxicity Detection - NLP

A machine-learning application that classifies customer reviews as **Toxic**
or **Non-Toxic**. It also estimates optional categories: insult, threat,
obscene language, and hate speech.

## Workflow

```text
Customer Review
	|
	v
Text Preprocessing
	|
	v
TF-IDF Vectorizer
	|
	v
Logistic Regression Classification
	|
	+--> Toxic / Non-Toxic
	|
	v
Toxicity Probability and Category Probabilities
	|
	v
Streamlit UI
```

## Project Structure

```text
Customer-Review-Toxicity-Detection/
|
|-- dataset/
|   `-- reviews.csv
|
|-- models/
|   |-- toxicity_model.pkl
|   `-- tfidf_vectorizer.pkl
|
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- train.py
|   `-- predict.py
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- LICENSE
```

### File Purpose

| File or folder | Purpose |
|---|---|
| `dataset/reviews.csv` | Customer reviews and toxicity/category labels |
| `models/toxicity_model.pkl` | Trained toxicity and category classifiers |
| `models/tfidf_vectorizer.pkl` | Saved TF-IDF vocabulary and weights |
| `src/preprocessing.py` | Text cleaning and model setup |
| `src/train.py` | Training and accuracy/F1 evaluation |
| `src/predict.py` | Load artifacts and make predictions |
| `app.py` | Streamlit web interface |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Ignore generated files and environments |
| `LICENSE` | MIT open-source license |

## Setup

```bash
python -m pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

The training command prints accuracy, precision, recall, and F1 score. It then
saves the classifier and vectorizer in the `models/` directory.

## Example

Input:

```text
This product is terrible and the seller is an idiot.
```

Output:

```text
Label: Toxic
Toxicity probability: 60.3%
Categories: insult, threat, obscene, hate
```

The included CSV is a small starter dataset for demonstration. Replace it with
a larger, consistently annotated dataset for production use. Model
probabilities should be treated as assistance signals, not final moderation
decisions.