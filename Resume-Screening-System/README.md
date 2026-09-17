# Resume Screening & Job Recommendation System

An NLP resume screening project that extracts resume text, detects common
skills, compares the resume with job descriptions using TF-IDF and cosine
similarity, and ranks recommended jobs in a Streamlit interface.

## Structure

```text
dataset/resumes.csv
dataset/job_descriptions.csv
models/tfidf_vectorizer.pkl
src/resume_parser.py
src/preprocessing.py
src/job_matcher.py
src/recommender.py
data/sample_jobs.json
app.py
```

## Run locally

```bash
cd Resume-Screening-System
python -m pip install -r requirements.txt
streamlit run app.py
```

Upload a PDF, DOCX, or TXT resume. The app extracts its text, shows detected
skills, and ranks the sample jobs by match percentage. The TF-IDF vectorizer is
created automatically on first use and saved to `models/tfidf_vectorizer.pkl`.

This repository contains a small demonstration dataset. Replace it with
carefully prepared job and resume data before using the system for real hiring
decisions. Similarity scores are relevance signals, not fair or complete
candidate evaluations.