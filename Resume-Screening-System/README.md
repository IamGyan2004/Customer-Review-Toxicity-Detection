# Resume Screening & Job Recommendation System

An NLP-based resume screening application that extracts text from resumes,
detects skills, compares candidates with job descriptions, and recommends the
most relevant roles using TF-IDF and cosine similarity.

## Project Overview

Recruiters often need to review many resumes against several job descriptions.
This project provides a lightweight screening workflow:

1. Upload a resume in PDF, DOCX, or TXT format.
2. Extract and clean the resume text.
3. Detect common technical skills.
4. Convert the resume and job descriptions into TF-IDF vectors.
5. Calculate cosine similarity between the resume and each job.
6. Rank jobs by match percentage.
7. Display recommendations in a Streamlit web application.

## Workflow

```text
Resume Upload
	|
	v
PDF / DOCX / TXT Text Extraction
	|
	v
Text Cleaning and Skill Detection
	|
	v
TF-IDF Vectorization
	|
	v
Cosine Similarity Matching
	|
	v
Job Ranking and Match Percentage
	|
	v
Streamlit Recommendations
```

## Features

- Upload resumes in PDF, DOCX, and TXT formats
- Extract text from uploaded documents
- Detect common skills such as Python, SQL, React, NLP, and Docker
- Compare resumes with multiple job descriptions
- Calculate a similarity-based match percentage
- Rank and display recommended jobs
- Save and reuse the TF-IDF vectorizer
- Simple Streamlit interface suitable for a portfolio project

## Project Structure

```text
Resume-Screening-System/
|
|-- dataset/
|   |-- resumes.csv
|   `-- job_descriptions.csv
|
|-- models/
|   `-- tfidf_vectorizer.pkl
|
|-- src/
|   |-- __init__.py
|   |-- preprocessing.py
|   |-- resume_parser.py
|   |-- job_matcher.py
|   `-- recommender.py
|
|-- data/
|   `-- sample_jobs.json
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
`-- LICENSE
```

## Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Streamlit
- PyPDF for PDF extraction
- python-docx for DOCX extraction
- Joblib for saving the vectorizer

## Installation

From the project directory, run:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

If that port is already in use, run the application on another port:

```bash
streamlit run app.py --server.port 8504
```

## Example Result

For a resume containing Python, SQL, pandas, statistics, and Tableau, the app
may produce a ranked result such as:

```text
Skills Detected:
Python | SQL | pandas | statistics | Tableau

Job Recommendations:
1. Data Analyst       82.4% Match
2. ML Engineer        41.7% Match
3. Python Developer   25.9% Match
```

The exact percentages depend on the job descriptions and vocabulary in the
dataset.

## Dataset Format

`dataset/job_descriptions.csv` expects these columns:

```text
job_id,title,description,skills
```

`dataset/resumes.csv` contains example resume text:

```text
resume_id,resume_text
```

To add jobs, append rows to `job_descriptions.csv`. The `description` column is
used for similarity matching and `skills` stores a readable list of expected
skills separated by `|`.

## How Matching Works

The application represents each job description and the uploaded resume as a
TF-IDF vector. It then uses cosine similarity to compare the resume with every
job description:

```text
match percentage = cosine similarity x 100
```

The vectorizer is created on the first recommendation request and saved at:

```text
models/tfidf_vectorizer.pkl
```

## Main Modules

- `resume_parser.py`: Extracts text from PDF, DOCX, and TXT files.
- `preprocessing.py`: Cleans text and detects known skills.
- `job_matcher.py`: Loads jobs, manages TF-IDF, and calculates similarity.
- `recommender.py`: Provides the ranked recommendation function.
- `app.py`: Provides the Streamlit frontend.

## Limitations and Responsible Use

This is an educational portfolio project using a small demonstration dataset.
Similarity scores are relevance signals, not hiring decisions. Real-world use
would require a larger and representative dataset, stronger skill extraction,
careful validation, privacy controls, and fairness testing. Do not use the
system as the sole basis for accepting or rejecting candidates.

## Future Improvements

- Add a database of live job postings
- Support more resume formats
- Use named entity recognition for richer skill extraction
- Add semantic embeddings with a transformer model
- Add recruiter filters for location, experience, and work authorization
- Add evaluation metrics using labeled resume-job pairs
- Add authentication and secure document deletion

## License

This project is available under the MIT License. See [LICENSE](LICENSE).