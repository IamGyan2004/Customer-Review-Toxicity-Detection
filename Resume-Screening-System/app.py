import streamlit as st

from src.preprocessing import detect_skills
from src.recommender import recommend_jobs
from src.resume_parser import extract_text


st.set_page_config(page_title="Resume Screening System", page_icon="Resume")
st.title("Resume Screening & Job Recommendations")
st.write("Upload a resume to rank the most relevant sample jobs.")

uploaded_file = st.file_uploader("Upload resume", type=["pdf", "docx", "txt"])
resume_text = ""
if uploaded_file:
    try:
        resume_text = extract_text(uploaded_file.getvalue(), uploaded_file.name)
        st.success(f"Resume loaded: {uploaded_file.name}")
    except ValueError as error:
        st.error(str(error))

if resume_text:
    skills = detect_skills(resume_text)
    st.subheader("Skills Detected")
    st.write(" • ".join(skills) if skills else "No known skills detected")

    try:
        recommendations = recommend_jobs(resume_text)
    except (OSError, ValueError) as error:
        st.error(str(error))
    else:
        st.subheader("Job Recommendations")
        for index, job in enumerate(recommendations, start=1):
            st.markdown(f"**{index}. {job['title']}** - {job['match_percent']:.1f}% Match")
            st.caption(job["description"])