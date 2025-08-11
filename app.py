import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from io import BytesIO

# ----------- Skill Keywords (Edit as Needed) -----------
TECHNICAL_SKILLS = ["python", "r", "sql", "pyspark", "spark", "hadoop", "nlp", "tensorflow", "deep learning"]
TOOLS = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "power bi", "tableau"]
SOFT_SKILLS = ["communication", "leadership", "teamwork", "problem solving", "critical thinking"]

# ----------- PDF Text Extractor -----------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.lower()

# ----------- Experience Extractor -----------
def extract_experience(text):
    match = re.search(r'(\d+)\+?\s+years?', text)
    return int(match.group(1)) if match else 0

# ----------- Skill Counter -----------
def count_skills(text, skill_list):
    return [skill for skill in skill_list if skill in text]

# ----------- ATS Score Calculator -----------
def calculate_ats_score(text):
    all_keywords = TECHNICAL_SKILLS + TOOLS + SOFT_SKILLS
    found = sum(1 for skill in all_keywords if skill in text)
    return round((found / len(all_keywords)) * 100, 2)

# ----------- CV Analyzer -----------
def analyze_resume(file):
    text = extract_text_from_pdf(file)

    tech_found = count_skills(text, TECHNICAL_SKILLS)
    tools_found = count_skills(text, TOOLS)
    soft_found = count_skills(text, SOFT_SKILLS)
    experience_years = extract_experience(text)
    ats_score = calculate_ats_score(text)

    suggestions = []
    if experience_years == 0:
        suggestions.append("Add work experience details with years.")
    if ats_score < 80:
        suggestions.append("Add more job-relevant keywords.")
    if len(text.split()) < 300:
        suggestions.append("Add more details to make resume more comprehensive.")

    return {
        "filename": file.name,
        "skills_count": len(tech_found) + len(tools_found) + len(soft_found),
        "experience_years_est": experience_years,
        "ats_score": ats_score,
        "technical_skills": tech_found,
        "tools": tools_found,
        "soft_skills": soft_found,
        "suggestions": suggestions
    }

# ----------- Graphs -----------
def plot_graphs(df):
    df['experience_years_est'] = pd.to_numeric(df['experience_years_est'], errors='coerce').fillna(0)

    # Skills (Bar) + Experience (Line)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))

    ax1.bar(x, df['skills_count'], width=0.4, color='blue', label='Skills Count')
    ax1.set_ylabel("Skills Count", color='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['filename'], rotation=45, ha="right")
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(x, df['experience_years_est'], color='orange', marker='o', linewidth=2, label='Experience (yrs)')
    ax2.set_ylabel("Experience (Years)", color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle("Skills vs Experience Comparison", fontsize=14)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1))
    st.pyplot(fig)

    # ATS Score Comparison
    fig2, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(df['filename'], df['ats_score'], color='green')
    ax3.set_ylabel("ATS Score (%)")
    ax3.set_title("ATS Score Comparison")
    st.pyplot(fig2)

    # Pie chart for skills distribution (sum of all skills)
    total_skills = df['skills_count'].sum()
    if total_skills > 0:
        labels = df['filename']
        sizes = df['skills_count']
        fig3, ax4 = plt.subplots()
        ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax4.set_title("Skills Distribution Across Resumes")
        st.pyplot(fig3)

# ----------- Streamlit App -----------
st.set_page_config(page_title="SmartCV Analyzer Pro", layout="wide")

st.title("📄 SmartCV Analyzer Pro")
st.write("Upload one or multiple resumes in PDF format to analyze skills, ATS score, experience, and improvement suggestions.")

uploaded_files = st.file_uploader("Upload Resume(s)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    for file in uploaded_files:
        results.append(analyze_resume(file))

    df = pd.DataFrame(results)

    # Best CV
    best_cv = df.loc[df['ats_score'].idxmax()]

    st.subheader("📊 Resume Analysis Results")
    st.dataframe(df)

    plot_graphs(df)

    st.subheader("🏆 Best Resume")
    st.write(f"**{best_cv['filename']}** with ATS Score: **{best_cv['ats_score']}%**")
    st.write("Suggestions:", ", ".join(best_cv['suggestions']) if best_cv['suggestions'] else "Looks perfect!")

    st.subheader("💡 Suggestions for All Resumes")
    for i, row in df.iterrows():
        st.markdown(f"**{row['filename']}**: " + (", ".join(row['suggestions']) if row['suggestions'] else "No major improvements needed."))
