# app.py
import streamlit as st
import fitz
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os

# ===============================
# 📌 Helper Functions
# ===============================
def extract_text_from_pdf(file):
    text = ""
    pdf_doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in pdf_doc:
        text += page.get_text()
    return text

def analyze_resume(text):
    text_lower = text.lower()

    # Education
    education_keywords = ["bachelor", "master", "ms", "phd", "bs", "mba"]
    education = [word for word in education_keywords if word in text_lower]

    # Experience Years
    exp_pattern = re.findall(r'(\d+)\s+year', text_lower)
    exp_years = int(exp_pattern[0]) if exp_pattern else 0

    # Skills
    technical_skills = ["python", "r", "sql", "pyspark", "spark", "hadoop", "nlp", "tensorflow", "java", "c++"]
    tools = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "power bi", "tableau", "excel"]
    soft_skills = ["communication", "teamwork", "leadership", "problem-solving", "critical thinking"]

    found_tech = [skill for skill in technical_skills if skill in text_lower]
    found_tools = [tool for tool in tools if tool in text_lower]
    found_soft = [skill for skill in soft_skills if skill in text_lower]

    # ATS Score
    ats_keywords = technical_skills + tools + soft_skills
    ats_score = int((sum(1 for word in ats_keywords if word in text_lower) / len(ats_keywords)) * 100)

    # Word count
    word_count = len(text.split())

    # Final Score for Ranking
    final_score = ats_score + (len(found_tech) * 2) + (exp_years * 3)

    return {
        "education": education,
        "experience_years_est": exp_years,
        "skills_count": len(found_tech + found_tools + found_soft),
        "technical_skills": found_tech,
        "tools": found_tools,
        "soft_skills": found_soft,
        "ats_score": ats_score,
        "word_count": word_count,
        "final_score": final_score
    }

def analyze_multiple_resumes(uploaded_files):
    all_data = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        analysis = analyze_resume(text)
        analysis["filename"] = file.name
        all_data.append(analysis)
    return pd.DataFrame(all_data)

def plot_graphs(df):
    df['experience_years_est'] = pd.to_numeric(df['experience_years_est'], errors='coerce').fillna(0)

    # Bar Chart: Skills vs Experience
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df))
    ax1.bar(x - 0.2, df['skills_count'], width=0.4, label='Skills Count', color='blue')
    ax1.bar(x + 0.2, df['experience_years_est'], width=0.4, label='Experience (yrs)', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['filename'], rotation=45)
    ax1.set_ylabel("Count / Years")
    ax1.set_title("Skills vs Experience Comparison")
    ax1.legend()

    # ATS Score Chart
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(df['filename'], df['ats_score'], color='green')
    ax2.set_ylabel("ATS Score (%)")
    ax2.set_title("ATS Score Comparison")

    # Pie Chart: Skills Breakdown of Best Resume
    best_resume = df.loc[df['final_score'].idxmax()]
    labels = ['Technical Skills', 'Tools', 'Soft Skills']
    sizes = [len(best_resume['technical_skills']), len(best_resume['tools']), len(best_resume['soft_skills'])]
    colors = ['#ff9999','#66b3ff','#99ff99']
    fig3, ax3 = plt.subplots(figsize=(4,4))
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax3.set_title(f"Skills Breakdown - {best_resume['filename']}")

    return fig1, fig2, fig3

# ===============================
# 📌 Streamlit UI
# ===============================
st.set_page_config(page_title="SmartCV Analyzer", layout="wide")
st.title("📄 SmartCV Analyzer")
st.write("Upload one or multiple resumes in PDF format to get a detailed ATS-friendly analysis and ranking.")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    df = analyze_multiple_resumes(uploaded_files)
    st.subheader("📊 Resume Analysis Results")
    st.dataframe(df)

    # Ranking
    df_sorted = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
    best_resume = df_sorted.iloc[0]

    st.subheader("🏆 Best Resume")
    st.write(f"**File:** {best_resume['filename']}")
    st.write(f"**Final Score:** {best_resume['final_score']}")
    st.write(f"**ATS Score:** {best_resume['ats_score']}%")
    st.write(f"**Experience:** {best_resume['experience_years_est']} years")
    st.write(f"**Skills Found:** {best_resume['technical_skills'] + best_resume['tools'] + best_resume['soft_skills']}")

    st.subheader("❌ Weak Resumes & Reasons")
    for idx, row in df_sorted.iloc[1:].iterrows():
        reasons = []
        if row['ats_score'] < 80:
            reasons.append("Low ATS Score")
        if row['experience_years_est'] < best_resume['experience_years_est']:
            reasons.append("Less Experience")
        if row['skills_count'] < best_resume['skills_count']:
            reasons.append("Fewer Skills")
        st.write(f"- **{row['filename']}**: {', '.join(reasons) if reasons else 'Other issues'}")

    # Detailed Suggestions
    st.subheader("📄 Detailed Suggestions")
    for idx, row in df.iterrows():
        st.markdown(f"### {row['filename']}")
        st.write(f"🎓 Education: {', '.join(row['education']) if row['education'] else 'Not mentioned'}")
        st.write(f"💼 Estimated Experience: {row['experience_years_est']} years")
        st.write(f"🛠 Skills Found: {row['technical_skills'] + row['tools'] + row['soft_skills']}")
        st.write(f"📈 ATS Score: {row['ats_score']}%")
        st.write(f"📑 Word Count: {row['word_count']}")
        suggestions = []
        if row['experience_years_est'] == 0:
            suggestions.append("Add clear years of experience.")
        if row['ats_score'] < 80:
            suggestions.append("Include more job-relevant keywords.")
        if row['word_count'] < 300:
            suggestions.append("Add more detailed project/work descriptions.")
        st.write("💡 Suggestions: " + (", ".join(suggestions) if suggestions else "Looks good!"))

    # Graphs
    st.subheader("📊 Visual Analysis")
    fig1, fig2, fig3 = plot_graphs(df)
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
