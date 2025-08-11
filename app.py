# app.py
import streamlit as st
from parser import parse_uploaded_file
from matcher import jd_resume_match_score, gap_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

st.set_page_config(page_title="SmartCV Pro", layout="wide")
st.title("SmartCV Pro — Resume Analyzer & Job Matcher (Pro)")

st.sidebar.header("Upload / Settings")
uploaded_files = st.sidebar.file_uploader("Upload resume files (PDF/DOCX/TXT)", accept_multiple_files=True)
job_desc = st.sidebar.text_area("Optional: Paste Job Description (for JD matching)", height=150)
run_button = st.sidebar.button("Analyze")

# optional: user selects plan (for UI/testing)
user_plan = st.sidebar.selectbox("Plan", ["Free (demo)", "Pro", "Enterprise"])

def generate_pdf_report_simple(report):
    """Return bytes of a simple PDF report (text-only)"""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    w, h = A4
    left = 40
    y = h - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, f"SmartCV Pro — Resume Analysis")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Name: {report.get('name','-')}   File: {report.get('filename')}")
    y -= 18
    c.drawString(left, y, f"ATS Score: {report.get('ats_score', '-')}, Final Score: {report.get('final_score','-')}")
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Contacts:")
    y -= 12
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"Emails: {', '.join(report.get('emails',[]))}")
    y -= 12
    c.drawString(left, y, f"Phones: {', '.join(report.get('phones',[]))}")
    y -= 18
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Top Skills:")
    y -= 12
    c.setFont("Helvetica", 10)
    c.drawString(left, y, ", ".join(report.get('top_skills',[]))[:200])
    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Suggestions:")
    y -= 12
    c.setFont("Helvetica", 10)
    for s in report.get('suggestions', [])[:8]:
        c.drawString(left+10, y, "- " + s)
        y -= 12
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

if run_button and uploaded_files:
    st.info("Analyzing... this can take a few seconds per file.")
    reports = []
    for f in uploaded_files:
        parsed = parse_uploaded_file(f)
        # compute simple ATS-like score (heuristic)
        skills_total = sum(len(v) for v in parsed['skills'].values())
        word_count = len(parsed['raw_text'].split())
        # simple ats_score heuristic
        ats_score = min(100, skills_total * 6 + (10 if parsed['education'] else 0) + (10 if parsed['emails'] else 0) + (5 if parsed['estimated_experience'] else 0))
        final_score = ats_score + skills_total*2 + (parsed['estimated_experience'] or 0)*3
        top_skills = []
        for cat, items in parsed['skills'].items():
            top_skills += items
        suggestions=[]
        if not parsed['emails']:
            suggestions.append("Add email.")
        if not parsed['phones']:
            suggestions.append("Add phone number.")
        if not parsed['education']:
            suggestions.append("Add education details.")
        if parsed['estimated_experience'] is None:
            suggestions.append("Add dates for jobs to estimate experience.")
        if skills_total < 4:
            suggestions.append("List more technical tools and libraries.")
        report = {
            "filename": f.name,
            "name": parsed['name'],
            "emails": parsed['emails'],
            "phones": parsed['phones'],
            "top_skills": top_skills,
            "skills_total": skills_total,
            "education": parsed['education'],
            "experiences": parsed['experiences'],
            "estimated_experience": parsed['estimated_experience'],
            "word_count": word_count,
            "ats_score": ats_score,
            "final_score": int(final_score),
            "suggestions": suggestions,
            "raw_text": parsed['raw_text']
        }
        # If job_desc present, compute matching
        if job_desc and job_desc.strip():
            similarity = jd_resume_match_score(job_desc, parsed['raw_text'])
            missing = gap_analysis(job_desc, parsed['raw_text'], top_k=10)
            report["jd_similarity"] = float(similarity)
            report["jd_missing"] = missing
        reports.append(report)

    # DataFrame summary
    df = pd.DataFrame([{
        "Filename": r["filename"],
        "Name": r.get("name",""),
        "Skills": r["skills_total"],
        "Experience(yrs)": r["estimated_experience"] or "",
        "ATS": r["ats_score"],
        "Final": r["final_score"],
        "JD Match": r.get("jd_similarity","") if job_desc else ""
    } for r in reports])

    st.subheader("Summary")
    st.dataframe(df.sort_values(by="Final", ascending=False).reset_index(drop=True))

    # Allow CSV download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download summary CSV", data=csv, file_name="smartcv_summary.csv", mime="text/csv")

    # Show best, detailed and graphs
    best = max(reports, key=lambda x: x["final_score"])
    st.success(f"Best resume (auto): {best['filename']} — Final score: {best['final_score']}")

    # Plots
    st.subheader("Visual Comparison")
    fig, ax = plt.subplots(figsize=(8,4))
    x = np.arange(len(reports))
    ax.bar(x - 0.2, [r['skills_total'] for r in reports], width=0.4, label="Skills Count")
    ax.bar(x + 0.2, [r['estimated_experience'] or 0 for r in reports], width=0.4, label="Experience (yrs)")
    ax.set_xticks(x); ax.set_xticklabels([r['filename'] for r in reports], rotation=30)
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8,3))
    ax2.bar([r['filename'] for r in reports], [r['ats_score'] for r in reports], color='green')
    ax2.set_ylabel("ATS Score")
    st.pyplot(fig2)

    # Per-resume expanders
    for r in reports:
        with st.expander(f"{r['filename']} — Score {r['final_score']}"):
            st.write("Name:", r['name'])
            st.write("Emails:", r['emails'])
            st.write("Phones:", r['phones'])
            st.write("Top skills:", r['top_skills'])
            st.write("Education lines:", r['education'])
            st.write("Experiences (snippets):", [e['text'][:200] for e in r['experiences']])
            st.write("Estimated experience years:", r['estimated_experience'])
            st.write("ATS score:", r['ats_score'])
            st.write("Suggestions:")
            for s in r['suggestions']:
                st.write("-", s)
            if "jd_similarity" in r:
                st.write("Job description match:", f"{r['jd_similarity']*100:.1f}%")
                st.write("Missing JD keywords:", r['jd_missing'])
            # download per resume report
            pdf_bytes = generate_pdf_report_simple(r)
            st.download_button("Download PDF report", data=pdf_bytes, file_name=f"{r['filename']}_analysis.pdf", mime="application/pdf")

else:
    st.info("Upload one or more resumes and click Analyze in the sidebar.")
