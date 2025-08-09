import streamlit as st
import fitz  # pymupdf
import re, io, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(page_title="Resume Analyzer & Comparator", layout="wide")

# ----------------------
# Configuration / Keywords
# ----------------------
SKILL_CATEGORIES = {
    "Technical": ["python","r","java","c++","sql","pyspark","spark","hadoop","nlp","tensorflow","pytorch","scikit-learn"],
    "Tools": ["pandas","numpy","matplotlib","seaborn","power bi","tableau","excel","git","docker","aws","azure","gcp"],
    "Soft": ["communication","leadership","teamwork","problem solving","presentation","management","collaboration"]
}
EDU_KEYWORDS = ["bachelor", "b.sc", "bsc", "bs", "bachelor of", "master", "m.sc", "msc", "ms", "mba", "phd", "doctor"]

# ----------------------
# Utility functions
# ----------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Return extracted text from PDF bytes using PyMuPDF."""
    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
    except Exception as e:
        st.warning(f"PDF read error: {e}")
    return text

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{6,}\d)')

def extract_contacts(text):
    emails = EMAIL_RE.findall(text)
    phones_raw = PHONE_RE.findall(text)
    phones = []
    for p in phones_raw:
        p_clean = re.sub(r'[^\d+]', '', p)
        if 7 <= len(re.sub(r'\D','', p_clean)) <= 15:
            phones.append(p_clean)
    return list(dict.fromkeys(emails)), list(dict.fromkeys(phones))

def extract_name(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:10]:
        low = line.lower()
        if len(line.split()) <= 4 and not re.search(r'email|@|phone|contact|address|linkedin|github|cv|resume', low):
            if not re.search(r'\b(profile|summary|objective|experience|education)\b', low):
                return line
    return lines[0] if lines else None

def extract_skills_categorized(text):
    t = text.lower()
    found = {cat: [] for cat in SKILL_CATEGORIES}
    for cat, kws in SKILL_CATEGORIES.items():
        for kw in kws:
            if kw in t:
                found[cat].append(kw)
    for cat in found:
        found[cat] = sorted(list(set(found[cat])))
    return found

def extract_education(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found = []
    for line in lines:
        low = line.lower()
        for kw in EDU_KEYWORDS:
            if kw in low:
                found.append(line)
                break
    # dedupe preserve order
    seen=[]
    for e in found:
        if e not in seen:
            seen.append(e)
    return seen

YEAR_RE = re.compile(r'((?:19|20)\d{2})')
RANGE_RE = re.compile(r'(\d{4})\s*[-–to]{1,3}\s*(\d{4}|present)', re.I)

def extract_experiences(text):
    lines=[l.strip() for l in text.splitlines() if l.strip()]
    exps=[]
    for i,line in enumerate(lines):
        low=line.lower()
        if re.search(r'\b(engineer|developer|analyst|scientist|manager|consultant|intern|lead|director)\b', low) or re.search(r'\b(at|@|inc|ltd|company|co\.|llc)\b', low):
            block=line
            if i+1 < len(lines):
                block += " " + lines[i+1]
            if i+2 < len(lines):
                block += " " + lines[i+2]
            years = YEAR_RE.findall(block)
            ranges = RANGE_RE.findall(block)
            exps.append({"text": block, "years_found": years, "ranges_found": ranges})
    # dedupe
    seen=set(); unique=[]
    for e in exps:
        k=e['text'][:200]
        if k not in seen:
            unique.append(e); seen.add(k)
    return unique

def estimate_total_experience(experiences):
    years=[]
    for e in experiences:
        for y in e.get('years_found',[]):
            try:
                years.append(int(y))
            except: pass
        for r in e.get('ranges_found',[]):
            try:
                start=int(r[0])
                end = datetime.now().year if str(r[1]).lower().startswith('p') else int(r[1])
                if end>=start:
                    years.append(start); years.append(end)
            except: pass
    if not years:
        return None
    start=min(years); end=max(years)
    return max(0, end-start)

def compute_ats_score(contacts, skills_found, education, experiences, word_count):
    score=0
    if contacts[0] or contacts[1]:
        score += 15
    skill_count = sum(len(v) for v in skills_found.values())
    score += min(40, skill_count*4)
    if education:
        score += 15
    if experiences:
        score += 15
    if word_count >= 300:
        score += 15
    return min(100, score)

def analyze_bytes_and_build_report(file_bytes, filename):
    text = extract_text_from_pdf_bytes(file_bytes)
    name = extract_name(text) or filename.replace('.pdf','')
    emails, phones = extract_contacts(text)
    skills_found = extract_skills_categorized(text)
    education = extract_education(text)
    experiences = extract_experiences(text)
    total_exp = estimate_total_experience(experiences)
    word_count = len(text.split())
    top_keywords = pd.Series(re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())).value_counts().drop(labels=list(set(['the','and','for','with','that','this','from','are','was','will','have'])), errors='ignore').head(10)

    ats_score = compute_ats_score((emails, phones), skills_found, education, experiences, word_count)
    final_score = ats_score + (sum(len(v) for v in skills_found.values()) * 2) + ((total_exp or 0) * 3)

    suggestions=[]
    if not emails and not phones:
        suggestions.append("Add clear contact info (email and phone) at the top.")
    if not education:
        suggestions.append("Mention highest degree and institute.")
    if not experiences:
        suggestions.append("Add detailed work experience lines (company, role, dates).")
    if total_exp is None:
        suggestions.append("Specify start/end years for roles to calculate total experience.")
    if sum(len(v) for v in skills_found.values()) < 6:
        suggestions.append("Add more relevant technical/tools skills with proficiency levels.")
    if word_count < 300:
        suggestions.append("Expand CV with achievements and quantifiable results (use numbers).")
    suggestions.append("Use bullet points and quantify achievements (e.g., improved X by 20%).")

    report = {
        "filename": filename,
        "name": name,
        "emails": emails,
        "phones": phones,
        "skills_found": skills_found,
        "education": education,
        "experiences": experiences,
        "total_experience": total_exp,
        "word_count": word_count,
        "top_keywords": top_keywords,
        "ats_score": ats_score,
        "final_score": final_score,
        "suggestions": suggestions,
        "raw_text": text
    }
    return report

# ----------------------
# PDF report generator (in-memory)
# ----------------------
def generate_pdf_bytes(report):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    left = 40
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left, y, f"Resume Analysis: {report['name']}")
    y -= 25
    c.setFont("Helvetica", 10)
    c.drawString(left, y, f"File: {report['filename']}   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 20
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left, y, f"ATS Score: {report['ats_score']}%    Final Score: {int(report['final_score'])}")
    y -= 25
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Contacts:")
    c.setFont("Helvetica", 10); y-=15
    c.drawString(left, y, f"Emails: {', '.join(report['emails']) if report['emails'] else 'Not found'}"); y-=14
    c.drawString(left, y, f"Phones: {', '.join(report['phones']) if report['phones'] else 'Not found'}"); y-=18

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Education:")
    c.setFont("Helvetica", 10); y-=15
    if report['education']:
        for ed in report['education'][:6]:
            c.drawString(left+10, y, f"- {ed}"); y-=12
    else:
        c.drawString(left+10, y, "Not found"); y-=12
    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Top Skills:")
    c.setFont("Helvetica", 10); y-=15
    all_skills = []
    for cat, items in report['skills_found'].items():
        all_skills += items
    c.drawString(left+10, y, ", ".join(all_skills[:80])); y-=40

    c.setFont("Helvetica-Bold", 11)
    c.drawString(left, y, "Improvement Suggestions:")
    c.setFont("Helvetica", 10); y-=15
    for s in report['suggestions'][:8]:
        c.drawString(left+10, y, f"- {s}"); y-=12

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()

# ----------------------
# Streamlit UI
# ----------------------
st.title("📄 Resume Analyzer & Comparator (Streamlit)")
st.markdown("Upload **one or more** PDF resumes. The app will analyze each resume and show detailed results, graphs, and downloadable reports.")

uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Analyzing resumes..."):
        reports = []
        for u in uploaded_files:
            bytes_data = u.read()
            r = analyze_bytes_and_build_report(bytes_data, u.name)
            reports.append(r)

    # Build summary DataFrame
    summary_rows=[]
    for r in reports:
        summary_rows.append({
            "Filename": r['filename'],
            "Name": r['name'],
            "Emails": ", ".join(r['emails']),
            "Phones": ", ".join(r['phones']),
            "Skills Count": sum(len(v) for v in r['skills_found'].values()),
            "Experience (yrs)": r['total_experience'] if r['total_experience'] is not None else "",
            "Word Count": r['word_count'],
            "ATS Score": r['ats_score'],
            "Final Score": int(r['final_score'])
        })
    df = pd.DataFrame(summary_rows)

    # Show summary table
    st.subheader("Summary")
    st.dataframe(df.style.format({"ATS Score": "{:.0f}", "Final Score": "{:.0f}"}), height=300)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Summary CSV", data=csv, file_name="resumes_summary.csv", mime="text/csv")

    # Ranking & best resume
    best_idx = df['Final Score'].idxmax()
    best_row = df.loc[best_idx]
    st.subheader("🏆 Best Resume (auto-selected)")
    st.markdown(f"**{best_row['Filename']}** — Final Score: **{best_row['Final Score']}**, ATS Score: **{best_row['ATS Score']}**")

    # Show reasons why others are weaker
    st.subheader("❌ Weak Resumes & Reasons")
    weaker = df.drop(index=best_idx).reset_index(drop=True)
    reason_list=[]
    for i, row in weaker.iterrows():
        reasons=[]
        if row['ATS Score'] < best_row['ATS Score']:
            reasons.append("Lower ATS Score")
        if (row['Experience (yrs)'] == "" or float(row['Experience (yrs)'] if row['Experience (yrs)']!="" else 0) < float(best_row['Experience (yrs)'] if best_row['Experience (yrs)']!="" else 0)):
            reasons.append("Less experience")
        if row['Skills Count'] < best_row['Skills Count']:
            reasons.append("Fewer detected skills")
        if not reasons:
            reasons.append("Other issues (formatting, missing sections)")
        reason_list.append({"filename": row['Filename'], "reasons": ", ".join(reasons)})
    if reason_list:
        st.table(pd.DataFrame(reason_list))

    # Per-resume expandable details
    for r in reports:
        with st.expander(f"Details: {r['filename']} — Score {int(r['final_score'])}", expanded=False):
            col1, col2 = st.columns([2,1])
            with col1:
                st.write(f"**Name:** {r['name']}")
                st.write(f"**Emails:** {', '.join(r['emails']) if r['emails'] else 'Not found'}")
                st.write(f"**Phones:** {', '.join(r['phones']) if r['phones'] else 'Not found'}")
                st.write(f"**Education (lines):**")
                if r['education']:
                    for ed in r['education'][:5]:
                        st.write("-", ed)
                else:
                    st.write("Not found")
                st.write("**Experience snippets:**")
                if r['experiences']:
                    for ex in r['experiences'][:6]:
                        st.write("-", ex['text'][:200])
                else:
                    st.write("Not found")
                st.write("**Top keywords:**")
                if not r['top_keywords'].empty:
                    st.write(r['top_keywords'].to_dict())
                st.write("**Suggestions:**")
                for s in r['suggestions']:
                    st.write("-", s)

            with col2:
                # Skills by category
                cat_counts = {cat: len(v) for cat,v in r['skills_found'].items()}
                fig1, ax1 = plt.subplots(figsize=(4,2.4))
                ax1.bar(cat_counts.keys(), cat_counts.values(), color=['#1f77b4','#ff7f0e','#2ca02c'])
                ax1.set_title("Skills by Category")
                ax1.set_ylabel("Count")
                plt.xticks(rotation=30)
                st.pyplot(fig1)

                # ATS score bar
                fig2, ax2 = plt.subplots(figsize=(4,1.5))
                color = "#2ca02c" if r['ats_score']>=75 else ("#ff7f0e" if r['ats_score']>=50 else "#d62728")
                ax2.barh(["ATS"], [r['ats_score']], color=color)
                ax2.set_xlim(0,100)
                ax2.set_title("ATS Score")
                st.pyplot(fig2)

                # Experience bar
                exp_val = r['total_experience'] or 0
                fig3, ax3 = plt.subplots(figsize=(4,2.2))
                ax3.bar(["Experience (yrs)"], [exp_val], color="#9467bd")
                ax3.set_ylim(0, max(5, exp_val+2))
                ax3.set_title("Estimated Experience")
                st.pyplot(fig3)

            # Download per-resume PDF report
            pdf_bytes = generate_pdf_bytes(r)
            st.download_button(f"⬇️ Download PDF Report — {r['filename']}", data=pdf_bytes, file_name=f"{r['filename'].replace('.pdf','')}_analysis.pdf", mime="application/pdf")

    # Comparison charts for multiple resumes
    if len(reports) > 1:
        st.subheader("Comparison Charts")
        df_plot = df.copy()
        df_plot['Experience (yrs)'] = pd.to_numeric(df_plot['Experience (yrs)'], errors='coerce').fillna(0)
        # Bar chart skills vs experience
        fig, ax = plt.subplots(figsize=(10,4))
        x = np.arange(len(df_plot))
        ax.bar(x - 0.2, df_plot['Skills Count'], width=0.4, label='Skills Count', color='#1f77b4')
        ax.bar(x + 0.2, df_plot['Experience (yrs)'], width=0.4, label='Experience (yrs)', color='#ff7f0e')
        ax.set_xticks(x); ax.set_xticklabels(df_plot['Filename'], rotation=45)
        ax.set_ylabel("Count / Years")
        ax.legend()
        st.pyplot(fig)

        # ATS comparison
        figb, axb = plt.subplots(figsize=(10,3))
        axb.bar(df_plot['Filename'], df_plot['ATS Score'], color='#2ca02c')
        axb.set_ylabel("ATS Score (%)")
        axb.set_title("ATS Score Comparison")
        plt.xticks(rotation=45)
        st.pyplot(figb)

        # Pie chart of best resume skill breakdown
        best_report = reports[[r['final_score'] for r in reports].index(max([r['final_score'] for r in reports]))]
        best_skills = best_report['skills_found']
        labels = list(best_skills.keys())
        sizes = [len(best_skills[l]) for l in labels]
        figc, axc = plt.subplots(figsize=(5,5))
        if sum(sizes) > 0:
            axc.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            axc.set_title(f"Skill Breakdown — {best_report['filename']}")
            st.pyplot(figc)
        else:
            st.info("No skills detected for pie chart.")

st.sidebar.markdown("## Tips\n- Add clear dates to experience lines (e.g., 2019-2021).\n- Include measurable achievements.\n- Add more keywords relevant to target role.\n\nAdvanced: add OCR (pytesseract) for scanned PDFs, or use spaCy/transformers for better extraction.")
