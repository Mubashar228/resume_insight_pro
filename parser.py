# parser.py
import fitz  # pymupdf
import pytesseract
from PIL import Image
import io
import re
import docx
import spacy

nlp = spacy.load("en_core_web_sm")

EMAIL_RE = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
PHONE_RE = re.compile(r'(\+?\d[\d\-\s\(\)]{6,}\d)')
YEAR_RE = re.compile(r'((?:19|20)\d{2})')

def read_pdf_bytes(pdf_bytes):
    """Extract text from PDF bytes using PyMuPDF. If pages have no text, fallback to OCR."""
    text_parts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        page_text = page.get_text("text")
        if page_text and page_text.strip():
            text_parts.append(page_text)
        else:
            # fallback to image -> OCR
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes()))
            ocr_text = pytesseract.image_to_string(img)
            text_parts.append(ocr_text)
    return "\n".join(text_parts)

def read_docx_bytes(docx_bytes):
    """Extract text from docx bytes."""
    tmp = io.BytesIO(docx_bytes)
    doc = docx.Document(tmp)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def normalize_text(text):
    return re.sub(r'\r', '\n', text).strip()

# Basic contact & name extraction
def extract_contacts_and_name(text):
    emails = list(dict.fromkeys(EMAIL_RE.findall(text)))
    phones_raw = PHONE_RE.findall(text)
    phones = []
    for p in phones_raw:
        p_clean = re.sub(r'[^\d+]', '', p)
        if 7 <= len(re.sub(r'\D','', p_clean)) <= 15:
            phones.append(p_clean)
    # name: use spaCy NER for first PERSON entity in first 1000 chars
    doc = nlp(text[:1000])
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    name = persons[0] if persons else None
    # fallback: first non-empty short line
    if not name:
        for line in text.splitlines()[:10]:
            line = line.strip()
            if line and len(line.split()) <= 4 and not re.search(r'email|@|phone|contact|address', line, re.I):
                name = line
                break
    return {"name": name, "emails": emails, "phones": phones}

# Skills extraction (expandable skill lists)
SKILL_BUCKETS = {
    "technical": ["python","r","java","c++","sql","pyspark","spark","hadoop","nlp","tensorflow","keras","pytorch","scikit-learn"],
    "libraries_tools": ["pandas","numpy","matplotlib","seaborn","plotly","tableau","power bi","excel","git","docker","kubernetes","aws","azure","gcp"],
    "soft": ["communication","leadership","teamwork","management","problem solving","presentation"]
}

def extract_skills(text):
    text_lower = text.lower()
    found = {k: [] for k in SKILL_BUCKETS}
    for cat, keywords in SKILL_BUCKETS.items():
        for kw in keywords:
            if kw in text_lower:
                found[cat].append(kw)
    return found

def extract_education(text):
    EDU = ["bachelor","master","ms","phd","b.sc","bsc","mba","degree"]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found = [ln for ln in lines if any(k in ln.lower() for k in EDU)]
    return found

def extract_experience_segments(text):
    """Heuristic to pull role/company lines and years mentioned nearby."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    exps = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        if re.search(r'\b(engineer|developer|analyst|scientist|manager|consultant|lead|intern)\b', low) or re.search(r'\b(at|@|inc|ltd|company|co\.)\b', low):
            block = ln
            if i+1 < len(lines): block += " " + lines[i+1]
            years = YEAR_RE.findall(block)
            exps.append({"text": block, "years": years})
    return exps

def estimate_total_experience(exps):
    years = []
    for e in exps:
        for y in e.get("years", []) if "years" in e else e.get("years_found", []):
            try:
                years.append(int(y))
            except:
                pass
    if not years:
        return None
    return max(years) - min(years)

# Helper to handle uploaded file bytes
def parse_uploaded_file(file):
    """file is a Streamlit UploadedFile or similar object"""
    name = file.name.lower()
    raw_bytes = file.read()
    if name.endswith(".pdf"):
        text = read_pdf_bytes(raw_bytes)
    elif name.endswith(".docx"):
        text = read_docx_bytes(raw_bytes)
    else:
        text = raw_bytes.decode(errors='ignore')
    text = normalize_text(text)
    contact = extract_contacts_and_name(text)
    skills = extract_skills(text)
    edu = extract_education(text)
    exps = extract_experience_segments(text)
    total_exp = estimate_total_experience(exps)
    return {
        "raw_text": text,
        "name": contact["name"],
        "emails": contact["emails"],
        "phones": contact["phones"],
        "skills": skills,
        "education": edu,
        "experiences": exps,
        "estimated_experience": total_exp
    }
