# matcher.py corrected
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def normalize(text):
    return re.sub(r'\s+', ' ', text.lower().strip())

def jd_resume_match_score(job_desc, resume_text):
    docs = [normalize(job_desc), normalize(resume_text)]
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(docs)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0,0]
    return float(sim)  # between 0 and 1

def gap_analysis(job_desc, resume_text, top_k=12):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf = vectorizer.fit_transform([job_desc, resume_text])
    feature_names = np.array(vectorizer.get_feature_names_out())
    jd_vec = tfidf[0].toarray().ravel()
    resume_vec = tfidf[1].toarray().ravel()
    jd_top_idx = np.argsort(-jd_vec)[:top_k]
    jd_top_terms = feature_names[jd_top_idx]
    missing = [t for t in jd_top_terms if resume_vec[np.where(feature_names==t)[0][0]] == 0]
    return missing
