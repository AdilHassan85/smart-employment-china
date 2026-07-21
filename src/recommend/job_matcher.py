"""
Job matching module.
TF-IDF + cosine similarity se user profile ko job postings se match karta hai.
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def match_jobs(user_skills_text: str, job_postings_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """User ki skills text ko job postings ke required skills se compare kar ke top matches deta hai."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    all_texts = [user_skills_text] + job_postings_df["required_skills"].str.replace("|", " ").tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    result = job_postings_df.copy()
    result["match_score"] = (similarities * 100).round(1)
    return result.sort_values("match_score", ascending=False).head(top_n)
