"""
Smart Employment Prediction & Skill Recommendation System - Streamlit app.
Real trained models (AI Jobs Market 2025-2026 dataset) se connected.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from src.recommend.job_matcher import match_jobs
from src.recommend.skill_recommender import (build_skill_demand_index,
                                             recommend_skills)
from src.utils.predict_utils import build_feature_vector

st.set_page_config(page_title="Smart Employment Predictor - China", layout="wide")


@st.cache_resource
def load_models():
    employment_model = joblib.load("models/employment_model.pkl")
    salary_model = joblib.load("models/salary_model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    return employment_model, salary_model, feature_columns


@st.cache_data
def load_reference_data():
    return pd.read_csv("data/raw/ai_jobs_market.csv")


employment_model, salary_model, feature_columns = load_models()
jobs_df = load_reference_data()

st.title("Smart Employment Prediction & Skill Recommendation")
st.caption("AI/Tech job market data (14 countries, China focus) par based ML system")

with st.sidebar:
    st.header("Apna profile enter karo")
    job_category = st.selectbox("Job category", sorted(jobs_df["job_category"].unique()))
    experience_level = st.selectbox("Experience level", sorted(jobs_df["experience_level"].unique()))
    years_experience = st.slider("Years of experience", 0, 20, 2)
    education = st.selectbox("Education", ["Bootcamp/Self-taught", "Associate's", "Bachelor's", "Master's", "PhD"])
    country = st.selectbox("Country", sorted(jobs_df["country"].unique()), index=sorted(jobs_df["country"].unique()).index("China") if "China" in jobs_df["country"].unique() else 0)
    city = st.text_input("City", "Shanghai")
    remote_work = st.selectbox("Remote work", sorted(jobs_df["remote_work"].unique()))
    company_size = st.selectbox("Company size", sorted(jobs_df["company_size"].unique()))
    industry = st.selectbox("Industry", sorted(jobs_df["industry"].unique()))
    skills = st.text_area("Apni skills likho (comma se separate)", "Python, SQL, Cloud")
    is_llm_role = st.checkbox("LLM-related role hai?")
    submit = st.button("Predict karo", type="primary")

if submit:
    col1, col2 = st.columns(2)

    user_profile = {
        "job_category": job_category,
        "experience_level": experience_level,
        "education_required": education,
        "city": city,
        "country": country,
        "remote_work": remote_work,
        "company_size": company_size,
        "industry": industry,
        "years_of_experience": years_experience,
        "skills": [s.strip() for s in skills.split(",") if s.strip()],
        "ai_salary_premium_pct": 5.0,
        "benefits_score_10": 6.0,
        "is_llm_role": int(is_llm_role),
    }

    X_user = build_feature_vector(user_profile, feature_columns)

    with col1:
        st.subheader("High-demand probability")
        prob = float(employment_model.predict_proba(X_user)[0][1])
        st.metric("Probability", f"{prob*100:.1f}%")
        st.progress(float(min(max(prob, 0.0), 1.0)))

        st.subheader("Salary estimate")
        salary_pred = float(salary_model.predict(X_user)[0])
        st.metric("Estimated annual salary (USD)", f"{salary_pred:,.0f}")

    with col2:
        st.subheader("Kyun yeh prediction aayi (SHAP)")
        explainer = shap.TreeExplainer(employment_model)
        shap_values = explainer.shap_values(X_user)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_user, show=False, plot_type="bar")
        st.pyplot(fig)

    st.subheader("Recommended skills to learn")
    demand_index = build_skill_demand_index(jobs_df, skills_column="required_skills")
    # Same job_category ke top postings ki skills ko target list ki tarah use karo
    target_skills = (
        jobs_df[jobs_df["job_category"] == job_category]["required_skills"]
        .str.split("|").explode().str.strip().unique().tolist()
    )
    recs = recommend_skills(user_profile["skills"], target_skills, demand_index, top_n=5)
    if recs:
        st.write(", ".join(recs))
    else:
        st.write("Tumhari skills already is category ke top requirements cover kar rahi hain.")

    st.subheader("Top matched jobs")
    user_skills_text = " ".join(user_profile["skills"])
    matches = match_jobs(user_skills_text, jobs_df, top_n=5)
    st.dataframe(matches[["job_title", "country", "company_size", "annual_salary_usd", "match_score"]])