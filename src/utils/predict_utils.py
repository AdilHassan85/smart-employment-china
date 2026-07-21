"""
Prediction utility.
User ke raw form-input (job_category, experience_level, etc.) ko us exact
57-column one-hot encoded format mein convert karta hai jo trained models
expect karte hain. Yeh alignment step app ke bina crash hue predictions
dene ke liye zaroori hai.
"""
import pandas as pd

TIER1_CITIES = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou"]
EDUCATION_ORDER = {
    "Bootcamp/Self-taught": 1,
    "Associate's": 2,
    "Bachelor's": 3,
    "Master's": 4,
    "PhD": 5,
}


def build_feature_vector(user_input: dict, feature_columns: list) -> pd.DataFrame:
    """
    user_input keys: job_category, experience_level, education_required, city,
    country, remote_work, company_size, industry, years_of_experience, skills (list),
    ai_salary_premium_pct, benefits_score_10, is_senior, is_remote_friendly, is_llm_role
    """
    row = {col: 0 for col in feature_columns}

    row["years_of_experience"] = user_input.get("years_of_experience", 0)
    row["ai_salary_premium_pct"] = user_input.get("ai_salary_premium_pct", 0)
    row["benefits_score_10"] = user_input.get("benefits_score_10", 5)
    row["posting_year"] = 2026
    row["posting_month"] = 1
    row["is_senior"] = int(user_input.get("years_of_experience", 0) >= 6)
    row["is_remote_friendly"] = int(user_input.get("remote_work") in ["Hybrid", "Fully Remote"])
    row["is_llm_role"] = int(user_input.get("is_llm_role", 0))
    row["is_outlier_salary"] = 0
    row["skill_count"] = len(user_input.get("skills", []))
    row["is_tier1_city"] = int(user_input.get("city") in TIER1_CITIES)
    row["is_china"] = int(user_input.get("country") == "China")
    row["education_score"] = EDUCATION_ORDER.get(user_input.get("education_required"), 3)

    one_hot_fields = {
        "job_category": user_input.get("job_category"),
        "experience_level": user_input.get("experience_level"),
        "remote_work": user_input.get("remote_work"),
        "company_size": user_input.get("company_size"),
        "industry": user_input.get("industry"),
        "country": user_input.get("country"),
    }
    for prefix, value in one_hot_fields.items():
        col_name = f"{prefix}_{value}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row])[feature_columns]
