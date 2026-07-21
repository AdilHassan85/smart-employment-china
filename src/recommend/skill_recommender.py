"""
Skill recommendation module.
User ki missing skills ko market demand ke hisab se rank karta hai.
"""
import pandas as pd


def build_skill_demand_index(job_postings_df: pd.DataFrame, skills_column: str = "required_skills") -> dict:
    """Job postings mein har skill kitni baar maangi gayi hai, usay 0-1 range mein normalize karta hai.
    Is dataset mein skills pipe (|) se separated hain, comma se nahi."""
    all_skills = job_postings_df[skills_column].str.split("|").explode()
    all_skills = all_skills.str.strip().str.lower()
    demand_counts = all_skills.value_counts()
    demand_score = demand_counts / demand_counts.max()
    return demand_score.to_dict()


def recommend_skills(user_skills: list, target_role_skills: list, demand_index: dict, top_n: int = 5) -> list:
    """User ki current skills aur target role skills ke gap ko demand ke hisab se rank karta hai."""
    user_set = set(s.strip().lower() for s in user_skills)
    missing = set(s.strip().lower() for s in target_role_skills) - user_set

    ranked = sorted(missing, key=lambda s: demand_index.get(s, 0), reverse=True)
    return ranked[:top_n]
