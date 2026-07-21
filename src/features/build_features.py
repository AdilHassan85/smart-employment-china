"""
Feature engineering module.
Real dataset columns se features banata hai (skills are pipe-separated: "Python|SQL|Cloud").
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

TIER1_CITIES = ["Beijing", "Shanghai", "Shenzhen", "Guangzhou"]
EDUCATION_ORDER = {
    "Bootcamp/Self-taught": 1,
    "Associate's": 2,
    "Bachelor's": 3,
    "Master's": 4,
    "PhD": 5,
}


def add_skill_count(df: pd.DataFrame) -> pd.DataFrame:
    """required_skills column pipe (|) se separated hai, comma se nahi."""
    df["skill_count"] = df["required_skills"].apply(lambda x: len(str(x).split("|")))
    return df


def add_tier1_city_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["is_tier1_city"] = df["city"].isin(TIER1_CITIES).astype(int)
    df = df.drop(columns=["city"])
    return df


def add_is_china(df: pd.DataFrame) -> pd.DataFrame:
    """China-focus analysis ke liye ek direct binary flag."""
    df["is_china"] = (df["country"] == "China").astype(int)
    return df


def add_education_score(df: pd.DataFrame) -> pd.DataFrame:
    df["education_score"] = df["education_required"].map(EDUCATION_ORDER)
    df = df.drop(columns=["education_required"])
    return df


def encode_categoricals(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    return pd.get_dummies(df, columns=columns, drop_first=True)


def scale_numeric_features(df: pd.DataFrame, numeric_features: list) -> tuple:
    """Scaler fit karta hai aur scaled dataframe + fitted scaler dono return karta hai.
    IMPORTANT: yeh sirf training data par fit karo, test data par sirf transform."""
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    return df, scaler


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Poori feature engineering pipeline ek sath chalata hai."""
    df = add_skill_count(df)
    df = add_tier1_city_flag(df)
    df = add_is_china(df)
    df = add_education_score(df)
    df = encode_categoricals(
        df, columns=["job_category", "experience_level", "remote_work", "company_size", "industry", "country"]
    )
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/interim/cleaned_data.csv")
    df = build_all_features(df)
    numeric_features = ["skill_count", "education_score", "years_of_experience", "demand_score"]
    df, scaler = scale_numeric_features(df, numeric_features)
    df.to_csv("data/processed/model_ready_data.csv", index=False)
    print(f"Final feature count: {df.shape[1]}")
