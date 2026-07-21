"""
Data cleaning module.
Real dataset: AI Jobs Market 2025-2026 (1500 rows, 14 countries, already clean -
no missing values, no duplicates - lekin phir bhi defensive checks rakhte hain).
"""
import pandas as pd
import numpy as np


def load_raw_data(path: str) -> pd.DataFrame:
    """Raw CSV load karta hai aur basic inspection print karta hai."""
    df = pd.read_csv(path)
    print(df.info())
    print(df.isnull().sum().sort_values(ascending=False))
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"China rows: {(df['country'] == 'China').sum()}")
    return df


def flag_outliers(series: pd.Series) -> pd.Series:
    """IQR method se outliers ko flag karta hai (delete nahi karta)."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def add_high_demand_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    'Employment probability' proxy target banata hai.
    Dataset mein direct employed/unemployed label nahi hai (job-posting level data hai),
    isliye demand_score ko employability proxy banate hain: median se upar = high demand (1).
    """
    median_demand = df["demand_score"].median()
    df["high_demand"] = (df["demand_score"] >= median_demand).astype(int)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Poori cleaning pipeline: duplicates, missing values, text standardize, outliers, target."""
    df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    df["is_outlier_salary"] = flag_outliers(df["annual_salary_usd"])
    print(f"Salary outliers found: {df['is_outlier_salary'].sum()}")

    df = add_high_demand_target(df)
    print(f"High demand class balance:\n{df['high_demand'].value_counts(normalize=True)}")

    return df


if __name__ == "__main__":
    df = load_raw_data("data/raw/ai_jobs_market.csv")
    df_clean = clean_dataframe(df)
    df_clean.to_csv("data/interim/cleaned_data.csv", index=False)
    print(f"Cleaned data saved: {df_clean.shape}")
