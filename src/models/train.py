"""
Model training module.
Employment classification model aur salary regression model dono train karta hai.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor


def train_employment_model(X_train, y_train, params: dict = None):
    """Employment probability classifier train karta hai."""
    default_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
    params = params or default_params
    model = XGBClassifier(eval_metric="logloss", **params)
    model.fit(X_train, y_train)
    return model


def train_salary_model(X_train, y_train, params: dict = None):
    """Salary prediction regressor train karta hai."""
    default_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1, "random_state": 42}
    params = params or default_params
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    df = pd.read_csv("data/processed/model_ready_data.csv")

    # high_demand = "employment probability" proxy (see config.yaml comment)
    # demand_growth_yoy_pct excluded: 0.566 correlated with demand_score (source of the target) -
    # keeping it in would leak target information into the features.
    drop_cols = ["high_demand", "annual_salary_usd", "salary_min_usd", "salary_max_usd",
                 "job_id", "job_title", "required_skills", "salary_tier", "demand_score",
                 "demand_growth_yoy_pct"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["high_demand"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    employment_model = train_employment_model(X_train, y_train)
    joblib.dump(employment_model, "models/employment_model.pkl")
    joblib.dump(list(X.columns), "models/feature_columns.pkl")
    print("Employment (high-demand) model saved.")

    # Salary model
    if "annual_salary_usd" in df.columns:
        Xs = df.drop(columns=[c for c in drop_cols if c in df.columns])
        ys = df["annual_salary_usd"]
        Xs_train, Xs_test, ys_train, ys_test = train_test_split(
            Xs, ys, test_size=0.2, random_state=42
        )
        salary_model = train_salary_model(Xs_train, ys_train)
        joblib.dump(salary_model, "models/salary_model.pkl")
        print("Salary model saved.")
