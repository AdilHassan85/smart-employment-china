"""
Hyperparameter tuning module.
GridSearchCV se best XGBoost configuration dhoondta hai.
"""
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
}


def tune_employment_model(X_train, y_train, param_grid: dict = None, cv: int = 5):
    """GridSearchCV se best employment model dhoondta hai."""
    param_grid = param_grid or XGB_PARAM_GRID
    search = GridSearchCV(
        XGBClassifier(eval_metric="logloss", random_state=42),
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV F1 score: {search.best_score_:.3f}")
    return search.best_estimator_
