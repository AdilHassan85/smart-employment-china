"""
Model evaluation aur explainability module.
Classification metrics, regression metrics, aur SHAP explanations.
"""
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classifier(model, X_test, y_test):
    """Employment model ko evaluate karta hai - precision, recall, F1, ROC-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")
    print(confusion_matrix(y_test, y_pred))

    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def evaluate_regressor(model, X_test, y_test):
    """Salary model ko evaluate karta hai - MAE, RMSE, R2."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.0f} yuan")
    print(f"RMSE: {rmse:.0f} yuan")
    print(f"R2: {r2:.3f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def explain_with_shap(model, X_test, save_path: str = "reports/figures/shap_summary.png"):
    """SHAP summary plot banata hai - dikhata hai kaunse features prediction ko drive karte hain."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return shap_values
