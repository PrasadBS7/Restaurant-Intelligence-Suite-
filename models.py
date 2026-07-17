"""
models.py
---------
Model factory and evaluation helpers shared by Task 1 (regression) and
Task 3 (classification) notebooks, so training/eval code is not duplicated.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def get_regression_models(random_state: int = 42) -> dict:
    """Return a dict of {name: unfitted regressor} for Task 1."""
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12,
                                                random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                        learning_rate=0.05,
                                                        random_state=random_state),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=5,
                                          learning_rate=0.05, random_state=random_state,
                                          verbosity=0)
    return models


def get_classification_models(random_state: int = 42) -> dict:
    """Return a dict of {name: unfitted classifier} for Task 3."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15,
                                                 random_state=random_state, n_jobs=-1),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=6,
                                           learning_rate=0.1, random_state=random_state,
                                           eval_metric="mlogloss", verbosity=0)
    return models


def evaluate_regression(y_true, y_pred) -> dict:
    """Compute MAE, MSE, RMSE, and R2 for a regression model's predictions."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2 Score": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred) -> dict:
    """Compute Accuracy, weighted Precision/Recall/F1 for a classifier."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_and_compare(models: dict, X_train, y_train, X_test, y_test,
                       task: str = "regression") -> pd.DataFrame:
    """
    Fit every model, evaluate it, and return a ranked comparison table.

    Parameters
    ----------
    models : dict
        {name: unfitted estimator}
    task : {'regression', 'classification'}

    Returns
    -------
    pd.DataFrame
        One row per model with evaluation metrics, sorted best-first.
    """
    eval_fn = evaluate_regression if task == "regression" else evaluate_classification
    rank_metric = "R2 Score" if task == "regression" else "F1 Score"

    rows = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = eval_fn(y_test, preds)
        metrics["Model"] = name
        rows.append(metrics)
        fitted_models[name] = model

    results_df = pd.DataFrame(rows).set_index("Model").reset_index()
    results_df = results_df.sort_values(rank_metric, ascending=False).reset_index(drop=True)
    return results_df, fitted_models
