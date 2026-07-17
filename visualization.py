"""
visualization.py
-----------------
Reusable, publication-quality plotting functions for EDA and model
evaluation. Every function returns the matplotlib Figure so notebooks can
save it via utils.save_fig().
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_rating_distribution(df: pd.DataFrame, rating_col: str = "Aggregate rating"):
    """Histogram + KDE of the rating distribution."""
    fig, ax = plt.subplots()
    sns.histplot(df[rating_col], bins=20, kde=True, ax=ax, color="#2E86AB")
    ax.set_title("Distribution of Aggregate Ratings")
    ax.set_xlabel("Aggregate Rating")
    ax.set_ylabel("Number of Restaurants")
    return fig


def plot_top_categories(df: pd.DataFrame, column: str, top_n: int = 10,
                         title: str = None):
    """Horizontal bar chart of the top-N most frequent category values."""
    fig, ax = plt.subplots()
    counts = df[column].value_counts().head(top_n).sort_values()
    counts.plot(kind="barh", ax=ax, color=sns.color_palette("viridis", top_n))
    ax.set_title(title or f"Top {top_n} by {column}")
    ax.set_xlabel("Number of Restaurants")
    ax.set_ylabel(column)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, columns: list = None):
    """Correlation heatmap for numeric columns."""
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix of Numeric Features")
    return fig


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str = None):
    """Boxplot of a numeric variable grouped by a categorical variable."""
    fig, ax = plt.subplots()
    order = df.groupby(x)[y].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x=x, y=y, order=order, ax=ax, palette="viridis")
    ax.set_title(title or f"{y} by {x}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig


def plot_countplot(df: pd.DataFrame, column: str, title: str = None):
    """Count plot for a low-cardinality categorical column (e.g. Yes/No)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.countplot(data=df, x=column, ax=ax, palette="viridis")
    ax.set_title(title or f"Count of {column}")
    for container in ax.containers:
        ax.bar_label(container)
    return fig


def plot_actual_vs_predicted(y_true, y_pred, model_name: str = ""):
    """Scatter of actual vs predicted values with an identity reference line."""
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.4, color="#2E86AB", edgecolor="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Rating")
    ax.set_ylabel("Predicted Rating")
    ax.set_title(f"Actual vs Predicted Rating -- {model_name}")
    ax.legend()
    return fig


def plot_residuals(y_true, y_pred, model_name: str = ""):
    """Residual plot (prediction error vs predicted value)."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.4, color="#A23B72", edgecolor="none")
    ax.axhline(0, color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("Predicted Rating")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title(f"Residual Plot -- {model_name}")
    return fig


def plot_feature_importance(importances, feature_names, top_n: int = 15,
                             model_name: str = ""):
    """Horizontal bar chart of top-N feature importances."""
    fig, ax = plt.subplots()
    imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(top_n)
    imp_series.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("viridis", top_n))
    ax.set_title(f"Top {top_n} Feature Importances -- {model_name}")
    ax.set_xlabel("Importance")
    return fig


def plot_model_comparison(results_df: pd.DataFrame, metric: str = "R2 Score"):
    """Bar chart comparing models on a chosen evaluation metric."""
    fig, ax = plt.subplots()
    sorted_df = results_df.sort_values(metric, ascending=False)
    sns.barplot(data=sorted_df, x=metric, y="Model", ax=ax, palette="viridis")
    ax.set_title(f"Model Comparison -- {metric}")
    for i, v in enumerate(sorted_df[metric]):
        ax.text(v, i, f" {v:.3f}", va="center")
    return fig


def plot_confusion_matrix(cm, labels, model_name: str = ""):
    """Heatmap of a classification confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix -- {model_name}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return fig
