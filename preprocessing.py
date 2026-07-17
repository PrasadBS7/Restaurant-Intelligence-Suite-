"""
preprocessing.py
-----------------
Data cleaning, missing-value handling, outlier treatment, and encoding
utilities for the Restaurant Intelligence Suite.

Design decisions (explained inline):
- Rows with Aggregate rating == 0 AND Rating text == 'Not rated' are
  restaurants that have never been rated. They carry no signal for a
  *rating prediction* model and are dropped only for Task 1 (regression),
  never globally, so Tasks 2-4 keep full data.
- Missing 'Cuisines' (9 rows) are filled with 'Other' rather than dropped,
  since dropping would lose otherwise-complete location/pricing data.
- Outliers in 'Average Cost for two' are capped (winsorized) at the 1st and
  99th percentile rather than removed, because a handful of ultra-luxury or
  free-listing restaurants are legitimate, not data-entry errors.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply core cleaning steps: duplicate removal, missing-value handling,
    and type normalization. Safe to run on the full dataset for any task.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe (copy).
    """
    data = df.copy()

    # 1. Duplicate removal
    before = len(data)
    data = data.drop_duplicates()
    removed = before - len(data)
    if removed:
        print(f"Removed {removed} duplicate rows.")

    # 2. Missing 'Cuisines' -> 'Other' (only 9 rows, ~0.09%)
    if "Cuisines" in data.columns:
        data["Cuisines"] = data["Cuisines"].fillna("Other")

    # 3. Strip whitespace from key text columns
    text_cols = ["Restaurant Name", "City", "Locality", "Cuisines"]
    for col in text_cols:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip()

    # 4. Normalize Yes/No binary columns to 1/0 for modeling convenience
    binary_cols = ["Has Table booking", "Has Online delivery",
                   "Is delivering now", "Switch to order menu"]
    for col in binary_cols:
        if col in data.columns:
            data[col + " (binary)"] = data[col].map({"Yes": 1, "No": 0})

    return data


def remove_unrated_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop restaurants with no rating (Aggregate rating == 0, Rating text ==
    'Not rated'). Used exclusively for the Task 1 regression target, since
    these rows have no ground truth to learn or evaluate against.
    """
    data = df.copy()
    before = len(data)
    if "Rating text" in data.columns:
        data = data[data["Rating text"].str.lower() != "not rated"]
    else:
        data = data[data["Aggregate rating"] > 0]
    print(f"Removed {before - len(data)} unrated restaurants "
          f"({(before - len(data)) / before:.1%} of data).")
    return data


def cap_outliers(df: pd.DataFrame, column: str,
                  lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.DataFrame:
    """
    Winsorize a numeric column at the given percentiles to reduce the
    influence of extreme outliers without discarding rows.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
    lower_pct, upper_pct : float
        Percentile bounds (0-1).

    Returns
    -------
    pd.DataFrame
    """
    data = df.copy()
    low, high = data[column].quantile([lower_pct, upper_pct])
    n_capped = ((data[column] < low) | (data[column] > high)).sum()
    data[column] = data[column].clip(lower=low, upper=high)
    print(f"Capped {n_capped} outliers in '{column}' to [{low:.1f}, {high:.1f}].")
    return data


def encode_categorical(df: pd.DataFrame, columns: list,
                        method: str = "label") -> tuple:
    """
    Encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list of str
        Columns to encode.
    method : {'label', 'onehot'}
        'label'  -> LabelEncoder, best for tree-based models and
                    high-cardinality columns (e.g. City, Locality) where
                    one-hot would explode dimensionality.
        'onehot' -> pd.get_dummies, best for low-cardinality nominal
                    columns (e.g. binary Yes/No, Price range) fed into
                    linear models.

    Returns
    -------
    (pd.DataFrame, dict)
        Encoded dataframe and a dict of fitted LabelEncoders (empty for
        'onehot').
    """
    data = df.copy()
    encoders = {}

    if method == "label":
        for col in columns:
            le = LabelEncoder()
            data[col + "_encoded"] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    elif method == "onehot":
        data = pd.get_dummies(data, columns=columns, prefix=columns, drop_first=True)
    else:
        raise ValueError("method must be 'label' or 'onehot'")

    return data, encoders


def scale_features(df: pd.DataFrame, columns: list,
                    method: str = "standard") -> tuple:
    """
    Scale numeric features.

    Scaler choice rationale:
    - 'standard' (StandardScaler): for linear/regularized models
      (Linear Regression) and distance-based methods -- assumes
      roughly-Gaussian, zero-centers and unit-variances features.
    - 'minmax' (MinMaxScaler): for algorithms sensitive to bounded ranges
      (e.g. neural nets, cosine-similarity recommendation features) so all
      features contribute proportionally within [0, 1].
    - 'robust' (RobustScaler): for columns with heavy outliers (e.g.
      Votes, Average Cost for two) since it scales using the IQR and is
      not distorted by extreme values.

    Returns
    -------
    (pd.DataFrame, scaler)
    """
    data = df.copy()
    scaler_map = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if method not in scaler_map:
        raise ValueError("method must be 'standard', 'minmax', or 'robust'")

    scaler = scaler_map[method]
    data[columns] = scaler.fit_transform(data[columns])
    return data, scaler
