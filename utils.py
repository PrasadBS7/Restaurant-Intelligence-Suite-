"""
utils.py
--------
General-purpose helper functions shared across the Restaurant Intelligence
Suite notebooks: data loading, country-code mapping, quick profiling, and
consistent plot styling.

Author: Restaurant Intelligence Suite
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Country code lookup (Zomato dataset uses numeric ISO-style codes that do
# not map 1:1 to standard ISO-3166 numeric codes, so we hardcode the known
# mapping used by this dataset).
# ---------------------------------------------------------------------------
COUNTRY_CODE_MAP = {
    1: "India", 14: "Australia", 30: "Brazil", 37: "Canada", 94: "Indonesia",
    148: "New Zealand", 162: "Philippines", 166: "Qatar", 184: "Singapore",
    189: "South Africa", 191: "Sri Lanka", 208: "Turkey", 214: "UAE",
    215: "United Kingdom", 216: "United States"
}


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the restaurant dataset with the correct encoding.

    The raw CSV is UTF-8 with a BOM, so we use 'utf-8-sig' to avoid a
    stray '\ufeff' prefix on the first column name.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with a mapped 'Country' column added.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Country Code" in df.columns:
        df["Country"] = df["Country Code"].map(COUNTRY_CODE_MAP).fillna("Other")
    return df


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a professional one-glance data summary: dtype, missing count,
    missing %, unique count, and a sample value for every column.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Summary table indexed by column name.
    """
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_count": df.isnull().sum(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2),
        "unique_values": df.nunique(),
        "sample_value": df.iloc[0] if len(df) else np.nan,
    })
    return summary.sort_values("missing_count", ascending=False)


def set_plot_style():
    """Apply a consistent, professional visual theme to every figure."""
    sns.set_theme(style="whitegrid", palette="viridis")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.dpi"] = 100


def save_fig(fig, name: str, output_dir: str = "../outputs/figures"):
    """
    Save a matplotlib figure to the outputs/figures directory at
    publication quality (300 dpi), trimmed of excess whitespace.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File name without extension.
    output_dir : str
        Destination directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Figure saved -> {path}")
