"""
feature_engineering.py
-----------------------
Derives new predictive features from the cleaned restaurant dataset:
cuisine counts, primary cuisine extraction, votes-per-cost ratios, city
popularity, and rating-tier labels.
"""

import pandas as pd
import numpy as np


def add_cuisine_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive cuisine-related features from the raw comma-separated
    'Cuisines' string.

    New columns
    -----------
    Cuisine Count : int
        Number of distinct cuisines offered (restaurants offering more
        cuisines often signal versatility, which can correlate with
        rating).
    Primary Cuisine : str
        The first-listed cuisine, used as a simplified single-label
        target for Task 3 classification.
    """
    data = df.copy()
    data["Cuisine Count"] = data["Cuisines"].apply(
        lambda x: len(str(x).split(","))
    )
    data["Primary Cuisine"] = data["Cuisines"].apply(
        lambda x: str(x).split(",")[0].strip()
    )
    return data


def add_popularity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive popularity / value-for-money features.

    New columns
    -----------
    Votes per Cost : float
        Votes normalized by average cost -- a proxy for "buzz relative to
        price point". Cost is floored at 1 to avoid division by zero for
        free listings.
    City Restaurant Count : int
        Number of restaurants listed in the same city (market density).
    City Avg Rating : float
        Mean aggregate rating for the restaurant's city (used cautiously
        -- target-leakage risk is handled by only using it as a
        *reference* feature computed on the training fold in modeling
        notebooks).
    """
    data = df.copy()
    safe_cost = data["Average Cost for two"].replace(0, 1)
    data["Votes per Cost"] = data["Votes"] / safe_cost

    city_counts = data.groupby("City")["Restaurant ID"].transform("count")
    data["City Restaurant Count"] = city_counts

    city_avg_rating = data.groupby("City")["Aggregate rating"].transform("mean")
    data["City Avg Rating"] = city_avg_rating

    return data


def add_rating_tier(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket 'Aggregate rating' into human-readable tiers matching Zomato's
    own 'Rating text' bands, useful for stratified analysis and as an
    auxiliary classification target.
    """
    data = df.copy()

    def tier(r):
        if r == 0:
            return "Not Rated"
        elif r < 2.5:
            return "Poor"
        elif r < 3.5:
            return "Average"
        elif r < 4.0:
            return "Good"
        elif r < 4.5:
            return "Very Good"
        else:
            return "Excellent"

    data["Rating Tier"] = data["Aggregate rating"].apply(tier)
    return data


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience pipeline that applies every feature-engineering step in
    the correct order.
    """
    data = add_cuisine_features(df)
    data = add_popularity_features(data)
    data = add_rating_tier(data)
    return data
