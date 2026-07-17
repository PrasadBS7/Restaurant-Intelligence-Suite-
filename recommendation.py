"""
recommendation.py
------------------
Content-based restaurant recommendation engine.

Approach
--------
1. Build a composite "content profile" string per restaurant combining
   Cuisines, City, and Price range -- the categorical signal a user cares
   about when searching.
2. Vectorize the profiles with TF-IDF so rarer, more distinctive cuisines
   / cities contribute more to similarity than common ones.
3. Blend TF-IDF cosine similarity (categorical fit) with normalized
   numeric closeness on cost and rating (quality / budget fit) into a
   single weighted score.
4. Filter candidates by hard user constraints (budget ceiling, delivery,
   table booking) before ranking, so results always respect stated
   preferences even if their text similarity is high.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RestaurantRecommender:
    """Content-based recommender using TF-IDF + cosine similarity."""

    def __init__(self, df: pd.DataFrame):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Cleaned restaurant dataframe. Must contain: Restaurant Name,
            City, Cuisines, Average Cost for two, Price range,
            Has Table booking, Has Online delivery, Aggregate rating,
            Votes.
        """
        self.df = df.reset_index(drop=True).copy()
        self._build_index()

    def _build_index(self):
        """Fit the TF-IDF vectorizer over the composite content profile."""
        self.df["content_profile"] = (
            self.df["Cuisines"].astype(str) + " " +
            self.df["City"].astype(str) + " " +
            ("PriceRange" + self.df["Price range"].astype(str))
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["content_profile"])

        # Precompute normalized numeric columns for the quality/budget blend
        self.norm_cost = self._normalize(self.df["Average Cost for two"])
        self.norm_rating = self._normalize(self.df["Aggregate rating"])
        self.norm_votes = self._normalize(self.df["Votes"])

    @staticmethod
    def _normalize(series: pd.Series) -> pd.Series:
        rng = series.max() - series.min()
        if rng == 0:
            return series * 0
        return (series - series.min()) / rng

    def recommend_restaurants(self, cuisine: str = None, city: str = None,
                               budget: float = None, price_range: int = None,
                               online_delivery: bool = None,
                               table_booking: bool = None,
                               top_n: int = 10) -> pd.DataFrame:
        """
        Recommend restaurants matching the given preferences.

        Parameters
        ----------
        cuisine : str, optional
            Desired cuisine, e.g. 'Italian'.
        city : str, optional
            Desired city.
        budget : float, optional
            Maximum "Average Cost for two" the user is willing to pay.
        price_range : int, optional
            Desired Zomato price tier (1-4).
        online_delivery : bool, optional
            Require online delivery if True.
        table_booking : bool, optional
            Require table booking availability if True.
        top_n : int
            Number of recommendations to return.

        Returns
        -------
        pd.DataFrame
            Top-N restaurants with a 'Similarity Score' column, sorted
            descending.
        """
        candidates = self.df.copy()

        # --- Hard constraint filters (respected exactly, not just weighted) ---
        if city:
            candidates = candidates[candidates["City"].str.lower() == city.lower()]
        if budget is not None:
            candidates = candidates[candidates["Average Cost for two"] <= budget]
        if price_range is not None:
            candidates = candidates[candidates["Price range"] == price_range]
        if online_delivery is not None:
            wanted = "Yes" if online_delivery else "No"
            candidates = candidates[candidates["Has Online delivery"] == wanted]
        if table_booking is not None:
            wanted = "Yes" if table_booking else "No"
            candidates = candidates[candidates["Has Table booking"] == wanted]

        if candidates.empty:
            return pd.DataFrame(columns=list(self.df.columns) + ["Similarity Score"])

        idx = candidates.index

        # --- Build the query profile and compute TF-IDF cosine similarity ---
        query_parts = []
        if cuisine:
            query_parts.append(cuisine)
        if city:
            query_parts.append(city)
        if price_range is not None:
            query_parts.append(f"PriceRange{price_range}")
        query_text = " ".join(query_parts) if query_parts else "Restaurant"

        query_vec = self.vectorizer.transform([query_text])
        content_sim = cosine_similarity(query_vec, self.tfidf_matrix[idx]).flatten()

        # --- Blend with quality signal (rating, votes) so strong performers
        #     are favored among equally-relevant matches ---
        quality_score = (
            0.6 * self.norm_rating.loc[idx].values +
            0.4 * self.norm_votes.loc[idx].values
        )

        final_score = 0.7 * content_sim + 0.3 * quality_score

        results = candidates.copy()
        results["Similarity Score"] = np.round(final_score, 4)
        results = results.sort_values("Similarity Score", ascending=False).head(top_n)

        display_cols = ["Restaurant Name", "City", "Cuisines", "Average Cost for two",
                         "Price range", "Has Online delivery", "Has Table booking",
                         "Aggregate rating", "Votes", "Similarity Score"]
        display_cols = [c for c in display_cols if c in results.columns]
        return results[display_cols].reset_index(drop=True)
