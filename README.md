# 🍽️ Restaurant Intelligence Suite
### AI-Powered Restaurant Analytics & Recommendation System

An end-to-end machine learning project built for the **Cognifyz Technologies Machine Learning
Internship**, transforming raw restaurant data into predictive models, an intelligent
recommendation engine, and interactive geographic business intelligence.

---

## 📌 Project Overview

Restaurant Intelligence Suite covers four complete ML tasks on a 9,551-restaurant dataset
spanning 15 countries and 141 cities:

| Task | Notebook | What it does |
|---|---|---|
| 1 | `Task1_RatingPrediction.ipynb` | Predicts a restaurant's Aggregate Rating using regression |
| 2 | `Task2_RestaurantRecommendation.ipynb` | Recommends restaurants via content-based filtering (TF-IDF + cosine similarity) |
| 3 | `Task3_CuisineClassification.ipynb` | Classifies a restaurant's primary cuisine from structured attributes |
| 4 | `Task4_LocationAnalysis.ipynb` | Geographic analysis with interactive Folium maps and city/locality insights |

Every notebook follows the same 12-section structure: Introduction → Business Problem →
Objective → Dataset Understanding → Data Cleaning → EDA → Feature Engineering → Model
Development → Model Evaluation → Business Insights → Conclusion → Future Improvements.

---

## 📂 Project Structure

```
Restaurant-Intelligence-Suite/
├── data/
│   └── restaurant_dataset.csv          # Source dataset
├── notebooks/
│   ├── Task1_RatingPrediction.ipynb
│   ├── Task2_RestaurantRecommendation.ipynb
│   ├── Task3_CuisineClassification.ipynb
│   └── Task4_LocationAnalysis.ipynb
├── src/
│   ├── utils.py                        # Loading, profiling, plot styling
│   ├── preprocessing.py                # Cleaning, outlier capping, encoding, scaling
│   ├── feature_engineering.py          # Derived features (cuisine count, votes/cost, tiers)
│   ├── visualization.py                # Reusable, publication-quality plot functions
│   ├── models.py                       # Model factory + train/evaluate/compare helpers
│   └── recommendation.py               # RestaurantRecommender (TF-IDF + cosine similarity)
├── outputs/
│   ├── figures/                        # Saved PNG charts (300 dpi)
│   ├── models/                         # Trained model, scaler, and encoder artifacts (.pkl)
│   └── reports/                        # Model comparison CSVs + interactive HTML maps
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Restaurant-Intelligence-Suite.git
cd Restaurant-Intelligence-Suite

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter and run any notebook in order
jupyter notebook notebooks/
```

Each notebook is self-contained: run all cells top-to-bottom. `src/` modules are imported via a
relative path, so notebooks must be run from inside the `notebooks/` directory (the default when
opened through Jupyter).

---

## 🧠 Task 1 — Restaurant Rating Prediction

**Approach:** Cleaned data (removed unrated restaurants, capped outliers), engineered features
(cuisine count, votes-per-cost, city density), label-encoded categoricals, and trained five
regression models.

**Results (test set):**

| Model | MAE | RMSE | R² Score |
|---|---|---|---|
| **XGBoost** | **0.243** | **0.325** | **0.658** |
| Gradient Boosting | 0.245 | 0.331 | 0.647 |
| Random Forest | 0.247 | 0.332 | 0.645 |
| Decision Tree | 0.284 | 0.381 | 0.531 |
| Linear Regression | 0.334 | 0.425 | 0.416 |

**Winner: XGBoost** — explains ~66% of rating variance, with `Votes` and `Price range` emerging
as the top predictive features. Tree ensembles substantially outperform linear regression,
confirming non-linear interactions between engagement, price, and rating.

---

## 🔎 Task 2 — Restaurant Recommendation System

A **content-based recommender** (`src/recommendation.py`) that:
1. Hard-filters candidates by city, budget ceiling, price range, delivery/table-booking preference.
2. Vectorizes `Cuisines + City + PriceRange` with **TF-IDF** and scores cosine similarity to the
   user's query.
3. Blends 70% content similarity with 30% quality signal (rating + votes) into a final score.
4. Returns the top-N restaurants via `recommend_restaurants()`.

```python
from recommendation import RestaurantRecommender

recommender = RestaurantRecommender(df)
recommender.recommend_restaurants(
    cuisine="Italian", city="New Delhi", budget=1500, top_n=10
)
```

---

## 🏷️ Task 3 — Cuisine Classification

**Approach:** Used `Primary Cuisine` (top 15 classes + "Other") as a clean single-label target
to handle the multi-label, high-cardinality raw `Cuisines` field, then trained four classifiers
on non-cuisine attributes (location, cost, votes, rating, service options).

**Results (test set):**

| Model | Accuracy | Weighted F1 |
|---|---|---|
| **XGBoost** | **0.401** | **0.358** |
| Random Forest | 0.401 | 0.335 |
| Decision Tree | 0.358 | 0.322 |
| Logistic Regression | 0.360 | 0.249 |

**Winner: XGBoost.** Performance is naturally bounded — cuisine is primarily a menu attribute,
not a location/price attribute — but the model still meaningfully beats the majority-class
baseline, especially on the dominant cuisine classes.

---

## 🗺️ Task 4 — Location-Based Analysis

Analyzed restaurant density, pricing, and rating patterns across cities/localities and produced:
- **Restaurant density map** with marker clustering (`outputs/reports/task4_restaurant_density_map.html`)
- **Rating heatmap** highlighting quality clusters (`outputs/reports/task4_rating_heatmap.html`)

**Key insight:** Restaurant volume, cost, and rating are only loosely coupled by geography —
some high-volume cities post below-average ratings, and the most expensive cities are largely
distinct from the highest-volume ones, pointing to separate "scale" and "premium" market
strategies.

---

## 🛠️ Tech Stack

- **Data:** pandas, numpy
- **Modeling:** scikit-learn, XGBoost
- **Visualization:** matplotlib, seaborn, plotly, folium
- **Environment:** Jupyter Notebook

---

## 📈 Future Improvements

- Hybrid recommender combining content-based and collaborative filtering once interaction data exists
- Multi-label cuisine classification using the full `Cuisines` field
- Hyperparameter tuning (GridSearchCV / Optuna) on the winning models
- External demographic/income enrichment for deeper location insights

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as part of the **Cognifyz Technologies Machine Learning Internship** program.
