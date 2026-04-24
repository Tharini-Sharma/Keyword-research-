"""
================================================================================
 Keyword Research using Python — Machine Learning, Data Analytics & Visualization
================================================================================
 Author : Your Name
 Project: Keyword Research with Python
 Sources:
   - PaulHex6/google-trends            https://github.com/PaulHex6/google-trends
   - MuhammadAhmed-0/Keyword-Research-tool-python
       https://github.com/MuhammadAhmed-0/Keyword-Research-tool-python
   - SEO Sample Data (Kaggle)
       https://www.kaggle.com/datasets/muhammetvarl/seo-sample-data
   - Google Trends Dataset (Kaggle)
       https://www.kaggle.com/datasets/dhruvildave/google-trends-dataset

 How to run:
   1) Google Colab :  Upload this file (or the .ipynb), then run cells top to
                     bottom. The first cell installs all required libraries.
   2) VS Code      :  Create a virtual env, run `pip install -r requirements.txt`
                     and execute `python keyword_research.py`.

 What this script does:
   - Loads SEO keyword sample data
   - Pulls Google Trends data (via pytrends) for a list of seed keywords
   - Cleans, transforms and explores the data (EDA)
   - Produces 6+ visualizations (bar, line, heatmap, scatter, pie, wordcloud)
   - Trains 2 ML models:
        * KMeans clustering of keywords (TF-IDF features)
        * Linear Regression to predict Search Volume from CPC + Competition
   - Saves all charts to ./outputs/

================================================================================
"""

# -----------------------------------------------------------------------------
# 1. INSTALL DEPENDENCIES (Colab users: uncomment the next line)
# -----------------------------------------------------------------------------
# !pip install pandas numpy matplotlib seaborn plotly scikit-learn wordcloud pytrends

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams["figure.figsize"] = (10, 5)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 2. LOAD THE SEO SAMPLE DATASET (Kaggle: muhammetvarl/seo-sample-data)
# -----------------------------------------------------------------------------
# In Colab use:
#   from google.colab import files
#   files.upload()                           # upload seo_sample.csv
# In VS Code, place the CSV next to this script.
#
# If the CSV is not present we generate a realistic in-memory sample so the
# notebook always runs end-to-end.

CSV_PATH = "seo_sample.csv"

if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows from {CSV_PATH}")
else:
    print("seo_sample.csv not found — using built-in sample dataset.")
    sample_keywords = [
        "python tutorial", "machine learning", "data science", "deep learning",
        "pandas dataframe", "numpy array", "scikit learn", "tensorflow",
        "keras model", "data visualization", "matplotlib chart", "seaborn heatmap",
        "kaggle competition", "jupyter notebook", "google colab", "linear regression",
        "logistic regression", "kmeans clustering", "tfidf vectorizer",
        "natural language processing", "computer vision", "neural network",
        "decision tree", "random forest", "xgboost", "feature engineering",
        "data cleaning", "exploratory data analysis", "seo keywords",
        "keyword research tool", "google trends", "search volume", "long tail keyword",
        "low competition keyword", "content marketing", "blog post ideas",
        "youtube seo", "instagram seo", "ecommerce seo", "local seo",
        "backlink analysis", "domain authority", "page speed", "mobile first",
        "schema markup", "meta description", "title tag optimization",
        "google analytics", "search console", "ahrefs alternative", "semrush alternative"
    ]
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Keyword": sample_keywords,
        "Search Volume": rng.integers(150, 50000, size=len(sample_keywords)),
        "CPC": np.round(rng.uniform(15, 1000, size=len(sample_keywords)), 0),  # CPC in INR (₹)
        "Competition": np.round(rng.uniform(0.05, 0.95, size=len(sample_keywords)), 2),
        "Category": rng.choice(
            ["Programming", "ML", "Data", "SEO", "Marketing"],
            size=len(sample_keywords)
        )
    })

print(df.head())
print("\nShape:", df.shape)
print("\nNumeric summary:\n", df.describe())


# -----------------------------------------------------------------------------
# 3. (OPTIONAL) PULL LIVE GOOGLE TRENDS DATA WITH pytrends
# -----------------------------------------------------------------------------
# Comment this block out if you have no internet (Kaggle Trends CSV is used as
# a fallback in section 4).
try:
    from pytrends.request import TrendReq
    pytrends = TrendReq(hl="en-US", tz=360)
    seeds = ["python", "machine learning", "data science",
             "deep learning", "seo"]
    pytrends.build_payload(seeds, cat=0, timeframe="today 12-m", geo="", gprop="")
    trends_df = pytrends.interest_over_time()
    if "isPartial" in trends_df.columns:
        trends_df = trends_df.drop(columns=["isPartial"])
    print("\nLive Google Trends data:\n", trends_df.tail())
except Exception as e:
    print(f"\npytrends fetch skipped ({type(e).__name__}). "
          f"Falling back to Kaggle CSV in section 4.")
    trends_df = None


# -----------------------------------------------------------------------------
# 4. LOAD KAGGLE GOOGLE TRENDS DATASET (dhruvildave/google-trends-dataset)
# -----------------------------------------------------------------------------
# If you downloaded `trends.csv` from Kaggle, place it next to this script.
TRENDS_PATH = "trends.csv"
if trends_df is None and os.path.exists(TRENDS_PATH):
    trends_df = pd.read_csv(TRENDS_PATH)
    print(f"\nLoaded Kaggle trends data: {trends_df.shape}")
elif trends_df is None:
    # synthetic fallback so the rest of the notebook always runs
    dates = pd.date_range("2025-01-01", periods=52, freq="W")
    trends_df = pd.DataFrame({
        "date": dates,
        "python": np.clip(70 + 10 * np.sin(np.arange(52) / 4) + np.random.randn(52) * 4, 40, 100).astype(int),
        "machine learning": np.clip(60 + 8 * np.cos(np.arange(52) / 5) + np.random.randn(52) * 3, 30, 100).astype(int),
        "data science": np.clip(55 + 5 * np.sin(np.arange(52) / 6) + np.random.randn(52) * 3, 30, 100).astype(int),
        "deep learning": np.clip(45 + 12 * np.sin(np.arange(52) / 3) + np.random.randn(52) * 4, 20, 100).astype(int),
        "seo": np.clip(35 + 6 * np.cos(np.arange(52) / 7) + np.random.randn(52) * 3, 15, 90).astype(int),
    }).set_index("date")
    print("\nSynthetic trends data generated (no internet, no CSV).")


# -----------------------------------------------------------------------------
# 5. EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------------
print("\nNull values:\n", df.isnull().sum())
print("\nKeywords per category:\n", df["Category"].value_counts())


# -----------------------------------------------------------------------------
# 6. VISUALIZATION 1 — Top 15 keywords by Search Volume
# -----------------------------------------------------------------------------
top15 = df.nlargest(15, "Search Volume")
plt.figure(figsize=(11, 6))
sns.barplot(data=top15, x="Search Volume", y="Keyword", palette="viridis")
plt.title("Top 15 Keywords by Search Volume", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_top_keywords.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 7. VISUALIZATION 2 — Search interest over time (line chart)
# -----------------------------------------------------------------------------
plt.figure(figsize=(11, 5))
for col in trends_df.columns:
    plt.plot(trends_df.index, trends_df[col], label=col, linewidth=2)
plt.title("Google Trends — Interest Over Time", fontsize=14, weight="bold")
plt.xlabel("Date"); plt.ylabel("Interest (0–100)")
plt.legend(loc="upper right"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_trends_over_time.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 8. VISUALIZATION 3 — Correlation heatmap of numeric columns
# -----------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(df[["Search Volume", "CPC", "Competition"]].corr(),
            annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation between Search Volume, CPC and Competition")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_correlation_heatmap.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 9. VISUALIZATION 4 — Scatter (CPC vs Search Volume), bubble = Competition
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["CPC"], df["Search Volume"],
            s=df["Competition"] * 400, c=df["Competition"],
            cmap="viridis", alpha=0.75, edgecolors="white")
plt.colorbar(label="Competition")
plt.xlabel("CPC (INR \u20b9)"); plt.ylabel("Search Volume")
plt.title("CPC vs Search Volume (bubble size = Competition)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_cpc_vs_volume.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 10. VISUALIZATION 5 — Category distribution
# -----------------------------------------------------------------------------
plt.figure(figsize=(7, 7))
df["Category"].value_counts().plot.pie(autopct="%1.1f%%", colors=sns.color_palette("viridis"))
plt.title("Keyword Categories"); plt.ylabel("")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_category_pie.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 11. VISUALIZATION 6 — Word cloud of all keywords
# -----------------------------------------------------------------------------
wc_text = " ".join(df["Keyword"].tolist())
wc = WordCloud(width=1000, height=500, background_color="white",
               colormap="viridis", collocations=False).generate(wc_text)
plt.figure(figsize=(12, 6))
plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
plt.title("Keyword Cloud", fontsize=14, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_wordcloud.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 12. MACHINE LEARNING — KMeans clustering on TF-IDF keyword vectors
# -----------------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Keyword"])

K = 5
km = KMeans(n_clusters=K, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)

print("\nKeywords per cluster:\n", df["Cluster"].value_counts())
print("\nSample clusters:")
for k in range(K):
    members = df[df["Cluster"] == k]["Keyword"].head(5).tolist()
    print(f"  Cluster {k}: {members}")


# -----------------------------------------------------------------------------
# 13. VISUALIZATION 7 — Cluster sizes
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Cluster", palette="viridis")
plt.title("Number of Keywords per Cluster")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_clusters.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 14. MACHINE LEARNING — Linear Regression (predict Search Volume)
# -----------------------------------------------------------------------------
features = df[["CPC", "Competition"]]
target = df["Search Volume"]

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.25, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Linear Regression results ---")
print(f"  R² score        : {r2_score(y_test, y_pred):.3f}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.1f}")
print(f"  Coefficients    : CPC={model.coef_[0]:.1f}, Comp={model.coef_[1]:.1f}")
print(f"  Intercept       : {model.intercept_:.1f}")


# -----------------------------------------------------------------------------
# 15. VISUALIZATION 8 — Actual vs Predicted
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="#7e22ce")
mn, mx = y_test.min(), y_test.max()
plt.plot([mn, mx], [mn, mx], "r--", label="Perfect prediction")
plt.xlabel("Actual Search Volume"); plt.ylabel("Predicted Search Volume")
plt.title("Linear Regression — Actual vs Predicted")
plt.legend(); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_regression.png", dpi=140)
plt.show()


# -----------------------------------------------------------------------------
# 16. SAVE FINAL ENRICHED DATASET
# -----------------------------------------------------------------------------
df.to_csv(f"{OUTPUT_DIR}/keywords_enriched.csv", index=False)
print(f"\nDone! All charts and the enriched CSV are in ./{OUTPUT_DIR}/")
