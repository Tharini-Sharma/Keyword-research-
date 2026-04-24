# Keyword Research using Python

A complete, copy-paste-ready Machine Learning, Data Analytics and Visualization project on **Keyword Research** built with Python.

> Sources used:
> - [PaulHex6/google-trends](https://github.com/PaulHex6/google-trends)
> - [MuhammadAhmed-0/Keyword-Research-tool-python](https://github.com/MuhammadAhmed-0/Keyword-Research-tool-python)
> - [SEO Sample Data (Kaggle)](https://www.kaggle.com/datasets/muhammetvarl/seo-sample-data)
> - [Google Trends Dataset (Kaggle)](https://www.kaggle.com/datasets/dhruvildave/google-trends-dataset)

## What's inside

| File | Purpose |
|------|---------|
| `keyword_research.ipynb` | Jupyter notebook — best for Google Colab |
| `keyword_research.py`    | Plain Python script — best for VS Code |
| `seo_sample.csv`         | Sample SEO keyword dataset |
| `trends.csv`             | Sample Google Trends weekly data |
| `requirements.txt`       | Pip install list |
| `README.md`              | This file |

## Run on Google Colab (easiest, no install)

1. Open <https://colab.research.google.com> → **File → Upload notebook**
2. Upload `keyword_research.ipynb`
3. (Optional) **Files** panel → upload `seo_sample.csv` and `trends.csv`
4. **Runtime → Run all**

## Run locally in VS Code

```bash
git clone https://github.com/<your-username>/keyword-research-python.git
cd keyword-research-python

python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
python keyword_research.py
```

All charts and an enriched CSV will be written to `./outputs/`.

## Project workflow

```
Kaggle SEO data ─┐
                 ├─►  Pandas EDA ─►  Visualizations (8+ charts) ─►  ML models
Google Trends   ─┘                                                  - KMeans
(pytrends)                                                          - Regression
```

## Visualizations produced

1. Top 15 keywords by search volume (bar)
2. Google Trends interest over time (multi-line)
3. Correlation heatmap (Search Volume / CPC / Competition)
4. CPC vs Search Volume scatter (bubble = competition)
5. Category distribution (pie)
6. Word cloud of all keywords
7. Cluster sizes (KMeans)
8. Linear Regression — actual vs predicted

## Machine learning models

- **KMeans clustering** on TF-IDF features of the keywords
- **Linear Regression** to predict `Search Volume` from `CPC` and `Competition`

## Push to GitHub

```bash
git init
git add .
git commit -m "Keyword research with Python"
git branch -M main
git remote add origin https://github.com/<your-username>/keyword-research-python.git
git push -u origin main
```

## License

MIT
