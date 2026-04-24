# Keyword Research using Python

A complete, copy-paste-ready Machine Learning, Data Analytics and Visualization project on **Keyword Research** built with Python.

> Sources used:
> - [PaulHex6/google-trends](https://github.com/PaulHex6/google-trends)
> - [MuhammadAhmed-0/Keyword-Research-tool-python](https://github.com/MuhammadAhmed-0/Keyword-Research-tool-python)
> - [SEO Sample Data (Kaggle)](https://www.kaggle.com/datasets/muhammetvarl/seo-sample-data)
> - [Google Trends Dataset (Kaggle)](https://www.kaggle.com/datasets/dhruvildave/google-trends-dataset)


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



## License

MIT
