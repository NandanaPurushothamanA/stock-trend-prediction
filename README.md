# Stock Market Trend Prediction

Tests whether a stock's closing price can be predicted to go **UP or DOWN on the next trading day**, using **250,000+ daily records of 49 NIFTY 50 stocks (2005–2026)**. Engineers 13 technical features with pandas, explores them with Seaborn, and compares three classification models against a simple baseline using a time-based split.

**Tech stack:** Python · pandas · NumPy · scikit-learn · Seaborn · Matplotlib · yfinance

## How it works

| Step | What happens |
|---|---|
| 1. Load data | Daily OHLCV data for NIFTY 50 stocks downloaded with `yfinance` (cached to `data/`), or your own CSV |
| 2. Cleaning | Parse dates, sort by stock and date, remove duplicates, missing values and invalid rows |
| 3. Feature engineering | Daily and 5-day returns, price vs 5/20-day moving averages, 10-day volatility, high–low range, open–close change, volume vs 20-day average, RSI(14), lagged returns |
| 4. Target | `1` if the next day's close is higher than today's, otherwise `0` |
| 5. EDA | Price trends, distribution of daily returns, and a feature correlation heatmap with Seaborn |
| 6. Time-based split | Train on the earliest 80% of dates, test on the latest 20%, **no shuffling** |
| 7. Model training | Logistic Regression and Linear SVM (with `StandardScaler`), Random Forest |
| 8. Evaluation | Accuracy, precision, recall, F1-score and a confusion matrix, compared with an "always predict the majority class" baseline |

## Project structure

```
stock-trend-prediction/
├── data/              # downloaded price data is cached here
├── images/            # plots and results table saved by the notebook
├── stock_trend_prediction.ipynb   # full pipeline: load → clean → features → EDA → training → evaluation
└── requirements.txt
```

## How to run

1. Install the libraries: `pip install -r requirements.txt`
2. Start Jupyter: `jupyter notebook`, open `stock_trend_prediction.ipynb`, then **Kernel → Restart & Run All**
3. The first run downloads NIFTY 50 data from 2005 (a few minutes) and caches it in `data/`. For a quicker run, set `START_DATE = "2015-01-01"` in the first code cell.

## Results

Data: **251,229 rows** after cleaning and feature engineering. **Train:** 198,805 rows (before June 2022) · **Test:** 52,424 rows (June 2022 onwards).

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Baseline (always predict UP)** | **0.511** | 0.511 | 1.000 | **0.676** |
| Logistic Regression | 0.509 | 0.511 | 0.892 | 0.650 |
| SVM (Linear) | 0.510 | 0.511 | 0.891 | 0.650 |
| Random Forest | 0.508 | 0.512 | 0.751 | 0.609 |

### Key finding
**None of the models beat the simple baseline.** Every feature has a correlation below 0.02 with next-day direction, and the models mostly learn to predict "UP" (stocks rose on 50.8% of days).

This is consistent with next-day price movements being close to random: past prices and common technical indicators alone carry almost no predictive signal. It also shows why the evaluation setup matters. With a random (shuffled) split or no baseline, similar models can appear far more accurate than they really are.

### Feature correlation
![Correlation heatmap](images/correlation_heatmap.png)

### Model comparison
![Model comparison](images/model_comparison.png)

### Feature importance (Random Forest)
![Feature importance](images/feature_importance.png)

## Key decisions

- **Time-based split, not a random split:** shuffling would let the model train on future prices and test on the past, which inflates scores. Real predictions only ever use past data.
- **Baseline comparison:** daily stock direction is close to random, so a model is only useful if it beats "always predict UP". Accuracy around 50–55% is normal for this problem; anything much higher usually means data leakage.

## Possible next steps
- Predict longer horizons (e.g. 5- or 20-day direction), where trends may be more predictable
- Add market-wide and external features (NIFTY index, sector performance, news sentiment)
- Try gradient boosting (XGBoost / LightGBM) and walk-forward validation
- **Scaling for LR and SVM only:** both are sensitive to feature scale; Random Forest is not.
- **Ratio-based features:** features like price/moving-average are comparable across stocks with very different price levels.
