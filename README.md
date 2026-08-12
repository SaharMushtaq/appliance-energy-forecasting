# Appliance Energy Forecasting

Reproducible time-series forecasting pipeline for household appliance energy use, comparing
benchmark models, SARIMAX, a feature-based machine learning model, and a time-series
foundation model (Chronos).

## Project aim

Forecast short-term household appliance energy use 24 hours ahead, and evaluate whether
increasingly complex models improve on simple benchmark methods.

The main questions addressed:
1. How well do simple benchmark models forecast appliance energy use?
2. Does a SARIMAX model improve on the benchmark forecasts?
3. Do sensor, weather, and time-based covariates improve forecast accuracy?
4. Does a feature-based machine-learning model improve performance?
5. Does a time-series foundation model (Chronos) provide any additional benefit?
6. Which model is most suitable for practical smart-home energy forecasting?

## Dataset

**Appliances Energy Prediction** dataset (UCI Machine Learning Repository), sampled at
10-minute intervals and resampled to hourly resolution for this project.

Source: Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017). Data driven prediction
models of energy use of appliances in a low-energy house. *Energy and Buildings*, 140, pp.81-97.

Target variable: `Appliances` (Wh)

## Forecasting task

- Forecast horizon: 24 hours ahead
- Train/test split: final 14 days held out as test set
- Evaluation metrics: MAE, RMSE, MASE, Bias

## Repository structure

```
appliance-energy-forecasting/
├── README.md
├── notebooks/
│   ├── 01_data_download_and_cleaning.ipynb
│   ├── 02_benchmark_models.ipynb
│   ├── 03_sarimax_model.ipynb
│   ├── 04_feature_model.ipynb
│   └── 05_foundation_model_and_comparison.ipynb
├── data/
│   └── processed/
│       └── appliance_hourly.csv
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   └── metrics/
└── reports/
    └── Appliance_Energy_Forecasting_Report.docx
```

## Models

### 1. Benchmark models
Mean, naive, daily seasonal naive (lag 24), weekly seasonal naive (lag 168), and drift
forecasts.

### 2. SARIMAX
Non-seasonal order (p, d, q) selected via AIC grid search (p in [0,6], d in [0,2],
q in [0,6]). Seasonal order fixed at (1, 1, 1, 24) based on stationarity testing and
ACF/PACF analysis. Exogenous variables: outdoor temperature, humidity, windspeed,
visibility, dewpoint.

### 3. Feature-based model
`HistGradientBoostingRegressor` (scikit-learn), using lag features (1-168 hours), rolling
mean/std features, and time-of-day/day-of-week encodings. All lag and rolling features are
shifted before computation to prevent data leakage.

### 4. Foundation model
`chronos-t5-small` (Amazon Chronos), applied zero-shot with a 512-hour rolling context
window, target-only (no exogenous covariates).

## Results

| Model | MAE | RMSE | MASE | Bias |
|---|---|---|---|---|
| Feature-based (HistGB) | 33.042 | 55.395 | 0.641 | 3.749 |
| Foundation model (Chronos) | 36.677 | 74.502 | 0.687 | -29.050 |
| SARIMAX | 36.807 | 65.452 | 0.689 | -6.517 |
| Seasonal naive (weekly) | 43.457 | 81.409 | 0.813 | -13.160 |
| Seasonal naive (daily) | 48.309 | 85.565 | 0.904 | 1.751 |
| Mean | 50.258 | 74.938 | 0.941 | -3.287 |
| Naive | 85.551 | 110.390 | 1.601 | 50.977 |
| Drift | 85.802 | 110.679 | 1.606 | 51.368 |

Full discussion in `reports/Appliance_Energy_Forecasting_Report.docx`.

## Running the notebooks

Each notebook is self-contained and designed to run in Google Colab:

1. Open the notebook in Colab (File > Upload notebook, or open directly from GitHub)
2. Run all cells top to bottom
3. The first code cell mounts Google Drive and creates/uses a project folder at
   `/content/drive/MyDrive/appliance-energy-forecasting` for persistent storage of data,
   figures, forecasts and metrics between notebooks
4. Run notebooks in order: 01 -> 02 -> 03 -> 04 -> 05, since later notebooks load data
   and forecasts saved by earlier ones

## Data leakage precautions

- All lag and rolling features are computed with `.shift(1)` applied before any rolling
  window, so no feature can access the current or a future value of the target
- Train/test split is strictly chronological (final 14 days held out)
- Model selection (SARIMAX order via AIC) is performed on training data only

## Known limitations

- SARIMAX and the feature-based model use realised (actual) weather values from the test
  set as exogenous inputs rather than genuine forecasts, so results should be interpreted
  as conditional forecasts rather than fully operational forecasts
- Chronos was used zero-shot with no fine-tuning and without access to exogenous covariates
- The SARIMAX AIC grid search was run on a 60-day training subsample for computational
  tractability in Colab, then refit on the full training set

## References

- Ansari, A.F. et al. (2024). Chronos: Learning the Language of Time Series. arXiv:2403.07815.
- Box, G.E.P. et al. (2015). Time Series Analysis: Forecasting and Control. 5th ed. Wiley.
- Candanedo, L.M., Feldheim, V. and Deramaix, D. (2017). Energy and Buildings, 140, pp.81-97.
- Hyndman, R.J. and Athanasopoulos, G. (2021). Forecasting: Principles and Practice. 3rd ed. OTexts.
- Hyndman, R.J. and Koehler, A.B. (2006). International Journal of Forecasting, 22(4), pp.679-688.
