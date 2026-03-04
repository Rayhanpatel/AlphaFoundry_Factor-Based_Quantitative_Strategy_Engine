# AlphaFoundry

Quantitative investment framework combining Fama-French 5-factor analysis with XGBoost Learning-to-Rank. Implements a rolling-window strategy that outperforms the S&P 500 on backtested data.

## Strategy

Monthly rebalancing with walk-forward validation to prevent data leakage:

1. **Data Ingestion** — Daily market data (S&P 500 constituents) + Fama-French factors
2. **Feature Engineering** — Rolling betas for Market, Size (SMB), Value (HML), Profitability (RMW), Investment (CMA)
3. **Alpha Generation** — OLS rolling regression baseline → XGBoost Ranker on 36-month lookback windows
4. **Portfolio Construction** — Top-K decile spread, equal-weighted

## Results (2016–2025 out-of-sample)

| Metric | AlphaFoundry (XGB) | S&P 500 (SPY) |
|---|---|---|
| Annualized Return | **14.2%** | 11.8% |
| Sharpe Ratio | **0.95** | 0.85 |
| Max Drawdown | -18.4% | -24.5% |

## Architecture

- `inference.py` — FastAPI production API. Endpoints: `/topk`, `/health`. On-the-fly ranking of 500+ assets.
- `one.ipynb` — Research notebook: full backtest pipeline, EDA, model comparison (OLS vs XGBoost).

## Run

```bash
pip install -r requirements.txt

# Research notebook
jupyter notebook one.ipynb

# Production API
uvicorn inference:app --reload
curl "http://localhost:8000/topk?k=50&n_bins=5"
```

## Requirements

Python 3.8+, pandas, numpy, scikit-learn, xgboost, fastapi, uvicorn

## License

MIT
