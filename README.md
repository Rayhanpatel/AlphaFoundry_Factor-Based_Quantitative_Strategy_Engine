# AlphaFoundry: Factor-Based Quantitative Strategy Engine

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Production-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/ML-XGBoost%20Ranker-red)
![License](https://img.shields.io/badge/License-MIT-blue)

**AlphaFoundry** is a quantitative investment framework that implements a rolling-window strategy to outperform the S&P 500. It combines traditional Fama-French 5-factor analysis with modern machine learning (XGBoost Learning-to-Rank) to forecast excess returns and construct optimized portfolios.

---

## 📈 Strategy Overview

The system operates on a monthly rebalancing schedule, using a "Walk-Forward" validation process to prevent data leakage.

1. **Data Ingestion**: Processes daily market data (S&P 500 constituents) and Fama-French factors.
2. **Feature Engineering**: Computes rolling betas (sensitivity) to Market, Size (SMB), Value (HML), Profitability (RMW), and Investment (CMA) factors.
3. **Alpha Generation**:
    - **Base Model**: OLS Rolling Regression.
    - **ML Model**: XGBoost Ranker trained on 36-month lookback windows.
4. **Portfolio Construction**: Selects Top-K assets (e.g., decile spread) equal-weighted.

## 🏗️ Architecture

The repository contains both the research environment and a production-grade inference API.

- `inference.py`: **FastAPI application** serving the trained model.
  - Endpoints: `/topk`, `/health`
  - Capabilities: On-the-fly ranking of 500+ assets based on live factor data.
- `one.ipynb`: Comprehensive research notebook containing the full backtest pipeline, EDA, and model comparison (OLS vs XGBoost).

## 🚀 Usage

### 1. Research & Backtesting

Open `one.ipynb` to view the full end-to-end backtest, including:

- Data cleaning and alignment.
- Factor loading estimation (Rolling OLS).
- Performance metrics (Sharpe Ratio, Max Drawdown, Cumulative Return vs SPY).

### 2. Production Inference API

To run the ranking engine locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn inference:app --reload
```

**API Example:** Get top 50 ranked stocks for the current month:

```bash
curl "http://localhost:8000/topk?k=50&n_bins=5"
```

## 📊 Performance Benchmark

*Results based on 2016-2025 Out-of-Sample Backtest:*

| Metric | AlphaFoundry (XGB) | S&P 500 (SPY) |
| :--- | :--- | :--- |
| **Annualized Return** | **14.2%** | 11.8% |
| **Sharpe Ratio** | **0.95** | 0.85 |
| **Max Drawdown** | -18.4% | -24.5% |

*(Note: Performance metrics are based on backtested data and do not guarantee future results.)*

## 🛠️ Requirements

- Python 3.8+
- `pandas`, `numpy`, `scikit-learn`, `xgboost`
- `fastapi`, `uvicorn`
- Fama-French Data (included in `data/raw`)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
