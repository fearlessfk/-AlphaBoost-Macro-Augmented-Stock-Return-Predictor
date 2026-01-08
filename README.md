# 📈 AlphaBoost: Macro-Augmented Stock Return Predictor

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?style=flat&logo=xgboost&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **An industrial-grade time series forecasting engine designed to predict 5-day stock returns.**  
> By integrating Macroeconomic Indicators (VIX, S&P 500) with technical signals, this model achieves robust directional accuracy and superior risk management in volatile markets.

---

## 📊 Performance Highlights (核心业绩)

### 1. High Directional Accuracy (高胜率)
The model focuses on predicting the **direction** (Up/Down) of the 5-day future return, rather than just minimizing error.
- **Directional Accuracy:** Achieved **~72% - 76%** across major tech stocks (AAPL, MSFT, GOOG) during the test period.
- **RMSE Reduction:** Outperformed the Naive Baseline (Mean Forecast) by **~20%**.

### 2. Strategy Backtest & Risk Control (回测与风控)
A simulated trading strategy (Long when predicted `Up`, Cash when `Down`) was conducted to evaluate financial performance.

#### ✅ Case Study: TSLA (High Volatility Regime)
In highly volatile assets like Tesla, the model significantly outperformed the "Buy & Hold" strategy by capturing price swings.
- **Strategy Return:** **+138.69%** 🚀
- **Benchmark Return:** +76.46%

![TSLA Backtest](backtest_tsla.png)
*(Strategy Equity Curve vs. Buy & Hold Benchmark)*

#### 🛡️ Risk Management: AAPL (Bull Market Regime)
In strong linear bull markets, while the strategy may trail the benchmark in total return, it demonstrates superior capital preservation.
- **Max Drawdown Reduction:** Reduced Maximum Drawdown from **-33.36%** (Benchmark) to **-22.06%** (Strategy).
- **Insight:** The model successfully identified downturn risks and signaled to hold cash, smoothing out the equity curve.

---

## 🧠 Methodology (核心逻辑)

### 1. Data Pipeline & Macro Integration
Unlike simple autoregressive models, this project ingests a multi-dimensional dataset:
- **Target Asset:** OHLCV data (via `yfinance`).
- **Macro Factors:** **S&P 500 (`^GSPC`)** for market beta and **Volatility Index (`^VIX`)** for risk sentiment.
- **Data Engineering:** Automated caching system to prevent API rate limits.

### 2. Rigorous Feature Engineering
Constructed 20+ features to capture market dynamics:
- **Momentum:** RSI (14), MACD, Percentage Change.
- **Trend:** SMA/EMA ratios, Rolling Means.
- **Lags:** Shifted features (Lag_1, Lag_2) to capture temporal dependencies.
- **Alpha/Beta Separation:** Calculated excess returns relative to the S&P 500.

### 3. Walk-Forward Validation (No Data Leakage)
To strictly prevent **Look-ahead Bias**, the model uses **Rolling Window Validation** (TimeSeriesSplit) instead of random shuffling.
- **Target Variable:** `Return_5d = Price(t+5) / Price(t) - 1` (Strictly physically isolated future data).

---

## 🛠️ Key Technologies

- **Modeling:** `XGBoost` (Gradient Boosting Decision Trees)
- **Data Processing:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **Financial Data:** `yfinance` API

---

## 📉 Feature Importance

What drives the prediction? The model identifies **Moving Average Ratios** and **Lagged Returns** as the most significant predictors, followed by **RSI** and **VIX**.

![Feature Importance](feature_importance.png)

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fearlessfk/AlphaBoost-Stock-Prediction.git
