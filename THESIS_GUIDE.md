# NSE Stock Closing Price Prediction - Thesis Guide

## Overview

This system predicts the **next day's closing price** for stocks listed on the Nairobi Securities Exchange (NSE) using three deep learning and time-series models:

1. **LSTM** (Long Short-Term Memory) - Advanced RNN architecture for capturing long-term dependencies
2. **RNN** (Recurrent Neural Network) - Baseline sequential model
3. **Prophet** - Facebook's time-series forecasting tool designed for business forecasting

## Research Objective

**To compare the effectiveness of LSTM, RNN, and Prophet models in predicting daily closing prices of NSE stocks.**

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│         DATA COLLECTION LAYER                           │
│  - Selenium scraper (every 30 seconds)                  │
│  - Real-time market data from fib.co.ke                 │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         ETL & STORAGE LAYER                             │
│  - Parse and clean HTML data                            │
│  - Store in TimescaleDB (PostgreSQL)                    │
│  - Intraday data (30-second intervals)                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         AGGREGATION LAYER                               │
│  - Convert intraday to daily closing prices             │
│  - Extract: Close, High, Low, Volume                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         PREDICTION MODELS                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   LSTM   │  │   RNN    │  │ Prophet  │             │
│  └──────────┘  └──────────┘  └──────────┘             │
│                                                          │
│  - 30-day lookback window                               │
│  - Train/test split (80/20)                             │
│  - Early stopping & learning rate reduction             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         EVALUATION & COMPARISON                         │
│  - RMSE, MAE, MAPE, R², MSE                            │
│  - Model comparison dashboard                           │
│  - Best model selection per stock                       │
└─────────────────────────────────────────────────────────┘
```

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- TensorFlow 2.13+ (for LSTM/RNN)
- Prophet (for Prophet model)
- pandas, numpy (data processing)
- psycopg2 (PostgreSQL connection)
- scikit-learn (preprocessing & metrics)

### 2. Database Setup

Make sure PostgreSQL with TimescaleDB is running:

```bash
# Check if database is accessible
psql -U postgres -d nse -c "SELECT COUNT(*) FROM stocksdata;"
```

Your database should contain intraday stock data collected by the scraper.

### 3. Verify Data Availability

```bash
# Check how many days of data you have
python -c "
from closing_price_pipeline import DailyDataAggregator
import os
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

agg = DailyDataAggregator(DB_CONFIG)
df = agg.get_daily_closing_prices(days_back=90)
print(f'Total records: {len(df)}')
print(f'Symbols: {df[\"symbol\"].nunique()}')
print(f'Date range: {df[\"trading_date\"].min()} to {df[\"trading_date\"].max()}')
"
```

**Minimum Requirements:**
- At least **60 days** of historical data per stock
- Recommended: **90+ days** for better model training

---

## Usage

### Quick Start (Top 5 Stocks)

```bash
python run_predictions.py
```

This will:
1. Fetch daily closing prices for all stocks
2. Select top 5 stocks by data availability
3. Train LSTM, RNN, and Prophet models for each
4. Save models to `models/` directory
5. Save comparison results to `results/model_comparison.csv`

### Predict Specific Stocks

```bash
python run_predictions.py --symbols SCOM KCB EQTY BAT ABSA
```

### Process All Available Stocks

```bash
python run_predictions.py --all
```

### Custom Training Parameters

```bash
# Train with 100 epochs and 60-day lookback
python run_predictions.py --epochs 100 --lookback 60 --symbols SCOM
```

---

## Model Details

### 1. LSTM (Long Short-Term Memory)

**Architecture:**
```
Input Layer (30 days × 1 feature)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Dense Layer (1 unit - prediction)
```

**Key Features:**
- Handles long-term dependencies via memory cells
- Gates control information flow (forget, input, output)
- Best for capturing complex temporal patterns
- Optimizer: Adam (learning_rate=0.001)
- Loss: Mean Squared Error

**Why LSTM?**
- Stock prices have long-term trends and cycles
- LSTM remembers important historical patterns
- Proven effective in financial time-series forecasting

### 2. RNN (Simple Recurrent Neural Network)

**Architecture:**
```
Input Layer (30 days × 1 feature)
    ↓
SimpleRNN Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
SimpleRNN Layer (50 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
SimpleRNN Layer (50 units)
    ↓
Dropout (0.2)
    ↓
Dense Layer (25 units)
    ↓
Dense Layer (1 unit - prediction)
```

**Key Features:**
- Baseline sequential model
- Simpler than LSTM (no memory cells)
- Prone to vanishing gradient problem
- Optimizer: Adam (learning_rate=0.001)
- Loss: Mean Squared Error

**Why RNN?**
- Serves as baseline for comparison
- Demonstrates improvement of LSTM over simple RNN
- Computationally cheaper than LSTM

### 3. Prophet

**Configuration:**
```python
Prophet(
    daily_seasonality=False,    # Not needed for daily data
    weekly_seasonality=True,    # Captures day-of-week effects
    yearly_seasonality=False,   # Limited data for yearly patterns
    changepoint_prior_scale=0.05  # Flexibility in trend changes
)
```

**Key Features:**
- Additive model: Trend + Seasonality + Holidays
- Robust to missing data and outliers
- Handles weekly patterns (Monday effect, Friday effect)
- Automatic detection of trend changepoints

**Why Prophet?**
- Industry-standard for business forecasting
- Designed by Facebook for real-world time series
- Interpretable components (trend vs seasonality)
- Non-deep learning baseline

---

## Evaluation Metrics

### 1. RMSE (Root Mean Squared Error)
```
RMSE = √(Σ(y_true - y_pred)² / n)
```
- **Lower is better**
- Penalizes large errors heavily
- Same units as the target variable (KES)

### 2. MAE (Mean Absolute Error)
```
MAE = Σ|y_true - y_pred| / n
```
- **Lower is better**
- Average absolute difference
- Less sensitive to outliers than RMSE

### 3. MAPE (Mean Absolute Percentage Error)
```
MAPE = (100/n) × Σ|y_true - y_pred| / y_true
```
- **Lower is better**
- Percentage error (easier to interpret)
- Independent of scale

### 4. R² (R-Squared)
```
R² = 1 - (SS_res / SS_tot)
```
- **Higher is better** (max = 1.0)
- Proportion of variance explained
- 1.0 = perfect predictions, 0.0 = random

### 5. MSE (Mean Squared Error)
```
MSE = Σ(y_true - y_pred)² / n
```
- **Lower is better**
- Used as loss function during training

---

## Understanding the Results

### Output Files

1. **`results/model_comparison.csv`**
   - Comparison of all models across all stocks
   - Columns: symbol, model, prediction, RMSE, MAE, MAPE, R², MSE

2. **`models/`**
   - `{SYMBOL}_lstm_model.h5` - Trained LSTM model
   - `{SYMBOL}_lstm_scaler.pkl` - Scaler for LSTM
   - `{SYMBOL}_rnn_model.h5` - Trained RNN model
   - `{SYMBOL}_rnn_scaler.pkl` - Scaler for RNN
   - `{SYMBOL}_prophet_model.pkl` - Trained Prophet model

3. **`closing_price_predictions.log`**
   - Detailed training logs
   - Metric values for each epoch
   - Error messages if any

### Interpreting Predictions

**Example Output:**
```
SCOM - NEXT DAY PREDICTIONS:
  Current Price:  28.30
  LSTM:           28.85 (+1.94%)
  RNN:            28.50 (+0.71%)
  Prophet:        28.20 (-0.35%)

  Best Model (by RMSE): LSTM
```

**What this means:**
- LSTM predicts price will increase by 1.94%
- RNN predicts slight increase of 0.71%
- Prophet predicts slight decrease of 0.35%
- LSTM had lowest RMSE on test data (most accurate historically)

---

## For Your Thesis

### Research Questions to Address

1. **Which model performs best for NSE stock prediction?**
   - Compare average RMSE/MAE across all stocks
   - Analyze why one model outperforms others

2. **Does model performance vary by stock characteristics?**
   - High volatility vs low volatility stocks
   - High volume vs low volume stocks
   - Large cap vs small cap

3. **How does lookback window affect performance?**
   - Test with 15, 30, 60 day lookbacks
   - Find optimal window for each model

4. **What are the limitations?**
   - Small dataset (only 60-90 days)
   - Market efficiency hypothesis
   - External factors (news, economic events)

### Suggested Analysis

1. **Model Comparison Table**
   ```
   | Model   | Avg RMSE | Avg MAE | Avg MAPE | Avg R² | Training Time |
   |---------|----------|---------|----------|--------|---------------|
   | LSTM    | 0.45     | 0.32    | 1.2%     | 0.85   | 45s          |
   | RNN     | 0.52     | 0.38    | 1.5%     | 0.78   | 35s          |
   | Prophet | 0.48     | 0.35    | 1.3%     | 0.82   | 5s           |
   ```

2. **Prediction Visualization**
   - Plot actual vs predicted prices
   - Show prediction intervals
   - Highlight where models succeed/fail

3. **Error Analysis**
   - When do models make large errors?
   - Correlation with market volatility
   - Day-of-week effects

4. **Statistical Significance**
   - Paired t-test between models
   - Confidence intervals for metrics

### Example Thesis Structure

**Chapter 3: Methodology**
- 3.1 Data Collection (Your scraping system)
- 3.2 Data Preprocessing (Daily aggregation)
- 3.3 Model Architectures (LSTM, RNN, Prophet details)
- 3.4 Training Procedure (80/20 split, early stopping)
- 3.5 Evaluation Metrics (RMSE, MAE, MAPE, R²)

**Chapter 4: Results**
- 4.1 Descriptive Statistics
- 4.2 Model Performance Comparison
- 4.3 Per-Stock Analysis
- 4.4 Prediction Accuracy Over Time

**Chapter 5: Discussion**
- 5.1 Why LSTM/RNN/Prophet performed as they did
- 5.2 Practical implications for traders
- 5.3 Limitations and threats to validity
- 5.4 Future work

---

## Advanced Usage

### Custom Model Training

```python
from closing_price_pipeline import (
    DailyDataAggregator,
    LSTMPredictor,
    RNNPredictor,
    ProphetPredictor
)

# Get data
aggregator = DailyDataAggregator(DB_CONFIG)
df_all = aggregator.get_daily_closing_prices()
df_scom = df_all[df_all['symbol'] == 'SCOM']

# Train LSTM with custom parameters
lstm = LSTMPredictor('SCOM', lookback_days=60)
metrics, history = lstm.train(df_scom, epochs=200, batch_size=16)

# Predict next day
next_day_price = lstm.predict_next_day(df_scom)
print(f"Predicted closing price: {next_day_price:.2f}")

# Save model
lstm.save_model("models")
```

### Loading Pre-trained Models

```python
# Load previously trained model
lstm = LSTMPredictor('SCOM', lookback_days=30)
lstm.load_model("models")

# Make prediction
prediction = lstm.predict_next_day(df_scom)
```

---

## Troubleshooting

### "Insufficient data for {symbol}"

**Problem:** Stock has less than 60 days of data.

**Solution:**
1. Run data collection for longer
2. Reduce lookback window: `--lookback 15`
3. Use stocks with more history

### "Database connection failed"

**Problem:** Cannot connect to PostgreSQL.

**Solution:**
```bash
# Check if PostgreSQL is running
pg_isready

# Start PostgreSQL (macOS)
brew services start postgresql

# Verify connection
psql -U postgres -d nse -c "SELECT 1;"
```

### "TensorFlow not found"

**Problem:** TensorFlow not installed or wrong version.

**Solution:**
```bash
# Install TensorFlow
pip install tensorflow>=2.13.0

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Prophet installation issues

**Problem:** Prophet requires cmdstan compiler.

**Solution:**
```bash
# macOS
brew install cmdstan

# Ubuntu
sudo apt-get install cmdstan

# Or use conda (easier)
conda install -c conda-forge prophet
```

---

## Performance Tips

### 1. GPU Acceleration (Optional)

If you have NVIDIA GPU:
```bash
pip install tensorflow-gpu
```

Training will be 5-10x faster.

### 2. Reduce Training Time

```bash
# Use fewer epochs for quick testing
python run_predictions.py --epochs 20

# Process fewer stocks
python run_predictions.py --symbols SCOM KCB
```

### 3. Parallel Processing

The script processes stocks sequentially. For parallel processing, modify `run_predictions.py` to use `multiprocessing`.

---

## Next Steps for Thesis

1. **Collect More Data**
   - Run scraper for 3-6 months to get robust dataset
   - More data = better model performance

2. **Feature Engineering**
   - Add technical indicators (RSI, MACD)
   - Include volume and volatility
   - Market sentiment from news

3. **Hyperparameter Tuning**
   - Grid search for optimal lookback window
   - Optimize LSTM architecture (layers, units)
   - Cross-validation for robust evaluation

4. **Ensemble Methods**
   - Combine predictions from all three models
   - Weighted average based on historical accuracy

5. **Real-time Deployment**
   - Integrate with existing API (render_enhanced.py)
   - Serve predictions via REST endpoints
   - Daily automated retraining

---

## References for Thesis

### LSTM
- Hochreiter & Schmidhuber (1997). "Long Short-Term Memory"
- Fischer & Krauss (2018). "Deep learning with long short-term memory networks for financial market predictions"

### RNN
- Rumelhart et al. (1986). "Learning representations by back-propagating errors"

### Prophet
- Taylor & Letham (2018). "Forecasting at scale" (Facebook Prophet paper)

### Stock Market Prediction
- Hiransha et al. (2018). "NSE Stock Market Prediction Using Deep-Learning Models"
- Selvin et al. (2017). "Stock price prediction using LSTM, RNN and CNN-sliding window model"

---

## Contact & Support

For issues or questions:
1. Check logs: `closing_price_predictions.log`
2. Review code: `closing_price_pipeline.py`
3. Test with single stock first: `python run_predictions.py --symbols SCOM`

**Good luck with your thesis!** 🎓📈
