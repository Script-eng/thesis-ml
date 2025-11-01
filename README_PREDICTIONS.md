# NSE Stock Closing Price Prediction System

A comprehensive machine learning system for predicting next-day closing prices of stocks listed on the Nairobi Securities Exchange (NSE) using LSTM, RNN, and Prophet models.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Predictions (Top 5 Stocks)
```bash
python run_predictions.py
```

### 3. Visualize Results
```bash
python visualize_results.py
```

## What This System Does

This system takes your real-time NSE stock data and:

1. **Aggregates** intraday data into daily closing prices
2. **Trains** three different models (LSTM, RNN, Prophet) on historical data
3. **Predicts** the next day's closing price for each stock
4. **Compares** model performance using multiple metrics
5. **Visualizes** results for thesis presentation

## Models Used

### 1. LSTM (Long Short-Term Memory)
- **Best for**: Capturing long-term dependencies and complex patterns
- **Architecture**: 3-layer LSTM with dropout regularization
- **Lookback**: 30 days of historical prices

### 2. RNN (Recurrent Neural Network)
- **Best for**: Baseline comparison with LSTM
- **Architecture**: 3-layer SimpleRNN with dropout
- **Lookback**: 30 days of historical prices

### 3. Prophet (Facebook Prophet)
- **Best for**: Business forecasting with seasonality
- **Features**: Automatic trend detection and weekly patterns
- **Approach**: Additive time-series model

## Current Data Status

Based on your database:
- **Total Records**: 6,072 intraday records
- **Stocks**: 69 unique symbols
- **Date Range**: October 16 - October 25, 2025 (9 days)
- **Frequency**: 30-second intervals

**Note**: You currently have only 9 days of data. For robust predictions:
- **Minimum**: 60 days recommended
- **Optimal**: 90+ days

**Action Required**: Continue running your data collection ([src/getlivedata.py](src/getlivedata.py)) to accumulate more historical data.

## Usage Examples

### Example 1: Predict Specific Stocks
```bash
python run_predictions.py --symbols SCOM KCB EQTY
```

### Example 2: Process All Available Stocks
```bash
python run_predictions.py --all
```

### Example 3: Custom Training Parameters
```bash
# Train with 100 epochs and 60-day lookback
python run_predictions.py --epochs 100 --lookback 60 --symbols SCOM
```

### Example 4: Quick Test (20 epochs)
```bash
python run_predictions.py --epochs 20 --symbols SCOM
```

## Understanding the Output

### Console Output
```
=================================================================
[1/5] Processing SCOM
=================================================================
Data range: 2025-09-26 to 2025-10-25
Total days: 30
Current closing price: 28.30

Training LSTM model...
LSTM SCOM - RMSE: 0.45, MAE: 0.32, MAPE: 1.2%

Training RNN model...
RNN SCOM - RMSE: 0.52, MAE: 0.38, MAPE: 1.5%

Training Prophet model...
Prophet SCOM - RMSE: 0.48, MAE: 0.35, MAPE: 1.3%

SCOM - NEXT DAY PREDICTIONS:
  Current Price:  28.30
  LSTM:           28.85 (+1.94%)
  RNN:            28.50 (+0.71%)
  Prophet:        28.20 (-0.35%)

  Best Model (by RMSE): LSTM
```

### What the Metrics Mean

- **RMSE** (Root Mean Squared Error): Average prediction error in KES. Lower is better.
- **MAE** (Mean Absolute Error): Average absolute difference. Lower is better.
- **MAPE** (Mean Absolute Percentage Error): Percentage error. Lower is better.
- **R²** (R-Squared): Proportion of variance explained. Higher is better (max = 1.0).

### Output Files

1. **`results/model_comparison.csv`**
   - Complete comparison table
   - Use for thesis tables

2. **`models/`** directory
   - All trained models saved here
   - Can be loaded later without retraining

3. **`closing_price_predictions.log`**
   - Detailed training logs
   - Useful for debugging

## Visualization

After running predictions, generate charts:

```bash
python visualize_results.py
```

This creates:
- **model_comparison.png** - Bar charts comparing all models
- **error_distribution.png** - Box plots showing error spread
- **stock_wise_performance.png** - Heatmap of RMSE by stock
- **[SYMBOL]_predictions.png** - Prediction vs actual for each stock
- **summary_statistics.csv** - Statistical summary table

## For Your Thesis

### How to Use These Models

#### 1. Data Collection Phase
```bash
# Keep your scraper running to collect data
python src/getlivedata.py &
python src/processlivedata.py &
```
Run for **at least 60 days** before serious model training.

#### 2. Model Training Phase
```bash
# Train models on all available stocks
python run_predictions.py --all --epochs 100
```

#### 3. Analysis Phase
```bash
# Generate visualizations
python visualize_results.py

# Analyze results in results/model_comparison.csv
```

#### 4. Writing Phase
Use the generated charts and tables in your thesis:
- **Chapter 3 (Methodology)**: Describe LSTM, RNN, Prophet architectures
- **Chapter 4 (Results)**: Include comparison charts and tables
- **Chapter 5 (Discussion)**: Interpret which model performed best and why

### Research Questions to Address

1. **Which model is most accurate?**
   - Compare average RMSE across all stocks
   - Statistical significance testing (t-test)

2. **Does accuracy vary by stock?**
   - Some stocks more predictable than others?
   - Relationship with volatility?

3. **What's the optimal lookback window?**
   - Test 15, 30, 60 days
   - Plot RMSE vs lookback period

4. **Can we combine models?**
   - Ensemble: Average of LSTM + RNN + Prophet
   - Weighted average based on historical accuracy

## Project Structure

```
live.nse/
├── closing_price_pipeline.py    # Core ML pipeline
├── run_predictions.py            # Main execution script
├── visualize_results.py          # Visualization generator
├── THESIS_GUIDE.md              # Comprehensive thesis guide
├── README_PREDICTIONS.md         # This file
├── requirements.txt              # Dependencies
├── models/                       # Saved models
│   ├── SCOM_lstm_model.h5
│   ├── SCOM_rnn_model.h5
│   └── SCOM_prophet_model.pkl
├── results/                      # Results and charts
│   ├── model_comparison.csv
│   ├── model_comparison.png
│   └── summary_statistics.csv
└── src/                          # Data collection
    ├── getlivedata.py
    ├── processlivedata.py
    └── utilities.py
```

## Advanced Usage

### Programmatic Access

```python
from closing_price_pipeline import (
    DailyDataAggregator,
    LSTMPredictor,
    RNNPredictor,
    ProphetPredictor
)
import os
from dotenv import load_dotenv

load_dotenv()
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Get daily data
aggregator = DailyDataAggregator(DB_CONFIG)
df_all = aggregator.get_daily_closing_prices()
df_scom = df_all[df_all['symbol'] == 'SCOM']

# Train LSTM
lstm = LSTMPredictor('SCOM', lookback_days=30)
metrics, history = lstm.train(df_scom, epochs=50)

# Predict
prediction = lstm.predict_next_day(df_scom)
print(f"Next day prediction: {prediction:.2f}")

# Save model
lstm.save_model("models")

# Load later
lstm_loaded = LSTMPredictor('SCOM', lookback_days=30)
lstm_loaded.load_model("models")
```

## Troubleshooting

### "Insufficient data for {symbol}"

**Problem**: Less than 60 days of data.

**Solution**:
- Run data collection longer
- OR reduce lookback: `--lookback 15`

### "Database connection failed"

**Solution**:
```bash
# Check PostgreSQL is running
pg_isready

# Start if needed (macOS)
brew services start postgresql

# Test connection
psql -U postgres -d nse -c "SELECT COUNT(*) FROM stocksdata;"
```

### TensorFlow/Prophet Installation Issues

```bash
# Use conda for easier installation
conda create -n nse_ml python=3.10
conda activate nse_ml
conda install -c conda-forge tensorflow prophet
pip install -r requirements.txt
```

## Performance Expectations

With your current 9 days of data:
- **Models will train** but predictions won't be reliable
- **Expect high errors** (MAPE > 5%)
- **Use for testing** the pipeline only

With 60+ days of data:
- **LSTM**: MAPE typically 1-3%
- **RNN**: MAPE typically 1.5-4%
- **Prophet**: MAPE typically 2-5%

## Next Steps

1. **Immediate**: Let data collection run for 60+ days
2. **Week 1-2**: Test pipeline with small dataset
3. **Week 3-8**: Collect sufficient data (60-90 days)
4. **Week 9**: Full model training and analysis
5. **Week 10**: Visualization and thesis writing

## Academic References

### LSTM
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. European Journal of Operational Research, 270(2), 654-669.

### Prophet
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.

### Stock Prediction
- Hiransha, M., et al. (2018). NSE stock market prediction using deep-learning models. Procedia computer science, 132, 1351-1362.
- Selvin, S., et al. (2017). Stock price prediction using LSTM, RNN and CNN-sliding window model. International conference on advances in computing, communications and informatics (ICACCI).

## Support

- **Documentation**: See [THESIS_GUIDE.md](THESIS_GUIDE.md)
- **Logs**: Check `closing_price_predictions.log`
- **Test**: Start with single stock `--symbols SCOM`

---

**Good luck with your thesis!** 🎓📊📈
