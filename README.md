# NSE Stock Market Prediction System

A real-time stock market data collection and machine learning prediction system for the Nairobi Securities Exchange (NSE).

## Overview

This system collects live NSE stock data and uses deep learning models (LSTM, RNN) and time-series forecasting (Prophet) to predict next-day closing prices.

## Quick Start

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create database
python -c "from src.utilities import create_database_and_schema; create_database_and_schema()"
```

### 2. Data Collection (Run for 60+ days)
```bash
python main_enhanced.py
```

### 3. Train Models & Predict
```bash
python run_predictions.py
```

### 4. Visualize Results
```bash
python visualize_results.py
```

## Project Structure

```
├── src/
│   ├── getlivedata.py          # Selenium web scraper
│   ├── processlivedata.py      # ETL pipeline
│   └── utilities.py            # Database utilities
├── main_enhanced.py            # System orchestrator
├── closing_price_pipeline.py   # ML models (LSTM, RNN, Prophet)
├── run_predictions.py          # Main prediction script
├── visualize_results.py        # Generate charts/graphs
├── render_enhanced.py          # API server with predictions
└── ml_pipeline.py              # Original ML features
```

## Documentation

- **[PROJECT_EXECUTION_GUIDE.md](PROJECT_EXECUTION_GUIDE.md)** - Entry points and execution order
- **[THESIS_GUIDE.md](THESIS_GUIDE.md)** - Complete guide for thesis work
- **[README_PREDICTIONS.md](README_PREDICTIONS.md)** - Prediction system details

## Models

- **LSTM** - Long Short-Term Memory neural network
- **RNN** - Recurrent Neural Network
- **Prophet** - Facebook's time-series forecasting

## Requirements

- Python 3.8+
- PostgreSQL with TimescaleDB
- Chrome/Chromium (for web scraping)
- See [requirements.txt](requirements.txt) for Python packages

## License

Academic/Research Use
