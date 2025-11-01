# Project Execution Guide

## Entry Points & Execution Order

### System Overview
This project has TWO main workflows:

1. **Data Collection Workflow** (Continuous/Background)
2. **ML Prediction Workflow** (On-demand/Analysis)

---

## WORKFLOW 1: Data Collection (Run First, Keep Running)

### Purpose
Continuously scrape and store NSE stock market data for later analysis.

### Files Involved
1. `src/getlivedata.py` - Web scraper
2. `src/processlivedata.py` - ETL processor
3. `src/utilities.py` - Database setup helper

### Execution Order

#### Step 1: Database Setup (One-time)
```bash
python -c "from src.utilities import create_database_and_schema; create_database_and_schema()"
```

**What it does:**
- Creates PostgreSQL database `nse`
- Enables TimescaleDB extension
- Creates `stocksdata` hypertable
- Sets up indexes

#### Step 2: Start Data Scraper (Keep Running)
```bash
python src/getlivedata.py
```

**What it does:**
- Opens Chrome browser (headless)
- Navigates to https://fib.co.ke/live-markets/
- Extracts stock data from iframe every 30 seconds
- Saves to `.rendered_stock_data.html`
- Runs continuously until stopped (Ctrl+C)

**Keep this running 24/7 during data collection phase!**

#### Step 3: Start ETL Processor (Keep Running in parallel)
```bash
# In a separate terminal
python src/processlivedata.py
```

**What it does:**
- Reads `.rendered_stock_data.html` every 30 seconds
- Parses HTML to extract stock data
- Cleans and validates data
- Inserts into PostgreSQL database
- Runs continuously until stopped (Ctrl+C)

**Keep this running alongside the scraper!**

#### Alternative: Use Main Orchestrator (Recommended)
```bash
python main_enhanced.py
```

**What it does:**
- Runs scraper, ETL, and ML integration in coordinated threads
- Automatic error handling and restart
- Health monitoring
- Logs to console and file

**This is the easiest way to run data collection!**

---

## WORKFLOW 2: ML Prediction (Run After Data Collection)

### Purpose
Train models and predict next day's closing prices.

### Prerequisites
- At least 60 days of data collected
- PostgreSQL database populated with stock data

### Execution Order

#### Step 1: Check Data Availability
```bash
PGPASSWORD=postgres psql -U postgres -h localhost -d nse -c \
"SELECT COUNT(*) as records, COUNT(DISTINCT symbol) as stocks, \
MIN(time) as earliest, MAX(time) as latest FROM stocksdata;"
```

**Expected output:**
```
 records | stocks |        earliest        |         latest
---------+--------+------------------------+------------------------
   50000 |     69 | 2025-09-01 09:30:00... | 2025-10-31 15:00:00...
```

You need at least 60 days between `earliest` and `latest`.

#### Step 2: Run Predictions (Main Entry Point)
```bash
python run_predictions.py
```

**What it does:**
- Aggregates intraday data to daily closing prices
- Trains LSTM, RNN, and Prophet models
- Predicts next day's closing price for top 5 stocks
- Saves models to `models/` directory
- Saves results to `results/model_comparison.csv`

**Options:**
```bash
# Specific stocks
python run_predictions.py --symbols SCOM KCB EQTY

# All stocks
python run_predictions.py --all

# Custom parameters
python run_predictions.py --epochs 100 --lookback 60
```

#### Step 3: Visualize Results
```bash
python visualize_results.py
```

**What it does:**
- Reads `results/model_comparison.csv`
- Generates charts and graphs
- Saves visualizations to `results/` directory

**Output:**
- `model_comparison.png`
- `error_distribution.png`
- `stock_wise_performance.png`
- `{SYMBOL}_predictions.png` (for each stock)
- `summary_statistics.csv`

---

## Complete Execution Flow (Recommended for Thesis)

### Phase 1: Setup (One-time, 5 minutes)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create database
python -c "from src.utilities import create_database_and_schema; create_database_and_schema()"
```

### Phase 2: Data Collection (60-90 days, continuous)
```bash
# Option A: Use orchestrator (recommended)
python main_enhanced.py

# Option B: Manual (two terminals)
# Terminal 1:
python src/getlivedata.py

# Terminal 2:
python src/processlivedata.py
```

**Let this run for 60-90 days!**

### Phase 3: Model Training (Week 9)
```bash
# Train all models
python run_predictions.py --all --epochs 100
```

### Phase 4: Analysis & Visualization (Week 10)
```bash
# Generate all charts
python visualize_results.py
```

---

## Entry Points Summary

| File | Type | Purpose | When to Run |
|------|------|---------|-------------|
| `main_enhanced.py` | **Primary Entry Point** | Orchestrates entire data collection system | Start of data collection phase |
| `run_predictions.py` | **Primary Entry Point** | Trains models and generates predictions | After 60+ days of data collected |
| `visualize_results.py` | **Analysis Tool** | Creates thesis visualizations | After running predictions |
| `src/getlivedata.py` | Background Service | Scrapes live market data | Continuous during collection |
| `src/processlivedata.py` | Background Service | Processes and stores data | Continuous during collection |

---

## Quick Start Commands

### For Data Collection
```bash
# Start everything (easiest)
python main_enhanced.py
```

### For Predictions (After Data Collection)
```bash
# Train and predict
python run_predictions.py

# Visualize
python visualize_results.py
```

---

## File Dependencies

### Data Collection Dependencies
```
main_enhanced.py
├── src/getlivedata.py
│   └── .env (configuration)
├── src/processlivedata.py
│   └── .env (configuration)
└── src/utilities.py
    └── .env (configuration)
```

### ML Prediction Dependencies
```
run_predictions.py
├── closing_price_pipeline.py
│   ├── DailyDataAggregator
│   ├── LSTMPredictor
│   ├── RNNPredictor
│   ├── ProphetPredictor
│   └── ModelComparator
└── .env (database config)

visualize_results.py
├── results/model_comparison.csv
└── closing_price_pipeline.py
```

---

## Environment Variables (.env)

Required variables:
```bash
# Database
DB_NAME=nse
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432

# Table
TABLE_NAME=stocksdata

# Scraper
TARGET_URL=https://fib.co.ke/live-markets/
OUTPUT_FILENAME=.rendered_stock_data.html
SCRAPE_INTERVAL_SECONDS=30

# ETL
HTML_FILE_PATH=.rendered_stock_data.html
PROCESSING_INTERVAL_SECONDS=30

# Logging
LOG_FILENAME=etl_timescale.log
GETLIVEDATA_LOG_FILENAME=getlivedata.log

# Timezone
TZ=Africa/Nairobi
```

---

## Stopping Services

### Stop Data Collection
```bash
# If using main_enhanced.py
Ctrl+C (once, waits for graceful shutdown)

# If using separate processes
Ctrl+C in each terminal
```

### Check What's Running
```bash
# See Python processes
ps aux | grep python

# Kill specific process
kill <PID>
```

---

## Logs and Monitoring

### Log Files
- `etl_timescale.log` - ETL processing logs
- `getlivedata.log` - Scraper logs
- `closing_price_predictions.log` - Model training logs

### View Logs in Real-time
```bash
# ETL logs
tail -f etl_timescale.log

# Scraper logs
tail -f getlivedata.log

# Prediction logs
tail -f closing_price_predictions.log
```

---

## Troubleshooting

### No data in database
```bash
# Check if scraper is running
ps aux | grep getlivedata

# Check if ETL is running
ps aux | grep processlivedata

# Check database
PGPASSWORD=postgres psql -U postgres -d nse -c "SELECT COUNT(*) FROM stocksdata;"
```

### Models failing to train
```bash
# Check data availability
python -c "
from closing_price_pipeline import DailyDataAggregator
import os
from dotenv import load_dotenv
load_dotenv()
db = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}
agg = DailyDataAggregator(db)
df = agg.get_daily_closing_prices()
print(f'Total days: {len(df)}')
print(f'Symbols: {df[\"symbol\"].nunique()}')
print(f'Date range: {df[\"trading_date\"].min()} to {df[\"trading_date\"].max()}')
"
```

---

## Recommended Workflow for Thesis

1. **Today**: Start `python main_enhanced.py` and let it run
2. **Weeks 1-8**: Monitor logs, ensure data collection is working
3. **Week 9**: Run `python run_predictions.py --all --epochs 100`
4. **Week 9**: Run `python visualize_results.py`
5. **Week 10**: Write thesis using generated results

---

**This is your complete execution guide. Follow these steps for successful thesis completion!**
