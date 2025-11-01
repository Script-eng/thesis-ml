"""
Quick Test Script for NSE Predictions
======================================
Tests if your setup works with the imported data.

Usage:
    python3 test_predictions.py
"""

import os
import sys
from dotenv import load_dotenv

print("="*70)
print("NSE PREDICTION SYSTEM - SETUP TEST")
print("="*70)

# Step 1: Check dependencies
print("\n1. Checking dependencies...")
missing = []

try:
    import numpy as np
    print("  ✅ numpy")
except ImportError:
    missing.append("numpy")
    print("  ❌ numpy")

try:
    import pandas as pd
    print("  ✅ pandas")
except ImportError:
    missing.append("pandas")
    print("  ❌ pandas")

try:
    import tensorflow as tf
    print(f"  ✅ tensorflow ({tf.__version__})")
except ImportError:
    missing.append("tensorflow")
    print("  ❌ tensorflow")

try:
    from prophet import Prophet
    print("  ✅ prophet")
except ImportError:
    missing.append("prophet")
    print("  ❌ prophet")

try:
    import psycopg2
    print("  ✅ psycopg2")
except ImportError:
    missing.append("psycopg2")
    print("  ❌ psycopg2")

try:
    from sklearn.preprocessing import MinMaxScaler
    print("  ✅ scikit-learn")
except ImportError:
    missing.append("scikit-learn")
    print("  ❌ scikit-learn")

if missing:
    print(f"\n❌ Missing dependencies: {', '.join(missing)}")
    print("\nInstall with:")
    print(f"  pip install {' '.join(missing)}")
    sys.exit(1)

# Step 2: Check database connection
print("\n2. Checking database connection...")
load_dotenv()

DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

try:
    import psycopg2
    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    # Check total records
    cursor.execute("SELECT COUNT(*) FROM stocksdata;")
    total_records = cursor.fetchone()[0]

    # Check date range
    cursor.execute("SELECT MIN(time), MAX(time) FROM stocksdata;")
    date_range = cursor.fetchone()

    # Check unique symbols
    cursor.execute("SELECT COUNT(DISTINCT symbol) FROM stocksdata;")
    unique_symbols = cursor.fetchone()[0]

    # Check trading days
    cursor.execute("SELECT COUNT(DISTINCT DATE(time AT TIME ZONE 'Africa/Nairobi')) FROM stocksdata;")
    trading_days = cursor.fetchone()[0]

    cursor.close()
    conn.close()

    print(f"  ✅ Connected to database: {DB_CONFIG['dbname']}")
    print(f"  Total records: {total_records:,}")
    print(f"  Unique symbols: {unique_symbols}")
    print(f"  Trading days: {trading_days}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")

    if trading_days < 30:
        print(f"\n  ⚠️  Warning: Only {trading_days} days of data.")
        print(f"     Recommended lookback: {min(10, trading_days // 3)} days")

except Exception as e:
    print(f"  ❌ Database connection failed: {e}")
    sys.exit(1)

# Step 3: Test data aggregation
print("\n3. Testing daily data aggregation...")
try:
    from closing_price_pipeline import DailyDataAggregator

    aggregator = DailyDataAggregator(DB_CONFIG)
    df_all = aggregator.get_daily_closing_prices(symbol='SCOM', days_back=90)

    if df_all is None or df_all.empty:
        print("  ❌ No data returned for SCOM")
        sys.exit(1)

    print(f"  ✅ Aggregation works")
    print(f"  SCOM daily records: {len(df_all)}")
    print(f"  Date range: {df_all['trading_date'].min()} to {df_all['trading_date'].max()}")

    if len(df_all) < 30:
        print(f"\n  ⚠️  SCOM has only {len(df_all)} days of data")
        print(f"     Need at least (lookback + 10) days for training")
        print(f"     Recommended lookback: {max(5, len(df_all) // 3)} days")

except Exception as e:
    print(f"  ❌ Aggregation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Test model import
print("\n4. Testing model imports...")
try:
    from closing_price_pipeline import LSTMPredictor, RNNPredictor, ProphetPredictor
    print("  ✅ LSTM model")
    print("  ✅ RNN model")
    print("  ✅ Prophet model")
except Exception as e:
    print(f"  ❌ Model import failed: {e}")
    sys.exit(1)

# Step 5: Quick training test
print("\n5. Testing model training with SCOM...")
try:
    # Get data
    df_scom = aggregator.get_daily_closing_prices(symbol='SCOM', days_back=90)

    if len(df_scom) < 20:
        print(f"  ⚠️  Skipping training test - insufficient data ({len(df_scom)} days)")
    else:
        # Determine lookback based on available data
        lookback = min(10, len(df_scom) // 3)
        print(f"  Using lookback: {lookback} days")
        print(f"  Training data: {len(df_scom)} days")

        # Test LSTM
        print("\n  Testing LSTM...")
        lstm = LSTMPredictor('SCOM', lookback_days=lookback)
        lstm_metrics, _ = lstm.train(df_scom, test_size=0.2, epochs=5, batch_size=16)
        print(f"    ✅ LSTM trained - RMSE: {lstm_metrics['RMSE']:.4f}")

        # Test prediction
        prediction = lstm.predict_next_day(df_scom)
        current_price = df_scom['close_price'].iloc[-1]
        print(f"    Current price: {current_price:.2f}")
        print(f"    Predicted price: {prediction:.2f}")
        print(f"    Change: {((prediction/current_price - 1)*100):+.2f}%")

except Exception as e:
    print(f"  ❌ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    print("\n  This is likely due to insufficient data or incorrect configuration.")
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nYour system is ready for predictions!")
print("\nRecommended next steps:")
print(f"  1. Run predictions with lookback={lookback}:")
print(f"     python3 run_predictions.py --lookback {lookback} --epochs 50 --symbols SCOM KCB EQTY")
print(f"\n  2. Or use Jupyter notebook:")
print(f"     jupyter notebook NSE_Closing_Price_Prediction_Workflow.ipynb")
print(f"     (Remember to set lookback_days={lookback})")
print("\n" + "="*70)
