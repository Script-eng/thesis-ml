"""
Predictions Runner - Generate and Save EOD Predictions
======================================================
This script runs the trained ML models to generate end-of-day closing price predictions
and saves them to the database for serving via the API.

Usage:
    python predictions_runner.py                    # Run for all stocks
    python predictions_runner.py --symbol SCOM      # Run for specific stock
    python predictions_runner.py --top 5            # Run for top 5 most active stocks
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging

from closing_price_pipeline import (
    DailyDataAggregator,
    LSTMPredictor,
    RNNPredictor,
    ProphetPredictor,
    PredictionSaver
)

# Setup
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('predictions_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Model configuration
MODEL_PATH = "models"
LOOKBACK_DAYS = 30
MIN_DAYS_REQUIRED = 45  # Minimum days of data needed to train (lowered from 60 to work with available data)


def get_current_price(df_daily):
    """Get the current/latest price from daily data."""
    if df_daily.empty:
        return None
    return df_daily['close_price'].iloc[-1]


def run_predictions_for_symbol(symbol, aggregator, saver, force_retrain=False):
    """
    Run all three models for a specific symbol and save predictions.

    Args:
        symbol: Stock symbol
        aggregator: DailyDataAggregator instance
        saver: PredictionSaver instance
        force_retrain: Whether to retrain models even if they exist

    Returns:
        dict: Prediction results or None if failed
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {symbol}")
    logger.info(f"{'='*60}")

    # Get daily data
    df_symbol = aggregator.get_daily_closing_prices(symbol=symbol, days_back=90)

    if df_symbol is None or len(df_symbol) < MIN_DAYS_REQUIRED:
        logger.warning(f"Insufficient data for {symbol} ({len(df_symbol) if df_symbol is not None else 0} days). Skipping.")
        return None

    df_symbol = df_symbol.sort_values('trading_date')
    current_price = get_current_price(df_symbol)

    if current_price is None:
        logger.warning(f"Could not determine current price for {symbol}")
        return None

    logger.info(f"Current price: {current_price:.2f} KES")
    logger.info(f"Available data: {len(df_symbol)} days")

    predictions = {}

    # Try LSTM
    try:
        lstm = LSTMPredictor(symbol, lookback_days=LOOKBACK_DAYS)

        # Try to load existing model
        model_exists = os.path.exists(f"{MODEL_PATH}/{symbol}_lstm_model.h5")

        if model_exists and not force_retrain:
            logger.info("Loading existing LSTM model...")
            lstm.load_model(MODEL_PATH)
        else:
            logger.info("Training LSTM model...")
            lstm_metrics, _ = lstm.train(df_symbol, epochs=50, batch_size=32)
            lstm.save_model(MODEL_PATH)
            logger.info(f"LSTM trained - R²: {lstm_metrics['R2']:.4f}, RMSE: {lstm_metrics['RMSE']:.2f}")

        # Predict
        lstm_pred = lstm.predict_next_day(df_symbol)

        # Get model performance metric (if we just trained, use that, otherwise use a default)
        if model_exists and not force_retrain:
            # For loaded models, we'll use a moderate confidence
            lstm_conf = 0.70  # Default confidence for existing models
        else:
            lstm_conf = max(0.0, min(1.0, lstm_metrics['R2']))  # Clamp between 0 and 1

        predictions['lstm'] = {
            'prediction': float(lstm_pred),
            'confidence': float(lstm_conf)
        }

        logger.info(f"LSTM prediction: {lstm_pred:.2f} KES (confidence: {lstm_conf:.2%})")

    except Exception as e:
        logger.error(f"LSTM failed for {symbol}: {e}")

    # Try RNN
    try:
        rnn = RNNPredictor(symbol, lookback_days=LOOKBACK_DAYS)

        model_exists = os.path.exists(f"{MODEL_PATH}/{symbol}_rnn_model.h5")

        if model_exists and not force_retrain:
            logger.info("Loading existing RNN model...")
            rnn.load_model(MODEL_PATH)
        else:
            logger.info("Training RNN model...")
            rnn_metrics, _ = rnn.train(df_symbol, epochs=50, batch_size=32)
            rnn.save_model(MODEL_PATH)
            logger.info(f"RNN trained - R²: {rnn_metrics['R2']:.4f}, RMSE: {rnn_metrics['RMSE']:.2f}")

        rnn_pred = rnn.predict_next_day(df_symbol)

        if model_exists and not force_retrain:
            rnn_conf = 0.65
        else:
            rnn_conf = max(0.0, min(1.0, rnn_metrics['R2']))

        predictions['rnn'] = {
            'prediction': float(rnn_pred),
            'confidence': float(rnn_conf)
        }

        logger.info(f"RNN prediction: {rnn_pred:.2f} KES (confidence: {rnn_conf:.2%})")

    except Exception as e:
        logger.error(f"RNN failed for {symbol}: {e}")

    # Try Prophet
    try:
        prophet = ProphetPredictor(symbol)

        model_exists = os.path.exists(f"{MODEL_PATH}/{symbol}_prophet_model.pkl")

        if model_exists and not force_retrain:
            logger.info("Loading existing Prophet model...")
            prophet.load_model(MODEL_PATH)
        else:
            logger.info("Training Prophet model...")
            prophet_metrics = prophet.train(df_symbol)
            prophet.save_model(MODEL_PATH)
            logger.info(f"Prophet trained - R²: {prophet_metrics['R2']:.4f}, RMSE: {prophet_metrics['RMSE']:.2f}")

        prophet_pred = prophet.predict_next_day(df_symbol)

        if model_exists and not force_retrain:
            prophet_conf = 0.60
        else:
            prophet_conf = max(0.0, min(1.0, prophet_metrics['R2']))

        predictions['prophet'] = {
            'prediction': float(prophet_pred),
            'confidence': float(prophet_conf)
        }

        logger.info(f"Prophet prediction: {prophet_pred:.2f} KES (confidence: {prophet_conf:.2%})")

    except Exception as e:
        logger.error(f"Prophet failed for {symbol}: {e}")

    # Save predictions to database
    if predictions:
        success = saver.save_prediction(
            symbol=symbol,
            current_price=current_price,
            predictions_dict=predictions
        )

        if success:
            logger.info(f"✅ Successfully saved predictions for {symbol}")
            return predictions
        else:
            logger.error(f"❌ Failed to save predictions for {symbol}")
            return None
    else:
        logger.warning(f"No predictions generated for {symbol}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate ML predictions for NSE stocks')
    parser.add_argument('--symbol', type=str, help='Run for specific symbol')
    parser.add_argument('--top', type=int, help='Run for top N most active stocks')
    parser.add_argument('--retrain', action='store_true', help='Force retrain models')
    args = parser.parse_args()

    logger.info("="*60)
    logger.info("NSE Stock Predictions Runner")
    logger.info("="*60)

    # Initialize services
    aggregator = DailyDataAggregator(DB_CONFIG)
    saver = PredictionSaver(DB_CONFIG)

    # Determine which symbols to process
    if args.symbol:
        symbols = [args.symbol.upper()]
        logger.info(f"Running for symbol: {args.symbol}")
    else:
        # Get all symbols with sufficient data
        df_all = aggregator.get_daily_closing_prices(days_back=90)

        if df_all is None or df_all.empty:
            logger.error("No data available. Exiting.")
            sys.exit(1)

        # Get symbols with at least MIN_DAYS_REQUIRED days
        symbol_counts = df_all.groupby('symbol').size()
        symbols = symbol_counts[symbol_counts >= MIN_DAYS_REQUIRED].index.tolist()

        if args.top:
            # Sort by recent volume and take top N
            latest_volumes = df_all.groupby('symbol')['daily_volume'].last().sort_values(ascending=False)
            symbols = latest_volumes.head(args.top).index.tolist()
            logger.info(f"Running for top {args.top} most active stocks")
        else:
            logger.info(f"Running for {len(symbols)} symbols with sufficient data")

    # Process each symbol
    results = {}
    successful = 0
    failed = 0

    for symbol in symbols:
        try:
            result = run_predictions_for_symbol(
                symbol,
                aggregator,
                saver,
                force_retrain=args.retrain
            )

            if result:
                results[symbol] = result
                successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            failed += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PREDICTIONS SUMMARY")
    logger.info("="*60)
    logger.info(f"Symbols processed: {len(symbols)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")

    if results:
        logger.info("\nPrediction Results:")
        for symbol, preds in results.items():
            logger.info(f"\n{symbol}:")
            for model, data in preds.items():
                logger.info(f"  {model.upper()}: {data['prediction']:.2f} (conf: {data['confidence']:.2%})")

    logger.info("\n✅ Predictions runner completed!")


if __name__ == "__main__":
    main()
