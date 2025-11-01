"""
Quick Start Script for NSE Closing Price Predictions
=====================================================
This script trains and compares LSTM, RNN, and Prophet models
for predicting next day's closing prices.

Usage:
    python run_predictions.py

For specific symbols:
    python run_predictions.py --symbols SCOM KCB EQTY

For all symbols:
    python run_predictions.py --all
"""

import argparse
import os
import sys
from datetime import datetime
from closing_price_pipeline import (
    DailyDataAggregator,
    LSTMPredictor,
    RNNPredictor,
    ProphetPredictor,
    ModelComparator
)
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='NSE Stock Price Prediction')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to predict')
    parser.add_argument('--all', action='store_true', help='Process all available symbols')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--lookback', type=int, default=30, help='Lookback days (default: 30)')
    args = parser.parse_args()

    # Database configuration
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

    # Create models directory
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Initialize
    logger.info("="*70)
    logger.info("NSE CLOSING PRICE PREDICTION SYSTEM")
    logger.info("="*70)

    aggregator = DailyDataAggregator(DB_CONFIG)
    comparator = ModelComparator()

    # Get daily data
    logger.info("\nFetching daily closing prices from database...")
    df_all = aggregator.get_daily_closing_prices(days_back=90)

    if df_all is None or df_all.empty:
        logger.error("No data available. Make sure your data collection is running.")
        sys.exit(1)

    # Determine which symbols to process
    available_symbols = df_all['symbol'].unique()

    if args.all:
        symbols_to_process = available_symbols
        logger.info(f"Processing ALL {len(symbols_to_process)} symbols")
    elif args.symbols:
        symbols_to_process = [s for s in args.symbols if s in available_symbols]
        if len(symbols_to_process) != len(args.symbols):
            missing = set(args.symbols) - set(symbols_to_process)
            logger.warning(f"Symbols not found in database: {missing}")
        logger.info(f"Processing {len(symbols_to_process)} specified symbols")
    else:
        # Default: process top 5 symbols by data availability
        symbol_counts = df_all.groupby('symbol').size().sort_values(ascending=False)
        symbols_to_process = symbol_counts.head(5).index.tolist()
        logger.info(f"Processing top 5 symbols: {symbols_to_process}")

    logger.info("\nAvailable symbols: " + ", ".join(available_symbols))
    logger.info("\n" + "="*70 + "\n")

    # Process each symbol
    successful = 0
    failed = 0

    for idx, symbol in enumerate(symbols_to_process, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"[{idx}/{len(symbols_to_process)}] Processing {symbol}")
        logger.info(f"{'='*70}")

        df_symbol = df_all[df_all['symbol'] == symbol].sort_values('trading_date')

        # Check data availability
        if len(df_symbol) < 60:
            logger.warning(f"Insufficient data for {symbol} ({len(df_symbol)} days). Need at least 60 days. Skipping.")
            failed += 1
            continue

        logger.info(f"Data range: {df_symbol['trading_date'].min()} to {df_symbol['trading_date'].max()}")
        logger.info(f"Total days: {len(df_symbol)}")
        logger.info(f"Current closing price: {df_symbol['close_price'].iloc[-1]:.2f}\n")

        try:
            # Train LSTM
            logger.info("Training LSTM model...")
            lstm = LSTMPredictor(symbol, lookback_days=args.lookback)
            lstm_metrics, _ = lstm.train(df_symbol, epochs=args.epochs, batch_size=32)
            lstm_prediction = lstm.predict_next_day(df_symbol)
            lstm.save_model("models")
            comparator.add_result(symbol, "LSTM", lstm_metrics, lstm_prediction)

            # Train RNN
            logger.info("Training RNN model...")
            rnn = RNNPredictor(symbol, lookback_days=args.lookback)
            rnn_metrics, _ = rnn.train(df_symbol, epochs=args.epochs, batch_size=32)
            rnn_prediction = rnn.predict_next_day(df_symbol)
            rnn.save_model("models")
            comparator.add_result(symbol, "RNN", rnn_metrics, rnn_prediction)

            # Train Prophet
            logger.info("Training Prophet model...")
            prophet = ProphetPredictor(symbol)
            prophet_metrics = prophet.train(df_symbol)
            prophet_prediction = prophet.predict_next_day(df_symbol)
            prophet.save_model("models")
            comparator.add_result(symbol, "Prophet", prophet_metrics, prophet_prediction)

            # Display predictions
            current_price = df_symbol['close_price'].iloc[-1]
            logger.info(f"\n{symbol} - NEXT DAY PREDICTIONS:")
            logger.info(f"  Current Price:  {current_price:.2f}")
            logger.info(f"  LSTM:           {lstm_prediction:.2f} ({((lstm_prediction/current_price - 1)*100):+.2f}%)")
            logger.info(f"  RNN:            {rnn_prediction:.2f} ({((rnn_prediction/current_price - 1)*100):+.2f}%)")
            logger.info(f"  Prophet:        {prophet_prediction:.2f} ({((prophet_prediction/current_price - 1)*100):+.2f}%)")

            best_model = comparator.get_best_model(symbol, metric='RMSE')
            logger.info(f"\n  Best Model (by RMSE): {best_model}")

            successful += 1

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            failed += 1
            continue

    # Save and display results
    logger.info("\n" + "="*70)
    logger.info("SAVING RESULTS")
    logger.info("="*70)

    comparator.save_results("results")

    # Summary statistics
    df_comparison = comparator.get_comparison_df()

    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)
    logger.info(f"Successfully processed: {successful} symbols")
    logger.info(f"Failed: {failed} symbols")

    if not df_comparison.empty:
        logger.info("\nModel Performance Averages:")
        avg_metrics = df_comparison.groupby('model')[['RMSE', 'MAE', 'MAPE', 'R2']].mean()
        print("\n" + avg_metrics.to_string())

        logger.info("\nDetailed Results:")
        print("\n" + df_comparison.to_string(index=False))

        logger.info(f"\nResults saved to: results/model_comparison.csv")
        logger.info(f"Models saved to: models/")

    logger.info("\n" + "="*70)
    logger.info("DONE!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
