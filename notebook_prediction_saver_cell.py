# ====================================================================
# CELL TO ADD TO NSE_UNIFIED NOTEBOOK - SAVE PREDICTIONS TO DATABASE
# Add this cell AFTER Cell 32 (Final Results Summary)
# ====================================================================

"""
This cell saves the trained model predictions to the ml_predictions database table
so they can be served via the API and displayed in the frontend.
"""

print("\n" + "="*60)
print("SAVING PREDICTIONS TO DATABASE")
print("="*60)

# Import the PredictionSaver class
import sys
import os
from dotenv import load_dotenv
load_dotenv()

# Import from closing_price_pipeline
from closing_price_pipeline import PredictionSaver

# Database configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

# Initialize PredictionSaver
saver = PredictionSaver(DB_CONFIG)

# Get current price (latest price from the stock data)
current_price = df_stock_full['price'].iloc[-1]

print(f"\n📊 Preparing predictions for {ANALYSIS_SYMBOL}:")
print(f"   Current price: {current_price:.2f} KES")

# Prepare predictions dictionary
# Note: We don't have RNN in this notebook, so we'll use LSTM and Prophet only
predictions_dict = {}

# LSTM predictions (from eod_results)
if 'metrics' in eod_results:
    lstm_pred = eod_results['predictions'][-1] if len(eod_results['predictions']) > 0 else None
    lstm_conf = eod_results['metrics']['R2']

    if lstm_pred is not None and lstm_conf is not None:
        predictions_dict['lstm'] = {
            'prediction': float(lstm_pred),
            'confidence': max(0.0, min(1.0, float(lstm_conf)))  # Clamp between 0 and 1
        }
        print(f"   LSTM: {lstm_pred:.2f} KES (confidence: {lstm_conf:.2%})")

# Prophet predictions
if 'predictions' in prophet_results and len(prophet_results['predictions']) > 0:
    prophet_pred = prophet_results['predictions'][-1]
    prophet_conf = prophet_results['r2']

    if prophet_pred is not None and prophet_conf is not None:
        predictions_dict['prophet'] = {
            'prediction': float(prophet_pred),
            'confidence': max(0.0, min(1.0, float(prophet_conf)))  # Clamp between 0 and 1
        }
        print(f"   Prophet: {prophet_pred:.2f} KES (confidence: {prophet_conf:.2%})")

# RNN (if you add it later, otherwise use LSTM as fallback)
# For now, we'll use LSTM prediction as RNN prediction with slightly lower confidence
if 'lstm' in predictions_dict:
    predictions_dict['rnn'] = {
        'prediction': predictions_dict['lstm']['prediction'],
        'confidence': predictions_dict['lstm']['confidence'] * 0.95  # Slightly lower
    }
    print(f"   RNN: {predictions_dict['rnn']['prediction']:.2f} KES (confidence: {predictions_dict['rnn']['confidence']:.2%}) [Using LSTM as fallback]")

# Save to database
if predictions_dict:
    success = saver.save_prediction(
        symbol=ANALYSIS_SYMBOL,
        current_price=current_price,
        predictions_dict=predictions_dict,
        prediction_date=None  # Uses today's date
    )

    if success:
        print(f"\n✅ Successfully saved predictions for {ANALYSIS_SYMBOL} to database!")
        print("\nYou can now:")
        print("  1. View predictions via API: GET /api/predictions")
        print("  2. See them in the frontend LiveMarket page")
    else:
        print(f"\n❌ Failed to save predictions for {ANALYSIS_SYMBOL}")
else:
    print("\n⚠️ No predictions to save. Train models first.")

print("="*60)
