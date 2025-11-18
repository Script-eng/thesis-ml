

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz
import logging

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Prophet
from prophet import Prophet

# Scikit-learn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Setup
load_dotenv()
NAIROBI_TZ = pytz.timezone('Africa/Nairobi')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('closing_price_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DailyDataAggregator:
    """Aggregates intraday data into daily closing prices."""

    def __init__(self, db_config):
        self.db_config = db_config

    def get_daily_closing_prices(self, symbol=None, days_back=90):
        """
        Extract daily closing prices from intraday data.

        Args:
            symbol: Stock symbol (None for all stocks)
            days_back: Number of days of historical data to fetch

        Returns:
            DataFrame with daily closing prices per symbol
        """
        try:
            conn = psycopg2.connect(**self.db_config)

            # Build query
            if symbol:
                symbol_filter = f"symbol = '{symbol}' AND"
            else:
                symbol_filter = ""

            query = f"""
            WITH daily_data AS (
                SELECT
                    symbol,
                    DATE(time AT TIME ZONE 'Africa/Nairobi') as trading_date,
                    latest_price,
                    high,
                    low,
                    volume,
                    ROW_NUMBER() OVER (
                        PARTITION BY symbol, DATE(time AT TIME ZONE 'Africa/Nairobi')
                        ORDER BY time DESC
                    ) as rn
                FROM stocksdata
                WHERE {symbol_filter} time >= NOW() - INTERVAL '{days_back} days'
            )
            SELECT
                symbol,
                trading_date,
                MAX(CASE WHEN rn = 1 THEN latest_price END) as close_price,
                MAX(high) as daily_high,
                MIN(low) as daily_low,
                SUM(volume) as daily_volume
            FROM daily_data
            GROUP BY symbol, trading_date
            ORDER BY symbol, trading_date;
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            logger.info(f"Fetched {len(df)} daily records for {df['symbol'].nunique()} symbols")
            return df

        except Error as e:
            logger.error(f"Database error: {e}")
            return None


class ClosingPricePredictor:
    """Base class for all prediction models."""

    def __init__(self, symbol, lookback_days=30):
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_name = "BaseModel"

    def prepare_sequences(self, data, for_prediction=False):
        """
        Prepare sequences for time series prediction.

        Args:
            data: Array of closing prices
            for_prediction: If True, only return the last sequence for prediction

        Returns:
            X, y arrays for training or X array for prediction
        """
        X, y = [], []

        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i, 0])
            if not for_prediction:
                y.append(data[i, 0])

        X = np.array(X)
        if not for_prediction:
            y = np.array(y)
            return X, y
        return X

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }


class LSTMPredictor(ClosingPricePredictor):
    """LSTM model for closing price prediction."""

    def __init__(self, symbol, lookback_days=30):
        super().__init__(symbol, lookback_days)
        self.model_name = "LSTM"

    def build_model(self, input_shape):
        """Build LSTM architecture."""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        logger.info(f"LSTM model built for {self.symbol}")
        return model

    def train(self, df_daily, test_size=0.2, epochs=100, batch_size=32):
        """Train LSTM model."""
        # Prepare data
        prices = df_daily[['close_price']].values
        scaled_data = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = self.prepare_sequences(scaled_data)

        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Train/test split (time-series aware)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build model
        self.build_model(input_shape=(X_train.shape[1], 1))

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        # Train
        logger.info(f"Training LSTM for {self.symbol} with {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Evaluate
        y_pred = self.model.predict(X_test, verbose=0)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)

        metrics = self.evaluate(y_test_actual, y_pred_actual)
        logger.info(f"LSTM {self.symbol} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

        return metrics, history

    def predict_next_day(self, df_daily):
        """Predict next day's closing price."""
        prices = df_daily[['close_price']].values
        scaled_data = self.scaler.transform(prices)

        # Get last sequence
        last_sequence = scaled_data[-self.lookback_days:]
        last_sequence = np.reshape(last_sequence, (1, self.lookback_days, 1))

        # Predict
        scaled_prediction = self.model.predict(last_sequence, verbose=0)
        prediction = self.scaler.inverse_transform(scaled_prediction)

        return float(prediction[0][0])

    def save_model(self, path):
        """Save model and scaler."""
        self.model.save(f"{path}/{self.symbol}_lstm_model.h5")
        joblib.dump(self.scaler, f"{path}/{self.symbol}_lstm_scaler.pkl")
        logger.info(f"LSTM model saved for {self.symbol}")

    def load_model(self, path):
        """Load model and scaler."""
        self.model = keras.models.load_model(f"{path}/{self.symbol}_lstm_model.h5")
        self.scaler = joblib.load(f"{path}/{self.symbol}_lstm_scaler.pkl")
        logger.info(f"LSTM model loaded for {self.symbol}")


class RNNPredictor(ClosingPricePredictor):
    """Simple RNN model for closing price prediction."""

    def __init__(self, symbol, lookback_days=30):
        super().__init__(symbol, lookback_days)
        self.model_name = "RNN"

    def build_model(self, input_shape):
        """Build RNN architecture."""
        model = Sequential([
            SimpleRNN(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            SimpleRNN(units=50, return_sequences=True),
            Dropout(0.2),
            SimpleRNN(units=50),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.model = model
        logger.info(f"RNN model built for {self.symbol}")
        return model

    def train(self, df_daily, test_size=0.2, epochs=100, batch_size=32):
        """Train RNN model."""
        # Prepare data
        prices = df_daily[['close_price']].values
        scaled_data = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = self.prepare_sequences(scaled_data)

        # Reshape for RNN [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Train/test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Build model
        self.build_model(input_shape=(X_train.shape[1], 1))

        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

        # Train
        logger.info(f"Training RNN for {self.symbol} with {len(X_train)} samples...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )

        # Evaluate
        y_pred = self.model.predict(X_test, verbose=0)
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)

        metrics = self.evaluate(y_test_actual, y_pred_actual)
        logger.info(f"RNN {self.symbol} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

        return metrics, history

    def predict_next_day(self, df_daily):
        """Predict next day's closing price."""
        prices = df_daily[['close_price']].values
        scaled_data = self.scaler.transform(prices)

        # Get last sequence
        last_sequence = scaled_data[-self.lookback_days:]
        last_sequence = np.reshape(last_sequence, (1, self.lookback_days, 1))

        # Predict
        scaled_prediction = self.model.predict(last_sequence, verbose=0)
        prediction = self.scaler.inverse_transform(scaled_prediction)

        return float(prediction[0][0])

    def save_model(self, path):
        """Save model and scaler."""
        self.model.save(f"{path}/{self.symbol}_rnn_model.h5")
        joblib.dump(self.scaler, f"{path}/{self.symbol}_rnn_scaler.pkl")
        logger.info(f"RNN model saved for {self.symbol}")

    def load_model(self, path):
        """Load model and scaler."""
        self.model = keras.models.load_model(f"{path}/{self.symbol}_rnn_model.h5")
        self.scaler = joblib.load(f"{path}/{self.symbol}_rnn_scaler.pkl")
        logger.info(f"RNN model loaded for {self.symbol}")


class ProphetPredictor:
    """Facebook Prophet model for closing price prediction."""

    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None
        self.model_name = "Prophet"

    def train(self, df_daily, test_size=0.2):
        """Train Prophet model."""
        
        prophet_df = pd.DataFrame({
            'ds': df_daily['trading_date'],
            'y': df_daily['close_price']
        })

        # Train/test split
        split_idx = int(len(prophet_df) * (1 - test_size))
        train_df = prophet_df[:split_idx]
        test_df = prophet_df[split_idx:]

        # Build and train model
        logger.info(f"Training Prophet for {self.symbol} with {len(train_df)} samples...")
        self.model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.model.fit(train_df)

        # Evaluate on test set
        forecast = self.model.predict(test_df[['ds']])
        y_test = test_df['y'].values
        y_pred = forecast['yhat'].values

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }

        logger.info(f"Prophet {self.symbol} - RMSE: {metrics['RMSE']:.2f}, MAE: {metrics['MAE']:.2f}, MAPE: {metrics['MAPE']:.2f}%")

        return metrics

    def predict_next_day(self, df_daily):
        """Predict next day's closing price."""
        last_date = df_daily['trading_date'].max()
        next_date = last_date + timedelta(days=1)

        # Skip weekends
        while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
            next_date += timedelta(days=1)

        future_df = pd.DataFrame({'ds': [next_date]})
        forecast = self.model.predict(future_df)

        return float(forecast['yhat'].values[0])

    def save_model(self, path):
        """Save Prophet model."""
        joblib.dump(self.model, f"{path}/{self.symbol}_prophet_model.pkl")
        logger.info(f"Prophet model saved for {self.symbol}")

    def load_model(self, path):
        """Load Prophet model."""
        self.model = joblib.load(f"{path}/{self.symbol}_prophet_model.pkl")
        logger.info(f"Prophet model loaded for {self.symbol}")


class ModelComparator:
    """Compare performance of multiple models."""

    def __init__(self):
        self.results = []

    def add_result(self, symbol, model_name, metrics, prediction):
        """Add model result."""
        result = {
            'symbol': symbol,
            'model': model_name,
            'prediction': prediction,
            **metrics
        }
        self.results.append(result)

    def get_comparison_df(self):
        """Get comparison DataFrame."""
        return pd.DataFrame(self.results)

    def get_best_model(self, symbol, metric='RMSE'):
        """Get best performing model for a symbol."""
        symbol_results = [r for r in self.results if r['symbol'] == symbol]
        if not symbol_results:
            return None

        best = min(symbol_results, key=lambda x: x[metric])
        return best['model']

    def save_results(self, path):
        """Save comparison results."""
        df = self.get_comparison_df()
        df.to_csv(f"{path}/model_comparison.csv", index=False)
        logger.info(f"Model comparison results saved to {path}/model_comparison.csv")


if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

    # Create models directory
    os.makedirs("models", exist_ok=True)

    # Initialize
    aggregator = DailyDataAggregator(DB_CONFIG)
    comparator = ModelComparator()

    # Get daily data
    logger.info("Fetching daily closing prices...")
    df_all = aggregator.get_daily_closing_prices(days_back=90)

    if df_all is None or df_all.empty:
        logger.error("No data available. Exiting.")
        exit(1)

    # Process each symbol
    symbols = df_all['symbol'].unique()
    logger.info(f"Processing {len(symbols)} symbols...")

    for symbol in symbols[:5]:  # Start with first 5 symbols for testing
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {symbol}")
        logger.info(f"{'='*60}")

        df_symbol = df_all[df_all['symbol'] == symbol].sort_values('trading_date')

        # Need at least 60 days of data
        if len(df_symbol) < 60:
            logger.warning(f"Insufficient data for {symbol} ({len(df_symbol)} days). Skipping.")
            continue

        try:
            # LSTM
            lstm = LSTMPredictor(symbol, lookback_days=30)
            lstm_metrics, _ = lstm.train(df_symbol, epochs=50)
            lstm_prediction = lstm.predict_next_day(df_symbol)
            lstm.save_model("models")
            comparator.add_result(symbol, "LSTM", lstm_metrics, lstm_prediction)

            # RNN
            rnn = RNNPredictor(symbol, lookback_days=30)
            rnn_metrics, _ = rnn.train(df_symbol, epochs=50)
            rnn_prediction = rnn.predict_next_day(df_symbol)
            rnn.save_model("models")
            comparator.add_result(symbol, "RNN", rnn_metrics, rnn_prediction)

            # Prophet
            prophet = ProphetPredictor(symbol)
            prophet_metrics = prophet.train(df_symbol)
            prophet_prediction = prophet.predict_next_day(df_symbol)
            prophet.save_model("models")
            comparator.add_result(symbol, "Prophet", prophet_metrics, prophet_prediction)

            # Summary for this symbol
            logger.info(f"\n{symbol} Predictions:")
            logger.info(f"  LSTM:    {lstm_prediction:.2f}")
            logger.info(f"  RNN:     {rnn_prediction:.2f}")
            logger.info(f"  Prophet: {prophet_prediction:.2f}")
            logger.info(f"  Current: {df_symbol['close_price'].iloc[-1]:.2f}")

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}", exc_info=True)
            continue

    # Save comparison
    comparator.save_results("models")

    # Display summary
    df_comparison = comparator.get_comparison_df()
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    print(df_comparison.to_string())
