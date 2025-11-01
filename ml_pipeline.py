"""
NSE Stock Market ML Pipeline
Real-time prediction system for market movements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import psycopg2
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error
import joblib
import json

# Technical Analysis
import talib

class StockDataCleaner:
    """Handles real-time data cleaning and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.outlier_thresholds = {
            'price_change_pct': 20,  # Max 20% change in one tick
            'volume_spike': 10,      # Max 10x average volume
            'spread_pct': 5          # Max 5% bid-ask spread
        }
    
    def clean_stream_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean incoming data stream"""
        
        # 1. Handle missing values
        df = self._handle_missing_values(df)
        
        # 2. Detect and handle outliers
        df = self._detect_outliers(df)
        
        # 3. Validate data consistency
        df = self._validate_consistency(df)
        
        # 4. Normalize volume values
        df = self._normalize_volumes(df)
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Smart imputation for missing values"""
        
        # Forward fill for price data (last known price)
        price_cols = ['latest_price', 'prev_close', 'high', 'low', 'avg_price']
        df[price_cols] = df.groupby('symbol')[price_cols].fillna(method='ffill')
        
        # If still missing, use prev_close as fallback
        df['latest_price'] = df['latest_price'].fillna(df['prev_close'])
        
        # Calculate missing change values
        mask = df['change_abs'].isna()
        df.loc[mask, 'change_abs'] = df.loc[mask, 'latest_price'] - df.loc[mask, 'prev_close']
        df.loc[mask, 'change_pct'] = (df.loc[mask, 'change_abs'] / df.loc[mask, 'prev_close']) * 100
        
        # Fill missing change_direction
        df['change_direction'] = df.apply(
            lambda x: 'UP' if x['change_pct'] > 0 else 
                     'DOWN' if x['change_pct'] < 0 else 'NEUTRAL',
            axis=1
        )
        
        # Zero fill for missing volumes
        df['volume'] = df['volume'].fillna(0)
        
        return df
    
    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and flag outliers without removing them"""
        
        df['is_outlier'] = False
        
        # Check for extreme price changes
        df['is_outlier'] |= abs(df['change_pct']) > self.outlier_thresholds['price_change_pct']
        
        # Check for volume spikes
        avg_volume = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=1).mean())
        df['volume_ratio'] = df['volume'] / (avg_volume + 1)  # Avoid division by zero
        df['is_outlier'] |= df['volume_ratio'] > self.outlier_thresholds['volume_spike']
        
        # Check for suspicious spreads
        df['spread_pct'] = ((df['high'] - df['low']) / df['latest_price']) * 100
        df['is_outlier'] |= df['spread_pct'] > self.outlier_thresholds['spread_pct']
        
        if df['is_outlier'].any():
            self.logger.warning(f"Detected {df['is_outlier'].sum()} outliers in batch")
        
        return df
    
    def _validate_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate logical consistency of data"""
        
        # Ensure high >= low
        mask = df['high'] < df['low']
        df.loc[mask, 'high'], df.loc[mask, 'low'] = df.loc[mask, 'low'], df.loc[mask, 'high']
        
        # Ensure latest_price is within high-low range
        df['latest_price'] = df['latest_price'].clip(lower=df['low'], upper=df['high'])
        
        # Ensure avg_price is reasonable
        df['avg_price'] = df['avg_price'].clip(lower=df['low'], upper=df['high'])
        
        return df
    
    def _normalize_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure volumes are properly scaled"""
        
        # Already handled in processlivedata.py, but double-check
        df['volume'] = df['volume'].fillna(0).astype(np.int64)
        
        return df


class FeatureEngineer:
    """Creates ML features from clean stock data"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20]):
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(__name__)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML"""
        
        features = pd.DataFrame(index=df.index)
        
        # Group by symbol for feature calculation
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            symbol_data = df[mask].copy()
            
            # Sort by time for proper feature calculation
            symbol_data = symbol_data.sort_values('time')
            
            # 1. Price-based features
            symbol_features = self._create_price_features(symbol_data)
            
            # 2. Volume-based features
            symbol_features = pd.concat([symbol_features, 
                                        self._create_volume_features(symbol_data)], axis=1)
            
            # 3. Technical indicators
            symbol_features = pd.concat([symbol_features,
                                        self._create_technical_indicators(symbol_data)], axis=1)
            
            # 4. Market microstructure features
            symbol_features = pd.concat([symbol_features,
                                        self._create_microstructure_features(symbol_data)], axis=1)
            
            # 5. Time-based features
            symbol_features = pd.concat([symbol_features,
                                        self._create_time_features(symbol_data)], axis=1)
            
            # Add to main features dataframe
            features.loc[mask] = symbol_features
        
        # Add symbol encoding
        features = pd.concat([features, pd.get_dummies(df['symbol'], prefix='symbol')], axis=1)
        
        return features
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-related features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Returns at different scales
        for period in self.lookback_periods:
            features[f'return_{period}'] = df['latest_price'].pct_change(period)
            features[f'volatility_{period}'] = df['latest_price'].pct_change().rolling(period).std()
        
        # Price position within the day's range
        features['price_position'] = (df['latest_price'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        # Distance from VWAP
        features['vwap_distance'] = (df['latest_price'] - df['avg_price']) / df['avg_price']
        
        # Momentum
        features['momentum'] = df['change_pct']
        
        # Price acceleration
        features['price_acceleration'] = df['change_pct'].diff()
        
        return features
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-related features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Volume moving averages
        for period in self.lookback_periods:
            features[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            features[f'volume_ratio_{period}'] = df['volume'] / (features[f'volume_ma_{period}'] + 1)
        
        # Volume-price correlation
        features['volume_price_corr'] = df['volume'].rolling(20).corr(df['latest_price'])
        
        # On-Balance Volume (simplified)
        features['obv'] = (df['volume'] * np.where(df['change_pct'] > 0, 1, -1)).cumsum()
        
        return features
    
    def _create_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Technical analysis indicators using TA-Lib"""
        
        features = pd.DataFrame(index=df.index)
        
        try:
            # Ensure we have enough data
            if len(df) < 30:
                return features
            
            # Convert to numpy arrays
            high = df['high'].values
            low = df['low'].values
            close = df['latest_price'].values
            volume = df['volume'].values
            
            # RSI
            features['rsi'] = talib.RSI(close, timeperiod=14)
            
            # MACD
            macd, macd_signal, macd_hist = talib.MACD(close, 
                                                       fastperiod=12, 
                                                       slowperiod=26, 
                                                       signalperiod=9)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_histogram'] = macd_hist
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(close, timeperiod=20)
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = upper - lower
            features['bb_position'] = (close - lower) / (upper - lower + 0.0001)
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd
            
            # ATR (Average True Range)
            features['atr'] = talib.ATR(high, low, close, timeperiod=14)
            
            # MFI (Money Flow Index)
            features['mfi'] = talib.MFI(high, low, close, volume, timeperiod=14)
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators: {e}")
        
        return features
    
    def _create_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Spread metrics
        features['spread'] = df['high'] - df['low']
        features['spread_pct'] = (features['spread'] / df['latest_price']) * 100
        
        # Price efficiency
        features['price_efficiency'] = abs(df['latest_price'] - df['avg_price']) / df['avg_price']
        
        # Tick direction
        features['tick_direction'] = df['change_pct'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)
        
        # Cumulative tick direction (uptick/downtick ratio)
        features['cum_tick_direction'] = features['tick_direction'].rolling(10).sum()
        
        return features
    
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features"""
        
        features = pd.DataFrame(index=df.index)
        
        # Extract time components
        df['time'] = pd.to_datetime(df['time'])
        
        # Hour of day (market hours)
        features['hour'] = df['time'].dt.hour
        features['minute'] = df['time'].dt.minute
        
        # Time since market open (9:30 AM)
        features['minutes_since_open'] = (features['hour'] - 9) * 60 + features['minute'] - 30
        
        # Time until market close (3:00 PM)
        features['minutes_until_close'] = 330 - features['minutes_since_open']  # 5.5 hours = 330 minutes
        
        # Day of week
        features['day_of_week'] = df['time'].dt.dayofweek
        
        # Is first/last 30 minutes (high volatility periods)
        features['is_opening'] = features['minutes_since_open'] <= 30
        features['is_closing'] = features['minutes_until_close'] <= 30
        
        return features


class MLPredictor:
    """Handles model training, evaluation, and prediction"""
    
    def __init__(self, prediction_target: str = 'direction', prediction_horizon: int = 1):
        """
        Initialize predictor
        
        Args:
            prediction_target: What to predict ('direction', 'volatility', 'returns')
            prediction_horizon: How many periods ahead to predict
        """
        self.prediction_target = prediction_target
        self.prediction_horizon = prediction_horizon
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.logger = logging.getLogger(__name__)
        
    def prepare_training_data(self, df: pd.DataFrame, features: pd.DataFrame) -> Tuple:
        """Prepare features and targets for training"""
        
        # Remove NaN values
        valid_idx = ~features.isna().any(axis=1)
        features_clean = features[valid_idx]
        df_clean = df[valid_idx]
        
        # Create target variable based on prediction type
        if self.prediction_target == 'direction':
            # Predict if price will go up (1), down (-1), or stay neutral (0)
            future_returns = df_clean.groupby('symbol')['latest_price'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            y = pd.cut(future_returns, bins=[-np.inf, -0.001, 0.001, np.inf], labels=[-1, 0, 1])
            
        elif self.prediction_target == 'volatility':
            # Predict future volatility
            y = df_clean.groupby('symbol')['latest_price'].pct_change().rolling(5).std().shift(-self.prediction_horizon)
            
        elif self.prediction_target == 'returns':
            # Predict actual returns
            y = df_clean.groupby('symbol')['latest_price'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            
        else:
            raise ValueError(f"Unknown prediction target: {self.prediction_target}")
        
        # Remove rows where target is NaN
        valid_target = ~y.isna()
        X = features_clean[valid_target]
        y = y[valid_target]
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'ensemble'):
        """Train ML model with time series cross-validation"""
        
        # Scale features
        scaler = RobustScaler()  # Robust to outliers
        X_scaled = scaler.fit_transform(X)
        
        # Initialize model based on type and target
        if self.prediction_target == 'direction':
            if model_type == 'ensemble':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                # Could add XGBoost, LightGBM, etc.
                pass
                
        else:  # Regression tasks (volatility, returns)
            if model_type == 'ensemble':
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42
                )
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            predictions = model.predict(X_val)
            
            if self.prediction_target == 'direction':
                score = accuracy_score(y_val, predictions)
            else:
                score = -mean_absolute_error(y_val, predictions)  # Negative MAE for consistency
            
            scores.append(score)
            
        self.logger.info(f"Cross-validation scores: {scores}")
        self.logger.info(f"Mean CV score: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
        
        # Train final model on all data
        model.fit(X_scaled, y)
        
        # Store model and scaler
        self.models[model_type] = model
        self.scalers[model_type] = scaler
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_type] = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return model, scaler
    
    def predict(self, features: pd.DataFrame, model_type: str = 'ensemble') -> pd.DataFrame:
        """Make predictions on new data"""
        
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.models[model_type]
        scaler = self.scalers[model_type]
        
        # Scale features
        X_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Add prediction probabilities for classification
        if self.prediction_target == 'direction' and hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_scaled)
            pred_df = pd.DataFrame({
                'prediction': predictions,
                'prob_down': probas[:, 0] if probas.shape[1] > 0 else 0,
                'prob_neutral': probas[:, 1] if probas.shape[1] > 1 else 0,
                'prob_up': probas[:, 2] if probas.shape[1] > 2 else 0,
                'confidence': probas.max(axis=1)
            })
        else:
            pred_df = pd.DataFrame({'prediction': predictions})
        
        return pred_df
    
    def save_model(self, filepath: str, model_type: str = 'ensemble'):
        """Save trained model and scaler"""
        
        model_data = {
            'model': self.models[model_type],
            'scaler': self.scalers[model_type],
            'feature_importance': self.feature_importance.get(model_type, None),
            'prediction_target': self.prediction_target,
            'prediction_horizon': self.prediction_horizon
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, model_type: str = 'ensemble'):
        """Load trained model and scaler"""
        
        model_data = joblib.load(filepath)
        
        self.models[model_type] = model_data['model']
        self.scalers[model_type] = model_data['scaler']
        self.feature_importance[model_type] = model_data['feature_importance']
        self.prediction_target = model_data['prediction_target']
        self.prediction_horizon = model_data['prediction_horizon']
        
        self.logger.info(f"Model loaded from {filepath}")


class MLPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.cleaner = StockDataCleaner()
        self.engineer = FeatureEngineer()
        self.predictor = MLPredictor(prediction_target='direction')
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_historical_data(self, days_back: int = 30) -> pd.DataFrame:
        """Load historical data for training"""
        
        engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
        query = f"""
        SELECT * FROM {self.db_config['table']}
        WHERE time >= NOW() - INTERVAL '{days_back} days'
        ORDER BY time, symbol
        """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        return df
    
    def train_pipeline(self, days_back: int = 30):
        """Train the complete ML pipeline"""
        
        self.logger.info("Starting ML pipeline training...")
        
        # 1. Load historical data
        self.logger.info(f"Loading {days_back} days of historical data...")
        df = self.load_historical_data(days_back)
        self.logger.info(f"Loaded {len(df)} records")
        
        # 2. Clean data
        self.logger.info("Cleaning data...")
        df_clean = self.cleaner.clean_stream_data(df)
        
        # 3. Engineer features
        self.logger.info("Engineering features...")
        features = self.engineer.create_features(df_clean)
        
        # 4. Prepare training data
        self.logger.info("Preparing training data...")
        X, y = self.predictor.prepare_training_data(df_clean, features)
        self.logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
        
        # 5. Train model
        self.logger.info("Training model...")
        model, scaler = self.predictor.train_model(X, y)
        
        # 6. Display feature importance
        if 'ensemble' in self.predictor.feature_importance:
            top_features = self.predictor.feature_importance['ensemble'].head(20)
            self.logger.info("\nTop 20 Most Important Features:")
            print(top_features)
        
        # 7. Save model
        self.predictor.save_model('nse_stock_model.pkl')
        
        return model, scaler
    
    def predict_live(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on live streaming data"""
        
        # 1. Clean incoming data
        df_clean = self.cleaner.clean_stream_data(df)
        
        # 2. Engineer features
        features = self.engineer.create_features(df_clean)
        
        # 3. Make predictions
        predictions = self.predictor.predict(features)
        
        # 4. Combine with original data
        result = pd.concat([
            df[['time', 'symbol', 'name', 'latest_price', 'volume']],
            predictions
        ], axis=1)
        
        # 5. Add trading signals
        result['signal'] = result['prediction'].map({
            1: 'BUY',
            0: 'HOLD',
            -1: 'SELL'
        })
        
        # Only show high confidence predictions
        result.loc[result['confidence'] < 0.6, 'signal'] = 'HOLD'
        
        return result
    
    def backtest(self, days_back: int = 7) -> Dict:
        """Backtest the model on recent data"""
        
        self.logger.info("Starting backtest...")
        
        # Load recent data
        df = self.load_historical_data(days_back)
        
        # Split into train and test
        split_point = int(len(df) * 0.7)
        df_train = df[:split_point]
        df_test = df[split_point:]
        
        # Train on earlier data
        df_clean = self.cleaner.clean_stream_data(df_train)
        features_train = self.engineer.create_features(df_clean)
        X_train, y_train = self.predictor.prepare_training_data(df_clean, features_train)
        
        # Test on later data
        df_test_clean = self.cleaner.clean_stream_data(df_test)
        features_test = self.engineer.create_features(df_test_clean)
        X_test, y_test = self.predictor.prepare_training_data(df_test_clean, features_test)
        
        # Train model
        self.predictor.train_model(X_train, y_train)
        
        # Make predictions
        predictions = self.predictor.predict(X_test)
        
        # Calculate metrics
        if self.predictor.prediction_target == 'direction':
            accuracy = accuracy_score(y_test, predictions['prediction'])
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, predictions['prediction'], average='weighted'
            )
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_predictions': len(predictions),
                'prediction_distribution': predictions['prediction'].value_counts().to_dict()
            }
        else:
            mae = mean_absolute_error(y_test, predictions['prediction'])
            results = {'mae': mae, 'total_predictions': len(predictions)}
        
        self.logger.info(f"Backtest results: {results}")
        
        return results


# Example usage and integration point
if __name__ == "__main__":
    # Database configuration
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    db_config = {
        'dbname': os.getenv('DB_NAME', 'nse'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'password'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'table': os.getenv('TABLE_NAME', 'stocksdata')
    }
    
    # Initialize pipeline
    pipeline = MLPipeline(db_config)
    
    # Train model on historical data
    model, scaler = pipeline.train_pipeline(days_back=30)
    
    # Backtest
    backtest_results = pipeline.backtest(days_back=7)
    print(f"\nBacktest Results: {backtest_results}")
    
    # Example: Load latest data for prediction
    latest_data = pipeline.load_historical_data(days_back=1)
    if not latest_data.empty:
        predictions = pipeline.predict_live(latest_data.tail(100))
        print("\nLatest Predictions:")
        print(predictions[['time', 'symbol', 'latest_price', 'signal', 'confidence']].head(10))
