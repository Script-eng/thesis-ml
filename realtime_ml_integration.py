"""
Real-time ML Integration for NSE Stock System
Connects the ML pipeline with live data streaming
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import redis
from typing import Dict, Optional
import psycopg2
from sqlalchemy import create_engine
import threading
import time
from ml_pipeline import MLPipeline, StockDataCleaner, FeatureEngineer, MLPredictor
import joblib
import os
from dotenv import load_dotenv

load_dotenv()


class RealTimePredictor:
    """Handles real-time predictions on streaming data"""
    
    def __init__(self, db_config: Dict, redis_config: Optional[Dict] = None):
        self.db_config = db_config
        self.pipeline = MLPipeline(db_config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis for caching predictions (optional)
        if redis_config:
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=True
            )
        else:
            self.redis_client = None
        
        # Model update tracking
        self.last_model_update = datetime.now()
        self.model_update_interval = timedelta(hours=24)  # Retrain daily
        
        # Performance metrics
        self.prediction_count = 0
        self.prediction_latencies = []
        
        # Load or train model
        self._initialize_model()
    
    def _initialize_model(self):
        """Load existing model or train new one"""
        
        model_path = 'nse_stock_model.pkl'
        
        if os.path.exists(model_path):
            self.logger.info("Loading existing model...")
            self.pipeline.predictor.load_model(model_path)
        else:
            self.logger.info("No model found. Training new model...")
            self.pipeline.train_pipeline(days_back=30)
    
    def process_live_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of live data and return predictions"""
        
        start_time = time.time()
        
        try:
            # Get predictions
            predictions = self.pipeline.predict_live(df)
            
            # Track performance
            latency = time.time() - start_time
            self.prediction_latencies.append(latency)
            self.prediction_count += len(predictions)
            
            # Cache predictions if Redis is available
            if self.redis_client:
                self._cache_predictions(predictions)
            
            # Log statistics
            if self.prediction_count % 100 == 0:
                avg_latency = np.mean(self.prediction_latencies[-100:])
                self.logger.info(f"Predictions made: {self.prediction_count}, "
                               f"Avg latency: {avg_latency:.3f}s")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}")
            return df  # Return original data if prediction fails
    
    def _cache_predictions(self, predictions: pd.DataFrame):
        """Cache predictions in Redis for fast retrieval"""
        
        for _, row in predictions.iterrows():
            key = f"prediction:{row['symbol']}:{datetime.now().strftime('%Y%m%d%H%M')}"
            value = {
                'signal': row['signal'],
                'confidence': float(row.get('confidence', 0)),
                'price': float(row['latest_price']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Set with 1 hour expiry
            self.redis_client.setex(key, 3600, json.dumps(value))
    
    def get_latest_predictions(self, symbol: Optional[str] = None) -> Dict:
        """Get latest predictions from cache or database"""
        
        if self.redis_client:
            if symbol:
                pattern = f"prediction:{symbol}:*"
            else:
                pattern = "prediction:*"
            
            keys = self.redis_client.keys(pattern)
            
            if keys:
                # Get the most recent key
                latest_key = sorted(keys)[-1]
                return json.loads(self.redis_client.get(latest_key))
        
        # Fallback to database
        return self._get_predictions_from_db(symbol)
    
    def _get_predictions_from_db(self, symbol: Optional[str] = None) -> Dict:
        """Retrieve latest predictions from database"""
        
        engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
        if symbol:
            query = f"""
            SELECT * FROM {self.db_config['table']} 
            WHERE symbol = '{symbol}'
            ORDER BY time DESC LIMIT 1
            """
        else:
            query = f"""
            SELECT DISTINCT ON (symbol) * FROM {self.db_config['table']}
            ORDER BY symbol, time DESC
            """
        
        df = pd.read_sql(query, engine)
        engine.dispose()
        
        return df.to_dict('records')
    
    def should_retrain(self) -> bool:
        """Check if model should be retrained"""
        
        time_since_update = datetime.now() - self.last_model_update
        return time_since_update > self.model_update_interval
    
    def retrain_model(self):
        """Retrain model with recent data"""
        
        self.logger.info("Starting model retraining...")
        
        # Train on last 30 days
        self.pipeline.train_pipeline(days_back=30)
        
        # Update timestamp
        self.last_model_update = datetime.now()
        
        self.logger.info("Model retraining complete")


class StreamProcessor:
    """Processes streaming data with ML predictions"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.predictor = RealTimePredictor(db_config)
        self.logger = logging.getLogger(__name__)
        
        # Buffer for batch processing
        self.data_buffer = []
        self.buffer_size = 20  # Process every 20 records
        self.buffer_timeout = 10  # Or every 10 seconds
        self.last_buffer_flush = time.time()
    
    def process_record(self, record: Dict):
        """Process a single streaming record"""
        
        # Add to buffer
        self.data_buffer.append(record)
        
        # Check if we should process the buffer
        if len(self.data_buffer) >= self.buffer_size or \
           (time.time() - self.last_buffer_flush) > self.buffer_timeout:
            self.flush_buffer()
    
    def flush_buffer(self):
        """Process buffered data"""
        
        if not self.data_buffer:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        # Get predictions
        predictions = self.predictor.process_live_batch(df)
        
        # Store predictions
        self._store_predictions(predictions)
        
        # Emit signals for high-confidence predictions
        self._emit_trading_signals(predictions)
        
        # Clear buffer
        self.data_buffer = []
        self.last_buffer_flush = time.time()
    
    def _store_predictions(self, predictions: pd.DataFrame):
        """Store predictions in database"""
        
        engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
        # Add prediction columns to the predictions table
        predictions['prediction_time'] = datetime.now()
        
        # Store in a separate predictions table
        predictions[['time', 'symbol', 'prediction', 'signal', 'confidence', 'prediction_time']].to_sql(
            'ml_predictions',
            engine,
            if_exists='append',
            index=False
        )
        
        engine.dispose()
    
    def _emit_trading_signals(self, predictions: pd.DataFrame):
        """Emit trading signals for high-confidence predictions"""
        
        # Filter high-confidence signals
        high_confidence = predictions[predictions['confidence'] > 0.75]
        
        for _, row in high_confidence.iterrows():
            signal = {
                'symbol': row['symbol'],
                'signal': row['signal'],
                'confidence': row['confidence'],
                'price': row['latest_price'],
                'timestamp': datetime.now().isoformat()
            }
            
            # Log the signal (in production, this could send to a message queue)
            if row['signal'] in ['BUY', 'SELL']:
                self.logger.info(f"TRADING SIGNAL: {signal}")


class MLIntegration:
    """Main integration class for ML with existing system"""
    
    def __init__(self):
        load_dotenv()
        
        self.db_config = {
            'dbname': os.getenv('DB_NAME', 'nse'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'password'),
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'table': os.getenv('TABLE_NAME', 'stocksdata')
        }
        
        self.processor = StreamProcessor(self.db_config)
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Control flags
        self.is_running = False
        self.processing_thread = None
        
    def start(self):
        """Start the ML integration"""
        
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start retraining scheduler
        self._schedule_retraining()
        
        self.logger.info("ML Integration started successfully")
    
    def stop(self):
        """Stop the ML integration"""
        
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        self.logger.info("ML Integration stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        
        engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['dbname']}"
        )
        
        last_processed_time = datetime.now()
        
        while self.is_running:
            try:
                # Fetch new data since last check
                query = f"""
                SELECT * FROM {self.db_config['table']}
                WHERE time > '{last_processed_time}'
                ORDER BY time
                """
                
                df = pd.read_sql(query, engine)
                
                if not df.empty:
                    # Process each record
                    for _, row in df.iterrows():
                        self.processor.process_record(row.to_dict())
                    
                    # Update last processed time
                    last_processed_time = df['time'].max()
                    
                    # Ensure buffer is flushed
                    self.processor.flush_buffer()
                
                # Sleep before next check
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(10)  # Wait longer on error
        
        engine.dispose()
    
    def _schedule_retraining(self):
        """Schedule periodic model retraining"""
        
        def retrain_check():
            while self.is_running:
                if self.processor.predictor.should_retrain():
                    self.logger.info("Starting scheduled model retraining...")
                    self.processor.predictor.retrain_model()
                
                # Check every hour
                time.sleep(3600)
        
        retrain_thread = threading.Thread(target=retrain_check)
        retrain_thread.daemon = True
        retrain_thread.start()
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        
        return {
            'predictions_made': self.processor.predictor.prediction_count,
            'average_latency': np.mean(self.processor.predictor.prediction_latencies[-100:])
            if self.processor.predictor.prediction_latencies else 0,
            'last_model_update': self.processor.predictor.last_model_update.isoformat(),
            'buffer_size': len(self.processor.data_buffer)
        }


# Integration with existing main1.py
def integrate_with_main():
    """Function to add to main1.py for ML integration"""
    
    # Add this to main1.py after starting scraper and ETL threads
    
    # Start ML integration
    ml_integration = MLIntegration()
    ml_integration.start()
    
    # The ML integration will automatically process new data
    # as it arrives in the database from the ETL pipeline
    
    return ml_integration


if __name__ == "__main__":
    # Standalone testing
    integration = MLIntegration()
    
    try:
        integration.start()
        
        # Keep running
        while True:
            # Print metrics every minute
            time.sleep(60)
            metrics = integration.get_performance_metrics()
            print(f"Performance Metrics: {metrics}")
            
    except KeyboardInterrupt:
        integration.stop()
        print("ML Integration stopped")
