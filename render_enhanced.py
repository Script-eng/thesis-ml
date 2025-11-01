"""
Enhanced API with ML Predictions
Serves stock data and ML predictions via REST and WebSocket
"""

import datetime
import jwt
import json
import numpy as np
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from sqlalchemy import create_engine, text
import pandas as pd
import pytz
from dotenv import load_dotenv
import os
from functools import wraps
import threading
import time
from typing import Dict, Optional

# Load environment variables
load_dotenv()

# Flask App Initialization
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- CONFIGURATION ---
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD") 
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
TABLE_NAME = os.getenv("TABLE_NAME")
TZ = os.getenv("TZ", "Africa/Nairobi")

# ML Configuration
ML_ENABLED = os.getenv("ENABLE_ML", "true").lower() == "true"
PREDICTIONS_TABLE = "ml_predictions"

# Timezone
NAIROBI_TZ = pytz.timezone(TZ)

# Database connection
DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# JWT Keys
try:
    with open('private_key.pem', 'rb') as f:
        PRIVATE_KEY = f.read()
    with open('public_key.pem', 'rb') as f:
        PUBLIC_KEY = f.read()
    print("✅ Keys loaded successfully.")
except FileNotFoundError:
    print("⚠️ JWT keys not found. Running in development mode.")
    PRIVATE_KEY = b"dev_key"
    PUBLIC_KEY = b"dev_key"

# Create engine
try:
    engine = create_engine(DATABASE_URI, pool_pre_ping=True, pool_size=10)
    print("✅ Database engine created.")
except Exception as e:
    engine = None
    print(f"❌ Database connection failed: {e}")


class DataService:
    """Service layer for data operations"""
    
    @staticmethod
    def fetch_latest_data() -> Optional[pd.DataFrame]:
        """Fetch latest stock data"""
        if not engine:
            return None
        
        try:
            query = f"""
            SELECT DISTINCT ON (symbol) * FROM {TABLE_NAME} 
            ORDER BY symbol, time DESC
            """
            df = pd.read_sql_query(query, engine)
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    @staticmethod
    def fetch_predictions(symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch ML predictions"""
        if not engine or not ML_ENABLED:
            return None
        
        try:
            if symbol:
                query = f"""
                SELECT * FROM {PREDICTIONS_TABLE}
                WHERE symbol = '{symbol}'
                ORDER BY prediction_time DESC
                LIMIT 100
                """
            else:
                query = f"""
                SELECT DISTINCT ON (symbol) * FROM {PREDICTIONS_TABLE}
                ORDER BY symbol, prediction_time DESC
                """
            
            df = pd.read_sql_query(query, engine)
            return df
        except Exception as e:
            print(f"Error fetching predictions: {e}")
            return None
    
    @staticmethod
    def fetch_historical_data(symbol: str, days: int = 7) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol"""
        if not engine:
            return None
        
        try:
            query = f"""
            SELECT * FROM {TABLE_NAME}
            WHERE symbol = '{symbol}'
            AND time >= NOW() - INTERVAL '{days} days'
            ORDER BY time
            """
            df = pd.read_sql_query(query, engine)
            return df
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    @staticmethod
    def fetch_market_summary() -> Dict:
        """Get market summary statistics"""
        if not engine:
            return {}
        
        try:
            query = f"""
            WITH latest_data AS (
                SELECT DISTINCT ON (symbol) * FROM {TABLE_NAME}
                ORDER BY symbol, time DESC
            )
            SELECT 
                COUNT(CASE WHEN change_direction = 'UP' THEN 1 END) as gainers,
                COUNT(CASE WHEN change_direction = 'DOWN' THEN 1 END) as losers,
                COUNT(CASE WHEN change_direction = 'NEUTRAL' THEN 1 END) as unchanged,
                SUM(volume) as total_volume,
                AVG(change_pct) as avg_change_pct
            FROM latest_data
            """
            
            result = pd.read_sql_query(query, engine)
            return result.iloc[0].to_dict()
        except Exception as e:
            print(f"Error fetching market summary: {e}")
            return {}


class WebSocketService:
    """Handles WebSocket connections and real-time updates"""
    
    def __init__(self):
        self.connected_clients = set()
        self.is_broadcasting = False
        self.broadcast_thread = None
    
    def start_broadcasting(self):
        """Start broadcasting updates to connected clients"""
        if self.is_broadcasting:
            return
        
        self.is_broadcasting = True
        self.broadcast_thread = threading.Thread(target=self._broadcast_loop)
        self.broadcast_thread.daemon = True
        self.broadcast_thread.start()
    
    def stop_broadcasting(self):
        """Stop broadcasting"""
        self.is_broadcasting = False
        if self.broadcast_thread:
            self.broadcast_thread.join(timeout=5)
    
    def _broadcast_loop(self):
        """Main broadcast loop"""
        while self.is_broadcasting:
            if self.connected_clients:
                # Fetch latest data
                data = DataService.fetch_latest_data()
                predictions = DataService.fetch_predictions() if ML_ENABLED else None
                
                if data is not None:
                    # Prepare broadcast message
                    message = {
                        'type': 'market_update',
                        'timestamp': datetime.datetime.now().isoformat(),
                        'data': data.to_dict('records') if not data.empty else [],
                    }
                    
                    if predictions is not None and not predictions.empty:
                        message['predictions'] = predictions.to_dict('records')
                    
                    # Emit to all connected clients
                    socketio.emit('market_update', message, namespace='/')
            
            time.sleep(10)  # Broadcast every 10 seconds


# Initialize WebSocket service
ws_service = WebSocketService()


# --- JWT Authentication ---
def auth_required(f):
    """Decorator for JWT authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization required"}), 401
        
        token = auth_header.split(' ')[1]
        try:
            jwt.decode(token, PUBLIC_KEY, algorithms=['RS256'], issuer='nse.market.data.service')
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401
        
        return f(*args, **kwargs)
    return decorated


# --- REST API ROUTES ---

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'service': 'NSE Market Data & ML Predictions API',
        'version': '2.0',
        'endpoints': {
            'auth': {
                'POST /auth/token': 'Get JWT token'
            },
            'data': {
                'GET /api/data': 'Latest stock data',
                'GET /api/data/<symbol>': 'Data for specific symbol',
                'GET /api/historical/<symbol>': 'Historical data',
                'GET /api/summary': 'Market summary'
            },
            'predictions': {
                'GET /api/predictions': 'Latest ML predictions',
                'GET /api/predictions/<symbol>': 'Predictions for symbol',
                'GET /api/signals': 'Trading signals'
            },
            'websocket': {
                'ws://host/': 'Real-time market updates'
            }
        },
        'ml_enabled': ML_ENABLED
    })


@app.route('/auth/token', methods=['POST'])
def generate_token():
    """Generate JWT token"""
    now_utc = datetime.datetime.now(datetime.UTC)
    payload = {
        'exp': now_utc + datetime.timedelta(minutes=60),
        'iat': now_utc,
        'iss': 'nse.market.data.service'
    }
    token = jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')
    return jsonify({'token': token, 'expires_in': 3600})


@app.route('/api/data')
@auth_required
def get_stock_data():
    """Get latest stock data"""
    data = DataService.fetch_latest_data()
    
    if data is None:
        return jsonify({"error": "Could not fetch data"}), 500
    
    response = {
        'timestamp': datetime.datetime.now(NAIROBI_TZ).isoformat(),
        'count': len(data),
        'data': data.to_dict('records')
    }
    
    return jsonify(response)


@app.route('/api/data/<symbol>')
@auth_required
def get_symbol_data(symbol):
    """Get data for specific symbol"""
    data = DataService.fetch_latest_data()
    
    if data is None:
        return jsonify({"error": "Could not fetch data"}), 500
    
    symbol_data = data[data['symbol'] == symbol.upper()]
    
    if symbol_data.empty:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404
    
    return jsonify(symbol_data.iloc[0].to_dict())


@app.route('/api/historical/<symbol>')
@auth_required
def get_historical_data(symbol):
    """Get historical data for a symbol"""
    days = request.args.get('days', 7, type=int)
    data = DataService.fetch_historical_data(symbol.upper(), days)
    
    if data is None:
        return jsonify({"error": "Could not fetch data"}), 500
    
    response = {
        'symbol': symbol.upper(),
        'days': days,
        'count': len(data),
        'data': data.to_dict('records')
    }
    
    return jsonify(response)


@app.route('/api/summary')
@auth_required
def get_market_summary():
    """Get market summary"""
    summary = DataService.fetch_market_summary()
    
    # Add top movers
    data = DataService.fetch_latest_data()
    if data is not None and not data.empty:
        # Top gainers
        gainers = data.nlargest(5, 'change_pct')[['symbol', 'name', 'latest_price', 'change_pct']]
        summary['top_gainers'] = gainers.to_dict('records')
        
        # Top losers
        losers = data.nsmallest(5, 'change_pct')[['symbol', 'name', 'latest_price', 'change_pct']]
        summary['top_losers'] = losers.to_dict('records')
        
        # Most active
        active = data.nlargest(5, 'volume')[['symbol', 'name', 'latest_price', 'volume']]
        summary['most_active'] = active.to_dict('records')
    
    return jsonify(summary)


@app.route('/api/predictions')
@auth_required
def get_predictions():
    """Get latest ML predictions"""
    if not ML_ENABLED:
        return jsonify({"error": "ML predictions not enabled"}), 404
    
    predictions = DataService.fetch_predictions()
    
    if predictions is None:
        return jsonify({"error": "Could not fetch predictions"}), 500
    
    response = {
        'timestamp': datetime.datetime.now(NAIROBI_TZ).isoformat(),
        'count': len(predictions),
        'predictions': predictions.to_dict('records') if not predictions.empty else []
    }
    
    return jsonify(response)


@app.route('/api/predictions/<symbol>')
@auth_required
def get_symbol_predictions(symbol):
    """Get predictions for specific symbol"""
    if not ML_ENABLED:
        return jsonify({"error": "ML predictions not enabled"}), 404
    
    predictions = DataService.fetch_predictions(symbol.upper())
    
    if predictions is None:
        return jsonify({"error": "Could not fetch predictions"}), 500
    
    if predictions.empty:
        return jsonify({"error": f"No predictions for {symbol}"}), 404
    
    response = {
        'symbol': symbol.upper(),
        'predictions': predictions.to_dict('records')
    }
    
    return jsonify(response)


@app.route('/api/signals')
@auth_required
def get_trading_signals():
    """Get high-confidence trading signals"""
    if not ML_ENABLED:
        return jsonify({"error": "ML predictions not enabled"}), 404
    
    predictions = DataService.fetch_predictions()
    
    if predictions is None or predictions.empty:
        return jsonify({"signals": []})
    
    # Filter for high confidence signals
    if 'confidence' in predictions.columns:
        signals = predictions[predictions['confidence'] > 0.7]
        signals = signals[signals['signal'].isin(['BUY', 'SELL'])]
        
        response = {
            'timestamp': datetime.datetime.now(NAIROBI_TZ).isoformat(),
            'count': len(signals),
            'signals': signals.to_dict('records') if not signals.empty else []
        }
    else:
        response = {'signals': []}
    
    return jsonify(response)


# --- WEBSOCKET HANDLERS ---

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    ws_service.connected_clients.add(client_id)
    
    # Start broadcasting if this is the first client
    if len(ws_service.connected_clients) == 1:
        ws_service.start_broadcasting()
    
    emit('connected', {
        'message': 'Connected to NSE Market Data Stream',
        'client_id': client_id
    })
    
    print(f"Client {client_id} connected. Total clients: {len(ws_service.connected_clients)}")


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    ws_service.connected_clients.discard(client_id)
    
    # Stop broadcasting if no clients
    if len(ws_service.connected_clients) == 0:
        ws_service.stop_broadcasting()
    
    print(f"Client {client_id} disconnected. Total clients: {len(ws_service.connected_clients)}")


@socketio.on('subscribe')
def handle_subscribe(data):
    """Handle subscription to specific symbols"""
    client_id = request.sid
    symbols = data.get('symbols', [])
    
    # In a production system, you'd track subscriptions per client
    # and filter broadcasts accordingly
    
    emit('subscribed', {
        'message': f'Subscribed to {len(symbols)} symbols',
        'symbols': symbols
    })


@socketio.on('unsubscribe')
def handle_unsubscribe(data):
    """Handle unsubscription"""
    client_id = request.sid
    symbols = data.get('symbols', [])
    
    emit('unsubscribed', {
        'message': f'Unsubscribed from {len(symbols)} symbols',
        'symbols': symbols
    })


if __name__ == '__main__':
    print("=" * 50)
    print("NSE Market Data & ML Predictions API")
    print("=" * 50)
    
    # Run with WebSocket support
    socketio.run(app, host='0.0.0.0', port=8052, debug=False)
