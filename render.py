import datetime
import jwt
from flask import Flask, jsonify, request, Response
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pytz
from flask_cors import CORS
from src.utilities import market_status
from dotenv import load_dotenv
import os
from functools import wraps

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # 🌐 Enable CORS for all routes

# --- LOAD KEYS FROM FILES ---
try:
    with open('private_key.pem', 'rb') as f:
        PRIVATE_KEY = f.read()
    with open('public_key.pem', 'rb') as f:
        PUBLIC_KEY = f.read()
    print("✅ Private and public keys loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: private_key.pem or public_key.pem not found.")
    exit()

# --- LOAD ENV VARIABLES ---
load_dotenv()
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
TABLE_NAME = os.getenv("TABLE_NAME")
TZ = os.getenv("TZ", "Africa/Nairobi")

# Timezone object
NAIROBI_TZ = pytz.timezone(TZ)

# --- DATABASE CONNECTION ---
DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URI, pool_pre_ping=True)
    print("✅ SQLAlchemy engine created successfully.")
except Exception as e:
    engine = None
    print(f"❌ Failed to create SQLAlchemy engine: {e}")

# --- HELPER FUNCTIONS ---
def fetch_latest_data_from_db():
    if not engine:
        return None
    try:
        query = f"SELECT DISTINCT ON (symbol) * FROM {TABLE_NAME} ORDER BY symbol, time DESC;"
        df = pd.read_sql_query(query, engine)
        if df.empty:
            return None
        df = df.replace({np.nan: None})
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"❌ Database Query Error: {e}")
        return None


def auth_required(f):
    """Decorator for enforcing JWT auth on endpoints"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header is missing or invalid"}), 401

        token = auth_header.split(' ')[1]
        try:
            jwt.decode(
                token,
                PUBLIC_KEY,
                algorithms=['RS256'],
                issuer='nse.market.data.service'
            )
            print("✅ JWT is valid.")
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError as e:
            return jsonify({"error": f"Invalid token: {e}"}), 401

        return f(*args, **kwargs)
    return decorated

# --- FLASK ROUTES ---
@app.route('/')
def index():
    message = r"""

 ▗▄▄▖ ▗▄▖ ▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖ ▗▄▖ ▗▄▄▖ ▗▄▄▄▖    
▐▌   ▐▌ ▐▌▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌       
 ▝▀▚▖▐▌ ▐▌▐▛▀▀▘  █  ▐▌ ▐▌▐▛▀▜▌▐▛▀▚▖▐▛▀▀▘    
▗▄▄▞▘▝▚▄▞▘▐▌     █  ▐▙█▟▌▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖    
                                            
▗▄▄▖ ▗▖ ▗▖▗▖    ▗▄▄▖▗▄▄▄▖ ▗▄▄▖              
▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌                 
▐▛▀▘ ▐▌ ▐▌▐▌    ▝▀▚▖▐▛▀▀▘ ▝▀▚▖              
▐▌   ▝▚▄▞▘▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▗▄▄▞▘              
"""
    return Response(message, mimetype='text/plain')


@app.route('/auth/token', methods=['POST'])
def generate_token():
    print("[AUTH] Issuing a new JWT token.")
    now_utc = datetime.datetime.now(datetime.UTC)
    payload = {
        'exp': now_utc + datetime.timedelta(minutes=15),
        'iat': now_utc,
        'iss': 'nse.market.data.service'
    }
    token = jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')
    return jsonify({'token': token})


@app.route('/api/data')
@auth_required
def get_stock_data_api():
    stock_data = fetch_latest_data_from_db()
    if stock_data is None:
        return jsonify({"error": "Could not retrieve data from the database"}), 500

    latest_time_object = max(item['time'] for item in stock_data if item['time'])
    timestamp = latest_time_object.astimezone(NAIROBI_TZ) if latest_time_object else None

    response = {
        "status": market_status(),
        "data_timestamp": timestamp.isoformat() if timestamp else None,
        "data": stock_data
    }
    return jsonify(response)


@app.route('/api/status', methods=['GET'])
@auth_required
def get_market_status_api():
    return jsonify({"status": market_status()})


@app.route('/api/symbols', methods=['GET'])
@auth_required
def get_symbols():
    try:
        query = f"SELECT DISTINCT symbol FROM {TABLE_NAME};"
        df = pd.read_sql_query(query, engine)
        return jsonify({"symbols": df['symbol'].tolist()})
    except Exception as e:
        return jsonify({"error": f"DB Error: {e}"}), 500


if __name__ == '__main__':
    print("--- Flask JWT-Secured API Server ---")
    app.run(host='0.0.0.0', port=8052, debug=False)
