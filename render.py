# app.py (Corrected)

import datetime
import jwt
from flask import Flask, render_template, jsonify, request
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pytz
from flask_cors import CORS
from src.utilities import market_status
from flask import Response

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "127.0.0.1, http://192.168.0.147:8080"}})  # 🌐 Enable CORS for all routes
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

# --- DATABASE CONFIGURATION ---
DB_NAME = "nse"
DB_USER = "postgres"
DB_PASS = "secret"
DB_HOST = "localhost"
DB_PORT = "5432"
TABLE_NAME = "stocksdata"
NAIROBI_TZ = pytz.timezone('Africa/Nairobi')

# --- DATABASE CONNECTION ---
DATABASE_URI = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
try:
    engine = create_engine(DATABASE_URI, pool_pre_ping=True)
    print("✅ SQLAlchemy engine created successfully.")
except Exception as e:
    engine = None
    print(f"❌ Failed to create SQLAlchemy engine: {e}")

# --- HELPER FUNCTION ---
def fetch_latest_data_from_db():
    if not engine: return None
    try:
        query = f"SELECT DISTINCT ON (symbol) * FROM {TABLE_NAME} ORDER BY symbol, time DESC;"
        df = pd.read_sql_query(query, engine)
        if df.empty: return None
        df = df.replace({np.nan: None})
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"❌ Database Query Error: {e}")
        return None

# --- FLASK ROUTES ---
@app.route('/')
def index():
    message = r"""

"In the beginning" : "there was data."





 ▗▄▄▖ ▗▄▖ ▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖ ▗▄▖ ▗▄▄▖ ▗▄▄▄▖    
▐▌   ▐▌ ▐▌▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌       
 ▝▀▚▖▐▌ ▐▌▐▛▀▀▘  █  ▐▌ ▐▌▐▛▀▜▌▐▛▀▚▖▐▛▀▀▘    
▗▄▄▞▘▝▚▄▞▘▐▌     █  ▐▙█▟▌▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖    
                                            
                                            
                                            
▗▄▄▖ ▗▖ ▗▖▗▖    ▗▄▄▖▗▄▄▄▖ ▗▄▄▖              
▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌   ▐▌   ▐▌                 
▐▛▀▘ ▐▌ ▐▌▐▌    ▝▀▚▖▐▛▀▀▘ ▝▀▚▖              
▐▌   ▝▚▄▞▘▐▙▄▄▖▗▄▄▞▘▐▙▄▄▖▗▄▄▞▘              
                                            
                                            
                                                                                                                










Digital Art by [Nyabuto]
"""
    return Response(message, mimetype='text/plain')

@app.route('/auth/token', methods=['POST'])
def generate_token():
    print("[AUTH] Issuing a new JWT token.")
    
    # --- FIX 2: Corrected the deprecated utcnow() calls ---
    # Use timezone-aware datetime.now(datetime.UTC)
    now_utc = datetime.datetime.now(datetime.UTC)
    payload = {
        'exp': now_utc + datetime.timedelta(minutes=15),
        'iat': now_utc,
        'iss': 'nse.market.data.service'
    }
    
    token = jwt.encode(payload, PRIVATE_KEY, algorithm='RS256')
    return jsonify({'token': token})

@app.route('/api/data')
def get_stock_data_api():
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return jsonify({"error": "Authorization header is missing or invalid"}), 401
    
    token = auth_header.split(' ')[1]
    
    try:
        jwt.decode(token, PUBLIC_KEY, algorithms=['RS256'], issuer='nse.market.data.service')
        print("✅ JWT is valid. Proceeding to fetch data.")
    except jwt.ExpiredSignatureError:
        print("❌ JWT has expired.")
        return jsonify({"error": "Token has expired"}), 401
    except jwt.InvalidTokenError as e:
        print(f"❌ Invalid JWT: {e}")
        return jsonify({"error": f"Invalid token: {e}"}), 401

    stock_data = fetch_latest_data_from_db()
    
    if stock_data is None:
        return jsonify({"error": "Could not retrieve data from the database"}), 500

    # --- FIX 1: Corrected the TypeError ---
    # Find the max datetime object directly from the list
    latest_time_object = max(item['time'] for item in stock_data if item['time'])
    
    # The value is already a datetime object, so we just convert its timezone
    timestamp = latest_time_object.astimezone(NAIROBI_TZ) if latest_time_object else None
    # Processing market status and returning as JSON file
    market_status_result = market_status()

    response = {
        "status": market_status_result,
        "data_timestamp": timestamp.isoformat() if timestamp else None,
        "data": stock_data
    }

    return jsonify(response)

if __name__ == '__main__':
    print("--- Flask JWT-Secured API Server ---")
    app.run(host='0.0.0.0', port=8052, debug=False)