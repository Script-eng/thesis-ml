import logging
import pandas as pd
from psycopg2 import OperationalError, Error
import os
from dotenv import load_dotenv
import psycopg2
import datetime
from render import NAIROBI_TZ


# --- LOGGING SETUP ---
def setup_logging(LOG_FILENAME):
    """Configures logging to output to both the console and a file."""
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    # File Handler
    file_handler = logging.FileHandler(LOG_FILENAME, mode='a', encoding='utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)



def create_database_and_schema():
    load_dotenv() 
    # --- CONFIGURATION ---
    DB_CONFIG = {
        "dbname": "postgres",
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT"),
    }

    NEW_DB_NAME = os.getenv("NEW_DB_NAME")
    TABLE_NAME = os.getenv("TABLE_NAME")

    """
    Connects to the PostgreSQL server, creates a new database,
    then connects to the new database to enable TimescaleDB
    and create the hypertable.
    """

    conn = None
    try:
        # --- Step 1: Connect to the default 'postgres' database ---
        print("Connecting to the default 'postgres' database...")
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True  # Required to run CREATE DATABASE
        cursor = conn.cursor()

        # --- Step 2: Create the new 'nse' database ---
        print(f"Checking if database '{NEW_DB_NAME}' exists...")
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{NEW_DB_NAME}'")
        if not cursor.fetchone():
            print(f"Database '{NEW_DB_NAME}' not found. Creating it now...")
            cursor.execute(f"CREATE DATABASE {NEW_DB_NAME};")
            print(f"✅ Database '{NEW_DB_NAME}' created successfully.")
        else:
            print(f"Database '{NEW_DB_NAME}' already exists. Skipping creation.")

        cursor.close()
        conn.close()

        # --- Step 3: Connect to the NEW database to create the schema ---
        print(f"\nConnecting to the new '{NEW_DB_NAME}' database...")
        db_config_new = DB_CONFIG.copy()
        db_config_new["dbname"] = NEW_DB_NAME
        conn = psycopg2.connect(**db_config_new)
        cursor = conn.cursor()

        # --- Step 4: Enable the TimescaleDB extension ---
        print("Enabling the 'timescaledb' extension...")
        cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        print("✅ 'timescaledb' extension is enabled.")

        # --- Step 5: Create the historical data table ---
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            time            TIMESTAMPTZ       NOT NULL,
            symbol          VARCHAR(20)       NOT NULL,
            name            TEXT,
            latest_price    NUMERIC(12, 4),
            prev_close      NUMERIC(12, 4),
            change_abs      NUMERIC(12, 4),
            change_pct      NUMERIC(10, 4),
            change_direction VARCHAR(10),
            high            NUMERIC(12, 4),
            low             NUMERIC(12, 4),
            avg_price       NUMERIC(12, 4),
            volume          BIGINT,
            trade_time      VARCHAR(10)
        );
        """
        print(f"Creating table '{TABLE_NAME}'...")
        cursor.execute(create_table_query)
        print(f"✅ Table '{TABLE_NAME}' created successfully.")

        # --- Step 6: Convert the table into a TimescaleDB hypertable ---
        # This is the magic step for performance.
        hypertable_query = f"SELECT create_hypertable('{TABLE_NAME}', 'time', if_not_exists => TRUE);"
        print(f"Converting '{TABLE_NAME}' to a hypertable...")
        cursor.execute(hypertable_query)
        print(f"✅ '{TABLE_NAME}' is now a hypertable, partitioned by 'time'.")
        
        # --- (Optional) Step 7: Create a useful index ---
        print("Creating an index on (symbol, time) for fast lookups...")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS ix_symbol_time ON {TABLE_NAME} (symbol, time DESC);")
        print("✅ Index created.")

        conn.commit()
        cursor.close()
        print("\n🎉 Database schema setup is complete!")

    except OperationalError as e:
        print(f"❌ A connection error occurred: {e}")
        print("   Please ensure your Docker container is running and the DB_CONFIG is correct.")
    except Error as e:
        print(f"❌ A database error occurred: {e}")
    finally:
        if conn:
            conn.close()

def market_status():
    # This function determines the market status based on the current time.
    now = datetime.datetime.now(NAIROBI_TZ)
    if now.weekday() < 5:  # Monday to Friday
        if (now.hour == 9 and now.minute < 30):
            return "pre-open"
        elif (now.hour == 9 and now.minute >= 30) or (now.hour > 9 and now.hour < 15):
            return "open"
        elif now.hour == 15 and now.minute == 0:
            return "open"
        else:
            return "closed"
    return "closed"

