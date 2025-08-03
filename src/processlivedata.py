import time
import datetime
import logging
import psycopg2
from psycopg2 import OperationalError, Error
import pandas as pd
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DB_CONFIG = {
    "dbname": "nse",
    "user": "postgres",
    "password": "secret", # Your Docker password
    "host": "localhost",  # Use the service name from docker-compose
    "port": "5432"
}
TABLE_NAME = "stocksdata" # The new hypertable
HTML_FILE_PATH = ".rendered_stock_data.html"
LOG_FILENAME = "etl_timescale.log"
PROCESSING_INTERVAL_SECONDS = 30

# --- LOGGING SETUP ---
def setup_logging():
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

# --- PARSE & TRANSFORM FUNCTION (Identical to your original) ---
def parse_local_html_file(html_filepath: str):
    """Parses a locally saved HTML file to extract and clean stock data."""
    try:
        with open(html_filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        logging.warning(f"HTML file '{html_filepath}' not found. Waiting for scraper to create it.")
        return None
    soup = BeautifulSoup(html_content, "html.parser")
    data_table = soup.find("table", id="board_table")
    if not data_table:
        logging.warning("Could not find data table in HTML. File might be empty or malformed.")
        return None
    data_rows = data_table.find("tbody").find_all("tr")
    all_stocks_data = []
    for row in data_rows:
        if "footer" in row.get("class", []): continue
        cells = row.find_all("td")
        if len(cells) == 10:
            all_stocks_data.append({
                'SYMBOL': cells[0].text.strip(), 'NAME': row.get("title", "").strip(),
                'PREV_CLOSE': cells[1].text.strip(), 'LATEST_PRICE': cells[2].text.strip(),
                'CHANGE_ABS': cells[3].text.strip(), 'CHANGE_PCT_RAW': cells[4].text.strip(),
                'HIGH': cells[5].text.strip(), 'LOW': cells[6].text.strip(),
                'VOLUME': cells[7].text.strip(), 'AVG_PRICE': cells[8].text.strip(),
                'TIME': cells[9].text.strip()
            })
    if not all_stocks_data:
        logging.warning("HTML parsed, but no stock data rows were extracted.")
        return None
    df = pd.DataFrame(all_stocks_data)
    def clean_volume(v_str):
        v_str = v_str.lower().replace(',', '')
        if 'm' in v_str: return float(v_str.replace('m', '')) * 1_000_000
        if 'k' in v_str: return float(v_str.replace('k', '')) * 1_000
        return pd.to_numeric(v_str, errors='coerce')
    df['VOLUME'] = df['VOLUME'].apply(clean_volume).fillna(0).astype(int)
    numeric_cols = ['PREV_CLOSE', 'LATEST_PRICE', 'CHANGE_ABS', 'HIGH', 'LOW', 'AVG_PRICE']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['CHANGE_PCT'] = df['CHANGE_PCT_RAW'].str.extract(r'([-+]?\d+\.?\d*)').astype(float)
    df['CHANGE_DIRECTION'] = 'NEUTRAL'
    df.loc[df['CHANGE_PCT_RAW'].str.contains('▲', na=False), 'CHANGE_DIRECTION'] = 'UP'
    df.loc[df['CHANGE_PCT_RAW'].str.contains('▼', na=False), 'CHANGE_DIRECTION'] = 'DOWN'
    final_cols = ['SYMBOL', 'NAME', 'LATEST_PRICE', 'PREV_CLOSE', 'CHANGE_ABS', 'CHANGE_PCT', 'CHANGE_DIRECTION', 'HIGH', 'LOW', 'AVG_PRICE', 'VOLUME', 'TIME']
    return df[final_cols]


# --- DATABASE INSERT FUNCTION ---
def insert_stock_data(conn, df):
    """Inserts new stock data records into the TimescaleDB hypertable."""
    # Note the change in columns to match our new table schema
    insert_query = f"""
    INSERT INTO {TABLE_NAME} (time, symbol, name, latest_price, prev_close, change_abs, change_pct, change_direction, high, low, avg_price, volume, trade_time)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
    """
    with conn.cursor() as cursor:
        # Prepare data for efficient batch insertion
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        data_to_insert = []
        for row in df.itertuples(index=False):
            # Prepend the timestamp and handle potential NaN values from pandas
            record = [now_utc] + [None if pd.isna(x) else x for x in row]
            data_to_insert.append(record)
        
        # Use executemany for a highly efficient batch insert
        if data_to_insert:
            cursor.executemany(insert_query, data_to_insert)
            logging.info(f"Successfully prepared to insert {len(data_to_insert)} records.")
        else:
            logging.warning("No data records were prepared for insertion.")
        
        conn.commit()


# --- MAIN EXECUTION LOOP ---
def main_etl_cycle():
    """Performs one full cycle of parsing and loading."""
    logging.info("--- Starting ETL cycle ---")
    
    stock_df = parse_local_html_file(HTML_FILE_PATH)
    
    if stock_df is None or stock_df.empty:
        logging.warning("Parsing failed or produced no data. Skipping database load.")
        return

    logging.info(f"Successfully parsed and cleaned {len(stock_df)} records.")

    conn = None
    try:
        logging.info(f"Connecting to the '{DB_CONFIG['dbname']}' database...")
        conn = psycopg2.connect(**DB_CONFIG)
        
        insert_stock_data(conn, stock_df)
        logging.info("✅ Data successfully inserted into the TimescaleDB hypertable.")
        
    except OperationalError as e:
        logging.critical(f"DB connection failed: {e}. Check if the Docker container is running.")
    except Error as e:
        logging.error(f"A database error occurred during the cycle: {e}")
    finally:
        if conn:
            conn.close()
            logging.info("Database connection closed for this cycle.")


if __name__ == "__main__":
    setup_logging()
    logging.info("--- Continuous TimescaleDB ETL Process Started ---")
    logging.info(f"Watching '{HTML_FILE_PATH}' and loading to DB every {PROCESSING_INTERVAL_SECONDS} seconds.")
    logging.info("Press Ctrl+C to stop.")

    try:
        while True:
            main_etl_cycle()
            logging.info(f"--- Cycle complete. Waiting for {PROCESSING_INTERVAL_SECONDS} seconds... ---")
            time.sleep(PROCESSING_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logging.warning("Process stopped by user. Exiting.")
    except Exception:
        logging.critical("An unexpected critical error occurred in the main loop.", exc_info=True)