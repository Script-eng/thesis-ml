import psycopg2
from psycopg2 import OperationalError, Error

# --- CONFIGURATION ---
# IMPORTANT: Connect to the default 'postgres' database to create a new one.
# You cannot be connected to a database that you are trying to create.
DB_ADMIN_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "secret", # Your Docker password
    "host": "localhost",  # Use the service name from docker-compose
    "port": "5432"
}

NEW_DB_NAME = "nse"
TABLE_NAME = "stocksdata"

def create_database_and_schema():
    """
    Connects to the PostgreSQL server, creates a new database,
    then connects to the new database to enable TimescaleDB
    and create the hypertable.
    """
    conn = None
    try:
        # --- Step 1: Connect to the default 'postgres' database ---
        print("Connecting to the default 'postgres' database...")
        conn = psycopg2.connect(**DB_ADMIN_CONFIG)
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
        db_config_new = DB_ADMIN_CONFIG.copy()
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

if __name__ == "__main__":
    create_database_and_schema()