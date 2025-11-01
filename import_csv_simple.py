"""
Simple CSV Data Importer (No external dependencies except psycopg2)
===================================================================
Imports historical CSV data into PostgreSQL database.

Usage:
    python3 import_csv_simple.py /path/to/data.csv
"""

import os
import sys
import csv
import psycopg2
from psycopg2 import Error
from datetime import datetime
import glob

# Load .env manually
def load_env():
    """Load environment variables from .env file."""
    env_vars = {}
    env_file = '.env'

    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

    return env_vars


def import_csv_to_db(csv_file, db_config, batch_size=10000):
    """Import CSV file into database."""

    print("\n" + "="*60)
    print(f"IMPORTING: {os.path.basename(csv_file)}")
    print("="*60)

    # Check file exists
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return False

    # Get file info
    file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    try:
        # Connect to database
        print("Connecting to database...")
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        print("✅ Connected")

        # Check existing data
        print("\nChecking existing data...")
        cursor.execute("SELECT COUNT(*), MIN(time), MAX(time) FROM stocksdata;")
        existing = cursor.fetchone()
        print(f"  Existing records: {existing[0]:,}")
        if existing[0] > 0:
            print(f"  Date range: {existing[1]} to {existing[2]}")

        # Prepare insert query
        insert_query = """
        INSERT INTO stocksdata
        (time, symbol, name, latest_price, prev_close, change_abs,
         change_pct, change_direction, high, low, avg_price, volume, trade_time)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Read and insert CSV
        print(f"\nReading CSV file...")

        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            batch_data = []
            total_inserted = 0
            total_rows = 0
            batch_count = 0

            for row in reader:
                total_rows += 1

                # Convert NULL strings to None
                def convert_value(val):
                    if val == 'NULL' or val == '' or val is None:
                        return None
                    return val

                # Prepare record
                record = [
                    row['time'],
                    row['symbol'],
                    row['name'],
                    convert_value(row['latest_price']),
                    convert_value(row['prev_close']),
                    convert_value(row['change_abs']),
                    convert_value(row['change_pct']),
                    convert_value(row['change_direction']),
                    convert_value(row['high']),
                    convert_value(row['low']),
                    convert_value(row['avg_price']),
                    int(row['volume']) if convert_value(row['volume']) else 0,
                    convert_value(row['trade_time'])
                ]

                batch_data.append(record)

                # Insert batch when it reaches batch_size
                if len(batch_data) >= batch_size:
                    try:
                        cursor.executemany(insert_query, batch_data)
                        conn.commit()
                        total_inserted += len(batch_data)
                        batch_count += 1
                        progress = (total_rows / 2371960) * 100  # Approximate total
                        print(f"  Batch {batch_count}: Inserted {total_inserted:,} records ({progress:.1f}% complete)")
                        batch_data = []
                    except Error as e:
                        print(f"  ⚠️  Batch {batch_count} error: {e}")
                        conn.rollback()
                        batch_data = []

            # Insert remaining records
            if batch_data:
                try:
                    cursor.executemany(insert_query, batch_data)
                    conn.commit()
                    total_inserted += len(batch_data)
                    batch_count += 1
                    print(f"  Batch {batch_count}: Inserted {total_inserted:,} records (100% complete)")
                except Error as e:
                    print(f"  ⚠️  Final batch error: {e}")
                    conn.rollback()

        # Final summary
        print("\n" + "="*60)
        print("IMPORT COMPLETE")
        print("="*60)
        print(f"Total rows processed: {total_rows:,}")
        print(f"✅ Successfully inserted: {total_inserted:,} records")
        print("="*60)

        # Check updated database state
        cursor.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(time), MAX(time) FROM stocksdata;")
        final = cursor.fetchone()
        print("\nDATABASE STATUS:")
        print(f"  Total records: {final[0]:,}")
        print(f"  Unique symbols: {final[1]}")
        print(f"  Date range: {final[2]} to {final[3]}")
        print("="*60 + "\n")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python3 import_csv_simple.py <csv_file>")
        print("\nExample:")
        print("  python3 import_csv_simple.py /Users/lesalon/Desktop/fn/data-1758802562185.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    # Load environment variables
    env = load_env()

    # Database configuration
    DB_CONFIG = {
        "dbname": env.get("DB_NAME"),
        "user": env.get("DB_USER"),
        "password": env.get("DB_PASSWORD"),
        "host": env.get("DB_HOST"),
        "port": env.get("DB_PORT")
    }

    # Validate configuration
    if not all(DB_CONFIG.values()):
        print("❌ Database configuration incomplete. Check your .env file.")
        print(f"Config: {DB_CONFIG}")
        sys.exit(1)

    print("\n" + "="*60)
    print("NSE CSV DATA IMPORTER (SIMPLE)")
    print("="*60)
    print(f"Database: {DB_CONFIG['dbname']}")
    print(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    print(f"User: {DB_CONFIG['user']}")
    print(f"File: {csv_file}")
    print("="*60)

    # Import
    if import_csv_to_db(csv_file, DB_CONFIG):
        print("\n✅ Import successful!")
    else:
        print("\n❌ Import failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
