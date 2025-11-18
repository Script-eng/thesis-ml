
import os
import sys
import pandas as pd
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import logging
from datetime import datetime
import glob

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('csv_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()


class CSVImporter:
    """Import CSV data into PostgreSQL database."""

    def __init__(self, db_config):
        self.db_config = db_config
        self.table_name = os.getenv("TABLE_NAME", "stocksdata")

    def connect(self):
        """Create database connection."""
        try:
            conn = psycopg2.connect(**self.db_config)
            logger.info("✅ Database connection established")
            return conn
        except Error as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise

    def check_existing_data(self, conn):
        """Check what data already exists in database."""
        try:
            cursor = conn.cursor()

            # Get count and date range
            query = f"""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT symbol) as unique_symbols,
                MIN(time) as earliest,
                MAX(time) as latest
            FROM {self.table_name};
            """

            cursor.execute(query)
            result = cursor.fetchone()
            cursor.close()

            logger.info(f"\n{'='*60}")
            logger.info("EXISTING DATA IN DATABASE")
            logger.info(f"{'='*60}")
            logger.info(f"Total records: {result[0]:,}")
            logger.info(f"Unique symbols: {result[1]}")
            logger.info(f"Date range: {result[2]} to {result[3]}")
            logger.info(f"{'='*60}\n")

            return result

        except Error as e:
            logger.warning(f"Could not check existing data: {e}")
            return None

    def import_csv(self, csv_file, batch_size=10000, skip_duplicates=True):
        """
        Import CSV file into database.

        Args:
            csv_file: Path to CSV file
            batch_size: Number of rows to insert at once
            skip_duplicates: Skip records that already exist (based on time + symbol)
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"IMPORTING: {os.path.basename(csv_file)}")
        logger.info(f"{'='*60}")

        # Check file exists
        if not os.path.exists(csv_file):
            logger.error(f"❌ File not found: {csv_file}")
            return False

        # Get file info
        file_size_mb = os.path.getsize(csv_file) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB")

        try:
            # Read CSV in chunks for memory efficiency
            logger.info("Reading CSV file...")

            # Read entire file (or use chunks for very large files)
            df = pd.read_csv(csv_file)

            logger.info(f"Total rows in CSV: {len(df):,}")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
            logger.info(f"Unique symbols: {df['symbol'].nunique()}")

            # Connect to database
            conn = self.connect()

            # Check existing data
            self.check_existing_data(conn)

            # Prepare insert query
            insert_query = f"""
            INSERT INTO {self.table_name}
            (time, symbol, name, latest_price, prev_close, change_abs,
             change_pct, change_direction, high, low, avg_price, volume, trade_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            if skip_duplicates:
                insert_query += " ON CONFLICT DO NOTHING"  # Requires unique constraint
                # Alternative: Check before inserting (slower but works without constraint)

            # Insert in batches
            cursor = conn.cursor()
            total_inserted = 0
            total_skipped = 0
            batch_count = 0

            logger.info(f"\nInserting data in batches of {batch_size:,}...")

            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                batch_df = df.iloc[start_idx:end_idx]

                # Prepare batch data
                batch_data = []
                for _, row in batch_df.iterrows():
                    record = [
                        row['time'],
                        row['symbol'],
                        row['name'],
                        row['latest_price'] if pd.notna(row['latest_price']) else None,
                        row['prev_close'] if pd.notna(row['prev_close']) else None,
                        row['change_abs'] if pd.notna(row['change_abs']) else None,
                        row['change_pct'] if pd.notna(row['change_pct']) else None,
                        row['change_direction'] if pd.notna(row['change_direction']) else None,
                        row['high'] if pd.notna(row['high']) else None,
                        row['low'] if pd.notna(row['low']) else None,
                        row['avg_price'] if pd.notna(row['avg_price']) else None,
                        int(row['volume']) if pd.notna(row['volume']) else 0,
                        row['trade_time'] if pd.notna(row['trade_time']) else None
                    ]
                    batch_data.append(record)

                # Execute batch insert
                try:
                    cursor.executemany(insert_query, batch_data)
                    conn.commit()
                    total_inserted += len(batch_data)
                    batch_count += 1

                    # Progress update
                    progress = (end_idx / len(df)) * 100
                    logger.info(f"  Batch {batch_count}: Inserted {total_inserted:,} records ({progress:.1f}% complete)")

                except Error as e:
                    logger.warning(f"  Batch {batch_count} failed: {e}")
                    conn.rollback()
                    total_skipped += len(batch_data)

            cursor.close()

            # Final summary
            logger.info(f"\n{'='*60}")
            logger.info("IMPORT COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"✅ Successfully inserted: {total_inserted:,} records")
            if total_skipped > 0:
                logger.info(f"⚠️  Skipped (duplicates): {total_skipped:,} records")
            logger.info(f"{'='*60}\n")

            # Check updated database state
            self.check_existing_data(conn)

            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Import failed: {e}", exc_info=True)
            return False

    def import_multiple_files(self, pattern):
        """Import multiple CSV files matching a pattern."""
        files = glob.glob(pattern)

        if not files:
            logger.error(f"❌ No files found matching pattern: {pattern}")
            return

        logger.info(f"\n{'='*60}")
        logger.info(f"FOUND {len(files)} CSV FILE(S)")
        logger.info(f"{'='*60}")
        for f in files:
            logger.info(f"  - {os.path.basename(f)} ({os.path.getsize(f)/(1024*1024):.2f} MB)")
        logger.info(f"{'='*60}\n")

        success_count = 0
        for idx, csv_file in enumerate(files, 1):
            logger.info(f"\n[{idx}/{len(files)}] Processing {os.path.basename(csv_file)}...")

            if self.import_csv(csv_file):
                success_count += 1
            else:
                logger.error(f"Failed to import: {csv_file}")

        logger.info(f"\n{'='*60}")
        logger.info("ALL IMPORTS COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✅ Successfully imported: {success_count}/{len(files)} files")
        logger.info(f"{'='*60}\n")


def main():
    """Main entry point."""

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python import_csv_data.py <csv_file_or_pattern>")
        print("\nExamples:")
        print("  python import_csv_data.py data.csv")
        print("  python import_csv_data.py /path/to/data-*.csv")
        print("  python import_csv_data.py '/Users/lesalon/Desktop/fn/data-*.csv'")
        sys.exit(1)

    csv_pattern = sys.argv[1]

    # Database configuration
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

    # Validate configuration
    if not all(DB_CONFIG.values()):
        logger.error("❌ Database configuration incomplete. Check your .env file.")
        sys.exit(1)

    logger.info(f"\n{'='*60}")
    logger.info("NSE CSV DATA IMPORTER")
    logger.info(f"{'='*60}")
    logger.info(f"Database: {DB_CONFIG['dbname']}")
    logger.info(f"Host: {DB_CONFIG['host']}:{DB_CONFIG['port']}")
    logger.info(f"User: {DB_CONFIG['user']}")
    logger.info(f"Pattern: {csv_pattern}")
    logger.info(f"{'='*60}\n")

    # Create importer
    importer = CSVImporter(DB_CONFIG)

    # Check if pattern contains wildcards
    if '*' in csv_pattern or '?' in csv_pattern:
        importer.import_multiple_files(csv_pattern)
    else:
        importer.import_csv(csv_pattern)

    logger.info("\n✅ Import process complete! Check csv_import.log for details.")


if __name__ == "__main__":
    main()
