import os
import threading
import time
import logging
from dotenv import load_dotenv
from src.getlivedata import run_continuous_scraper
from src.processlivedata import main_etl_cycle
from src.utilities import setup_logging

# --- Configuration Loading ---
# Load environment variables from a .env file in your project root.
load_dotenv()

# Scraper Configuration (from getlivedata.py context)
TARGET_URL = os.getenv("TARGET_URL")
OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME") # This file is the bridge between the two processes
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", 60))
RESTART_DELAY_SECONDS = int(os.getenv("RESTART_DELAY_SECONDS", 30))

# ETL Configuration (from processlivedata.py context)
PROCESSING_INTERVAL_SECONDS = int(os.getenv("PROCESSING_INTERVAL_SECONDS", 10))
# The ETL process reads the file the scraper writes. Its source must match the scraper's output.
HTML_FILE_PATH = OUTPUT_FILENAME

# Shared Configuration (using a single log file for the combined app)
LOG_FILENAME = os.getenv("APP_LOG_FILENAME", "app.log")


# --- Worker Functions ---

def scraper_worker():
    """
    Worker for the scraper thread.
    This logic mirrors your original getlivedata.py main block,
    providing a loop that restarts the scraper if it crashes.
    """
    logging.info("Scraper worker started.")
    while True:
        try:
            # run_continuous_scraper is expected to contain its own internal loop
            # for scraping and sleeping (scrape -> sleep -> repeat).
            run_continuous_scraper(TARGET_URL, OUTPUT_FILENAME, SCRAPE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            # This allows Ctrl+C to be passed up to the main thread.
            break
        except Exception:
            # This top-level catch will restart the scraper if the whole function crashes.
            logging.error(f"Scraper function crashed. Restarting in {RESTART_DELAY_SECONDS} seconds...", exc_info=True)
            time.sleep(RESTART_DELAY_SECONDS)

def etl_worker():
    """
    Worker for the database ETL thread.
    This logic mirrors your original processlivedata.py main block.
    """
    logging.info(f"ETL worker started. Watching '{HTML_FILE_PATH}' to process every {PROCESSING_INTERVAL_SECONDS} seconds.")
    while True:
        try:
            main_etl_cycle()
            logging.info(f"ETL cycle complete. Waiting for {PROCESSING_INTERVAL_SECONDS} seconds.")
            time.sleep(PROCESSING_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            # Allow Ctrl+C to stop this thread's loop.
            break
        except Exception:
            logging.critical("An unexpected critical error occurred in the ETL worker.", exc_info=True)
            # Avoid rapid-fire errors by waiting before the next cycle.
            time.sleep(PROCESSING_INTERVAL_SECONDS)


if __name__ == "__main__":
    # 1. Setup logging for the entire application
    setup_logging(LOG_FILENAME)
    logging.info("--- Main Application Starting ---")

    # 2. Create thread objects for each worker function
    # daemon=True ensures threads exit when the main program does.
    scraper_thread = threading.Thread(target=scraper_worker, name="ScraperThread", daemon=True)
    etl_thread = threading.Thread(target=etl_worker, name="ETLThread", daemon=True)

    # 3. Start the threads
    scraper_thread.start()
    etl_thread.start()

    logging.info("Scraper and ETL processes are now running in the background.")
    logging.info("Press Ctrl+C to stop the application.")

    # 4. Keep the main thread alive and wait for a shutdown signal (Ctrl+C)
    try:
        # The main thread will wait here until the daemon threads finish,
        # which they only will upon error or program exit.
        scraper_thread.join()
        etl_thread.join()
    except KeyboardInterrupt:
        logging.info("Shutdown signal received. Exiting gracefully.")
    finally:
        logging.info("--- Application Shutting Down ---")