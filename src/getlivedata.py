import time
import logging
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os
from dotenv import load_dotenv
from utilities import setup_logging


load_dotenv()
# --- CONFIGURATION ---
TARGET_URL = os.getenv("TARGET_URL")
OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME")
LOG_FILENAME = os.getenv("GETLIVEDATA_LOG_FILENAME")
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", 30))
RESTART_DELAY_SECONDS = int(os.getenv("RESTART_DELAY_SECONDS", 10))



def setup_driver():
    """Initializes and returns a headless Chrome WebDriver instance."""
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    # chrome_options.add_argument("--start-maximized")  # run chrome with gui
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.6723.116 Safari/537.36"
    )

    try:
        driver = webdriver.Chrome(options=chrome_options)
        logging.info("ChromeDriver initialized successfully.")
        return driver
    except Exception:
        logging.exception("Failed to initialize ChromeDriver. Check if chromedriver is in your PATH or installed correctly.")
        raise


def run_continuous_scraper(url: str, output_filename: str, interval: int):
    """
    Main function to run the scraping loop.

    Handles driver setup, page interaction, data extraction, and restarts.
    """
    logging.info("--- Continuous Scraping Initialized ---")
    logging.info(f"Target URL: {url}")
    logging.info(f"Output File: {output_filename}")
    logging.info(f"Scrape Interval: {interval} seconds")
    logging.info("Press Ctrl+C to stop the scraper.")

    driver = setup_driver()
    driver.get(url)

    try:
        while True:
            logging.info("Starting new scrape cycle...")

            try:
                wait = WebDriverWait(driver, 20)
                
                logging.info("Locating and switching to iframe 'mslFrame0'.")
                iframe = wait.until(EC.presence_of_element_located((By.ID, "mslFrame0")))
                driver.switch_to.frame(iframe)

                logging.info("Waiting for data table to load inside the iframe.")
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

                rendered_html = driver.page_source
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(rendered_html)

                logging.info(f"Successfully saved rendered HTML to '{output_filename}'.")

            except Exception as e:
                logging.error(f"A non-fatal error occurred during the scrape cycle: {e}")
                logging.exception("Traceback for the scrape cycle failure:")

            finally:
                driver.switch_to.default_content()

            logging.info(f"Cycle complete. Waiting for {interval} seconds...\n")
            time.sleep(interval)
            
            logging.info("Refreshing the page for the next cycle.")
            driver.refresh()

    # except KeyboardInterrupt:
    #     logging.warning("Scraper interrupted by user (Ctrl+C). Shutting down.")
    except Exception as e:
        logging.critical(f"A fatal error occurred in the main scraper loop: {e}")
        logging.exception("Traceback for the fatal error:")
        raise
    finally:
        logging.info("Closing browser and quitting WebDriver.")
        driver.quit()
        logging.info("Scraper has been shut down.")


if __name__ == "__main__":
    setup_logging(LOG_FILENAME)
    while True:
        try:
            run_continuous_scraper(TARGET_URL, OUTPUT_FILENAME, SCRAPE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            logging.info("Exiting application by user request.")
            break
        except Exception:
            logging.error(f"Scraper crashed. Restarting in {RESTART_DELAY_SECONDS} seconds...")
            time.sleep(RESTART_DELAY_SECONDS)