import time
import datetime
import traceback
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# --- CONFIGURATION ---
TARGET_URL = "https://fib.co.ke/live-markets/"
OUTPUT_FILENAME = ".rendered_stock_data.html"
SCRAPE_INTERVAL_SECONDS = 40
RESTART_DELAY_SECONDS = 10


def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/130.0.6723.116 Safari/537.36"
    )

    # Let Selenium find chromedriver from PATH
    try:
        driver = webdriver.Chrome(options=chrome_options)
        print("✅ ChromeDriver initialized successfully.")
        return driver
    except Exception as e:
        print("❌ Failed to initialize ChromeDriver.")
        traceback.print_exc()
        raise


def run_continuous_scraper(url: str, output_filename: str, interval: int):
    print("\n--- Continuous Scraping Initialized ---")
    print(f"Target: {url}")
    print(f"Output: {output_filename}")
    print(f"Interval: {interval} seconds")
    print("Press Ctrl+C to stop.\n")

    driver = setup_driver()
    driver.get(url)

    try:
        while True:
            cycle_start = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{cycle_start}] 🔁 New scrape cycle...")

            try:
                wait = WebDriverWait(driver, 20)
                print("🔍 Locating iframe 'mslFrame0'...")
                iframe = wait.until(EC.presence_of_element_located((By.ID, "mslFrame0")))
                driver.switch_to.frame(iframe)

                print("⏳ Waiting for data table inside iframe...")
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))

                rendered_html = driver.page_source
                with open(output_filename, "w", encoding="utf-8") as f:
                    f.write(rendered_html)

                print(f"✅ Saved HTML to '{output_filename}'")

            except Exception as e:
                print(f"⚠️  Scrape cycle failed: {e}")
                traceback.print_exc()

            finally:
                driver.switch_to.default_content()

            print(f"⏱ Waiting {interval} seconds...")
            time.sleep(interval)
            print("🔄 Refreshing page...")
            driver.refresh()

    except KeyboardInterrupt:
        print("\n🛑 Scraper interrupted by user.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
        raise
    finally:
        print("🔚 Closing browser...")
        driver.quit()
        print("✅ Scraper shut down.")


if __name__ == "__main__":
    while True:
        try:
            run_continuous_scraper(TARGET_URL, OUTPUT_FILENAME, SCRAPE_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("👋 Exiting by user request.")
            break
        except Exception as e:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{now}] 💥 Scraper crashed. Restarting in {RESTART_DELAY_SECONDS} seconds...")
            time.sleep(RESTART_DELAY_SECONDS)
