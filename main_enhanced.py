"""
Enhanced Main Module with ML Integration
Combines scraping, ETL, and ML prediction in one orchestrated system
"""

import os
import threading
import time
import logging
from dotenv import load_dotenv
from src.getlivedata import run_continuous_scraper
from src.processlivedata import main_etl_cycle
from src.utilities import setup_logging
from realtime_ml_integration import MLIntegration

# Load environment variables
load_dotenv()

# --- Configuration Loading ---
# Scraper Configuration
TARGET_URL = os.getenv("TARGET_URL")
OUTPUT_FILENAME = os.getenv("OUTPUT_FILENAME")
SCRAPE_INTERVAL_SECONDS = int(os.getenv("SCRAPE_INTERVAL_SECONDS", 60))
RESTART_DELAY_SECONDS = int(os.getenv("RESTART_DELAY_SECONDS", 30))

# ETL Configuration
PROCESSING_INTERVAL_SECONDS = int(os.getenv("PROCESSING_INTERVAL_SECONDS", 10))
HTML_FILE_PATH = OUTPUT_FILENAME

# ML Configuration
ENABLE_ML = os.getenv("ENABLE_ML", "true").lower() == "true"
ML_WARMUP_DELAY = int(os.getenv("ML_WARMUP_DELAY", 60))  # Wait for initial data

# Logging Configuration
LOG_FILENAME = os.getenv("APP_LOG_FILENAME", "app.log")


class EnhancedSystemOrchestrator:
    """Orchestrates all components of the stock data system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threads = {}
        self.ml_integration = None
        self.is_running = False
        self.startup_time = time.time()
        
        # Performance tracking
        self.scraper_cycles = 0
        self.etl_cycles = 0
        self.errors = {'scraper': 0, 'etl': 0, 'ml': 0}
    
    def scraper_worker(self):
        """Enhanced scraper worker with cycle counting"""
        self.logger.info("Scraper worker started.")
        
        while self.is_running:
            try:
                run_continuous_scraper(TARGET_URL, OUTPUT_FILENAME, SCRAPE_INTERVAL_SECONDS)
                self.scraper_cycles += 1
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.errors['scraper'] += 1
                self.logger.error(f"Scraper crashed (attempt #{self.errors['scraper']}): {e}")
                
                if self.errors['scraper'] > 10:
                    self.logger.critical("Scraper failed too many times. Stopping.")
                    break
                    
                time.sleep(RESTART_DELAY_SECONDS)
    
    def etl_worker(self):
        """Enhanced ETL worker with cycle counting"""
        self.logger.info(f"ETL worker started. Processing every {PROCESSING_INTERVAL_SECONDS} seconds.")
        
        while self.is_running:
            try:
                main_etl_cycle()
                self.etl_cycles += 1
                self.logger.info(f"ETL cycle #{self.etl_cycles} complete.")
                time.sleep(PROCESSING_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.errors['etl'] += 1
                self.logger.critical(f"ETL error (#{self.errors['etl']}): {e}")
                time.sleep(PROCESSING_INTERVAL_SECONDS)
    
    def ml_worker(self):
        """ML prediction worker"""
        self.logger.info("ML worker starting...")
        
        # Wait for initial data to accumulate
        self.logger.info(f"Waiting {ML_WARMUP_DELAY} seconds for initial data...")
        time.sleep(ML_WARMUP_DELAY)
        
        try:
            # Initialize ML integration
            self.ml_integration = MLIntegration()
            self.ml_integration.start()
            self.logger.info("ML Integration started successfully")
            
            # Monitor ML performance
            while self.is_running:
                time.sleep(60)  # Check every minute
                
                if self.ml_integration:
                    metrics = self.ml_integration.get_performance_metrics()
                    self.logger.info(f"ML Metrics: {metrics}")
                    
        except Exception as e:
            self.errors['ml'] += 1
            self.logger.error(f"ML worker error: {e}")
    
    def health_monitor(self):
        """Monitor system health and restart failed components"""
        self.logger.info("Health monitor started")
        
        while self.is_running:
            time.sleep(30)  # Check every 30 seconds
            
            # Check thread health
            for name, thread in self.threads.items():
                if not thread.is_alive() and self.is_running:
                    self.logger.warning(f"Thread {name} died. Attempting restart...")
                    
                    # Restart the dead thread
                    if name == 'scraper':
                        self.threads[name] = threading.Thread(
                            target=self.scraper_worker, name="ScraperThread", daemon=True
                        )
                    elif name == 'etl':
                        self.threads[name] = threading.Thread(
                            target=self.etl_worker, name="ETLThread", daemon=True
                        )
                    elif name == 'ml' and ENABLE_ML:
                        self.threads[name] = threading.Thread(
                            target=self.ml_worker, name="MLThread", daemon=True
                        )
                    
                    self.threads[name].start()
                    self.logger.info(f"Thread {name} restarted successfully")
            
            # Log system stats
            uptime = time.time() - self.startup_time
            self.logger.info(
                f"System Health - Uptime: {uptime/3600:.1f}h, "
                f"Scraper cycles: {self.scraper_cycles}, "
                f"ETL cycles: {self.etl_cycles}, "
                f"Errors: {self.errors}"
            )
    
    def start(self):
        """Start all system components"""
        self.is_running = True
        
        # Setup logging
        setup_logging(LOG_FILENAME)
        self.logger.info("=" * 50)
        self.logger.info("--- Enhanced Stock Data System Starting ---")
        self.logger.info("=" * 50)
        
        # Create and start threads
        self.threads['scraper'] = threading.Thread(
            target=self.scraper_worker, name="ScraperThread", daemon=True
        )
        self.threads['etl'] = threading.Thread(
            target=self.etl_worker, name="ETLThread", daemon=True
        )
        
        if ENABLE_ML:
            self.threads['ml'] = threading.Thread(
                target=self.ml_worker, name="MLThread", daemon=True
            )
        
        # Start health monitor
        self.threads['health'] = threading.Thread(
            target=self.health_monitor, name="HealthMonitor", daemon=True
        )
        
        # Start all threads
        for name, thread in self.threads.items():
            thread.start()
            self.logger.info(f"Started {name} thread")
        
        self.logger.info("\nAll components started successfully!")
        self.logger.info("Press Ctrl+C to stop the application.\n")
    
    def stop(self):
        """Gracefully stop all components"""
        self.logger.info("Initiating shutdown...")
        self.is_running = False
        
        # Stop ML integration
        if self.ml_integration:
            self.ml_integration.stop()
        
        # Wait for threads to finish
        for name, thread in self.threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
                self.logger.info(f"Thread {name} stopped")
        
        self.logger.info("=" * 50)
        self.logger.info("--- System Shutdown Complete ---")
        self.logger.info("=" * 50)
    
    def run(self):
        """Main run method"""
        self.start()
        
        try:
            # Keep main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("\nShutdown signal received (Ctrl+C)")
        finally:
            self.stop()


def main():
    """Main entry point"""
    orchestrator = EnhancedSystemOrchestrator()
    orchestrator.run()


if __name__ == "__main__":
    main()
