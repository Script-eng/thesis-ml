"""
Inject ML Predictions into HTML Table
======================================
This module reads the scraped HTML, fetches predictions from the database,
and injects a "Predicted Close" column next to the "Prev Close" column.

Author: Thesis Project
Date: November 2025
"""

import logging
import psycopg2
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz

load_dotenv()

# Database Configuration
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT")
}

HTML_FILE_PATH = os.getenv("HTML_FILE_PATH")
NAIROBI_TZ = pytz.timezone('Africa/Nairobi')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fetch_latest_predictions():
    """
    Fetch the latest predictions for all symbols from the database.

    Returns:
        dict: {symbol: predicted_close, ...}
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Get today's date in Nairobi timezone
        today = datetime.now(NAIROBI_TZ).date()

        # Fetch latest predictions (today's or most recent)
        query = """
        SELECT DISTINCT ON (symbol)
            symbol,
            predicted_close,
            ensemble_pred,
            trading_date
        FROM ml_predictions
        WHERE trading_date >= %s - INTERVAL '7 days'
        ORDER BY symbol, trading_date DESC, prediction_time DESC
        """

        cursor.execute(query, (today,))
        results = cursor.fetchall()

        # Create a dictionary mapping symbol to predicted close
        predictions = {}
        for row in results:
            symbol, predicted_close, ensemble_pred, trading_date = row
            # Use predicted_close if available, otherwise ensemble_pred
            price = predicted_close if predicted_close else ensemble_pred
            predictions[symbol] = {
                'price': float(price) if price else None,
                'date': trading_date
            }

        cursor.close()
        conn.close()

        logger.info(f"Fetched predictions for {len(predictions)} symbols")
        return predictions

    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        return {}


def inject_prediction_column(html_filepath: str, predictions: dict):
    """
    Injects a "Predicted Close" column into the HTML table.

    Args:
        html_filepath: Path to the HTML file
        predictions: Dictionary of predictions {symbol: {'price': float, 'date': date}}

    Returns:
        str: Modified HTML content
    """
    try:
        with open(html_filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        logger.warning(f"HTML file '{html_filepath}' not found.")
        return None

    soup = BeautifulSoup(html_content, "html.parser")

    # Step 1: Modify the table header
    header_table = soup.find("table", id="board_table_head")
    if header_table:
        header_row = header_table.find("tr", class_="row")
        if header_row:
            # Find the "Prev." header (2nd th element)
            headers = header_row.find_all("th")
            if len(headers) >= 2:
                # Insert new header after "Prev." header
                new_header = soup.new_tag("th", class_="btth")
                new_header['title'] = "ML Predicted Today's Close"
                new_header['style'] = "cursor: pointer; background-color: #e8f5e9;"
                new_header.string = "Pred. Close"

                # Insert after the 2nd header (Prev.)
                headers[1].insert_after(new_header)
                logger.info("Added 'Pred. Close' header to table")

    # Step 2: Add column width in the spacing row
    spacing_table = soup.find("table", id="board_table_head")
    if spacing_table:
        spacing_rows = spacing_table.find_all("tr")
        if len(spacing_rows) >= 2:
            spacing_row = spacing_rows[1]  # Second row with widths
            spacing_cells = spacing_row.find_all("td")
            if len(spacing_cells) >= 2:
                # Add new spacing cell
                new_spacing = soup.new_tag("td", class_="n")
                new_spacing['width'] = "70"
                new_spacing.string = "\u00A0"  # Non-breaking space
                spacing_cells[1].insert_after(new_spacing)

    # Step 3: Modify the data table rows
    data_table = soup.find("table", id="board_table")
    if data_table:
        data_rows = data_table.find("tbody").find_all("tr")
        rows_updated = 0

        for row in data_rows:
            if "footer" in row.get("class", []):
                continue

            # Get the symbol from row ID or first cell
            symbol = row.get("id", "").strip()
            if not symbol:
                cells = row.find_all("td")
                if cells:
                    symbol = cells[0].text.strip()

            if symbol and symbol in predictions:
                pred_info = predictions[symbol]
                pred_price = pred_info['price']

                # Create new cell for prediction
                new_cell = soup.new_tag("td", class_="n pred")
                new_cell['style'] = "background-color: #e8f5e9; font-weight: 500;"

                if pred_price:
                    new_cell.string = f"{pred_price:.2f}"
                else:
                    new_cell.string = "--"

                # Insert after the 2nd cell (Prev Close)
                cells = row.find_all("td")
                if len(cells) >= 2:
                    cells[1].insert_after(new_cell)
                    rows_updated += 1
            else:
                # No prediction available, add empty cell
                new_cell = soup.new_tag("td", class_="n pred")
                new_cell['style'] = "background-color: #f5f5f5; color: #999;"
                new_cell.string = "--"

                cells = row.find_all("td")
                if len(cells) >= 2:
                    cells[1].insert_after(new_cell)

        logger.info(f"Updated {rows_updated} rows with predictions")

    return str(soup)


def process_html_with_predictions():
    """
    Main function to fetch predictions and inject them into the HTML.
    This should be called after the HTML is scraped and saved.
    """
    logger.info("Starting prediction injection process...")

    # Fetch predictions
    predictions = fetch_latest_predictions()

    if not predictions:
        logger.warning("No predictions found in database. Skipping injection.")
        return False

    # Inject predictions into HTML
    modified_html = inject_prediction_column(HTML_FILE_PATH, predictions)

    if modified_html:
        # Save the modified HTML back to the file
        with open(HTML_FILE_PATH, "w", encoding="utf-8") as f:
            f.write(modified_html)
        logger.info(f"Successfully injected predictions into {HTML_FILE_PATH}")
        return True
    else:
        logger.error("Failed to inject predictions")
        return False


if __name__ == "__main__":
    process_html_with_predictions()
