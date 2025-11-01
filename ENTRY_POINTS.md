# Project Entry Points - Quick Reference

## Primary Entry Points

### 1. Data Collection (Run First)
```bash
python main_enhanced.py
```
**Purpose**: Continuously collect NSE stock data
**Duration**: Run for 60-90 days
**What it does**: Scrapes → Processes → Stores data in database

### 2. Model Training & Prediction (Run After Data Collection)
```bash
python run_predictions.py
```
**Purpose**: Train LSTM, RNN, Prophet models and predict closing prices
**Prerequisites**: At least 60 days of data
**What it does**: Aggregates daily prices → Trains models → Generates predictions

### 3. Visualization (Run After Predictions)
```bash
python visualize_results.py
```
**Purpose**: Generate thesis charts and graphs
**Prerequisites**: Must run `run_predictions.py` first
**What it does**: Creates comparison charts, heatmaps, prediction plots

---

## Execution Order

1. **Setup** (one-time)
   ```bash
   pip install -r requirements.txt
   python -c "from src.utilities import create_database_and_schema; create_database_and_schema()"
   ```

2. **Data Collection** (60-90 days)
   ```bash
   python main_enhanced.py
   ```

3. **Model Training** (after data collection)
   ```bash
   python run_predictions.py --all --epochs 100
   ```

4. **Visualization** (for thesis)
   ```bash
   python visualize_results.py
   ```

---

## Files Deleted Today
- `demo_ml.py` - Old demo script
- `main1.py` - Redundant orchestrator
- `script.py` - Database sync script (not needed)
- `quickstart.py` - Replaced by documentation
- `output.csv` - Old test data
- `nse_stock_model.pkl` - Old model file
- `app.log` - Old log file
- `.DS_Store` - macOS system file
- `ml-requirements.txt` - Merged into requirements.txt
- `IMPLEMENTATION_SUMMARY.md` - Replaced by THESIS_GUIDE.md
- `ML_PREDICTION_GUIDE.md` - Replaced by THESIS_GUIDE.md

---

## Current Project Structure

### Core Files (Use These!)
- `main_enhanced.py` - Data collection orchestrator
- `closing_price_pipeline.py` - ML models (LSTM, RNN, Prophet)
- `run_predictions.py` - Main prediction script
- `visualize_results.py` - Visualization generator

### Documentation
- `README.md` - Project overview
- `PROJECT_EXECUTION_GUIDE.md` - Detailed execution guide
- `THESIS_GUIDE.md` - Complete thesis documentation
- `README_PREDICTIONS.md` - Prediction system reference
- `ENTRY_POINTS.md` - This file

### Supporting Files
- `ml_pipeline.py` - Feature engineering (original system)
- `realtime_ml_integration.py` - Real-time ML integration
- `render_enhanced.py` - API server with predictions
- `render.py` - Basic API server

### Data Collection
- `src/getlivedata.py` - Web scraper
- `src/processlivedata.py` - ETL processor
- `src/utilities.py` - Database utilities

---

## Git Changes Committed

**Commit**: `8f5a54f`
**Branch**: `main`
**Remote**: Pushed to `origin/main`

**Summary**:
- Added 21,405 lines of new code
- Removed 101 lines of old code
- Created 9 new files
- Updated 3 existing files
- Deleted 1 redundant file

---

## What to Do Next

1. **Start data collection**:
   ```bash
   python main_enhanced.py
   ```

2. **Monitor it's working**:
   ```bash
   # Check logs
   tail -f etl_timescale.log

   # Check database
   PGPASSWORD=postgres psql -U postgres -d nse -c "SELECT COUNT(*) FROM stocksdata;"
   ```

3. **Wait 60-90 days** for sufficient data

4. **Then run predictions**:
   ```bash
   python run_predictions.py --all --epochs 100
   python visualize_results.py
   ```

5. **Use results for thesis**

---

**All done! Your project is clean, documented, and ready for thesis work.**
