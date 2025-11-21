# Notebook Execution Guide

## Quick Reference

**Total Cells:** 33 (17 markdown + 16 code)
**Expected Runtime:** ~10-15 minutes (depending on LSTM training)
**Figures Generated:** 7 (Figure1.png through Figure7.png)

---

## Execution Order

Simply execute cells **sequentially from top to bottom** (Cell 0 → Cell 32).

### Section Breakdown:

1. **Setup (Cells 0-2)**
   - Title & imports

2. **Data Loading (Cells 3-6)**
   - Configuration & load CSV

3. **Data Cleaning (Cells 7-10)**
   - Quality checks & deduplication

4. **Exploratory Analysis (Cells 11-18)**
   - Data availability analysis
   - **→ Figure 1** (Cell 14): Snapshot availability
   - **→ Figure 2** (Cell 16): Intraday patterns
   - Stock selection

5. **LSTM Model (Cells 19-22)**
   - Training on intraday sequences
   - **→ Figure 3** (Cell 22): Training history
   - **→ Figure 4** (Cell 22): Predictions

6. **EOD Predictions (Cells 23-26)**
   - Extract end-of-day forecasts
   - **→ Figure 5** (Cell 26): EOD predictions
   - **→ Figure 6** (Cell 26): Prediction errors

7. **Prophet Baseline (Cells 27-28)**
   - Train on daily aggregated data

8. **Model Comparison (Cells 29-30)**
   - **→ Figure 7** (Cell 30): LSTM vs Prophet

9. **Final Results (Cells 31-32)**
   - Comprehensive summary & CSV exports

---

## After Execution

### Save Figures
Right-click on each figure and save as:
- Cell 14 → `Figure1.png`
- Cell 16 → `Figure2.png`
- Cell 22 → `Figure3.png` (training history)
- Cell 22 → `Figure4.png` (predictions)
- Cell 26 → `Figure5.png` (EOD predictions)
- Cell 26 → `Figure6.png` (errors)
- Cell 30 → `Figure7.png` (comparison)

Place all figures in `figures/` directory for LaTeX inclusion.

### Check Results
Two CSV files will be created in `results/`:
- `eod_predictions_comparison.csv` - Both models' predictions
- `model_metrics_comparison.csv` - Performance metrics

---

## Expected Performance

### LSTM (Intraday Approach)
- Training sequences: ~48,775
- R²: 0.65-0.75
- RMSE: 0.8-1.2 KES
- MAPE: 3-4%
- Directional Accuracy: 65-70%

### Prophet (Daily Approach)
- Training days: 60
- R²: 0.35-0.55
- RMSE: 1.0-1.4 KES
- MAPE: 3.5-4.5%
- Directional Accuracy: 55-65%

### Improvement
- Data utilization: 812× more data points
- RMSE improvement: 15-30% better
- R² improvement: +0.20 to +0.30

---

## Troubleshooting

**If you encounter errors:**
1. Ensure CSV file exists at `~/Desktop/stock_data_2025-11-19.csv`
2. Check that all required libraries are installed (pandas, numpy, matplotlib, tensorflow, prophet)
3. Verify Python environment has sufficient memory for LSTM training

**Cell-specific notes:**
- Cell 20 (LSTM training): Takes longest (~5-8 minutes)
- Cell 28 (Prophet): May show warnings about seasonality - this is normal
- Cell 32: Creates `results/` directory if it doesn't exist

---

## For Thesis Inclusion

1. Export all 7 figures to `figures/` directory
2. Use naming convention: `Figure1.png` through `Figure7.png`
3. In LaTeX, include with:
   ```latex
   \includegraphics[width=0.8\textwidth]{figures/Figure1.png}
   ```
4. Results tables available in generated CSV files
