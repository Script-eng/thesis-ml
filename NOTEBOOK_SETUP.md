# Jupyter Notebook Setup Guide

## Installation

### 1. Install All Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Install Jupyter
pip install jupyter notebook ipykernel

# Optional: Install Jupyter Lab (more modern interface)
pip install jupyterlab
```

### 2. Start Jupyter

```bash
# Option A: Classic Notebook
jupyter notebook

# Option B: Jupyter Lab (recommended)
jupyter lab
```

Your browser will open automatically at `http://localhost:8888`

## Using the Workflow Notebook

### File: `NSE_Closing_Price_Prediction_Workflow.ipynb`

This notebook contains the complete ML workflow:

1. **Data Collection** - Fetch from database
2. **EDA** - Exploratory analysis with charts
3. **Model Training** - LSTM, RNN, Prophet
4. **Comparison** - Side-by-side metrics
5. **Results** - Export for thesis

### Running the Notebook

1. Open `NSE_Closing_Price_Prediction_Workflow.ipynb`
2. Run cells sequentially (Shift+Enter)
3. First cell will check if dependencies are installed
4. Subsequent cells load data and train models

**Important**: The notebook requires:
- Database running with data
- At least 60 days of data per stock
- All dependencies installed

### Common Issues & Solutions

#### Issue 1: "No module named 'numpy'"

**Solution**:
```bash
pip install numpy pandas matplotlib seaborn tensorflow prophet scikit-learn
```

#### Issue 2: "Database connection failed"

**Solution**:
```bash
# Check PostgreSQL is running
pg_isready

# Start if needed
brew services start postgresql  # macOS
sudo service postgresql start   # Linux
```

#### Issue 3: "Insufficient data"

**Solution**:
- Run data collection for longer
- OR reduce lookback in notebook: change `lookback_days=30` to `lookback_days=15`

#### Issue 4: Kernel dies during training

**Solution**:
- Reduce epochs: Change `epochs=100` to `epochs=20`
- Reduce batch size: Change `batch_size=32` to `batch_size=16`
- Process fewer stocks at once

## Workflow: Scripts vs Notebook

### Use Scripts For:

**Production & Automation**
```bash
# Train all models on all stocks
python run_predictions.py --all --epochs 100

# Generate all visualizations
python visualize_results.py
```

### Use Notebook For:

**Analysis & Thesis Work**
- Detailed EDA
- Experimenting with hyperparameters
- Creating custom visualizations
- Documenting methodology
- Showing step-by-step process to reviewers

## Tips for Thesis

### 1. Save Figures

In notebook cells:
```python
plt.savefig('thesis_figures/stock_trends.png', dpi=300, bbox_inches='tight')
```

### 2. Export Results

```python
# Save metrics table
results_df.to_csv('thesis_results/model_comparison.csv', index=False)

# Save as LaTeX table
print(results_df.to_latex(index=False))
```

### 3. Include Code Snippets

You can copy code from notebook to thesis:
- Format as code blocks
- Explain key algorithms
- Show model architectures

### 4. Convert Notebook to PDF

```bash
# Convert to HTML first
jupyter nbconvert --to html NSE_Closing_Price_Prediction_Workflow.ipynb

# Or to PDF (requires LaTeX)
jupyter nbconvert --to pdf NSE_Closing_Price_Prediction_Workflow.ipynb
```

## Recommended Workflow for Thesis

### Week 1-8: Data Collection
```bash
# Keep running
python main_enhanced.py
```

### Week 9: Analysis Phase

**Day 1-2**: Run automated predictions
```bash
python run_predictions.py --all --epochs 100
```

**Day 3-5**: Deep dive in notebook
```bash
jupyter lab
# Open NSE_Closing_Price_Prediction_Workflow.ipynb
# Run all cells
# Experiment with different stocks
# Create custom visualizations
```

**Day 6-7**: Generate thesis materials
```bash
python visualize_results.py
# Export notebook results
# Convert notebook to PDF
```

### Week 10: Writing

Use:
- Charts from `results/` directory
- Tables from CSV exports
- Code snippets from notebook
- Methodology documented in notebook markdown cells

## Shortcuts (Jupyter)

| Action | Shortcut |
|--------|----------|
| Run cell | Shift + Enter |
| Insert cell below | B |
| Insert cell above | A |
| Delete cell | DD |
| Change to markdown | M |
| Change to code | Y |
| Save notebook | Cmd/Ctrl + S |
| Show shortcuts | H |

## Best Practices

1. **Run cells in order** - Don't skip around
2. **Restart kernel regularly** - Kernel → Restart & Clear Output
3. **Save often** - Auto-save isn't always reliable
4. **Comment your code** - Future you will thank you
5. **Use markdown cells** - Explain what you're doing
6. **Clear outputs before commit** - Keep git diffs clean

## File Organization

```
thesis_work/
├── notebooks/
│   └── NSE_Closing_Price_Prediction_Workflow.ipynb
├── thesis_results/
│   ├── model_comparison.csv
│   ├── average_performance.csv
│   └── ...
├── thesis_figures/
│   ├── stock_trends.png
│   ├── model_comparison.png
│   └── ...
└── models/
    ├── SCOM_lstm_model.h5
    └── ...
```

## Next Steps

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install jupyter
   ```

2. **Start notebook**:
   ```bash
   jupyter lab
   ```

3. **Open workflow notebook**:
   - `NSE_Closing_Price_Prediction_Workflow.ipynb`

4. **Run cells sequentially**:
   - Check for errors
   - Adjust parameters if needed

5. **Export results**:
   - Save figures
   - Export tables
   - Convert to PDF

---

**You now have both automated scripts AND interactive notebooks for your thesis work!** 🎓
