"""
Visualization Script for Thesis Results
========================================
Creates charts and graphs for thesis presentation.

Usage:
    python visualize_results.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from closing_price_pipeline import DailyDataAggregator, LSTMPredictor
from dotenv import load_dotenv

load_dotenv()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def plot_model_comparison():
    """Plot comparison of all models across metrics."""
    # Load results
    if not os.path.exists('results/model_comparison.csv'):
        print("No results found. Run 'python run_predictions.py' first.")
        return

    df = pd.read_csv('results/model_comparison.csv')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    titles = ['Root Mean Squared Error (Lower is Better)',
              'Mean Absolute Error (Lower is Better)',
              'Mean Absolute Percentage Error (Lower is Better)',
              'R-Squared Score (Higher is Better)']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        # Group by model and calculate mean
        model_perf = df.groupby('model')[metric].mean().sort_values()

        # Create bar plot
        bars = ax.bar(range(len(model_perf)), model_perf.values,
                      color=['#3498db', '#e74c3c', '#2ecc71'])

        # Customize
        ax.set_xticks(range(len(model_perf)))
        ax.set_xticklabels(model_perf.index, fontweight='bold')
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel(metric)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, model_perf.values)):
            ax.text(i, val, f'{val:.3f}',
                   ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/model_comparison.png")
    plt.show()


def plot_predictions_vs_actual(symbol='SCOM'):
    """Plot predictions vs actual prices for a specific stock."""
    # Load results
    df_results = pd.read_csv('results/model_comparison.csv')
    symbol_results = df_results[df_results['symbol'] == symbol]

    if symbol_results.empty:
        print(f"No results for {symbol}")
        return

    # Get actual closing prices
    DB_CONFIG = {
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    }

    aggregator = DailyDataAggregator(DB_CONFIG)
    df_all = aggregator.get_daily_closing_prices(days_back=90)
    df_stock = df_all[df_all['symbol'] == symbol].sort_values('trading_date')

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot historical prices
    ax.plot(df_stock['trading_date'], df_stock['close_price'],
           label='Actual Price', linewidth=2, color='black', marker='o', markersize=3)

    # Mark current price
    current_price = df_stock['close_price'].iloc[-1]
    current_date = df_stock['trading_date'].iloc[-1]

    # Plot predictions (next day)
    next_date = current_date + pd.Timedelta(days=1)

    colors = {'LSTM': '#3498db', 'RNN': '#e74c3c', 'Prophet': '#2ecc71'}

    for _, row in symbol_results.iterrows():
        model_name = row['model']
        prediction = row['prediction']

        ax.plot([current_date, next_date], [current_price, prediction],
               '--', label=f'{model_name} Prediction',
               color=colors.get(model_name, 'gray'),
               linewidth=2, marker='o', markersize=8)

        # Add annotation
        ax.annotate(f'{prediction:.2f}',
                   xy=(next_date, prediction),
                   xytext=(10, 0), textcoords='offset points',
                   fontweight='bold', color=colors.get(model_name, 'gray'))

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Closing Price (KES)', fontsize=12, fontweight='bold')
    ax.set_title(f'{symbol} - Price Predictions', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/{symbol}_predictions.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: results/{symbol}_predictions.png")
    plt.show()


def plot_error_distribution():
    """Plot error distribution across all models."""
    df = pd.read_csv('results/model_comparison.csv')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Error Distribution by Model', fontsize=16, fontweight='bold')

    models = df['model'].unique()
    colors = {'LSTM': '#3498db', 'RNN': '#e74c3c', 'Prophet': '#2ecc71'}

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = df[df['model'] == model]

        # Box plot
        bp = ax.boxplot([model_data['RMSE'], model_data['MAE'], model_data['MAPE']],
                        labels=['RMSE', 'MAE', 'MAPE'],
                        patch_artist=True)

        # Color
        for patch in bp['boxes']:
            patch.set_facecolor(colors.get(model, 'gray'))

        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_ylabel('Error Value')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/error_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/error_distribution.png")
    plt.show()


def create_summary_table():
    """Create summary statistics table."""
    df = pd.read_csv('results/model_comparison.csv')

    summary = df.groupby('model').agg({
        'RMSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'MAPE': ['mean', 'std', 'min', 'max'],
        'R2': ['mean', 'std', 'min', 'max']
    }).round(3)

    print("\n" + "="*80)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    print(summary.to_string())
    print("="*80)

    # Save to CSV
    summary.to_csv('results/summary_statistics.csv')
    print("\n✅ Saved: results/summary_statistics.csv")


def plot_stock_wise_performance():
    """Plot performance for each stock across models."""
    df = pd.read_csv('results/model_comparison.csv')

    # Pivot for heatmap
    pivot_rmse = df.pivot(index='symbol', columns='model', values='RMSE')

    fig, ax = plt.subplots(figsize=(10, len(pivot_rmse) * 0.5))

    sns.heatmap(pivot_rmse, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'RMSE'}, ax=ax)

    ax.set_title('RMSE by Stock and Model (Lower is Better)',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stock Symbol', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/stock_wise_performance.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: results/stock_wise_performance.png")
    plt.show()


def generate_all_visualizations():
    """Generate all visualizations for thesis."""
    print("\n" + "="*60)
    print("GENERATING THESIS VISUALIZATIONS")
    print("="*60 + "\n")

    os.makedirs('results', exist_ok=True)

    if not os.path.exists('results/model_comparison.csv'):
        print("❌ No results found. Please run:")
        print("   python run_predictions.py")
        return

    print("1. Creating model comparison chart...")
    plot_model_comparison()

    print("\n2. Creating error distribution chart...")
    plot_error_distribution()

    print("\n3. Creating summary statistics table...")
    create_summary_table()

    print("\n4. Creating stock-wise performance heatmap...")
    plot_stock_wise_performance()

    # Create prediction plots for top 3 stocks
    df = pd.read_csv('results/model_comparison.csv')
    top_symbols = df.groupby('symbol')['R2'].mean().nlargest(3).index

    print(f"\n5. Creating prediction plots for top stocks: {list(top_symbols)}")
    for symbol in top_symbols:
        print(f"   - {symbol}")
        plot_predictions_vs_actual(symbol)

    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'results/' directory:")
    print("  - model_comparison.png")
    print("  - error_distribution.png")
    print("  - summary_statistics.csv")
    print("  - stock_wise_performance.png")
    print("  - [SYMBOL]_predictions.png (for each stock)")
    print("\nUse these for your thesis presentation and report!")


if __name__ == "__main__":
    generate_all_visualizations()
