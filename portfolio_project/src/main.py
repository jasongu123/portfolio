"""
Portfolio Optimization Project
==============================
Comparing ERI, GARCH-EVT-CVaR, and classical approaches on NASDAQ stocks.

Based on:
- Mainik, Mitov & Rüschendorf (2015): ERI optimization
- Bedoui et al. (2023): GARCH-EVT-Vine Copula-CVaR
- Damjanovic et al. (2025): Diversification metrics

Usage:
    python main.py --download    # Download real data from Yahoo Finance
    python main.py --synthetic   # Use synthetic data (for testing)
    python main.py --backtest    # Run full backtest
    python main.py --visualize   # Create visualizations
    python main.py --all         # Run everything
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def download_data():
    """Download real data from Yahoo Finance."""
    print("\n" + "="*70)
    print("DOWNLOADING DATA FROM YAHOO FINANCE")
    print("="*70)
    
    from data_module import prepare_dataset
    prepare_dataset()


def generate_synthetic():
    """Generate synthetic data for testing."""
    print("\n" + "="*70)
    print("GENERATING SYNTHETIC DATA")
    print("="*70)
    
    from synthetic_data import prepare_synthetic_dataset
    prepare_synthetic_dataset()


def run_backtest():
    """Run the full backtest."""
    print("\n" + "="*70)
    print("RUNNING BACKTEST")
    print("="*70)
    
    from backtester import run_full_backtest
    run_full_backtest()


def create_visualizations():
    """Create all visualizations."""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    import pandas as pd
    from visualization import create_all_visualizations
    
    try:
        portfolio_values = pd.read_csv("data/portfolio_values.csv", 
                                        index_col=0, parse_dates=True)
        perf_metrics = pd.read_csv("data/performance_metrics.csv", index_col=0)
        div_metrics = pd.read_csv("data/diversification_metrics.csv", index_col=0)
        
        create_all_visualizations(portfolio_values, perf_metrics, div_metrics,
                                   output_dir="results")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run backtest first with: python main.py --backtest")


def run_all():
    """Run the complete pipeline."""
    print("\n" + "="*70)
    print("RUNNING COMPLETE PIPELINE")
    print("="*70)
    
    # Check if real data exists, otherwise use synthetic
    if not os.path.exists("data/returns.csv"):
        print("\nNo data found. Generating synthetic data...")
        print("(For real data, run: python main.py --download)")
        generate_synthetic()
    
    run_backtest()
    create_visualizations()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  - data/portfolio_values.csv")
    print("  - data/performance_metrics.csv")
    print("  - data/diversification_metrics.csv")
    print("  - results/*.png (visualizations)")


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio Optimization Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--download', action='store_true',
                        help='Download real data from Yahoo Finance')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate synthetic data for testing')
    parser.add_argument('--backtest', action='store_true',
                        help='Run full backtest')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualizations')
    parser.add_argument('--all', action='store_true',
                        help='Run complete pipeline')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    if args.download:
        download_data()
    elif args.synthetic:
        generate_synthetic()
    elif args.backtest:
        run_backtest()
    elif args.visualize:
        create_visualizations()
    elif args.all:
        run_all()
    else:
        # Default: run all
        run_all()


if __name__ == "__main__":
    main()
