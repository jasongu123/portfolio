"""
Visualization Module
====================
Creates plots for portfolio comparison and analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_cumulative_returns(portfolio_values: pd.DataFrame,
                            title: str = "Cumulative Portfolio Returns",
                            save_path: Optional[str] = None):
    """
    Plot cumulative returns for all strategies.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for i, col in enumerate(portfolio_values.columns):
        ax.plot(portfolio_values.index, portfolio_values[col], 
                label=col, linewidth=2, color=colors[i % len(colors)])
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drawdowns(portfolio_values: pd.DataFrame,
                   title: str = "Portfolio Drawdowns",
                   save_path: Optional[str] = None):
    """
    Plot drawdowns for all strategies.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for i, col in enumerate(portfolio_values.columns):
        values = portfolio_values[col]
        rolling_max = values.expanding().max()
        drawdown = (values / rolling_max - 1) * 100  # Percentage
        
        ax.fill_between(drawdown.index, drawdown.values, 0,
                        alpha=0.3, color=colors[i % len(colors)])
        ax.plot(drawdown.index, drawdown.values, 
                label=col, linewidth=1.5, color=colors[i % len(colors)])
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_rolling_sharpe(portfolio_values: pd.DataFrame,
                        window: int = 252,
                        title: str = "Rolling Sharpe Ratio (1-Year)",
                        save_path: Optional[str] = None):
    """
    Plot rolling Sharpe ratio.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    
    for i, col in enumerate(portfolio_values.columns):
        returns = portfolio_values[col].pct_change().dropna()
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std
        
        ax.plot(rolling_sharpe.index, rolling_sharpe.values,
                label=col, linewidth=1.5, color=colors[i % len(colors)])
    
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_performance_comparison(metrics: pd.DataFrame,
                                 save_path: Optional[str] = None):
    """
    Create bar chart comparing key performance metrics.
    """
    set_plot_style()
    
    # Select key metrics
    key_metrics = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    titles = ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown']
    
    for idx, (metric, title) in enumerate(zip(key_metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = metrics[metric].values
        
        if metric == 'max_drawdown':
            values = values * 100  # Convert to percentage
            ylabel = 'Drawdown (%)'
        elif metric in ['annual_return', 'annual_volatility']:
            values = values * 100
            ylabel = 'Percentage (%)'
        else:
            ylabel = 'Ratio'
        
        bars = ax.bar(metrics.index, values, color=colors[:len(metrics)])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Performance Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_diversification_comparison(div_metrics: pd.DataFrame,
                                     save_path: Optional[str] = None):
    """
    Create bar chart comparing diversification metrics.
    """
    set_plot_style()
    
    metrics_to_plot = ['avg_ENC', 'avg_ENB', 'avg_EDP', 'avg_div_ratio']
    titles = ['Effective Number of\nConstituents (ENC)', 
              'Effective Number\nof Bets (ENB)',
              'Exploited Diversification\nPotential (EDP)',
              'Diversification\nRatio']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx // 2, idx % 2]
        
        if metric in div_metrics.columns:
            values = div_metrics[metric].values
            bars = ax.bar(div_metrics.index, values, color=colors[:len(div_metrics)])
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Diversification Metrics Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_weight_distribution(weights: np.ndarray,
                              tickers: list,
                              title: str = "Portfolio Weight Distribution",
                              save_path: Optional[str] = None):
    """
    Plot portfolio weight distribution.
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by weight
    sorted_idx = np.argsort(weights)[::-1]
    sorted_weights = weights[sorted_idx]
    sorted_tickers = [tickers[i] for i in sorted_idx]
    
    # Only show non-zero weights
    mask = sorted_weights > 0.001
    sorted_weights = sorted_weights[mask]
    sorted_tickers = [t for t, m in zip(sorted_tickers, mask) if m]
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(sorted_weights)))
    
    ax.bar(range(len(sorted_weights)), sorted_weights * 100, color=colors)
    ax.set_xticks(range(len(sorted_weights)))
    ax.set_xticklabels(sorted_tickers, rotation=90)
    ax.set_xlabel('Asset')
    ax.set_ylabel('Weight (%)')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_all_visualizations(portfolio_values: pd.DataFrame,
                               perf_metrics: pd.DataFrame,
                               div_metrics: pd.DataFrame,
                               output_dir: str = "results"):
    """
    Create and save all visualizations.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    # Cumulative returns
    plot_cumulative_returns(
        portfolio_values,
        save_path=f"{output_dir}/cumulative_returns.png"
    )
    print("  Saved: cumulative_returns.png")
    
    # Drawdowns
    plot_drawdowns(
        portfolio_values,
        save_path=f"{output_dir}/drawdowns.png"
    )
    print("  Saved: drawdowns.png")
    
    # Rolling Sharpe
    plot_rolling_sharpe(
        portfolio_values,
        save_path=f"{output_dir}/rolling_sharpe.png"
    )
    print("  Saved: rolling_sharpe.png")
    
    # Performance comparison
    plot_performance_comparison(
        perf_metrics,
        save_path=f"{output_dir}/performance_comparison.png"
    )
    print("  Saved: performance_comparison.png")
    
    # Diversification comparison
    if len(div_metrics) > 0:
        plot_diversification_comparison(
            div_metrics,
            save_path=f"{output_dir}/diversification_comparison.png"
        )
        print("  Saved: diversification_comparison.png")
    
    plt.close('all')
    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Test with saved results
    try:
        portfolio_values = pd.read_csv("data/portfolio_values.csv", 
                                        index_col=0, parse_dates=True)
        perf_metrics = pd.read_csv("data/performance_metrics.csv", index_col=0)
        div_metrics = pd.read_csv("data/diversification_metrics.csv", index_col=0)
        
        create_all_visualizations(portfolio_values, perf_metrics, div_metrics)
    except FileNotFoundError:
        print("Run backtester.py first to generate results.")
