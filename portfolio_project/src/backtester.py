"""
Backtesting Framework
=====================
Monthly rebalancing backtest comparing:
1. Equal Weight (EW)
2. Minimum Variance (MV)
3. ERI Optimization (Mainik et al.)
4. GARCH-EVT-CVaR (Bedoui et al.)

With diversification analysis from Damjanovic et al.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from benchmark_portfolios import EqualWeightPortfolio, MinimumVariancePortfolio
from eri_optimizer import ERIPortfolio
from garch_evt_cvar import GARCHEVTCVaRPortfolio
from diversification_metrics import DiversificationMetrics, compute_portfolio_stats


class Backtester:
    """
    Portfolio backtesting framework with monthly rebalancing.
    """
    
    def __init__(self,
                 returns: pd.DataFrame,
                 backtest_start: str = "2015-01-01",
                 rebalance_freq: str = "Q",
                 min_history_days: int = 500,
                 max_weight: float = 0.20,
                 transaction_cost: float = 0.001):
        """
        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns with DatetimeIndex
        backtest_start : str
            Start date for backtest
        rebalance_freq : str
            Rebalancing frequency ('M' for monthly, 'Q' for quarterly)
        min_history_days : int
            Minimum history required before first optimization
        max_weight : float
            Maximum weight per asset
        transaction_cost : float
            Round-trip transaction cost as fraction
        """
        self.returns = returns
        self.backtest_start = pd.to_datetime(backtest_start)
        self.rebalance_freq = rebalance_freq
        self.min_history_days = min_history_days
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost
        
        # Results storage
        self.portfolio_values = {}
        self.weights_history = {}
        self.rebalance_dates = []
        self.diversification_history = {}
    
    def _get_rebalance_dates(self) -> List[pd.Timestamp]:
        """Get list of rebalancing dates."""
        # Filter to backtest period
        mask = self.returns.index >= self.backtest_start
        backtest_returns = self.returns[mask]
        
        if self.rebalance_freq == 'M':
            # Monthly: last trading day of each month
            rebalance_dates = backtest_returns.groupby(
                pd.Grouper(freq='M')
            ).apply(lambda x: x.index[-1] if len(x) > 0 else None)
        elif self.rebalance_freq == 'Q':
            # Quarterly
            rebalance_dates = backtest_returns.groupby(
                pd.Grouper(freq='Q')
            ).apply(lambda x: x.index[-1] if len(x) > 0 else None)
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_freq}")
        
        return rebalance_dates.dropna().tolist()
    
    def _get_estimation_data(self, date: pd.Timestamp) -> pd.DataFrame:
        """Get historical data for estimation (expanding window)."""
        mask = self.returns.index < date
        return self.returns[mask]
    
    def run_backtest(self, 
                     strategies: Dict[str, object],
                     verbose: bool = True) -> pd.DataFrame:
        """
        Run backtest for multiple strategies.
        
        Parameters
        ----------
        strategies : dict
            Dictionary of strategy name -> optimizer object
        verbose : bool
            Print progress updates
            
        Returns
        -------
        pd.DataFrame
            Portfolio values over time
        """
        self.rebalance_dates = self._get_rebalance_dates()
        n_assets = self.returns.shape[1]
        
        # Initialize
        for name in strategies:
            self.portfolio_values[name] = []
            self.weights_history[name] = []
            self.diversification_history[name] = []
        
        # Current weights (start with equal weight)
        current_weights = {name: np.ones(n_assets) / n_assets for name in strategies}
        
        # Current portfolio values
        portfolio_value = {name: 1.0 for name in strategies}
        
        if verbose:
            print(f"Running backtest from {self.backtest_start.date()}")
            print(f"Rebalancing dates: {len(self.rebalance_dates)}")
            print(f"Strategies: {list(strategies.keys())}")
            print("-" * 60)
        
        # Get all trading days in backtest period
        mask = self.returns.index >= self.backtest_start
        trading_days = self.returns.index[mask]
        
        rebalance_idx = 0
        div_metrics = DiversificationMetrics()
        total_rebalances = len(self.rebalance_dates)
        cvar_ratios_concentrated = []
        cvar_ratios_mv = []
        cvar_values_equal = []
        cvar_values_concentrated = []
        cvar_values_mv = []
        
        for i, date in enumerate(trading_days):
            # Check if rebalancing day
            if rebalance_idx < len(self.rebalance_dates) and \
               date >= self.rebalance_dates[rebalance_idx]:
                
                # Get estimation data
                est_data = self._get_estimation_data(date)
                
                if len(est_data) >= self.min_history_days:
                    if verbose:
                        print(f"Rebalancing on {date.date()} "
                            f"(history: {len(est_data)} days)")
                    
                    # Optimize each strategy
                    for name, optimizer in strategies.items():
                        try:
                            import time
                            start_time = time.time()
                            # Create fresh optimizer instance
                            if name == 'EW':
                                opt = EqualWeightPortfolio()
                            elif name == 'MV':
                                opt = MinimumVariancePortfolio(max_weight=self.max_weight)
                            elif name == 'ERI':
                                opt = ERIPortfolio(
                                    tail_fraction=0.10,
                                    max_weight=self.max_weight
                                )
                            elif name == 'GARCH-EVT-CVaR':
                                opt = GARCHEVTCVaRPortfolio(
                                    cvar_alpha=0.95,
                                    n_simulations=1000,
                                    max_weight=self.max_weight,
                                    start_from_mv=False
                                )
                            elif name == 'GARCH-EVT-CVaR-MV':
                                opt = GARCHEVTCVaRPortfolio(
                                    cvar_alpha=0.95,
                                    n_simulations=1000,
                                    max_weight=self.max_weight,
                                    start_from_mv=True
                                )
                            else:
                                opt = optimizer
                            
                            new_weights = opt.fit(est_data)
                            if name == 'GARCH-EVT-CVaR' and hasattr(opt, 'cvar_diagnostics'):
                                cvar_ratios_concentrated.append(opt.cvar_diagnostics['ratio_concentrated'])
                                cvar_ratios_mv.append(opt.cvar_diagnostics['ratio_mv'])
                                cvar_values_equal.append(opt.cvar_diagnostics['cvar_equal'])
                                cvar_values_concentrated.append(opt.cvar_diagnostics['cvar_concentrated'])
                                cvar_values_mv.append(opt.cvar_diagnostics['cvar_mv'])
                            elapsed = time.time() - start_time
                            if verbose:
                                print(f"    {name}: {elapsed:.1f}s, non-zero weights: {np.sum(new_weights > 0.001)}")
                                                
                            # Transaction costs
                            turnover = np.sum(np.abs(new_weights - current_weights[name]))
                            tc = turnover * self.transaction_cost / 2
                            portfolio_value[name] *= (1 - tc)
                            
                            current_weights[name] = new_weights
                            
                            # Compute diversification metrics
                            metrics = div_metrics.compute_all(
                                new_weights, 
                                cov_matrix=est_data.cov().values
                            )
                            self.diversification_history[name].append({
                                'date': date,
                                **metrics
                            })
                            
                        except Exception as e:
                            if verbose:
                                print(f"  Warning: {name} optimization failed: {e}")
                            # Keep previous weights
                
                rebalance_idx += 1
            
            # Daily portfolio update
            daily_returns = self.returns.loc[date].values
            
            for name in strategies:
                # Portfolio return
                port_return = current_weights[name] @ daily_returns
                portfolio_value[name] *= (1 + port_return)
                
                self.portfolio_values[name].append({
                    'date': date,
                    'value': portfolio_value[name]
                })
                
                self.weights_history[name].append({
                    'date': date,
                    'weights': current_weights[name].copy()
                })
        
        # Convert to DataFrames
        results = {}
        for name in strategies:
            df = pd.DataFrame(self.portfolio_values[name])
            df.set_index('date', inplace=True)
            results[name] = df['value']
        if len(cvar_ratios_concentrated) > 0:
            print("\n" + "="*60)
            print("CVaR RATIO AVERAGES (across all rebalancing periods)")
            print("="*60)
            print(f"Equal Weight:              1.00 (baseline)")
            print(f"Concentrated (10 stocks):  {np.mean(cvar_ratios_concentrated):.2f}")
            print(f"MV Optimal Weights:        {np.mean(cvar_ratios_mv):.2f}")
            print("\nCVaR VALUES AVERAGES")
            print("="*60)
            print(f"Equal Weight CVaR:         {np.mean(cvar_values_equal):.6f}")
            print(f"Concentrated CVaR:         {np.mean(cvar_values_concentrated):.6f}")
            print(f"MV Optimal Weights CVaR:   {np.mean(cvar_values_mv):.6f}")
            print("="*60)
            
        
        return pd.DataFrame(results)
    
    def compute_performance_metrics(self, 
                                     portfolio_values: pd.DataFrame) -> pd.DataFrame:
        """
        Compute performance metrics for all strategies.
        """
        metrics = []
        
        for name in portfolio_values.columns:
            values = portfolio_values[name]
            returns = values.pct_change().dropna()
            
            # Annualized return
            total_return = values.iloc[-1] / values.iloc[0] - 1
            n_years = len(returns) / 252
            ann_return = (1 + total_return)**(1/n_years) - 1
            
            # Annualized volatility
            ann_vol = returns.std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            sharpe = (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0
            
            # Maximum drawdown
            rolling_max = values.expanding().max()
            drawdowns = values / rolling_max - 1
            max_dd = drawdowns.min()
            
            # Calmar ratio
            calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
            
            # Sortino ratio
            neg_returns = returns[returns < 0]
            downside_vol = neg_returns.std() * np.sqrt(252)
            sortino = (ann_return - 0.02) / downside_vol if downside_vol > 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            metrics.append({
                'strategy': name,
                'total_return': total_return,
                'annual_return': ann_return,
                'annual_volatility': ann_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_dd,
                'calmar_ratio': calmar,
                'var_95_daily': var_95,
                'cvar_95_daily': cvar_95,
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            })
        
        return pd.DataFrame(metrics).set_index('strategy')
    
    def get_diversification_summary(self) -> pd.DataFrame:
        """
        Get average diversification metrics for each strategy.
        """
        summaries = []
        
        for name, history in self.diversification_history.items():
            if len(history) == 0:
                continue
            
            df = pd.DataFrame(history)
            
            # Average metrics
            summary = {
                'strategy': name,
                'avg_ENC': df['ENC'].mean(),
                'avg_ENB': df['ENB'].mean(),
                'avg_EDP': df['EDP'].mean(),
                'avg_div_ratio': df['diversification_ratio'].mean(),
                'avg_HHI': df['HHI'].mean()
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries).set_index('strategy')


def run_full_backtest():
    """
    Run complete backtest with all strategies.
    """
    print("="*70)
    print("PORTFOLIO OPTIMIZATION BACKTEST")
    print("="*70)
    
    # Load data
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    # Use subset for faster testing (adjust for full run)
    
    print(f"\nData: {returns.shape[0]} days, {returns.shape[1]} assets")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Initialize backtester
    backtester = Backtester(
        returns=returns,
        backtest_start="2015-01-01",
        rebalance_freq="Q",
        min_history_days=500,
        max_weight=0.20,
        transaction_cost=0.001
    )
    
    # Define strategies
    strategies = {
        'EW': EqualWeightPortfolio(),
        'MV': MinimumVariancePortfolio(max_weight=0.20),
        'ERI': ERIPortfolio(tail_fraction=0.10, max_weight=0.20),
        'GARCH-EVT-CVaR': GARCHEVTCVaRPortfolio(
            cvar_alpha=0.95,
            n_simulations=500,
            max_weight=0.20,
            start_from_mv=False
        ),
        'GARCH-EVT-CVaR-MV': GARCHEVTCVaRPortfolio(
        cvar_alpha=0.95,
        n_simulations=1000,
        max_weight=0.20,
        start_from_mv=True  # MV weights start
    )
    }
    
    # Run backtest
    print("\n" + "-"*70)
    portfolio_values = backtester.run_backtest(strategies, verbose=True)
    
    # Performance metrics
    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)
    
    perf_metrics = backtester.compute_performance_metrics(portfolio_values)
    print("\n" + perf_metrics.round(4).to_string())
    
    # Diversification metrics
    print("\n" + "="*70)
    print("DIVERSIFICATION METRICS (Averages)")
    print("="*70)
    
    div_metrics = backtester.get_diversification_summary()
    print("\n" + div_metrics.round(4).to_string())
    
    # Save results
    portfolio_values.to_csv("data/portfolio_values.csv")
    perf_metrics.to_csv("data/performance_metrics.csv")
    div_metrics.to_csv("data/diversification_metrics.csv")
    
    print("\n" + "-"*70)
    print("Results saved to data/")
    
    return portfolio_values, perf_metrics, div_metrics


if __name__ == "__main__":
    run_full_backtest()
