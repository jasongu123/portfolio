"""
Benchmark Portfolio Optimizers
==============================
Minimum Variance (MV) and Equal Weight (EW) portfolios.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional


class EqualWeightPortfolio:
    """
    Equal weight (1/N) portfolio.
    """
    
    def __init__(self):
        self.weights = None
        self.n_assets = None
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute equal weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns (not actually used, but kept for API consistency)
            
        Returns
        -------
        np.ndarray
            Portfolio weights
        """
        self.n_assets = returns.shape[1]
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights


class MinimumVariancePortfolio:
    """
    Minimum variance portfolio using quadratic programming.
    """
    
    def __init__(self, 
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 allow_short: bool = False):
        """
        Parameters
        ----------
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        allow_short : bool
            Whether to allow short selling
        """
        self.min_weight = min_weight if not allow_short else -1.0
        self.max_weight = max_weight
        self.weights = None
        self.cov_matrix = None
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute minimum variance portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
            
        Returns
        -------
        np.ndarray
            Portfolio weights
        """
        n_assets = returns.shape[1]
        
        # Estimate covariance matrix
        self.cov_matrix = returns.cov().values
        
        # Objective: minimize portfolio variance w' Σ w
        def portfolio_variance(w):
            return w @ self.cov_matrix @ w
        
        # Gradient of objective
        def variance_gradient(w):
            return 2 * self.cov_matrix @ w
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess: equal weight
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            jac=variance_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            # Fall back to equal weight if optimization fails
            self.weights = np.ones(n_assets) / n_assets
        else:
            self.weights = result.x
            # Clean up small weights
            self.weights[np.abs(self.weights) < 1e-6] = 0
            self.weights = self.weights / self.weights.sum()
        
        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_portfolio_stats(self, returns: pd.DataFrame) -> dict:
        """
        Compute portfolio statistics.
        """
        if self.weights is None:
            raise ValueError("Must call fit() first")
        
        mean_returns = returns.mean().values
        
        port_return = self.weights @ mean_returns * 252  # Annualized
        port_vol = np.sqrt(self.weights @ self.cov_matrix @ self.weights) * np.sqrt(252)
        
        return {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': port_return / port_vol if port_vol > 0 else 0
        }


class MeanVariancePortfolio:
    """
    Mean-variance portfolio with target return or maximum Sharpe ratio.
    """
    
    def __init__(self,
                 target_return: Optional[float] = None,
                 risk_free_rate: float = 0.02,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0):
        """
        Parameters
        ----------
        target_return : float, optional
            Target annualized return (if None, maximize Sharpe ratio)
        risk_free_rate : float
            Annual risk-free rate
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        """
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weights = None
        self.cov_matrix = None
        self.mean_returns = None
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute optimal portfolio weights.
        """
        n_assets = returns.shape[1]
        
        self.cov_matrix = returns.cov().values
        self.mean_returns = returns.mean().values * 252  # Annualized
        
        if self.target_return is not None:
            # Minimize variance for target return
            return self._fit_target_return(n_assets)
        else:
            # Maximize Sharpe ratio
            return self._fit_max_sharpe(n_assets)
    
    def _fit_target_return(self, n_assets: int) -> np.ndarray:
        """Minimize variance for given target return."""
        
        def portfolio_variance(w):
            return w @ self.cov_matrix @ w * 252
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ self.mean_returns - self.target_return}
        ]
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            portfolio_variance,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.weights = result.x if result.success else np.ones(n_assets) / n_assets
        return self.weights
    
    def _fit_max_sharpe(self, n_assets: int) -> np.ndarray:
        """Maximize Sharpe ratio."""
        
        def neg_sharpe(w):
            port_return = w @ self.mean_returns
            port_vol = np.sqrt(w @ self.cov_matrix @ w * 252)
            if port_vol < 1e-10:
                return 1e10
            return -(port_return - self.risk_free_rate) / port_vol
        
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        w0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            neg_sharpe,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        self.weights = result.x if result.success else np.ones(n_assets) / n_assets
        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights


def test_benchmarks():
    """Test benchmark optimizers."""
    # Load synthetic data
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    # Use subset for testing
    returns = returns.iloc[-500:]  # Last 500 days
    
    print("Testing Benchmark Portfolios")
    print("="*60)
    
    # Equal Weight
    ew = EqualWeightPortfolio()
    ew_weights = ew.fit(returns)
    print(f"\nEqual Weight Portfolio:")
    print(f"  Number of assets: {len(ew_weights)}")
    print(f"  Weight per asset: {ew_weights[0]:.4f}")
    
    # Minimum Variance
    mv = MinimumVariancePortfolio(max_weight=0.20)
    mv_weights = mv.fit(returns)
    mv_stats = mv.get_portfolio_stats(returns)
    
    print(f"\nMinimum Variance Portfolio:")
    print(f"  Non-zero weights: {np.sum(mv_weights > 0.001)}")
    print(f"  Max weight: {mv_weights.max():.4f}")
    print(f"  Expected return: {mv_stats['expected_return']:.2%}")
    print(f"  Volatility: {mv_stats['volatility']:.2%}")
    print(f"  Sharpe ratio: {mv_stats['sharpe_ratio']:.2f}")
    
    # Max Sharpe
    ms = MeanVariancePortfolio(risk_free_rate=0.02, max_weight=0.20)
    ms_weights = ms.fit(returns)
    
    print(f"\nMax Sharpe Portfolio:")
    print(f"  Non-zero weights: {np.sum(ms_weights > 0.001)}")
    print(f"  Max weight: {ms_weights.max():.4f}")


if __name__ == "__main__":
    test_benchmarks()
