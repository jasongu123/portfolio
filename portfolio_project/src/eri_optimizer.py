"""
Extreme Risk Index (ERI) Portfolio Optimizer
=============================================
Based on Mainik, Mitov & Rüschendorf (2015):
"Portfolio optimization for heavy-tailed assets: Extreme Risk Index vs. Markowitz"

The ERI approach uses multivariate extreme value theory to minimize
the probability of large portfolio losses.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class HillEstimator:
    """
    Hill estimator for the tail index of heavy-tailed distributions.
    
    The tail index α characterizes how heavy the tail is:
    - α < 2: Infinite variance
    - α ∈ [2, 4]: Heavy tails (finite variance, infinite kurtosis for α < 4)
    - α > 4: Moderate tails
    """
    
    def __init__(self, tail_fraction: float = 0.10):
        """
        Parameters
        ----------
        tail_fraction : float
            Fraction of observations to use for tail estimation (default 10%)
        """
        self.tail_fraction = tail_fraction
        self.alpha = None
        self.alpha_se = None
        self.k = None
    
    def fit(self, data: np.ndarray) -> float:
        """
        Estimate tail index using Hill estimator.
        
        Parameters
        ----------
        data : np.ndarray
            Data series (typically absolute returns or radial parts)
            
        Returns
        -------
        float
            Estimated tail index α
        """
        # Use absolute values
        abs_data = np.abs(data[~np.isnan(data)])
        n = len(abs_data)
        self.k = max(10, int(n * self.tail_fraction))
        
        # Sort in descending order
        sorted_data = np.sort(abs_data)[::-1]
        
        # Hill estimator: α = k / Σ log(X_(i) / X_(k+1))
        log_ratios = np.log(sorted_data[:self.k] / sorted_data[self.k])
        
        self.alpha = self.k / np.sum(log_ratios)
        
        # Standard error (asymptotic)
        self.alpha_se = self.alpha / np.sqrt(self.k)
        
        return self.alpha
    
    def get_statistics(self) -> dict:
        return {
            'alpha': self.alpha,
            'se': self.alpha_se,
            'k': self.k,
            '95_ci_lower': self.alpha - 1.96 * self.alpha_se,
            '95_ci_upper': self.alpha + 1.96 * self.alpha_se
        }


class SpectralMeasureEstimator:
    """
    Estimator for the spectral measure in multivariate regular variation.
    
    For a regularly varying random vector X with tail index α,
    the spectral measure Ψ describes the angular distribution of extremes.
    """
    
    def __init__(self, tail_fraction: float = 0.10):
        """
        Parameters
        ----------
        tail_fraction : float
            Fraction of observations to use for estimation
        """
        self.tail_fraction = tail_fraction
        self.spectral_data = None
        self.n_extreme = None
    
    def fit(self, returns: np.ndarray) -> np.ndarray:
        """
        Estimate empirical spectral measure.
        
        Parameters
        ----------
        returns : np.ndarray
            Multivariate return matrix (n_obs x n_assets)
            
        Returns
        -------
        np.ndarray
            Angular components of extreme observations (n_extreme x n_assets)
        """
        n_obs, n_assets = returns.shape
        self.n_extreme = max(10, int(n_obs * self.tail_fraction))
        
        # Compute radial parts (L2 norm of each observation)
        radial = np.linalg.norm(returns, axis=1)
        
        # Get indices of largest radial parts
        extreme_indices = np.argsort(radial)[-self.n_extreme:]
        
        # Compute angular parts: θ = x / ||x||
        extreme_returns = returns[extreme_indices, :]
        extreme_radial = radial[extreme_indices].reshape(-1, 1)
        
        # Avoid division by zero
        extreme_radial = np.maximum(extreme_radial, 1e-10)
        
        self.spectral_data = extreme_returns / extreme_radial
        
        return self.spectral_data
    
    def compute_eri(self, weights: np.ndarray) -> float:
        """
        Compute Extreme Risk Index for given portfolio weights.
        
        ERI(w) = E[|w'Θ|^α] where Θ follows the spectral measure
        
        For the empirical version:
        ERI(w) ≈ (1/k) Σ |w'θ_i|^α
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
            
        Returns
        -------
        float
            Extreme Risk Index
        """
        if self.spectral_data is None:
            raise ValueError("Must call fit() first")
        
        # Portfolio projections onto angular components
        projections = self.spectral_data @ weights
        
        # Return mean of absolute projections (with α=2 for simplicity)
        # In practice, α should be estimated from the data
        return np.mean(np.abs(projections)**2)


class ERIPortfolio:
    """
    Portfolio optimizer based on Extreme Risk Index minimization.
    
    This implements the methodology from Mainik et al. (2015).
    """
    
    def __init__(self,
                 tail_fraction: float = 0.10,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 tail_index: Optional[float] = None):
        """
        Parameters
        ----------
        tail_fraction : float
            Fraction of observations for tail estimation
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        tail_index : float, optional
            Fixed tail index (if None, estimate from data)
        """
        self.tail_fraction = tail_fraction
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.fixed_tail_index = tail_index
        
        self.weights = None
        self.tail_index = None
        self.spectral_estimator = None
        self.asset_tail_indices = None
    
    def _estimate_tail_index(self, returns: np.ndarray) -> float:
        """
        Estimate common tail index from multivariate data.
        Uses the radial parts of the return vectors.
        """
        # Compute radial parts
        radial = np.linalg.norm(returns, axis=1)
        
        # Apply Hill estimator to radial parts
        hill = HillEstimator(tail_fraction=self.tail_fraction)
        return hill.fit(radial)
    
    def _estimate_asset_tail_indices(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estimate tail index for each individual asset.
        """
        indices = []
        for col in returns.columns:
            hill = HillEstimator(tail_fraction=self.tail_fraction)
            alpha = hill.fit(returns[col].values)
            indices.append(alpha)
        return np.array(indices)
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute ERI-optimal portfolio weights.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns
            
        Returns
        -------
        np.ndarray
            Optimal portfolio weights
        """
        returns_array = returns.values
        n_assets = returns.shape[1]
        
        # Estimate tail index
        if self.fixed_tail_index is not None:
            self.tail_index = self.fixed_tail_index
        else:
            self.tail_index = self._estimate_tail_index(returns_array)
        
        # Estimate individual asset tail indices (for diagnostics)
        self.asset_tail_indices = self._estimate_asset_tail_indices(returns)
        
        # Fit spectral measure
        self.spectral_estimator = SpectralMeasureEstimator(
            tail_fraction=self.tail_fraction
        )
        self.spectral_estimator.fit(returns_array)
        
        # Objective: minimize ERI
        def eri_objective(w):
            # Compute ERI with estimated tail index
            projections = self.spectral_estimator.spectral_data @ w
            # Use absolute value raised to tail index power
            return np.mean(np.abs(projections)**self.tail_index)
        
        # Gradient (numerical for stability)
        def eri_gradient(w):
            eps = 1e-7
            grad = np.zeros(n_assets)
            f0 = eri_objective(w)
            for i in range(n_assets):
                w_plus = w.copy()
                w_plus[i] += eps
                grad[i] = (eri_objective(w_plus) - f0) / eps
            return grad
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess
        w0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            eri_objective,
            w0,
            method='SLSQP',
            jac=eri_gradient,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            self.weights = result.x
            # Clean up small weights
            self.weights[np.abs(self.weights) < 1e-6] = 0
            self.weights = self.weights / self.weights.sum()
        else:
            # Fall back to equal weight
            self.weights = np.ones(n_assets) / n_assets
        
        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_diagnostics(self) -> dict:
        """
        Return diagnostic information.
        """
        return {
            'portfolio_tail_index': self.tail_index,
            'asset_tail_indices': self.asset_tail_indices,
            'n_extreme_obs': self.spectral_estimator.n_extreme if self.spectral_estimator else None,
            'eri_value': self._compute_portfolio_eri() if self.weights is not None else None
        }
    
    def _compute_portfolio_eri(self) -> float:
        """Compute ERI for current weights."""
        projections = self.spectral_estimator.spectral_data @ self.weights
        return np.mean(np.abs(projections)**self.tail_index)


class ERIPortfolioHeavyTailsOnly(ERIPortfolio):
    """
    ERI Portfolio that filters to assets with heavy tails only.
    
    Based on Mainik et al. finding that ERI outperforms on assets with α ≤ 2.2
    """
    
    def __init__(self,
                 tail_threshold: float = 3.0,
                 **kwargs):
        """
        Parameters
        ----------
        tail_threshold : float
            Maximum tail index for inclusion (lower = heavier tails)
        """
        super().__init__(**kwargs)
        self.tail_threshold = tail_threshold
        self.selected_assets = None
        self.full_weights = None
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Fit ERI portfolio on heavy-tailed assets only.
        """
        # First estimate tail indices for all assets
        asset_tails = self._estimate_asset_tail_indices(returns)
        
        # Select assets with heavy tails
        self.selected_assets = asset_tails <= self.tail_threshold
        n_selected = np.sum(self.selected_assets)
        
        if n_selected < 2:
            # Not enough heavy-tailed assets, use all
            self.selected_assets = np.ones(len(asset_tails), dtype=bool)
            n_selected = len(asset_tails)
        
        # Fit on selected assets
        selected_returns = returns.iloc[:, self.selected_assets]
        weights_selected = super().fit(selected_returns)
        
        # Map back to full weight vector
        self.full_weights = np.zeros(returns.shape[1])
        self.full_weights[self.selected_assets] = weights_selected
        
        return self.full_weights
    
    def get_weights(self) -> np.ndarray:
        return self.full_weights


def test_eri_optimizer():
    """Test ERI optimizer."""
    # Load synthetic data
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    # Use subset for testing
    returns = returns.iloc[-1000:]
    
    print("Testing ERI Portfolio Optimizer")
    print("="*60)
    
    # Standard ERI
    eri = ERIPortfolio(tail_fraction=0.10, max_weight=0.20)
    eri_weights = eri.fit(returns)
    
    diagnostics = eri.get_diagnostics()
    
    print(f"\nERI Portfolio:")
    print(f"  Estimated tail index: {diagnostics['portfolio_tail_index']:.2f}")
    print(f"  Extreme observations used: {diagnostics['n_extreme_obs']}")
    print(f"  ERI value: {diagnostics['eri_value']:.6f}")
    print(f"  Non-zero weights: {np.sum(eri_weights > 0.001)}")
    print(f"  Max weight: {eri_weights.max():.4f}")
    
    print(f"\nAsset tail indices (sample):")
    for i, (ticker, alpha) in enumerate(zip(returns.columns[:5], 
                                            diagnostics['asset_tail_indices'][:5])):
        print(f"  {ticker}: α = {alpha:.2f}")
    
    # Heavy tails only
    print("\n" + "-"*60)
    eri_heavy = ERIPortfolioHeavyTailsOnly(
        tail_threshold=4.5, 
        tail_fraction=0.10, 
        max_weight=0.20
    )
    eri_heavy_weights = eri_heavy.fit(returns)
    
    print(f"\nERI Portfolio (Heavy Tails Only, α ≤ 4.5):")
    print(f"  Assets selected: {np.sum(eri_heavy.selected_assets)}/{len(returns.columns)}")
    print(f"  Non-zero weights: {np.sum(eri_heavy_weights > 0.001)}")


if __name__ == "__main__":
    test_eri_optimizer()
