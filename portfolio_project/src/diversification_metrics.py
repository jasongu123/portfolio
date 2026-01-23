"""
Diversification Metrics
=======================
Based on Damjanovic et al. (2025):
"Price vs. market-cap-weighted portfolio diversification: does it matter?"

Implements various diversification measures:
1. Weight-based: ENC, Shannon Entropy, HHI, Gini
2. Covariance-based: PDI (Portfolio Diversification Index)
3. Combined: ENB (Effective Number of Bets)
4. Novel: EDP (Exploited Diversification Potential)
"""

import numpy as np
import pandas as pd
from scipy import linalg
from typing import Dict, Tuple


class DiversificationMetrics:
    """
    Compute various diversification metrics for a portfolio.
    """
    
    def __init__(self):
        self.metrics = {}
    
    def compute_all(self, 
                    weights: np.ndarray,
                    returns: pd.DataFrame = None,
                    cov_matrix: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all diversification metrics.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        returns : pd.DataFrame, optional
            Historical returns (for covariance estimation)
        cov_matrix : np.ndarray, optional
            Pre-computed covariance matrix
            
        Returns
        -------
        dict
            Dictionary of all metrics
        """
        # Clean weights
        weights = np.array(weights)
        weights = weights / weights.sum()  # Ensure sum to 1
        
        # Weight-based metrics (always computable)
        self.metrics['ENC'] = self.effective_number_constituents(weights)
        self.metrics['shannon_entropy'] = self.shannon_entropy(weights)
        self.metrics['HHI'] = self.herfindahl_hirschman_index(weights)
        self.metrics['gini'] = self.gini_coefficient(weights)
        
        # Covariance-based metrics
        if returns is not None:
            cov_matrix = returns.cov().values
        
        if cov_matrix is not None:
            self.metrics['PDI'] = self.portfolio_diversification_index(cov_matrix)
            self.metrics['ENB'] = self.effective_number_bets(weights, cov_matrix)
            self.metrics['diversification_ratio'] = self.diversification_ratio(weights, cov_matrix)
            
            # EDP requires both weights and covariance
            self.metrics['EDP'] = self.exploited_diversification_potential(
                weights, cov_matrix
            )
        
        return self.metrics
    
    # ==========================================================================
    # Weight-Based Metrics
    # ==========================================================================
    
    @staticmethod
    def effective_number_constituents(weights: np.ndarray) -> float:
        """
        Effective Number of Constituents (ENC).
        
        ENC = 1 / Σ w_i²
        
        Also known as the inverse Herfindahl-Hirschman Index.
        Ranges from 1 (concentrated) to N (equally weighted).
        """
        weights = weights[weights > 0]  # Only positive weights
        return 1.0 / np.sum(weights**2)
    
    @staticmethod
    def shannon_entropy(weights: np.ndarray) -> float:
        """
        Shannon Entropy of portfolio weights.
        
        H = -Σ w_i * log(w_i)
        
        Higher values indicate more diversification.
        Maximum = log(N) for equal weights.
        """
        weights = weights[weights > 0]  # Avoid log(0)
        return -np.sum(weights * np.log(weights))
    
    @staticmethod
    def herfindahl_hirschman_index(weights: np.ndarray) -> float:
        """
        Herfindahl-Hirschman Index (HHI).
        
        HHI = Σ w_i²
        
        Lower values indicate more diversification.
        Ranges from 1/N (equal weights) to 1 (single asset).
        """
        return np.sum(weights**2)
    
    @staticmethod
    def gini_coefficient(weights: np.ndarray) -> float:
        """
        Gini coefficient of weight distribution.
        
        Measures inequality in weight distribution.
        0 = perfect equality, 1 = maximum inequality
        """
        weights = np.sort(weights)
        n = len(weights)
        cumsum = np.cumsum(weights)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    # ==========================================================================
    # Covariance-Based Metrics
    # ==========================================================================
    
    @staticmethod
    def portfolio_diversification_index(cov_matrix: np.ndarray) -> float:
        """
        Portfolio Diversification Index (PDI).
        
        Based on principal component analysis of the covariance matrix.
        PDI = 2 * Σ(i * λ_i / Σλ_j) - 1
        
        Ranges from 1 (no diversification) to N (perfect diversification).
        """
        # Eigenvalues of covariance matrix
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues[eigenvalues > 0]  # Only positive
        eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
        
        n = len(eigenvalues)
        total_var = np.sum(eigenvalues)
        
        if total_var == 0:
            return 1.0
        
        # Normalized eigenvalues
        norm_eigenvalues = eigenvalues / total_var
        
        # PDI formula
        pdi = 2 * np.sum(np.arange(1, n+1) * norm_eigenvalues) - 1
        
        return pdi
    
    @staticmethod
    def effective_number_bets(weights: np.ndarray, 
                               cov_matrix: np.ndarray) -> float:
        """
        Effective Number of Bets (ENB).
        
        Combines weight concentration with correlation structure.
        Uses principal portfolios to identify independent risk factors.
        
        ENB = exp(-Σ p_i * log(p_i))
        where p_i is the variance contribution of each principal portfolio.
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Only positive eigenvalues
        pos_mask = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_mask]
        eigenvectors = eigenvectors[:, pos_mask]
        
        n_factors = len(eigenvalues)
        
        # Portfolio variance
        port_var = weights @ cov_matrix @ weights
        
        if port_var == 0:
            return 1.0
        
        # Contribution of each principal portfolio to total variance
        contributions = np.zeros(n_factors)
        for i in range(n_factors):
            # Loading of portfolio on i-th principal portfolio
            loading = weights @ eigenvectors[:, i]
            contributions[i] = (loading**2 * eigenvalues[i]) / port_var
        
        # Normalize
        contributions = contributions / contributions.sum()
        contributions = contributions[contributions > 0]
        
        # Shannon entropy-based ENB
        enb = np.exp(-np.sum(contributions * np.log(contributions)))
        
        return enb
    
    @staticmethod
    def diversification_ratio(weights: np.ndarray,
                               cov_matrix: np.ndarray) -> float:
        """
        Diversification Ratio (DR).
        
        DR = (w' σ) / sqrt(w' Σ w)
        
        Ratio of weighted average volatility to portfolio volatility.
        Higher values indicate more diversification benefit from correlations.
        """
        # Individual volatilities
        vols = np.sqrt(np.diag(cov_matrix))
        
        # Portfolio volatility
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        if port_vol == 0:
            return 1.0
        
        # Weighted average volatility
        weighted_vol = weights @ vols
        
        return weighted_vol / port_vol
    
    # ==========================================================================
    # Combined Metrics
    # ==========================================================================
    
    @staticmethod
    def exploited_diversification_potential(weights: np.ndarray,
                                             cov_matrix: np.ndarray) -> float:
        """
        Exploited Diversification Potential (EDP).
        
        Novel metric from Damjanovic et al. (2025).
        
        EDP = Actual_Diversification / Maximum_Possible_Diversification
        
        Measures how efficiently the portfolio exploits available
        diversification opportunities given the correlation structure.
        """
        n_assets = len(weights)
        
        n_assets = len(weights)
        enc = DiversificationMetrics.effective_number_constituents(weights)
        return enc / n_assets


def compute_portfolio_stats(weights: np.ndarray,
                            returns: pd.DataFrame) -> Dict[str, float]:
    """
    Compute portfolio performance statistics.
    """
    # Portfolio returns
    port_returns = (returns * weights).sum(axis=1)
    
    # Annualized metrics
    ann_return = port_returns.mean() * 252
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
    
    # Downside metrics
    neg_returns = port_returns[port_returns < 0]
    sortino = ann_return / (neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + port_returns).cumprod()
    rolling_max = cum_returns.expanding().max()
    drawdowns = cum_returns / rolling_max - 1
    max_drawdown = drawdowns.min()
    
    # VaR and CVaR
    var_95 = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()
    
    return {
        'annual_return': ann_return,
        'annual_volatility': ann_vol,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': port_returns.skew(),
        'kurtosis': port_returns.kurtosis()
    }


def compare_portfolio_diversification(portfolios: Dict[str, np.ndarray],
                                       returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compare diversification metrics across multiple portfolios.
    
    Parameters
    ----------
    portfolios : dict
        Dictionary of portfolio name -> weights
    returns : pd.DataFrame
        Historical returns
        
    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    cov_matrix = returns.cov().values
    metrics_calc = DiversificationMetrics()
    
    results = []
    for name, weights in portfolios.items():
        # Diversification metrics
        metrics = metrics_calc.compute_all(weights, cov_matrix=cov_matrix)
        
        # Performance metrics
        perf = compute_portfolio_stats(weights, returns)
        
        # Combine
        row = {'portfolio': name}
        row.update(metrics)
        row.update(perf)
        results.append(row)
    
    return pd.DataFrame(results).set_index('portfolio')


def test_diversification_metrics():
    """Test diversification metrics."""
    # Load synthetic data
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    returns = returns.iloc[-500:]  # Last 500 days
    
    n_assets = returns.shape[1]
    
    print("Testing Diversification Metrics")
    print("="*60)
    
    # Create test portfolios
    portfolios = {
        'Equal Weight': np.ones(n_assets) / n_assets,
        'Concentrated (5 assets)': np.array([0.2]*5 + [0]*(n_assets-5)),
        'Single Asset': np.array([1] + [0]*(n_assets-1)),
    }
    
    # Compare
    comparison = compare_portfolio_diversification(portfolios, returns)
    
    print("\nDiversification Metrics:")
    print("-"*60)
    div_cols = ['ENC', 'shannon_entropy', 'HHI', 'PDI', 'ENB', 'EDP', 'diversification_ratio']
    print(comparison[div_cols].round(3).to_string())
    
    print("\n\nPerformance Metrics:")
    print("-"*60)
    perf_cols = ['annual_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']
    print(comparison[perf_cols].round(4).to_string())


if __name__ == "__main__":
    test_diversification_metrics()
