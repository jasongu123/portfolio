"""
GARCH-EVT-CVaR Portfolio Optimizer
==================================
Based on Bedoui et al. (2023):
"Portfolio optimization through hybrid deep learning and genetic algorithms
vine Copula-GARCH-EVT-CVaR model"

This module implements:
1. GARCH(1,1) filtering for volatility
2. GPD (Generalized Pareto Distribution) for tail modeling
3. CVaR (Conditional Value at Risk) optimization
4. Vine Copula for dependency structure (simplified version)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats
from typing import Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


class GARCHModel:
    """
    GARCH(1,1) model for volatility estimation.
    
    r_t = μ + ε_t
    ε_t = σ_t * z_t,  z_t ~ N(0,1) or Student-t
    σ²_t = ω + α*ε²_{t-1} + β*σ²_{t-1}
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """
        Parameters
        ----------
        p : int
            ARCH order
        q : int
            GARCH order
        """
        self.p = p
        self.q = q
        self.params = None
        self.conditional_vol = None
        self.standardized_residuals = None
        self.mu = None
    
    def fit(self, returns: np.ndarray) -> Dict:
        """
        Fit GARCH(1,1) model using quasi-MLE.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
            
        Returns
        -------
        dict
            Fitted parameters
        """
        returns = returns[~np.isnan(returns)]
        n = len(returns)
        
        # Estimate mean
        self.mu = np.mean(returns)
        eps = returns - self.mu
        
        # Initial parameter guesses
        # [omega, alpha, beta]
        var_eps = np.var(eps)
        omega0 = var_eps * 0.05
        alpha0 = 0.08
        beta0 = 0.90
        
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            
            # Stationarity constraint
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
                return 1e10
            
            # Initialize variance
            sigma2 = np.zeros(n)
            sigma2[0] = var_eps
            
            # Compute conditional variance
            for t in range(1, n):
                sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
            
            # Log-likelihood (Gaussian)
            ll = -0.5 * np.sum(np.log(sigma2) + eps**2 / sigma2)
            
            return -ll
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            [omega0, alpha0, beta0],
            method='L-BFGS-B',
            bounds=[(1e-10, None), (0, 0.5), (0, 0.999)]
        )
        
        if result.success:
            omega, alpha, beta = result.x
        else:
            # Use defaults
            omega, alpha, beta = omega0, 0.05, 0.90
        
        self.params = {'omega': omega, 'alpha': alpha, 'beta': beta}
        
        # Compute final conditional volatility and residuals
        sigma2 = np.zeros(n)
        sigma2[0] = var_eps
        for t in range(1, n):
            sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
        
        self.conditional_vol = np.sqrt(sigma2)
        self.standardized_residuals = eps / self.conditional_vol
        
        return self.params
    
    def get_standardized_residuals(self) -> np.ndarray:
        return self.standardized_residuals
    
    def forecast_volatility(self, h: int = 1) -> float:
        """
        Forecast volatility h steps ahead.
        """
        omega = self.params['omega']
        alpha = self.params['alpha']
        beta = self.params['beta']
        
        sigma2_last = self.conditional_vol[-1]**2
        eps_last = (self.standardized_residuals[-1] * self.conditional_vol[-1])**2
        
        sigma2_forecast = omega + alpha * eps_last + beta * sigma2_last
        
        # Multi-step forecast
        persistence = alpha + beta
        unconditional_var = omega / (1 - persistence)
        
        for _ in range(1, h):
            sigma2_forecast = omega + persistence * sigma2_forecast
        
        return np.sqrt(sigma2_forecast)


class GPDTailModel:
    """
    Generalized Pareto Distribution for tail modeling.
    
    Used for peaks-over-threshold (POT) method.
    """
    
    def __init__(self, threshold_quantile: float = 0.90):
        """
        Parameters
        ----------
        threshold_quantile : float
            Quantile for threshold selection
        """
        self.threshold_quantile = threshold_quantile
        self.threshold = None
        self.xi = None  # Shape parameter
        self.beta = None  # Scale parameter
        self.n_exceedances = None
    
    def fit(self, data: np.ndarray, tail: str = 'lower') -> Dict:
        """
        Fit GPD to tail exceedances.
        
        Parameters
        ----------
        data : np.ndarray
            Data series (typically standardized residuals)
        tail : str
            'lower' for left tail (losses), 'upper' for right tail
            
        Returns
        -------
        dict
            Fitted parameters
        """
        data = data[~np.isnan(data)]
        
        if tail == 'lower':
            # For losses (left tail), use negative values
            data_tail = -data
            self.threshold = np.percentile(data_tail, self.threshold_quantile * 100)
        else:
            self.threshold = np.percentile(data, self.threshold_quantile * 100)
            data_tail = data
        
        # Get exceedances
        exceedances = data_tail[data_tail > self.threshold] - self.threshold
        self.n_exceedances = len(exceedances)
        
        if self.n_exceedances < 10:
            # Not enough exceedances, use exponential (xi=0)
            self.xi = 0
            self.beta = np.mean(exceedances) if len(exceedances) > 0 else 1.0
            return {'xi': self.xi, 'beta': self.beta, 'threshold': self.threshold}


        # Use scipy's MLE estimator (more robust than PWM)
        from scipy.stats import genpareto

        # Fit GPD using MLE
        xi_fit, loc_fit, beta_fit = genpareto.fit(exceedances, floc=0)

        self.xi = xi_fit
        self.beta = beta_fit

        # Ensure valid parameters for CVaR (xi < 1)
        if self.xi < -0.5:
            self.xi = -0.5
        elif self.xi > 0.9:
            self.xi = 0.9

        if self.beta <= 0:
            self.beta = np.std(exceedances)

        return {'xi': self.xi, 'beta': self.beta, 'threshold': self.threshold}
    
    def var(self, p: float) -> float:
        """
        Compute VaR at probability level p.
        
        Parameters
        ----------
        p : float
            Probability level (e.g., 0.95 for 95% VaR)
            
        Returns
        -------
        float
            VaR estimate
        """
        if self.xi == 0:
            var = self.threshold + self.beta * np.log(
                self.n_exceedances / (1 - p)
            )
        else:
            var = self.threshold + (self.beta / self.xi) * (
                (self.n_exceedances / (1 - p))**self.xi - 1
            )
        return var
    
    def cvar(self, p: float) -> float:
        """
        Compute CVaR (Expected Shortfall) at probability level p.
        
        Parameters
        ----------
        p : float
            Probability level
            
        Returns
        -------
        float
            CVaR estimate
        """
        var_p = self.var(p)
        
        if self.xi >= 1:
            # CVaR is infinite for xi >= 1
            return np.inf
        
        cvar = var_p / (1 - self.xi) + (self.beta - self.xi * self.threshold) / (1 - self.xi)
        return cvar


class GARCHEVTCVaRPortfolio:
    """
    Portfolio optimizer using GARCH-EVT-CVaR methodology.
    
    Implements the approach from Bedoui et al. (2023) without Vine Copula
    (simplified version using empirical correlation of standardized residuals).
    """
    
    def __init__(self,
                 cvar_alpha: float = 0.95,
                 threshold_quantile: float = 0.90,
                 n_simulations: int = 10000,
                 min_weight: float = 0.0,
                 max_weight: float = 1.0,
                 start_from_mv: bool = False):
        """
        Parameters
        ----------
        cvar_alpha : float
            Confidence level for CVaR (e.g., 0.95)
        threshold_quantile : float
            Quantile for GPD threshold
        n_simulations : int
            Number of Monte Carlo simulations
        min_weight : float
            Minimum weight per asset
        max_weight : float
            Maximum weight per asset
        """
        self.cvar_alpha = cvar_alpha
        self.threshold_quantile = threshold_quantile
        self.n_simulations = n_simulations
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.start_from_mv = start_from_mv
        
        self.weights = None
        self.garch_models = {}
        self.gpd_models = {}
        self.correlation_matrix = None
        self._simulation_cache = {}  # Cache for simulation parameters
        self._cholesky_L = None  # Cached Cholesky decomposition

    def _fit_single_asset(self, col: str, data: np.ndarray) -> Tuple:
        """Fit GARCH and GPD for a single asset (for parallel execution)."""
        garch = GARCHModel(p=1, q=1)
        garch.fit(data)
        gpd = GPDTailModel(threshold_quantile=self.threshold_quantile)
        gpd.fit(garch.get_standardized_residuals(), tail='lower')
        return col, garch, gpd

    def _fit_marginals(self, returns: pd.DataFrame, n_workers: int = 4) -> None:
        """
        Fit GARCH and GPD models to each asset (parallelized).
        """
        columns = list(returns.columns)

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._fit_single_asset, col, returns[col].values): col
                for col in columns
            }
            for future in as_completed(futures):
                col, garch, gpd = future.result()
                self.garch_models[col] = garch
                self.gpd_models[col] = gpd

    def _precompute_simulation_params(self, tickers: list) -> None:
        """Pre-compute and cache parameters needed for simulation."""
        self._simulation_cache = {}
        for ticker in tickers:
            garch = self.garch_models[ticker]
            gpd = self.gpd_models[ticker]
            std_resid = garch.get_standardized_residuals()
            sorted_resid = np.sort(std_resid)
            n_resid = len(sorted_resid)

            self._simulation_cache[ticker] = {
                'sigma': garch.forecast_volatility(h=1),
                'xi': gpd.xi,
                'beta': gpd.beta,
                'threshold': gpd.threshold,
                'sorted_resid': sorted_resid,
                'empirical_cdf': np.linspace(0, 1, n_resid),
            }
    
    def _estimate_correlation(self, returns: pd.DataFrame) -> None:
        """
        Estimate correlation matrix from standardized residuals.
        """
        std_resid = pd.DataFrame()
        for col in returns.columns:
            std_resid[col] = self.garch_models[col].get_standardized_residuals()
        
        self.correlation_matrix = std_resid.corr().values
    
    def _simulate_portfolio_returns(self,
                                     weights: np.ndarray,
                                     tickers: list) -> np.ndarray:
        """
        Simulate portfolio returns using fitted models.

        Uses Gaussian copula with correlation from standardized residuals.
        Fully vectorized for performance, uses cached parameters.
        """
        n_assets = len(weights)

        # Generate correlated standard normals
        Z = np.random.standard_normal((self.n_simulations, n_assets))
        correlated_Z = Z @ self._cholesky_L.T

        # Transform to uniform using standard normal CDF
        U = stats.norm.cdf(correlated_Z)

        # Transform to asset returns using semi-parametric distribution (VECTORIZED)
        simulated_returns = np.zeros((self.n_simulations, n_assets))

        for i, ticker in enumerate(tickers):
            cache = self._simulation_cache[ticker]
            sigma = cache['sigma']
            xi = cache['xi']
            beta = cache['beta']
            threshold = cache['threshold']
            sorted_resid = cache['sorted_resid']
            empirical_cdf = cache['empirical_cdf']

            # Get uniform values for this asset
            u = U[:, i]

            # Initialize z array
            z = np.zeros(self.n_simulations)

            # Vectorized masks for each region
            lower_mask = u < 0.1
            upper_mask = u > 0.9
            center_mask = ~lower_mask & ~upper_mask

            # Lower tail (vectorized)
            if np.any(lower_mask):
                u_lower = u[lower_mask]
                if xi == 0:
                    z[lower_mask] = -threshold - beta * np.log((1 - u_lower) / 0.1)
                else:
                    z[lower_mask] = -threshold - (beta / xi) * (
                        (0.1 / (1 - u_lower))**xi - 1
                    )

            # Upper tail (vectorized)
            if np.any(upper_mask):
                u_upper = u[upper_mask]
                if xi == 0:
                    z[upper_mask] = threshold + beta * np.log(u_upper / 0.1)
                else:
                    z[upper_mask] = threshold + (beta / xi) * (
                        (0.1 / (1 - u_upper))**xi - 1
                    )

            # Center: use interpolation instead of repeated percentile calls
            if np.any(center_mask):
                z[center_mask] = np.interp(u[center_mask], empirical_cdf, sorted_resid)

            simulated_returns[:, i] = z * sigma

        # Portfolio returns
        portfolio_returns = simulated_returns @ weights

        return portfolio_returns
    
    def _compute_cvar(self, weights: np.ndarray, tickers: list) -> float:
        """
        Compute CVaR using Monte Carlo simulation.
        """
        portfolio_returns = self._simulate_portfolio_returns(weights, tickers)
        
        # VaR threshold
        var_threshold = np.percentile(portfolio_returns, (1 - self.cvar_alpha) * 100)
        
        # CVaR = expected loss given loss exceeds VaR
        losses = -portfolio_returns
        cvar = np.mean(losses[losses >= -var_threshold])
        
        return cvar
    
    def fit(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Compute CVaR-optimal portfolio weights.

        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns

        Returns
        -------
        np.ndarray
            Optimal portfolio weights
        """
        tickers = list(returns.columns)
        n_assets = len(tickers)

        # Step 1: Fit marginal models (parallelized)
        print("  Fitting GARCH models...")
        self._fit_marginals(returns)

        # Step 2: Estimate correlation and cache Cholesky decomposition
        self._estimate_correlation(returns)
        self._cholesky_L = np.linalg.cholesky(self.correlation_matrix)

        # Step 3: Precompute simulation parameters (avoids repeated computation)
        self._precompute_simulation_params(tickers)

        # Step 4: Optimize
        print("  Optimizing CVaR...")
        
        if self.start_from_mv:
    # Compute MV weights as starting point
            from benchmark_portfolios import MinimumVariancePortfolio
            mv_opt = MinimumVariancePortfolio(max_weight=self.max_weight)
            w0 = mv_opt.fit(returns)
            print(f"  Starting from MV weights (ENC: {1/np.sum(w0**2):.1f})")
        else:
            w0 = np.ones(n_assets) / n_assets
            print(f"  Starting from equal weights")
        rng_state = np.random.get_state()

        def objective(w):
            np.random.seed(42)
            return self._compute_cvar(w, tickers)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]

        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Optimize (using fewer simulations for gradient estimation)
        original_sims = self.n_simulations
        self.n_simulations = 2000  # Reduced for faster optimization

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 50, 'ftol': 1e-3}
        )

        self.n_simulations = original_sims
        np.random.set_state(rng_state)
        
        # Diagnostic: Compare CVaR at different allocations
        print("\n  CVaR Comparison:")
        self.n_simulations = 5000

        # Equal weights
        w_equal = np.ones(n_assets) / n_assets
        np.random.seed(42)
        cvar_equal = self._compute_cvar(w_equal, tickers)
        print(f"    Equal weights: CVaR = {cvar_equal:.6f}")

        # Concentrated in top 10 lowest-beta stocks
        betas = [self.gpd_models[t].beta for t in tickers]
        top10_idx = np.argsort(betas)[:10]
        w_concentrated = np.zeros(n_assets)
        w_concentrated[top10_idx] = 0.10
        np.random.seed(42)
        cvar_conc = self._compute_cvar(w_concentrated, tickers)
        print(f"    Concentrated (10 stocks): CVaR = {cvar_conc:.6f}")

        returns_cov = returns.cov().values
        
        from benchmark_portfolios import MinimumVariancePortfolio
        mv_opt = MinimumVariancePortfolio(max_weight=self.max_weight)
        w_mv = mv_opt.fit(returns)
        np.random.seed(42)
        cvar_mv = self._compute_cvar(w_mv, tickers)
        print(f"    MV weights: CVaR = {cvar_mv:.6f}")
        print(f"    MV weights ENC: {1/np.sum(w_mv**2):.1f}")

        # Optimizer result
        np.random.seed(42)
        cvar_opt = self._compute_cvar(result.x, tickers)
        print(f"    Optimizer result: CVaR = {cvar_opt:.6f}")
        
        self.cvar_diagnostics = {
            'cvar_equal': cvar_equal,
            'cvar_concentrated': cvar_conc,
            'cvar_mv': cvar_mv,
            'ratio_concentrated': cvar_conc / cvar_equal,
            'ratio_mv': cvar_mv / cvar_equal
        }
        
        self.n_simulations = original_sims

        if result.success:
            self.weights = result.x
            self.weights[np.abs(self.weights) < 1e-6] = 0
            self.weights = self.weights / self.weights.sum()
        else:
            self.weights = np.ones(n_assets) / n_assets

        return self.weights
    
    def get_weights(self) -> np.ndarray:
        return self.weights
    
    def get_diagnostics(self, returns: pd.DataFrame) -> dict:
        """
        Return diagnostic information.
        """
        tickers = list(returns.columns)

        # Compute final CVaR with more simulations
        original_sims = self.n_simulations
        self.n_simulations = 5000
        final_cvar = self._compute_cvar(self.weights, tickers)
        self.n_simulations = original_sims

        # GARCH parameters
        garch_params = {
            ticker: model.params
            for ticker, model in self.garch_models.items()
        }

        # GPD parameters
        gpd_params = {
            ticker: {'xi': model.xi, 'beta': model.beta}
            for ticker, model in self.gpd_models.items()
        }

        return {
            'cvar': final_cvar,
            'cvar_alpha': self.cvar_alpha,
            'garch_params_sample': dict(list(garch_params.items())[:3]),
            'gpd_params_sample': dict(list(gpd_params.items())[:3])
        }


def test_garch_evt_cvar():
    """Test GARCH-EVT-CVaR optimizer."""
    # Load synthetic data
    returns = pd.read_csv("data/returns.csv", index_col=0, parse_dates=True)
    
    # Use smaller subset for faster testing
    returns = returns.iloc[-500:, :20]  # 500 days, 20 stocks
    
    print("Testing GARCH-EVT-CVaR Portfolio Optimizer")
    print("="*60)
    
    # Fit optimizer
    optimizer = GARCHEVTCVaRPortfolio(
        cvar_alpha=0.95,
        threshold_quantile=0.90,
        n_simulations=5000,
        max_weight=0.20
    )
    
    print("\nFitting model...")
    weights = optimizer.fit(returns)
    
    print(f"\nResults:")
    print(f"  Non-zero weights: {np.sum(weights > 0.001)}")
    print(f"  Max weight: {weights.max():.4f}")
    print(f"  Min non-zero weight: {weights[weights > 0.001].min():.4f}")
    
    # Get diagnostics
    diagnostics = optimizer.get_diagnostics(returns)
    print(f"\n  CVaR (95%): {diagnostics['cvar']:.4f}")
    
    print("\nSample GARCH parameters:")
    for ticker, params in diagnostics['garch_params_sample'].items():
        print(f"  {ticker}: ω={params['omega']:.2e}, α={params['alpha']:.3f}, β={params['beta']:.3f}")


if __name__ == "__main__":
    test_garch_evt_cvar()
