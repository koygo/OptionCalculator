"""
American Option implementation with LSM pricing and finite difference Greeks.
"""
from .option import Option
from .monte_carlo import MonteCarloEngine
import numpy as np


class AmericanOption(Option):
    """
    American option with early exercise capability.

    Pricing uses the Longstaff-Schwartz Least Squares Monte Carlo (LSM) algorithm.
    Greeks are calculated using finite differences with Common Random Numbers (CRN)
    for variance reduction.

    Inherits all attributes from Option base class.
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call', num_simulations=10000, num_steps=252):
        """Initialize American option."""
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)

    def price(self):
        """
        Calculate American option price using LSM algorithm.

        Returns
        -------
        float
            Option price
        """
        return self.mc_engine.price_american(
            self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type
        )

    # Helper Methods

    def _generate_paths_for_greek(self, Z, S0, r, sigma):
        """
        Generate price paths for Greek calculations.

        Centralizes path generation logic to eliminate code duplication
        across different Greek calculations.

        Parameters
        ----------
        Z : ndarray
            Pre-generated random numbers
        S0 : float
            Initial stock price
        r : float
            Risk-free rate
        sigma : float
            Volatility

        Returns
        -------
        ndarray
            Simulated price paths
        """
        dt = self.T / self.num_steps
        num_sims = Z.shape[0]
        paths = np.zeros((num_sims, self.num_steps + 1))
        paths[:, 0] = S0

        for t in range(1, self.num_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - self.q - 0.5 * sigma ** 2) * dt +
                sigma * np.sqrt(dt) * Z[:, t - 1]
            )
        return paths

    def _price_from_paths(self, paths, r):
        """
        Price option from generated paths using LSM.

        Parameters
        ----------
        paths : ndarray
            Simulated price paths
        r : float
            Risk-free rate for discounting

        Returns
        -------
        float
            Option price
        """
        return self.mc_engine._lsm_pricing(paths, self.K, r, self.T, self.option_type)

    # Greeks

    def delta(self, bump=1.0, num_sims=100000):
        """
        Calculate Delta: ∂V/∂S.

        Measures sensitivity of option value to $1 change in underlying price.
        Uses central finite differences with Common Random Numbers.

        Parameters
        ----------
        bump : float, optional
            Size of price bump in dollars (default: 1.0)
        num_sims : int, optional
            Number of simulations (default: 100000)

        Returns
        -------
        float
            Delta value
        """
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S + bump, self.r, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S - bump, self.r, self.sigma)

        price_up = self._price_from_paths(paths_up, self.r)
        price_down = self._price_from_paths(paths_down, self.r)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=1.0, num_sims=100000):
        """
        Calculate Gamma: ∂²V/∂S².

        Measures rate of change of delta with respect to underlying price.
        Uses second-order finite differences with Common Random Numbers.

        Parameters
        ----------
        bump : float, optional
            Size of price bump in dollars (default: 1.0)
        num_sims : int, optional
            Number of simulations (default: 100000)

        Returns
        -------
        float
            Gamma value (non-negative due to MC noise flooring)
        """
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S + bump, self.r, self.sigma)
        paths_center = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S - bump, self.r, self.sigma)

        price_up = self._price_from_paths(paths_up, self.r)
        price_center = self._price_from_paths(paths_center, self.r)
        price_down = self._price_from_paths(paths_down, self.r)

        gamma = (price_up - 2 * price_center + price_down) / (bump ** 2)
        return max(gamma, 1e-6)

    def vega(self, bump=0.01, num_sims=100000):
        """
        Calculate Vega: ∂V/∂σ.

        Measures sensitivity of option value to 1% (0.01) change in volatility.
        Uses central finite differences with Common Random Numbers.

        Parameters
        ----------
        bump : float, optional
            Size of volatility bump (default: 0.01 = 1%)
        num_sims : int, optional
            Number of simulations (default: 100000)

        Returns
        -------
        float
            Vega value (change in price per 1% volatility change)
        """
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma + bump)
        paths_down = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma - bump)

        price_up = self._price_from_paths(paths_up, self.r)
        price_down = self._price_from_paths(paths_down, self.r)

        return (price_up - price_down) / (2 * bump)

    def theta(self, bump=1 / 365, num_sims=100000):
        """
        Calculate Theta: ∂V/∂t.

        Measures time decay of option value (change per day).
        Uses forward finite differences.

        Parameters
        ----------
        bump : float, optional
            Size of time bump in years (default: 1/365 = 1 day)
        num_sims : int, optional
            Number of simulations (default: 100000)

        Returns
        -------
        float
            Theta value (typically negative for long options)
        """
        # Price at current T
        np.random.seed(42)
        temp_engine_current = MonteCarloEngine(num_sims, self.num_steps, seed=42)
        price_current = temp_engine_current.price_american(
            self.S, self.K, self.T, self.r, self.sigma, self.q, self.option_type
        )

        # Price at T - bump (one day earlier expiration)
        np.random.seed(42)
        temp_engine_minus = MonteCarloEngine(num_sims, self.num_steps, seed=42)
        T_minus = max(self.T - bump, 0)
        price_minus = temp_engine_minus.price_american(
            self.S, self.K, T_minus, self.r, self.sigma, self.q, self.option_type
        )

        return (price_minus - price_current) / bump

    def rho(self, bump=0.01, num_sims=100000):
        """
        Calculate Rho: ∂V/∂r.

        Measures sensitivity of option value to 1% (0.01) change in interest rate.
        Uses central finite differences with Common Random Numbers.

        Parameters
        ----------
        bump : float, optional
            Size of rate bump (default: 0.01 = 1%)
        num_sims : int, optional
            Number of simulations (default: 100000)

        Returns
        -------
        float
            Rho value (change in price per 1% rate change)
        """
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S, self.r + bump, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S, self.r - bump, self.sigma)

        price_up = self._price_from_paths(paths_up, self.r + bump)
        price_down = self._price_from_paths(paths_down, self.r - bump)

        return (price_up - price_down) / (2 * bump)

    def get_all_greeks(self):
        """
        Calculate all Greeks and return as dictionary.

        Returns
        -------
        dict
            Dictionary containing all Greeks: delta, gamma, vega, theta, rho
        """
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }