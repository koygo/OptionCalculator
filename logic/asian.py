"""
Asian Option implementation with path-dependent averaging.
"""
from .option import Option
import numpy as np


class AsianOption(Option):
    """
    Asian option with payoff based on average underlying price.

    Supports both arithmetic and geometric averaging. Greeks are calculated
    using finite differences with Common Random Numbers (CRN).

    Attributes
    ----------
    average_type : str
        'arithmetic' or 'geometric' averaging method
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call',
                 average_type='arithmetic', num_simulations=10000, num_steps=252):
        """
        Initialize Asian option.

        Parameters
        ----------
        average_type : str, optional
            'arithmetic' or 'geometric' (default: 'arithmetic')
        """
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)
        self.average_type = average_type

    def price(self):
        """Calculate Asian option price using Monte Carlo."""
        return self.mc_engine.price_asian(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.average_type
        )

    # Helper Methods

    def _generate_paths_for_greek(self, Z, S0, r, sigma):
        """Generate price paths for Greek calculations with CRN."""
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

    def _price_from_paths(self, paths):
        """Calculate Asian option price from generated paths."""
        # Calculate average
        if self.average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        # Calculate payoff
        if self.option_type == 'call':
            payoffs = np.maximum(avg_prices - self.K, 0)
        else:
            payoffs = np.maximum(self.K - avg_prices, 0)

        return np.exp(-self.r * self.T) * np.mean(payoffs)

    # Greeks

    def delta(self, bump=1.0, num_sims=100000):
        """Calculate Delta: ∂V/∂S."""
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S + bump, self.r, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S - bump, self.r, self.sigma)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def gamma(self, bump=1.0, num_sims=100000):
        """Calculate Gamma: ∂²V/∂S²."""
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S + bump, self.r, self.sigma)
        paths_center = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S - bump, self.r, self.sigma)

        price_up = self._price_from_paths(paths_up)
        price_center = self._price_from_paths(paths_center)
        price_down = self._price_from_paths(paths_down)

        gamma = (price_up - 2 * price_center + price_down) / (bump ** 2)
        return max(gamma, 1e-6)

    def vega(self, bump=0.01, num_sims=100000):
        """Calculate Vega: ∂V/∂σ."""
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma + bump)
        paths_down = self._generate_paths_for_greek(Z, self.S, self.r, self.sigma - bump)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def theta(self, bump=1 / 365, num_sims=100000):
        """Calculate Theta: ∂V/∂t."""
        from .monte_carlo import MonteCarloEngine

        np.random.seed(42)
        temp_engine_current = MonteCarloEngine(num_sims, self.num_steps, seed=42)
        price_current = temp_engine_current.price_asian(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.average_type
        )

        np.random.seed(42)
        temp_engine_minus = MonteCarloEngine(num_sims, self.num_steps, seed=42)
        T_minus = max(self.T - bump, 0)
        price_minus = temp_engine_minus.price_asian(
            self.S, self.K, T_minus, self.r, self.sigma, self.q,
            self.option_type, self.average_type
        )

        return (price_minus - price_current) / bump

    def rho(self, bump=0.01, num_sims=100000):
        """Calculate Rho: ∂V/∂r."""
        np.random.seed(42)
        Z = np.random.standard_normal((num_sims, self.num_steps))

        paths_up = self._generate_paths_for_greek(Z, self.S, self.r + bump, self.sigma)
        paths_down = self._generate_paths_for_greek(Z, self.S, self.r - bump, self.sigma)

        price_up = self._price_from_paths(paths_up)
        price_down = self._price_from_paths(paths_down)

        return (price_up - price_down) / (2 * bump)

    def get_all_greeks(self):
        """Calculate all Greeks and return as dictionary."""
        return {
            'delta': self.delta(),
            'gamma': self.gamma(),
            'vega': self.vega(),
            'theta': self.theta(),
            'rho': self.rho()
        }