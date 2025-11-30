"""
Barrier Option implementation with knock-in/knock-out features.
"""
from .option import Option
import numpy as np
from scipy.stats import norm


class BarrierOption(Option):
    """
    Barrier option with knock-in or knock-out features.

    Supports four types: up-and-out, up-and-in, down-and-out, down-and-in.
    Includes closed-form pricing for specific cases (down-and-out call).

    Attributes
    ----------
    barrier_type : str
        Type of barrier: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
    barrier_level : float
        Barrier price level
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call',
                 num_simulations=10000, num_steps=252,
                 barrier_type='down-and-out', barrier_level=None):
        """
        Initialize Barrier option.

        Parameters
        ----------
        barrier_type : str, optional
            Barrier type (default: 'down-and-out')
        barrier_level : float
            Required barrier level (raises ValueError if None)

        Raises
        ------
        ValueError
            If barrier_level is not provided
        """
        super().__init__(S, K, T, r, sigma, q, option_type, num_simulations, num_steps)
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level

        if barrier_level is None:
            raise ValueError("barrier_level is required for barrier options")

    def price(self):
        """Calculate Barrier option price using Monte Carlo."""
        return self.mc_engine.price_barrier(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )

    def price_closed_form(self):
        """
        Calculate closed-form price for down-and-out call (if applicable).

        Returns
        -------
        float or None
            Closed-form price if available, None otherwise
        """
        if self.barrier_type != 'down-and-out' or self.option_type != 'call':
            return None

        H = self.barrier_level

        # Check if already knocked out
        if self.S <= H:
            return 0

        # Check if formula is valid (strike above barrier)
        if self.K <= H:
            return 0

        # Calculate auxiliary parameters
        lambda_val = (self.r - self.q + 0.5 * self.sigma ** 2) / (self.sigma ** 2)
        y = (np.log(H ** 2 / (self.S * self.K)) / (self.sigma * np.sqrt(self.T)) +
             lambda_val * self.sigma * np.sqrt(self.T))

        # Get vanilla call price
        from .black_scholes import BlackScholesModel
        bs = BlackScholesModel(self.S, self.K, self.T, self.r, self.sigma, self.q)
        vanilla_call = bs.call_price()

        # Calculate barrier correction term
        correction = (
                self.S * np.exp(-self.q * self.T) * (H / self.S) ** (2 * lambda_val) * norm.cdf(y) -
                self.K * np.exp(-self.r * self.T) * (H / self.S) ** (2 * lambda_val - 2) *
                norm.cdf(y - self.sigma * np.sqrt(self.T))
        )

        return vanilla_call - correction

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
        """Calculate Barrier option price from generated paths."""
        ST = paths[:, -1]

        # Check barrier condition
        if self.barrier_type == 'up-and-out' or self.barrier_type == 'up-and-in':
            knocked = np.max(paths, axis=1) >= self.barrier_level
        else:  # down barriers
            knocked = np.min(paths, axis=1) <= self.barrier_level

        # Calculate vanilla payoff
        if self.option_type == 'call':
            payoffs = np.maximum(ST - self.K, 0)
        else:
            payoffs = np.maximum(self.K - ST, 0)

        # Apply barrier condition
        if 'out' in self.barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:  # 'in' options
            payoffs = np.where(knocked, payoffs, 0)

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
        price_current = temp_engine_current.price_barrier(
            self.S, self.K, self.T, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
        )

        np.random.seed(42)
        temp_engine_minus = MonteCarloEngine(num_sims, self.num_steps, seed=42)
        T_minus = max(self.T - bump, 0)
        price_minus = temp_engine_minus.price_barrier(
            self.S, self.K, T_minus, self.r, self.sigma, self.q,
            self.option_type, self.barrier_type, self.barrier_level
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