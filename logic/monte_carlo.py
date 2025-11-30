"""
Monte Carlo Engine for option pricing.

This module provides Monte Carlo simulation capabilities for pricing various
types of options including European, American, Asian, and Barrier options.
"""
import numpy as np
from .black_scholes import BlackScholesModel


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for option pricing.

    This class handles path generation and pricing for various option types
    using Monte Carlo simulation. It supports both standard pricing and
    Common Random Numbers (CRN) for Greeks calculation.

    Attributes
    ----------
    num_simulations : int
        Number of simulation paths
    num_steps : int
        Number of time steps per path
    """

    def __init__(self, num_simulations=10000, num_steps=252, seed=None):
        """
        Initialize Monte Carlo engine.

        Parameters
        ----------
        num_simulations : int, optional
            Number of paths to simulate (default: 10000)
        num_steps : int, optional
            Time steps per path (default: 252)
        seed : int, optional
            Random seed for reproducibility (default: None)
        """
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        if seed is not None:
            np.random.seed(seed)

    #Path Generation

    def simulate_paths(self, S0, T, r, sigma, q=0):
        """
        Generate price paths using Black-Scholes dynamics.

        Parameters
        ----------
        S0 : float
            Initial stock price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        q : float, optional
            Dividend yield (default: 0)

        Returns
        -------
        ndarray
            Simulated price paths (num_simulations x num_steps+1)
        """
        return BlackScholesModel.simulate_paths(
            S0, T, r, sigma, q, self.num_simulations, self.num_steps
        )

    def _generate_paths_from_randoms(self, Z, S0, T, r, sigma, q):
        """
        Generate paths from pre-generated random numbers (for CRN).

        This method is used for Greeks calculation to ensure the same
        random numbers are used across different parameter values.

        Parameters
        ----------
        Z : ndarray
            Pre-generated standard normal randoms (num_sims x num_steps)
        S0 : float
            Initial stock price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        q : float
            Dividend yield

        Returns
        -------
        ndarray
            Simulated price paths (num_sims x num_steps+1)
        """
        dt = T / self.num_steps
        num_sims = Z.shape[0]
        paths = np.zeros((num_sims, self.num_steps + 1))
        paths[:, 0] = S0

        for t in range(1, self.num_steps + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(
                (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
            )

        return paths

    # LSM

    def _lsm_pricing(self, paths, K, r, T, option_type):
        """
        Price American option using Least Squares Monte Carlo (LSM).

        Implements the Longstaff-Schwartz algorithm for American option pricing.

        Parameters
        ----------
        paths : ndarray
            Simulated price paths
        K : float
            Strike price
        r : float
            Risk-free rate
        T : float
            Time to maturity
        option_type : str
            'call' or 'put'

        Returns
        -------
        float
            Option price
        """
        dt = T / self.num_steps

        # Calculate intrinsic values
        if option_type.lower() == 'call':
            intrinsic_value = np.maximum(paths - K, 0)
        else:
            intrinsic_value = np.maximum(K - paths, 0)

        # Initialize with terminal payoff
        cash_flows = intrinsic_value[:, -1].copy()

        # Backward induction
        for t in range(self.num_steps - 1, 0, -1):
            discounted_cf = cash_flows * np.exp(-r * dt)
            itm = intrinsic_value[:, t] > 0

            if np.sum(itm) > 0:
                # Regression for continuation value
                X = paths[itm, t]
                Y = discounted_cf[itm]
                regression = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(regression, X)

                # Exercise decision
                exercise = intrinsic_value[itm, t] > continuation_value
                cash_flows[itm] = np.where(
                    exercise,
                    intrinsic_value[itm, t],
                    discounted_cf[itm]
                )

        return np.exp(-r * dt) * np.mean(cash_flows)

    # European

    def price_european(self, S0, K, T, r, sigma, q, option_type):
        """Price European option using Monte Carlo."""
        paths = self.simulate_paths(S0, T, r, sigma, q)
        ST = paths[:, -1]

        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    # American

    def price_american(self, S0, K, T, r, sigma, q, option_type):
        """Price American option using LSM algorithm."""
        paths = self.simulate_paths(S0, T, r, sigma, q)
        return self._lsm_pricing(paths, K, r, T, option_type)

    def price_american_with_randoms(self, Z, S0, K, T, r, sigma, q, option_type):
        """Price American option with pre-generated randoms (for CRN)."""
        paths = self._generate_paths_from_randoms(Z, S0, T, r, sigma, q)
        return self._lsm_pricing(paths, K, r, T, option_type)

    # Asian

    def price_asian(self, S0, K, T, r, sigma, q, option_type, average_type='arithmetic'):
        """Price Asian option with path-dependent payoff."""
        paths = self.simulate_paths(S0, T, r, sigma, q)
        return self._calculate_asian_payoff(paths, K, T, r, option_type, average_type)

    def price_asian_with_randoms(self, Z, S0, K, T, r, sigma, q, option_type,
                                 average_type='arithmetic'):
        """Price Asian option with pre-generated randoms (for CRN)."""
        paths = self._generate_paths_from_randoms(Z, S0, T, r, sigma, q)
        return self._calculate_asian_payoff(paths, K, T, r, option_type, average_type)

    def _calculate_asian_payoff(self, paths, K, T, r, option_type, average_type):
        """Calculate Asian option payoff from paths."""
        # Calculate average
        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:  # geometric
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))

        # Calculate payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    # Barrier

    def price_barrier(self, S0, K, T, r, sigma, q, option_type,
                      barrier_type, barrier_level):
        """Price Barrier option with knock-in/knock-out features."""
        paths = self.simulate_paths(S0, T, r, sigma, q)
        return self._calculate_barrier_payoff(
            paths, K, T, r, option_type, barrier_type, barrier_level
        )

    def price_barrier_with_randoms(self, Z, S0, K, T, r, sigma, q, option_type,
                                   barrier_type, barrier_level):
        """Price Barrier option with pre-generated randoms (for CRN)."""
        paths = self._generate_paths_from_randoms(Z, S0, T, r, sigma, q)
        return self._calculate_barrier_payoff(
            paths, K, T, r, option_type, barrier_type, barrier_level
        )

    def _calculate_barrier_payoff(self, paths, K, T, r, option_type,
                                  barrier_type, barrier_level):
        """Calculate barrier option payoff from paths."""
        ST = paths[:, -1]

        # Check barrier condition
        knocked = self._check_barrier(paths, barrier_type, barrier_level)

        # Calculate vanilla payoff
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        # Apply barrier condition
        if 'out' in barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:  # 'in' options
            payoffs = np.where(knocked, payoffs, 0)

        return np.exp(-r * T) * np.mean(payoffs)

    def _check_barrier(self, paths, barrier_type, barrier_level):
        """Check if barrier has been hit for each path."""
        if barrier_type == 'up-and-out' or barrier_type == 'up-and-in':
            return np.max(paths, axis=1) >= barrier_level
        elif barrier_type == 'down-and-out' or barrier_type == 'down-and-in':
            return np.min(paths, axis=1) <= barrier_level
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")