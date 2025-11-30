"""
Base Option class for all option types.
"""
from .monte_carlo import MonteCarloEngine


class Option:
    """
    Base class for all option types.

    This class provides common functionality and attributes shared by all option types.
    Subclasses should implement their specific pricing and Greeks calculation methods.

    Attributes
    ----------
    S : float
        Current underlying price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate
    sigma : float
        Volatility of the underlying asset
    q : float
        Continuous dividend yield (default: 0)
    option_type : str
        'call' or 'put'
    num_simulations : int
        Number of Monte Carlo simulations (default: 10000)
    num_steps : int
        Number of time steps in simulation (default: 252)
    mc_engine : MonteCarloEngine
        Monte Carlo simulation engine instance
    """

    def __init__(self, S, K, T, r, sigma, q=0, option_type='call',
                 num_simulations=10000, num_steps=252):
        """
        Initialize option with market parameters.

        Parameters
        ----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        q : float, optional
            Dividend yield (default: 0)
        option_type : str, optional
            'call' or 'put' (default: 'call')
        num_simulations : int, optional
            Number of MC simulations (default: 10000)
        num_steps : int, optional
            Time steps per simulation (default: 252)
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()
        self.num_simulations = num_simulations
        self.num_steps = num_steps

        # Initialize Monte Carlo engine
        self.mc_engine = MonteCarloEngine(num_simulations, num_steps)

    def price(self):
        """Calculate option price. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement price()")

    def get_all_greeks(self):
        """Calculate all Greeks. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_all_greeks()")