import numpy as np
from .black_scholes import BlackScholesModel


class MonteCarloEngine:
    '''
    Performs all functions related to Monte Carlo simulation needed for option pricing

    Parameters
    ----------
    num_simulations: int, optional
        The number of times to randomly sample (default is 10_000)

    num_steps: int, optional
        The length of time to simulate (default is 252)

    seed: int, optional
        Set a custom rng seed (default is random)
    '''

    def __init__(self, num_simulations=10000, num_steps=252, seed=None):
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        if seed is not None:
            np.random.seed(seed)

    def simulate_paths(self, S0, T, r, sigma, q=0):
        '''
        '''
        return BlackScholesModel.simulate_paths(S0, T, r, sigma, q, self.num_simulations, self.num_steps)

    def price_european(self, S0, K, T, r, sigma, q, option_type):
        '''
        Prices a european option

        Parameters
        ----------
        S0: float
            Initial asset price
        K: float
            Strike price
        T: float
            Time till maturity (in years)
        r: float
            Risk free interest rate
        sigma:
            Asset volatility
        q: float
            Dividend yield
        option_type: str
            Call or put
        '''

        #Generate paths
        paths = self.simulate_paths(S0, T, r, sigma, q)
        ST = paths[:, -1]

        #Option type handling
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        #Calculate the price
        price = np.exp(-r * T) * np.mean(payoffs)
        return price

    def price_american(self, S0, K, T, r, sigma, q, option_type):
        '''
        Prices an american option

        Parameters
        ----------
        S0: float
            Initial asset price
        K: float
            Strike price
        T: float
            Time till maturity (in years)
        r: float
            Risk free interest rate
        sigma:
            Asset volatility
        q: float
            Dividend yield
        option_type: str
            Call or put
        '''

        #Generate paths
        paths = self.simulate_paths(S0, T, r, sigma, q)
        dt = T / self.num_steps

        #Option type handling
        if option_type.lower() == 'call':
            intrinsic_value = np.maximum(paths - K, 0)
        else:
            intrinsic_value = np.maximum(K - paths, 0)


        cash_flows = intrinsic_value[:, -1].copy()


        for t in range(self.num_steps - 1, 0, -1):

            discounted_cf = cash_flows * np.exp(-r * dt)


            itm = intrinsic_value[:, t] > 0

            if np.sum(itm) > 0:
                X = paths[itm, t]
                Y = discounted_cf[itm]

                regression = np.polyfit(X, Y, 2)
                continuation_value = np.polyval(regression, X)

                exercise = intrinsic_value[itm, t] > continuation_value

                cash_flows[itm] = np.where(exercise,
                                          intrinsic_value[itm, t],
                                          discounted_cf[itm])

        price = np.exp(-r * dt) * np.mean(cash_flows)
        return price

    def price_asian(self, S0, K, T, r, sigma, q, option_type, average_type='arithmetic'):
        '''
        Prices an asian option

        Parameters
        ----------
        S0: float
            Initial asset price
        K: float
            Strike price
        T: float
            Time till maturity (in years)
        r: float
            Risk free interest rate
        sigma:
            Asset volatility
        q: float
            Dividend yield
        option_type: str
            Call or put
        average_type: str, optional
            type of average (default is arithmetic)
        '''

        #Generate paths
        paths = self.simulate_paths(S0, T, r, sigma, q)

        #Average type handling
        if average_type == 'arithmetic':
            avg_prices = np.mean(paths, axis=1)
        else:
            avg_prices = np.exp(np.mean(np.log(paths), axis=1))
        
        #Option type handling
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        else:
            payoffs = np.maximum(K - avg_prices, 0)

        #Calculate the price
        price = np.exp(-r * T) * np.mean(payoffs)
        return price

    def price_barrier(self, S0, K, T, r, sigma, q, option_type, barrier_type, barrier_level):
        '''
        Prices a barrier option

        Parameters
        ----------
        S0: float
            Initial asset price
        K: float
            Strike price
        T: float
            Time till maturity (in years)
        r: float
            Risk free interest rate
        sigma:
            Asset volatility
        q: float
            Dividend yield
        option_type: str
            Call or put
        barrier_type: str
            
        barrier_level: str
        '''

        #Generate paths
        paths = self.simulate_paths(S0, T, r, sigma, q)
        #Set ST as final value
        ST = paths[:, -1]

        #Barrier type handling
        if barrier_type == 'up-and-out':
            knocked = np.max(paths, axis=1) >= barrier_level
        elif barrier_type == 'up-and-in':
            knocked = np.max(paths, axis=1) >= barrier_level
        elif barrier_type == 'down-and-out':
            knocked = np.min(paths, axis=1) <= barrier_level
        elif barrier_type == 'down-and-in':
            knocked = np.min(paths, axis=1) <= barrier_level
        else:
            raise ValueError(f"Unknown barrier type: {barrier_type}")

        #Option type handling
        if option_type.lower() == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)

        if 'out' in barrier_type:
            payoffs = np.where(knocked, 0, payoffs)
        else:
            payoffs = np.where(knocked, payoffs, 0)

        #Calculate the price
        price = np.exp(-r * T) * np.mean(payoffs)
        return price