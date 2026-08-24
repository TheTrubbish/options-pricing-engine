import numpy as np


def monte_carlo_price(S, K, T, r, sigma, simulations, option_type= "call", seed = None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(simulations)
    stock_prices = S * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(stock_prices - K, 0)
    if option_type == 'put':
        payoffs = np.maximum(K - stock_prices, 0)
    discounted_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(simulations)
    lower = float(price - 1.96 * standard_error)
    upper = float(price + 1.96 * standard_error)
    return price, standard_error, (lower, upper)
