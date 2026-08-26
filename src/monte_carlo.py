import numpy as np


def monte_carlo_price(S, K, T, r, sigma, simulations, option_type= "call", seed = None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(simulations)
    stock_prices = S * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(stock_prices - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - stock_prices, 0)
    else:
        raise ValueError("option_type must be call or put")
    discounted_payoffs = np.exp(-r * T) * payoffs
    price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(simulations)
    lower = float(price - 1.96 * standard_error)
    upper = float(price + 1.96 * standard_error)
    return price, standard_error, (lower, upper)


def monte_carlo_antithetical(S, K, T, r, sigma, simulations, option_type = 'call', seed = None):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(simulations)
    stock_prices = S * np.exp((r - 0.5 * sigma**2)*T + sigma * np.sqrt(T) * Z)
    stock_prices_minus = S * np.exp((r - 0.5 * sigma**2)*T - sigma * np.sqrt(T) * Z)
    if option_type == 'call':
        payoffs = np.maximum(stock_prices - K, 0)
        payoffs_minus = np.maximum(stock_prices_minus - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - stock_prices, 0)
        payoffs_minus = np.maximum(K - stock_prices_minus, 0)
    else:
        raise ValueError("option_type must be call or put")
    averages_payoffs = (payoffs + payoffs_minus) / 2
    discounted_payoffs = np.exp(-r * T) * averages_payoffs
    price = np.mean(discounted_payoffs)
    standard_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(simulations)
    lower = float(price - 1.96 * standard_error)
    upper = float(price + 1.96 * standard_error)
    return price, standard_error, (lower, upper)

for simulation_count in [10000, 100_000, 1_000_000]:
    stored_array = []
    for _ in range(100):
        mc_single_SE = monte_carlo_price(100, 100, 1, 0.05, 0.2, simulation_count)[1]
        mc_antih_SE = monte_carlo_antithetical(100, 100, 1, 0.05, 0.2, int(simulation_count/2))[1]
        stored_array.append(1 - mc_antih_SE/mc_single_SE)
    print(f'Mean SE reduction is {np.mean(stored_array)}')