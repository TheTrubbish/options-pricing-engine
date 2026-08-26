from src.monte_carlo import monte_carlo_price, monte_carlo_antithetical
from src.black_scholes import call_price, put_price

S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

def test_reproducibility():
    price1 = monte_carlo_price(S, K, T, r, sigma, 10000, seed=100)[0]
    price2 = monte_carlo_price(S, K, T, r, sigma, 10000, seed=100)[0]
    assert price1 == price2

def test_monte_carlo_non_negative():
    price_call = monte_carlo_price(S, K, T, r, sigma, 10000)[0]
    price_put = monte_carlo_price(S, K, T, r, sigma, 10000, 'put')[0]
    assert price_call >=0 and price_put >=0

def test_monte_carlo_call_approximately_matches_black_scholes():
    mc_price = monte_carlo_price(S, K, T, r, sigma, simulations=100_000, option_type="call")[0]
    bs_price = call_price(S, K, T, r, sigma)
    assert abs(mc_price - bs_price) < 0.25


def test_monte_carlo_put_approximately_matches_black_scholes():
    mc_price = monte_carlo_price(S, K, T, r, sigma, simulations=100_000, option_type="put", seed=42)[0]
    bs_price = put_price(S, K, T, r, sigma)
    assert abs(mc_price - bs_price) < 0.25

def test_reproducibility_antithetical():
    price1 = monte_carlo_antithetical(S, K, T, r, sigma, 10000, seed=100)[0]
    price2 = monte_carlo_antithetical(S, K, T, r, sigma, 10000, seed=100)[0]
    assert price1 == price2

