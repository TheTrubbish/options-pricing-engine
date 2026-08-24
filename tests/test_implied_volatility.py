import numpy as np

from src.black_scholes import call_price, put_price
from src.implied_volatility import (
    implied_volatility_bisection,
    implied_volatility_newton_raphson,
)


def test_bisection_recovers_call_volatility():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    market_price = call_price(S, K, T, r, sigma)

    recovered, _ = implied_volatility_bisection(
        market_price, S, K, T, r, option_type="call"
    )

    assert np.isclose(recovered, sigma, atol=1e-8)


def test_newton_recovers_put_volatility():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    market_price = put_price(S, K, T, r, sigma)

    recovered, _ = implied_volatility_newton_raphson(
        market_price, S, K, T, r, option_type="put"
    )

    assert np.isclose(recovered, sigma, atol=1e-8)
