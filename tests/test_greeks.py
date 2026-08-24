import numpy as np

from src.black_scholes import call_price
from src.greeks import call_delta, put_delta, gamma, vega


def test_delta_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    assert np.isclose(
        call_delta(S, K, T, r, sigma) - put_delta(S, K, T, r, sigma),
        1.0,
        atol=1e-12,
    )


def test_delta_against_finite_difference():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    h = 0.01
    numerical = (
        call_price(S + h, K, T, r, sigma)
        - call_price(S - h, K, T, r, sigma)
    ) / (2 * h)

    assert np.isclose(numerical, call_delta(S, K, T, r, sigma), atol=1e-6)


def test_gamma_positive_and_vega_positive():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    assert gamma(S, K, T, r, sigma) > 0
    assert vega(S, K, T, r, sigma) > 0
