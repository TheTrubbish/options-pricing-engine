import numpy as np

from src.black_scholes import call_price, put_price


def test_known_black_scholes_prices():
    call = call_price(100, 100, 1, 0.05, 0.20)
    put = put_price(100, 100, 1, 0.05, 0.20)

    assert np.isclose(call, 10.45058357, atol=1e-8)
    assert np.isclose(put, 5.57352602, atol=1e-8)


def test_put_call_parity():
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20
    call = call_price(S, K, T, r, sigma)
    put = put_price(S, K, T, r, sigma)

    assert np.isclose(call - put, S - K * np.exp(-r * T))
