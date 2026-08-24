import numpy as np

from .black_scholes import call_price, put_price
from .greeks import vega


def _option_price(S, K, T, r, sigma, option_type):
    option_type = option_type.lower()
    if option_type == "call":
        return call_price(S, K, T, r, sigma)
    if option_type == "put":
        return put_price(S, K, T, r, sigma)
    raise ValueError("option_type must be 'call' or 'put'")


def implied_volatility_bisection(market_price, S, K, T, r, option_type = "call", tolerance = 1e-10, max_iterations = 1000, lower_bound = 0.0, upper_bound = 1.0):
    low = lower_bound
    high = upper_bound

    low_price = _option_price(S, K, T, r, low, option_type)
    high_price = _option_price(S, K, T, r, high, option_type)

    if market_price < low_price or market_price > high_price:
        raise ValueError("Market price is outside the supplied volatility bracket.")

    for iterations in range(1, max_iterations + 1):
        mid = (low + high) / 2
        model_price = _option_price(S, K, T, r, mid, option_type)
        error = model_price - market_price

        if abs(error) < tolerance:
            return mid, iterations

        if model_price < market_price:
            low = mid
        else:
            high = mid

    raise RuntimeError("Bisection did not converge within max_iterations.")


def implied_volatility_newton_raphson(market_price, S, K, T, r, option_type = "call", initial_guess = 0.2, tolerance = 1e-10, max_iterations = 1000, min_vega = 1e-12):
    sigma = initial_guess

    for iterations in range(1, max_iterations + 1):
        model_price = _option_price(S, K, T, r, sigma, option_type)
        error = model_price - market_price

        if abs(error) < tolerance:
            return sigma, iterations

        current_vega = vega(S, K, T, r, sigma)
        if current_vega < min_vega:
            raise RuntimeError("Vega is too small for a stable Newton-Raphson step.")

        sigma -= error / current_vega

        if sigma <= 0:
            raise RuntimeError("Newton-Raphson produced a non-positive volatility.")

    raise RuntimeError("Newton-Raphson did not converge within max_iterations.")
