import numpy as np
from scipy.stats import norm

from .black_scholes import d1, d2


def call_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma))


def put_delta(S, K, T, r, sigma):
    return norm.cdf(d1(S, K, T, r, sigma)) - 1


def gamma(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    return norm.pdf(d1(S, K, T, r, sigma)) * S * np.sqrt(T)


def call_theta(S, K, T, r, sigma):
    return (-S * norm.pdf(d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))


def put_theta(S, K, T, r, sigma):
    return (-S * norm.pdf(d1(S, K, T, r, sigma)) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)))


def call_rho(S, K, T, r, sigma):
    return K * T * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma))


def put_rho(S, K, T, r, sigma):
    return -K * T * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma))
