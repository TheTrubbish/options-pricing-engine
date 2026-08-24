import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    numerator = np.log(S / K) + (r + (sigma ** 2) / 2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator


def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    return (S * norm.cdf(d1(S, K, T, r, sigma)) - K * np.exp(-r * T) * norm.cdf(d2(S, K, T, r, sigma)))


def put_price(S, K, T, r, sigma):
    return (K * np.exp(-r * T) * norm.cdf(-d2(S, K, T, r, sigma)) - S * norm.cdf(-d1(S, K, T, r, sigma)))
