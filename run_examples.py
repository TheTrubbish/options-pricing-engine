from src.black_scholes import call_price, put_price
from src.greeks import call_delta, gamma, vega
from src.implied_volatility import (
    implied_volatility_bisection,
    implied_volatility_newton_raphson,
)
from src.binomial import binomial_price
from src.monte_carlo import monte_carlo_price_with_error


S, K, T, r, sigma = 100, 100, 1, 0.05, 0.20

print("Black-Scholes call:", call_price(S, K, T, r, sigma))
print("Black-Scholes put:", put_price(S, K, T, r, sigma))
print("Call delta:", call_delta(S, K, T, r, sigma))
print("Gamma:", gamma(S, K, T, r, sigma))
print("Vega:", vega(S, K, T, r, sigma))

market_price = call_price(S, K, T, r, sigma)
print(
    "Bisection IV:",
    implied_volatility_bisection(market_price, S, K, T, r, "call"),
)
print(
    "Newton IV:",
    implied_volatility_newton_raphson(market_price, S, K, T, r, "call"),
)

print(
    "Binomial European call:",
    binomial_price(S, K, T, r, sigma, 500, "call", "european"),
)
print(
    "Binomial American put:",
    binomial_price(S, K, T, r, sigma, 500, "put", "american"),
)

mc_price, se, ci = monte_carlo_price_with_error(
    S, K, T, r, sigma, 100_000, "call", seed=42
)
print("Monte Carlo call:", mc_price)
print("Monte Carlo standard error:", se)
print("Monte Carlo 95% CI:", ci)
