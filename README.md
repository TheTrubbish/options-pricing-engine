# Options Pricing & Quantitative Derivatives Engine

A Python project implementing and validating several standard derivatives-pricing
techniques.

## Current models

- Black-Scholes pricing for European calls and puts
- Analytical Greeks: Delta, Gamma, Vega, Theta and Rho
- Implied volatility using bisection and Newton-Raphson
- Cox-Ross-Rubinstein (CRR) binomial pricing
- European and American options using binomial backward induction
- Monte Carlo pricing under risk-neutral geometric Brownian motion

## Validation

The project includes tests for:

- Black-Scholes known values and put-call parity
- Greeks and finite-difference checks
- Implied-volatility round trips
- Binomial convergence toward Black-Scholes
- American-put value versus European-put value
- Monte Carlo reproducibility and confidence intervals

## Project structure

```text
options-pricing-engine/
├── src/
│   ├── black_scholes.py
│   ├── greeks.py
│   ├── implied_volatility.py
│   ├── binomial.py
│   └── monte_carlo.py
├── tests/
├── notebooks/
├── data/
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

```bash
python -m venv .venv
```

Activate the environment, then:

```bash
pip install -r requirements.txt
```

Run tests from the project root:

```bash
pytest
```

## Notes

Monte Carlo pricing currently handles European vanilla options. Future work can add
variance reduction, path-dependent products, market option-chain data and implied
volatility surfaces.
