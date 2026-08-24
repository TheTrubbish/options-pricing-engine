import numpy as np

from src.binomial import binomial_price, binomial_price_one_step


def test_one_step_matches_general_binomial():
    args = (100, 100, 1, 0.05, 0.20)
    assert np.isclose(
        binomial_price_one_step(*args, option_type="call"),
        binomial_price(*args, steps=1, option_type="call"),
    )


def test_european_binomial_converges_to_black_scholes():
    from src.black_scholes import call_price

    bs = call_price(100, 100, 1, 0.05, 0.20)
    tree = binomial_price(100, 100, 1, 0.05, 0.20, 1000, option_type="call")

    assert abs(tree - bs) < 0.02


def test_american_put_is_at_least_european_put():
    european = binomial_price(
        100, 100, 1, 0.05, 0.20, 500, option_type="put", exercise="european"
    )
    american = binomial_price(
        100, 100, 1, 0.05, 0.20, 500, option_type="put", exercise="american"
    )

    assert american >= european
