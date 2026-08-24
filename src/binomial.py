import numpy as np


def binomial_price_one_step(S, K, T, r, sigma, option_type = "call"):
    delta_t = T
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - d) / (u - d)

    S_u = S * u
    S_d = S * d

    if option_type.lower() == "call":
        V_u = max(S_u - K, 0)
        V_d = max(S_d - K, 0)
    elif option_type.lower() == "put":
        V_u = max(K - S_u, 0)
        V_d = max(K - S_d, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return np.exp(-r * delta_t) * (p * V_u + (1 - p) * V_d)


def binomial_price(S, K, T, r, sigma, steps, option_type = "call", exercise = "european"):
    if steps < 1:
        raise ValueError("steps must be at least 1")

    option_type = option_type.lower()
    exercise = exercise.lower()

    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if exercise not in {"european", "american"}:
        raise ValueError("exercise must be 'european' or 'american'")

    delta_t = T / steps
    u = np.exp(sigma * np.sqrt(delta_t))
    d = np.exp(-sigma * np.sqrt(delta_t))
    p = (np.exp(r * delta_t) - d) / (u - d)
    discount = np.exp(-r * delta_t)

    stock_prices = [
        S * (u ** j) * (d ** (steps - j))
        for j in range(steps + 1)
    ]

    if option_type == "call":
        option_values = [max(stock_price - K, 0) for stock_price in stock_prices]
    else:
        option_values = [max(K - stock_price, 0) for stock_price in stock_prices]

    for i in range(steps - 1, -1, -1):
        new_values = []

        for j in range(i + 1):
            continuation = discount * (
                (1 - p) * option_values[j] + p * option_values[j + 1]
            )

            if exercise == "american":
                stock_price = S * (u ** j) * (d ** (i - j))
                if option_type == "call":
                    exercise_value = max(stock_price - K, 0)
                else:
                    exercise_value = max(K - stock_price, 0)
                value = max(continuation, exercise_value)
            else:
                value = continuation

            new_values.append(value)

        option_values = new_values

    return option_values[0]
