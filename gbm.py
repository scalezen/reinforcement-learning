import numpy as np
import math
import random

import scipy


def simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Simulate geometric Brownian motion under the risk-neutral measure.

    Returns array of shape (n_paths, n_steps + 1), where column 0 is S0
    for every path and column k is the price at time k * (T / n_steps).
    """

    if seed is not None:
        random.seed(seed)

    output = np.empty((n_paths, n_steps + 1), dtype=np.float32)

    step_size = T / n_steps

    output[:, 0] = S0  # initial t=0

    for step in range(1, n_steps + 1):
        prev_state = output[:, step - 1]
        w = np.random.standard_normal(n_paths)
        output[:, step] = prev_state * np.exp(
            (r - (sigma * sigma) / 2.0) * step_size + sigma * math.sqrt(step_size) * w
        )

    return output


def simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """
    Simulate geometric Brownian motion under the risk-neutral measure.

    Fully numpy vectorised.

    Returns array of shape (n_paths, n_steps + 1), where column 0 is S0
    for every path and column k is the price at time k * (T / n_steps).
    """

    if seed is not None:
        np.random.seed(seed)

    output = np.empty((n_paths, n_steps + 1), dtype=np.float32)

    step_size = T / n_steps

    w = np.random.standard_normal((n_paths, n_steps))

    output[:, 0] = S0
    output[:, 1:] = S0 * np.exp(
        np.cumsum(
            ((r - sigma * sigma / 2.0) * step_size + sigma * math.sqrt(step_size) * w),
            axis=1,
        )
    )

    return output


def european_call_mc(S0, K, r, sigma, T, n_paths, seed=None) -> dict[str, float]:
    """
    Price a European call by Monte Carlo under GBM.

    Returns (price, se): the discounted mean payoff and its standard error.

    """
    n_steps = int(T / 0.002)

    paths = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )

    # price = exp(-r*T) * mean(max(S_T - K, 0))
    payoffs = np.exp(-r * T) * np.mean(np.maximum(paths[:, -1] - K, 0.0))
    mean_price = np.mean(payoffs)

    # se    = exp(-r*T) * std(payoff, ddof=1) / sqrt(n_paths)
    std_error = np.std(payoffs, ddof=1) / np.sqrt(n_paths)

    return (mean_price, std_error)


def black_scholes_call(S0, K, r, sigma, T):
    """
    Analytic Black-Scholes call price.

    Phi is the standard normal CDF: scipy.stats.norm.cdf.
    """

    d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    C = S0 * scipy.stats.norm.cdf(d1) - K * np.exp(-r * T) * scipy.stats.norm.cdf(d2)

    return C
