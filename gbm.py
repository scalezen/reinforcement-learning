import numpy as np
import math
import random

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


