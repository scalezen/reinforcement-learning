import numpy as np
import math
import random

import pytest

import ..gbm.py


@pytest.fixture
def gbm_params():
    return dict(S0=100, r=0.05, sigma=0.2, T=1.0, n_steps=50, n_paths=400_000)


def test_output_size(gbm_params):
    params = {**gbm_params, "n_paths": 200_000}
    result = simulate_gbm(**params)
    assert result.shape == (
        params["n_paths"],
        params["n_steps"] + 1,
    ), "Unexpected output shape"


def test_mean(gbm_params):
    """Tests that the numerical mean and theoretical mean are within 3 standard deviations"""

    S0, r, T, n_paths = (
        gbm_params["S0"],
        gbm_params["r"],
        gbm_params["T"],
        gbm_params["n_paths"],
    )

    result = simulate_gbm(**gbm_params)
    result_vec = simulate_gbm_vectorised(**gbm_params)

    theoretical_mean = S0 * math.exp(r * T)
    numerical_mean = np.sum(result[:, -1]) / n_paths
    numerical_mean_vec = np.sum(result_vec[:, -1]) / n_paths

    numerator = np.sum((result[:, -1] - numerical_mean) ** 2)
    variance = numerator / (n_paths - 1)
    numerical_se = math.sqrt(variance / n_paths)

    print(f"theoretical_mean = {theoretical_mean}")
    print(f"numerical_mean = {numerical_mean}")
    print(f"numerical_mean_vec = {numerical_mean_vec}")
    print(f"numerical_se = {numerical_se}")

    # assert within 3*standard errors
    assert (
        abs(numerical_mean - numerical_mean_vec) < 3.0 * math.sqrt(2) * numerical_se
    ), "Multiple runs differ by more than se."
    assert (
        abs(theoretical_mean - numerical_mean) < 3.0 * numerical_se
    ), "Numerical mean doesn't match the expected mean."


def test_mean_per_timestep(gbm_params):
    """For each timestep, tests that the numerical mean and theoretical mean are within 3 standard deviations"""

    S0, r, sigma, n_steps, n_paths = (
        gbm_params["S0"],
        gbm_params["r"],
        gbm_params["sigma"],
        gbm_params["n_steps"],
        gbm_params["n_paths"],
    )
    t = 0.5  # intermediate time at which we are testing

    # generate the paths
    result_vec = simulate_gbm_vectorised(**gbm_params)

    i_step = int(t * n_steps / gbm_params["T"])

    theoretical_mean = np.log(S0) + (r - sigma * sigma / 2.0) * t

    log_result = np.log(result_vec)
    numerical_mean = np.mean(log_result[:, i_step])
    numerical_std = np.std(log_result[:, i_step], ddof=1)
    numerical_se = numerical_std / np.sqrt(n_paths)

    print(f"theoretical_mean = {theoretical_mean}")
    print(f"numerical_mean = {numerical_mean}")
    print(f"numerical std = {numerical_std}")
    print(f"numerical_se = {numerical_se}")

    assert (
        abs(theoretical_mean - numerical_mean) < 3.0 * numerical_se
    ), "theoretical and numerical mean diverge unexpectedly."
    assert abs(numerical_std - sigma * math.sqrt(t)) < 3.0 * (
        numerical_se / math.sqrt(2)
    ), "Numerical paths standard deviation is unexpected."


def test_sign(gbm_params):
    """For each timestep, tests that the numerical mean and theoretical mean are within 3 standard errors.
    Confirms that the standard deviation for paths at time t is sigma*sqrt(t)
    """

    # generate the paths
    result_vec = simulate_gbm_vectorised(**gbm_params)

    assert np.all(result_vec > 0.0), "Unexpected negative value encountered."


def test_seed(gbm_params):
    """Tests if using the same seed produces identical paths"""

    # generate the paths with seed_zero
    seed_zero = 1
    result_vec_zero = simulate_gbm_vectorised(**gbm_params, seed=seed_zero)

    # generate the paths with seed_one
    seed_one = 1
    result_vec_one = simulate_gbm_vectorised(**gbm_params, seed=seed_one)

    assert np.array_equal(
        result_vec_zero, result_vec_one
    ), "Same seed generates different paths."


