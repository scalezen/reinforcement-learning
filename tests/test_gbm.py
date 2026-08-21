import numpy as np
import math
import random

import pytest

from gbm import (
    simulate_gbm,
    simulate_gbm_vectorised,
    european_call_mc,
    black_scholes_call,
)


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


def test_bs_call(gbm_params):
    # Tests, with S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=200_000:
    bs_call_price = black_scholes_call(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
    print(f"bs_call_price = {bs_call_price}")
    assert math.isclose(bs_call_price, 10.4506, abs_tol=1e-4)


def test_mc_call():

    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0

    n_paths = 200_000
    n_steps = int(T / 0.002)
    seed = 1

    mc_call_price, mc_std_error = european_call_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    print(f"mc price = {mc_call_price}")
    print(f"mc_std_error = {mc_std_error}")
    print(f"bs price = {bs_price}")

    print(f"se/price = {mc_std_error/mc_call_price}")

    assert (
        abs(mc_call_price - bs_price) < 3.0 * mc_std_error
    ), "MC price and BS price don't agree."

    # fraction of paths ending out of the money
    paths = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )
    payoffs = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
    n_positive = np.sum(payoffs > 0.0)

    print(f"fraction of paths in the money: {n_positive/n_paths}")


def test_oom():

    S0 = 100
    K = 150
    r = 0.05
    sigma = 0.2
    T = 1.0

    n_paths = 200_000
    n_steps = int(T / 0.002)
    seed = 1

    mc_call_price, mc_std_error = european_call_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T, n_paths=n_paths, n_steps=n_steps, seed=seed
    )

    bs_price = black_scholes_call(S0=S0, K=K, r=r, sigma=sigma, T=T)

    print(f"mc price = {mc_call_price}")
    print(f"mc_std_error = {mc_std_error}")
    print(f"bs price = {bs_price}")

    assert math.isclose(
        bs_price, 0.3599, abs_tol=1e-3
    ), "Unexpected price for out of the money option"
    assert (
        abs(mc_call_price - bs_price) < 3.0 * mc_std_error
    ), "MC price and BS price don't agree."

    print(f"se/price = {mc_std_error/mc_call_price}")

    # fraction of paths ending out of the money
    paths = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )
    payoffs = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
    n_positive = np.sum(payoffs > 0.0)

    print(f"fraction of paths that ended up in the money: {n_positive/n_paths}")


def test_put_call_parity():

    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = int(T / 0.002)
    n_paths = 200_000
    seed = 0

    # fraction of paths ending out of the money
    paths = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )

    # assuming payoff is max(S-K, 0.0), evaluate on the simulated paths
    call_payoffs = np.exp(-r * T) * np.maximum(paths[:, -1] - K, 0.0)
    mean_call_price = np.mean(call_payoffs)

    # assuming payoff is max(S-K, 0.0), evaluate on the simulated paths
    put_payoffs = np.exp(-r * T) * np.maximum(K - paths[:, -1], 0.0)
    mean_put_price = np.mean(put_payoffs)

    # check put call parity
    lhs = mean_call_price - mean_put_price
    rhs = S0 - K * np.exp(-r * T)
    print(f"put call parity, lhs, C-P= {lhs}")
    print(f"put call parity, rhs, (S_0-K*exp(-rT)) = {rhs}")

    standard_error = np.std(paths[:, -1], ddof=1) / np.sqrt(n_paths)
    assert abs(lhs - rhs) < 3 * standard_error, "Put-call parity doesn't hold"
    print(f"put call parity - test I passed.")

    # check put call parity - II
    rhs = np.exp(-r * T) * (np.mean(paths[:, -1]) - K)  # exp(-rT)*(S_T - K)
    print(f"lhs = {lhs}, rhs = {rhs}")
    assert abs(lhs - rhs) < 1e-12, "Put-call parity doesn't hold"

    print(f"put call parity - test II passed.")


def test_sanity_analytical_gbm():
    S0 = 100
    K = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = int(T / 0.002)
    n_paths = 200_000
    seed = 0

    # fraction of paths ending out of the money
    paths = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=n_steps, n_paths=n_paths, seed=seed
    )
    S_T = np.mean(paths[:, -1])

    paths_single_step = simulate_gbm_vectorised(
        S0=S0, r=r, sigma=sigma, T=T, n_steps=1, n_paths=n_paths, seed=seed
    )
    S_T_single_step = np.mean(paths_single_step[:, 1])

    print(f"S_T = {S_T}, S_T_single_step = {S_T_single_step}")

    statistical_sig = 1e-4  # TODO: calculate
    assert (
        np.abs(S_T - S_T_single_step) < statistical_sig
    ), "Number of time-steps seems to matter for analytical gbm."


if __name__ == "__main__":
    # force run tests

    # test_mc_call()
    # print("done running test_mc_call")

    # test_oom()
    # print("done running test_oom")

    # test_put_call_parity()
    # print("done running test_put_call_parity")

    test_sanity_analytical_gbm()
