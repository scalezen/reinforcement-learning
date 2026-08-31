import math
from loguru import logger

import torch
from gbm import simulate_gbm_vectorised
from StoppingNet import (
    price_final_date_only,
    train_stopping_net,
    StoppingNet,
    price_bermudan_two_dates,
)


def test_stopping_net():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 1
    n_paths = 1000
    seed = 1

    K = 100

    S_T = simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths, seed=seed)
    S_T = torch.tensor(S_T[:, -1], dtype=torch.float32).reshape((n_paths, 1))

    logger.info(f"shape of tensor S_T: {S_T.shape}")

    payoff = torch.clamp(K - S_T, min=0.0)
    continuation = torch.zeros(n_paths, dtype=torch.float32)

    train_stopping_net(
        StoppingNet(d=1), S_T, payoff, continuation, 0.3, n_epochs=10, batch=10
    )


def test_price_final_date_only():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    seed = 1

    K = 100

    mean_payoff, se = price_final_date_only(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths_train=100_000,
        n_paths_eval=200_000,
        seed=seed,
    )

    assert (
        abs(mean_payoff - 5.5735) < 3 * se
    ), "Payoff computed from trained network is not within 3*se of expected value."


def test_price_final_date_different_seeds():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100

    seed = 0
    mean_payoff, se = price_final_date_only(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths_train=100_000,
        n_paths_eval=200_000,
        seed=seed,
    )

    seed = 1
    mean_payoff_one, se_one = price_final_date_only(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths_train=100_000,
        n_paths_eval=200_000,
        seed=seed,
    )

    logger.info(f"mean_payoff with seed zero {mean_payoff}")
    logger.info(f"mean_payoff with seed one {mean_payoff_one}")

    assert abs(mean_payoff - mean_payoff_one) < 3 * (
        se + se_one
    ), "Payoff computed from trained network is not within 3*se of expected value."


def test_continuation_value():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100
    continuation_val = 20.0  # None # 20.0

    seed = 0
    mean_payoff, se = price_final_date_only(
        S0=S0,
        K=K,
        continuation_val=continuation_val,
        r=r,
        sigma=sigma,
        T=T,
        n_paths_train=100_000,
        n_paths_eval=200_000,
        seed=seed,
    )

    logger.info(f"mean_payoff = {mean_payoff}")

    assert math.isclose(
        mean_payoff, continuation_val
    ), "mean_payoff doesn't match continuation value, net training is off."


def test_price_bermudan_two_dates():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    K = 100

    mean_price, se = price_bermudan_two_dates(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths_train=300_000,
        n_paths_eval=300_000,
        seed_train=0,
        seed_eval=1,
        n_epochs=50,
    )

    # TODO - implement black-scholes put and show bermudan pv > european pv.
    logger.info(f"mean_price = {mean_price}")
    assert (
        mean_price > 5.3
    ), f"! bermudan PV: {mean_price} > european PV), check network stopping condition."


if __name__ == "__main__":
    # DEBUGGING only - force run tests
    # test_stopping_net()

    # test_price_final_date_only()

    # test_price_final_date_different_seeds()

    # test_continuation_value()

    test_price_bermudan_two_dates()
