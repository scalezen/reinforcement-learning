import torch
from gbm import simulate_gbm_vectorised
from StoppingNet import price_final_date_only, train_stopping_net, StoppingNet


def test_stopping_net():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 1
    n_paths = 1000
    seed = 1

    K = 100

    S_T = simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths, seed=None)
    S_T = torch.tensor(S_T[:, -1], dtype=torch.float32).reshape((n_paths, 1))

    # print(f"shape of tensor S_T: {S_T.shape}")

    payoff = torch.tensor(np.maximum(K - S_T, 0.0), dtype=torch.float32)
    continuation = torch.zeros(n_paths, dtype=torch.float32)

    train_stopping_net(
        StoppingNet(d=1), S_T, payoff, continuation, 0.3, n_epochs=10, batch=10
    )


def test_price_final_date_only():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 1
    n_paths = 1000
    seed = 1

    K = 100

    mean_payoff, se = StoppingNet.price_final_date_only(
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
    n_steps = 1
    n_paths = 1000
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

    # print(f"mean_payoff with seed zero {mean_payoff}")
    # print(f"mean_payoff with seed one {mean_payoff_one}")
    assert abs(mean_payoff - mean_payoff_one) < 3 * (
        se + se_one
    ), "Payoff computed from trained network is not within 3*se of expected value."

def test_continuation_value():
    S0 = 100
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_steps = 1
    n_paths = 1000
    K = 100
    continuation_val = 20.0 #None # 20.0

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

    print(f"mean_payoff = {mean_payoff}")


if __name__ == "__main__":
    # test_stopping_net()

    #test_price_final_date_only()

    # test_price_final_date_different_seeds()

    test_continuation_value()