import numpy as np
import torch
from torch import nn
from gbm import simulate_gbm_vectorised


class StoppingNet(nn.Module):
    """
    Decision network for one exercise date.

    Input:  (n_paths, d) state — the asset prices at this date.
    Output: (n_paths,) in [0, 1] — soft probability of stopping.

    Architecture (following Becker-Cheridito-Jentzen):
      Linear(d, d + 40) -> BatchNorm -> ReLU
      Linear(d + 40, d + 40) -> BatchNorm -> ReLU
      Linear(d + 40, 1) -> Sigmoid
    """

    def __init__(self, d):
        super().__init__()
        self._dimension = d

        # set up the network
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(d, d + 40),
            nn.ReLU(),
            nn.Linear(d + 40, d + 40),
            nn.ReLU(),
            nn.Linear(d + 40, 1),
            nn.Sigmoid(),
        )

    def forward(self, S_at_date):
        # S_at_date has dimensions n_paths x d

        logits = self.linear_relu_stack(S_at_date)
        logits = torch.squeeze(logits, -1)
        return logits


def train_stopping_net(
    net, S_at_date, curr_payoff_val, continuation_val, lr, n_epochs=50, batch=8192
):
    """
    Train one date's network to maximise expected value.

    Loss (to MINIMISE) = -mean( f * g(S) + c * (1 - g(S)) )
      f = immediate payoff at this date, shape (n_paths,)
      c = continuation_value, shape (n_paths,)
      g = net(S), the soft stopping probability

    Adam, lr=1e-3, batches of 8192.
    """
    n = S_at_date.shape[0]

    # Set-up optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()

    for epoch in range(n_epochs):
        perm = torch.randperm(n)

        for val in (0, n, batch):

            # randomised batch of data for the forward and backward pass
            indices = perm[val : val + batch]

            # Forward pass
            logits = net.forward(torch.tensor(S_at_date[indices]))
            # print(f"computed logits")

            # Initialize the loss function
            loss = -(
                curr_payoff_val[indices] * logits
                + continuation_val[indices] * (1 - logits)
            ).mean()
            # print(f"computed loss")

            # Backpropagation
            optimizer.zero_grad()
            # print(f"finished calling zero_grad() on optimizer.")

            loss.backward()
            # print(f"after calling loss.backward()")

            optimizer.step()
            # print(f"after calling optimizer.step()")

    net.eval()  # BatchNorm uses running stats
    return net


def price_final_date_only(S0, K, r, sigma, T, n_paths_train, n_paths_eval, seed):
    """
    Price a Bermudan that may only be exercised at T.

    Train the net on n_paths_train, then evaluate on a FRESH set of
    n_paths_eval paths using a HARD threshold: stop iff g(S) > 0.5.

    Returns (price, se).
    """
    n_steps = int(T / 0.002)
    S_T = simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths_train, seed=seed)
    S_T = torch.tensor(S_T[:, -1], dtype=torch.float32).reshape((n_paths_train, 1))

    # print(f"shape of training tensor S_T: {S_T.shape}")

    payoff = np.exp(-r * T) * torch.tensor(
        np.maximum(K - S_T, 0.0), dtype=torch.float32
    )
    continuation = torch.zeros(n_paths_train, dtype=torch.float32)

    trained_net = train_stopping_net(
        StoppingNet(d=1), S_T, payoff, continuation, 0.3, n_epochs=10, batch=8192
    )

    S_T_eval = simulate_gbm_vectorised(
        S0, r, sigma, T, n_steps, n_paths_eval, seed=seed
    )
    standard_error = np.std(S_T_eval[:, -1]) / np.sqrt(n_paths_eval)

    payoff_eval = np.exp(-r * T) * np.maximum(K - S_T_eval[:, -1], 0.0)
    payoff_eval = torch.tensor(payoff_eval, dtype=torch.float32).reshape(
        n_paths_eval, 1
    )

    with torch.no_grad():
        result = trained_net.forward(
            torch.tensor(S_T_eval[:, -1], dtype=torch.float32).reshape(n_paths_eval, 1)
        )
        payoff_conditioned = payoff_eval.squeeze(-1) * result

        # print(f'shape of logits result {result.shape}')
        # print(f'shape of payoff_eval result {payoff_eval.shape}')

        mean_payoff = torch.sum(payoff_conditioned) / n_paths_eval

        # print(f"mean_payoff = {mean_payoff}")

        # disagreement rate
        payoff_eval_squeezed = payoff_eval.squeeze(-1)
        positive_payoffs = payoff_eval_squeezed > 0.0
        positive_exercise = result > 0.5
        count_agreement = (positive_payoffs == positive_exercise).float().mean()

        # print(f"disagreement rate between the net and analytically = {1 - count_agreement}")

        disagreement_mask = positive_payoffs != positive_exercise
        disagreement_payoffs = payoff_eval_squeezed[disagreement_mask]

        if disagreement_payoffs.numel() > 0:
            min_disagreement = disagreement_payoffs.min().item()
            max_disagreement = disagreement_payoffs.max().item()
            print(
                f"min value of payoff where the network disagrees with analytics: {min_disagreement:.6f}"
            )
            print(
                f"max value of payoff where the network disagrees with analytics: {max_disagreement:.6f}"
            )
        else:
            print("No disagreements between the network and analytics.")

        return mean_payoff, standard_error
