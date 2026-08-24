import math
from loguru import logger
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
            logger.info(f"computed logits")

            # Initialize the loss function
            loss = -(
                curr_payoff_val[indices] * logits
                + continuation_val[indices] * (1 - logits)
            ).mean()
            logger.info(f"computed loss")

            # Backpropagation
            optimizer.zero_grad()
            logger.info(f"finished calling zero_grad() on optimizer.")

            loss.backward()
            logger.info(f"after calling loss.backward()")

            optimizer.step()
            logger.info(f"after calling optimizer.step()")

    net.eval()  # BatchNorm uses running stats
    return net


def price_final_date_only(S0, K, r, sigma, T, n_paths_train, n_paths_eval, seed, continuation_val=None):
    """
    Price a Bermudan that may only be exercised at T.

    Train the net on n_paths_train, then evaluate on a FRESH set of
    n_paths_eval paths using a HARD threshold: stop iff g(S) > 0.5.

    Returns (price, se).
    """
    n_steps = int(T / 0.002)
    S_T = simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths_train, seed=seed)
    S_T = torch.tensor(S_T[:, -1], dtype=torch.float32).reshape((n_paths_train, 1)) # S_T is a torch tensor

    logger.info(f"shape of training tensor S_T: {S_T.shape}")

    payoff = math.exp(-r * T) * torch.clamp(K - S_T, min=0.0)

    if not continuation_val:
        continuation_val = 0.0
        continuation = torch.zeros(n_paths_train, dtype=torch.float32)
    else:
        print(f"price_final_date, got continuation value {continuation_val}")
        continuation = torch.full((n_paths_train,), continuation_val, dtype=torch.float32)

    trained_net = train_stopping_net(
        StoppingNet(d=1), S_T, payoff, continuation, 0.3, n_epochs=10, batch=8192
    )

    S_T_eval = simulate_gbm_vectorised(
        S0, r, sigma, T, n_steps, n_paths_eval, seed=seed
    )
    S_T_eval = torch.tensor(S_T_eval[:,-1], dtype=torch.float32).reshape(n_paths_eval, 1) # S_T_eval is a torch tensor
    standard_error = S_T_eval.std() / math.sqrt(n_paths_eval)

    payoff_eval = math.exp(-r * T) * torch.clamp(K - S_T_eval, min=0.0)
   
    with torch.no_grad():
        result = trained_net.forward(S_T_eval)
        exercise = (result > 0.5).float()
        continuation = torch.full((n_paths_eval,), continuation_val, dtype=torch.float32)
        payoff_conditioned = payoff_eval.squeeze(-1) * exercise + continuation.squeeze(-1) * (1 - exercise)

        # logger.info(f'shape of logits result {result.shape}')
        # logger.info(f'shape of payoff_eval result {payoff_eval.shape}')

        mean_payoff = torch.sum(payoff_conditioned) / n_paths_eval

        # logger.info(f"mean_payoff = {mean_payoff}")

        # disagreement rate
        payoff_eval_squeezed = payoff_eval.squeeze(-1)
        positive_payoffs = payoff_eval_squeezed > 0.0
        positive_exercise = result > 0.5
        count_agreement = (positive_payoffs == positive_exercise).float().mean()

        logger.info(f"disagreement rate between the net and analytically = {1 - count_agreement}")

        disagreement_mask = positive_payoffs != positive_exercise
        disagreement_payoffs = payoff_eval_squeezed[disagreement_mask]

        if disagreement_payoffs.numel() > 0:
            min_disagreement = disagreement_payoffs.min().item()
            max_disagreement = disagreement_payoffs.max().item()
            logger.info(
                f"min value of payoff where the network disagrees with analytics: {min_disagreement:.6f}"
            )
            logger.info(
                f"max value of payoff where the network disagrees with analytics: {max_disagreement:.6f}"
            )
        else:
            logger.info("No disagreements between the network and analytics.")

        return mean_payoff, standard_error
