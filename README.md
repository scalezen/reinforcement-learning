# reinforcement-learning

**Optimal stopping decision.** Optimal stopping approached from two directions: a deep
stopping network in Python that prices Bermudan options, and Q-learning in C++ on
LibTorch. Same underlying question — *is the value of acting now greater than the
value of waiting?* — once as a pricing problem with a closed-form answer to check
against, once as a control problem with none.

---

## Track 1 — Deep optimal stopping (Python)

`python/StoppingNet.py` implements the network of
[Becker, Cheridito & Jentzen (2019)](https://arxiv.org/abs/1804.05394). One small
MLP per exercise date maps the state to a *soft* stopping probability
`g(S) ∈ [0,1]`, trained to maximise

```
E[ f·g(S) + c·(1 − g(S)) ]        f = immediate payoff, c = continuation value
```

`g` is smooth, so the expectation is
differentiable and Adam optimizer is works. At evaluation time `g` is converted to a binary stopping decision
`stop ⟺ g(S) > 0.5`, and the price is taken on a **new** set of brownian motion paths — training
and evaluation never share randomness, which is what keeps the estimator from
being biased high by its own fitted noise.

`python/gbm.py` supplies the paths (loop and vectorised implementations, kept side by
side for performance comparison) plus the Black–Scholes reference the tests measure against.

### Benchmarks

S₀ = K = 100, r = 5%, σ = 20%, T = 1.

| Quantity | Value | Source |
|---|---|---|
| European call | 10.4506 | closed form; Monte Carlo agrees within 3 s.e. |
| European put | 5.5735 | put–call parity — the *no early exercise* floor |
| Bermudan put, exercisable at {0.5, 1.0} | **5.8387** | numerical integration |
| Early exercise premium | 0.2652 | the part a stopping rule has to earn |
| Exercise boundary at t = 0.5 | **S\* = 90.38** | what the network discovers unaided |

## Track 2 — Q-learning (C++ / LibTorch)

A 4×4 grid world with a goal, a trap, and a per-step penalty.

- `cpp/q_learning_tabular.cpp` — Starter code. A 16×4 table, ε-greedy,
  a Bellman update per step. Small enough that the optimal policy can be read
  off by hand and compared.
- `cpp/q_learning.cpp` — the same problem as DQN. One-hot state encoding, an MLP
  (`cpp/include/DQN.h`), a circular replay buffer (`cpp/include/ReplayBuffer.h`), a target
  network synced every 500 steps, and ε annealed 1.0 → 0.01.

The tabular version exists to check the DQN. A DQN
on a 16-state problem is overkill — here the focus is on the
machinery (replay, target network, off-policy sampling).

Runs on Apple Silicon MPS when present, CPU otherwise.

---

## Quick start

**Python**

```bash
conda env create -f environment.yaml && conda activate ml_env
pytest -m "not slow"      # GBM and Monte Carlo — a few minutes
pytest                    # everything, including network training
```

**C++**

LibTorch is not committed. Unpack it into `./libtorch` (CMake finds it there) or
point CMake elsewhere:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release   # or -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Layout

```
python/
  StoppingNet.py           deep optimal stopping: network, training loop, pricers
  gbm.py                   GBM paths (loop + vectorised), MC pricer, Black-Scholes
cpp/
  q_learning_tabular.cpp   tabular Q-learning baseline
  q_learning.cpp           DQN on the same grid world
  splitmix64.cpp           SplitMix64 seed-expansion RNG (CI-only executable)
  include/DQN.h            MLP: 16 -> 64 -> 64 -> 4
  include/ReplayBuffer.h   fixed-capacity circular buffer with uniform sampling
  include/SplitMix64.h
tests/
  python/                  pytest; every assertion is against a closed form or an s.e.
  cpp/                     gtest — dqn_test, splitmix64_test
```

## TOFIXs

- **The two-date Bermudan test condition is too relaxed.** It asserts `price > 5.3`, but the
  European value is 5.5735 — so a network that never exercises early still passes.
  It should assert against 5.8387 within a few standard errors, which is the only
  version that actually tests the stopping rule.
- **Only two exercise dates.** The backward induction generalises to a full
  Bermudan schedule; the loop over dates is not written yet.
- **One dimension.** `StoppingNet` takes `d` as a parameter and the BCJ
  architecture is built for high `d` — currently, `d = 1`.
- **C++ tests:** TODO
- **No Longstaff–Schwartz comparison.** The regression approach is the obvious
  benchmark for the network and it isn't here yet.

