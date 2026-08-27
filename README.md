# reinforcement-learning

[![CI](https://github.com/scalezen/reinforcement-learning/actions/workflows/ci.yml/badge.svg)](https://github.com/scalezen/reinforcement-learning/actions/workflows/ci.yml)
[![Nightly](https://github.com/scalezen/reinforcement-learning/actions/workflows/nightly.yml/badge.svg)](https://github.com/scalezen/reinforcement-learning/actions/workflows/nightly.yml)

**Optimal stopping decision.** Optimal stopping approached from two directions: a deep
stopping network in Python that prices Bermudan options, and Q-learning in C++ on
LibTorch. Same underlying question — *is the value of acting now greater than the
value of waiting?* — once as a pricing problem with a closed-form answer to check
against, once as a control problem with none.

%Every claim below is pinned to an analytic benchmark. That is the point of the
%repo: a stopping rule that cannot be checked against something exact is a stopping
%rule you cannot trust.

---

## Track 1 — Deep optimal stopping (Python)

`StoppingNet.py` implements the network of
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

`gbm.py` supplies the paths (loop and vectorised implementations, kept side by
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

%The network is never told the boundary. It has to find it from paths and payoffs,
%and the 0.2652 of premium is the entire margin between a rule that works and a
%rule that has quietly learned to never exercise early.

## Track 2 — Q-learning (C++ / LibTorch)

A 4×4 grid world with a goal, a trap, and a per-step penalty.

- `q_learning_tabular.cpp` — the honest baseline. A 16×4 table, ε-greedy,
  a Bellman update per step. Small enough that the optimal policy can be read
  off by hand and compared.
- `q_learning.cpp` — the same problem as DQN. One-hot state encoding, an MLP
  (`include/DQN.h`), a circular replay buffer (`include/ReplayBuffer.h`), a target
  network synced every 500 steps, and ε annealed 1.0 → 0.01.

The tabular version exists so the DQN has something to be wrong against. A DQN
on a 16-state problem is deliberate overkill — the interesting content is the
machinery (replay, target network, off-policy sampling), not the result.

Runs on Apple Silicon MPS when present, CPU otherwise.

---

## Quick start

**Python**

```bash
conda env create -f environment.yaml && conda activate ml_env
pytest -m "not slow"      # GBM and Monte Carlo — a few minutes
pytest                    # everything, including network training
```

`pip install -r requirements-dev.txt` also works; see the header of that file for
the CPU-only torch index.

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
StoppingNet.py        deep optimal stopping: network, training loop, pricers
gbm.py                GBM paths (loop + vectorised), MC pricer, Black-Scholes
q_learning_tabular.cpp  tabular Q-learning baseline
q_learning.cpp        DQN on the same grid world
include/DQN.h         MLP: 16 -> 64 -> 64 -> 4
include/ReplayBuffer.h  fixed-capacity circular buffer with uniform sampling
tests/                pytest; every assertion is against a closed form or an s.e.
```

## Known gaps

Listed because they are the next commits, not because they are excuses.

- **The two-date Bermudan test is too loose.** It asserts `price > 5.3`, but the
  European value is 5.5735 — so a network that never exercises early still passes.
  It should assert against 5.8387 within a few standard errors, which is the only
  version that actually tests the stopping rule.
- **Only two exercise dates.** The backward induction generalises to a full
  Bermudan schedule; the loop over dates is not written yet.
- **One dimension.** `StoppingNet` takes `d` as a parameter and the BCJ
  architecture is built for high `d` — the whole reason the method exists is
  baskets where PDE methods die. Untested above `d = 1`.
- **C++ tests assert nothing.** `test_dqn.cpp` prints `[PASS]` lines and always
  returns 0. Converting them to GoogleTest `TEST()` cases is what unlocks a
  meaningful C++ CI job; the CMake wiring is already in place, commented.
- **No Longstaff–Schwartz comparison.** The regression approach is the obvious
  benchmark for the network and it isn't here yet.

## CI

`ci.yml` runs the fast tests on every push and pull request to `main`.
`nightly.yml` runs the full suite, including the `slow`-marked training tests, at
02:00 UTC. The C++ build is not yet in CI — see the gap above.
