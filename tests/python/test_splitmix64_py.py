import numpy as np
import pytest

import splitmix64_py as sm

# Same sequence as C++ gtest
KNOWN_ANSWER_SEED_ZERO = [
    16294208416658607535,
    7960286522194355700,
    487617019471545679,
    17909611376780542444,
    1961750202426094747,
]

def test_next_batch_zero_returns_empty_array():
    rng = sm.SplitMix64(seed=0)
    assert rng.next_batch(0).tolist() == []


def test_next_uniform_zero_returns_empty_array():
    rng = sm.SplitMix64(seed=0)
    assert rng.next_uniform(0).tolist() == []


def test_next_batch_rejects_negative_n():
    rng = sm.SplitMix64(seed=0)
    with pytest.raises(ValueError):
        rng.next_batch(-1)


def test_next_uniform_rejects_negative_n():
    rng = sm.SplitMix64(seed=0)
    with pytest.raises(ValueError):
        rng.next_uniform(-1)

def test_next_batch_dtype_and_shape():
    rng = sm.SplitMix64(seed=1)
    batch = rng.next_batch(10)
    assert batch.dtype == np.uint64
    assert batch.shape == (10,)

def test_next_uniform_dtype_and_range():
    rng = sm.SplitMix64(seed=2)
    u = rng.next_uniform(100_000)
    assert u.dtype == np.float64
    assert u.shape == (100_000,)
    assert np.all(u >= 0.0) and np.all(u < 1.0)

def test_next_batch_matches_known_answer_sequence():
    rng = sm.SplitMix64(seed=0)
    batch = rng.next_batch(5)
    assert batch.tolist() == KNOWN_ANSWER_SEED_ZERO

def test_next_batch_matches_sequential_next_calls():
    seed = 123
    n = 1000

    rng_single = sm.SplitMix64(seed=seed)
    sequential = [rng_single.next() for _ in range(n)]

    rng_batch = sm.SplitMix64(seed=seed)
    batch = rng_batch.next_batch(n)

    assert batch.tolist() == sequential

def test_next_uniform_matches_sequential_next_double_calls():
    seed = 456
    n = 1000

    rng_single = sm.SplitMix64(seed=seed)
    sequential = [rng_single.next_double() for _ in range(n)]

    rng_batch = sm.SplitMix64(seed=seed)
    batch = rng_batch.next_uniform(n)

    assert batch.tolist() == sequential

def test_batch_calls_continue_the_same_stream_as_single_calls():
    seed = 7

    rng_single = sm.SplitMix64(seed=seed)
    first = rng_single.next()
    rest = [rng_single.next() for _ in range(4)]

    rng_batch = sm.SplitMix64(seed=seed)
    rng_batch.next()  # advance by one, mirroring `first` above
    batch_rest = rng_batch.next_batch(4)

    assert batch_rest.tolist() == rest


