import numpy as np
import pytest

import splitmix64_py as sm

# Same known-answer sequence as the single-threaded SplitMix64 tests:
# SplitMix64Parallel is defined to reproduce SplitMix64's exact sequence.
KNOWN_ANSWER_SEED_ZERO = [
    16294208416658607535,
    7960286522194355700,
    487617019471545679,
    17909611376780542444,
    1961750202426094747,
]


def test_rejects_zero_threads():
    with pytest.raises(ValueError):
        sm.SplitMix64Parallel(seed=0, n_threads=0)


def test_next_batch_zero_returns_empty_array():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=4)
    assert rng.next_batch(0).tolist() == []


def test_next_uniform_zero_returns_empty_array():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=4)
    assert rng.next_uniform(0).tolist() == []


def test_next_batch_rejects_negative_n():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=4)
    with pytest.raises(ValueError):
        rng.next_batch(-1)


def test_next_uniform_rejects_negative_n():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=4)
    with pytest.raises(ValueError):
        rng.next_uniform(-1)


def test_n_threads_property():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=6)
    assert rng.n_threads == 6


def test_next_batch_dtype_and_shape():
    rng = sm.SplitMix64Parallel(seed=1, n_threads=4)
    batch = rng.next_batch(37)
    assert batch.dtype == np.uint64
    assert batch.shape == (37,)


def test_next_uniform_dtype_shape_and_range():
    rng = sm.SplitMix64Parallel(seed=2, n_threads=4)
    u = rng.next_uniform(100_000)
    assert u.dtype == np.float64
    assert u.shape == (100_000,)
    assert np.all(u >= 0.0) and np.all(u < 1.0)


def test_matches_known_answer_sequence():
    rng = sm.SplitMix64Parallel(seed=0, n_threads=4)
    assert rng.next_batch(5).tolist() == KNOWN_ANSWER_SEED_ZERO


@pytest.mark.parametrize("n_threads", [1, 2, 3, 4, 8, 16])
def test_single_call_matches_plain_splitmix64_regardless_of_thread_count(n_threads):
    seed = 123
    n = 1000

    parallel = sm.SplitMix64Parallel(seed=seed, n_threads=n_threads)
    got = parallel.next_batch(n).tolist()

    plain = sm.SplitMix64(seed=seed)
    expected = [plain.next() for _ in range(n)]

    assert got == expected


def test_multiple_calls_with_uneven_sizes_match_one_plain_sequence():
    seed = 7
    sizes = [3, 1, 4, 1, 5, 9, 2, 6]  # not multiples of n_threads below

    rng = sm.SplitMix64Parallel(seed=seed, n_threads=5)
    got = []
    for n in sizes:
        got.extend(rng.next_batch(n).tolist())

    plain = sm.SplitMix64(seed=seed)
    expected = [plain.next() for _ in range(sum(sizes))]

    assert got == expected


def test_next_uniform_matches_plain_next_double_sequence():
    seed = 3
    n = 500

    rng = sm.SplitMix64Parallel(seed=seed, n_threads=4)
    got = rng.next_uniform(n).tolist()

    plain = sm.SplitMix64(seed=seed)
    expected = [plain.next_double() for _ in range(n)]

    assert got == expected
