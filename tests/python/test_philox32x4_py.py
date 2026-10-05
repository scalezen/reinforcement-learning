import math

import numpy as np
import pytest

import philox32x4_py as px
from gbm import simulate_gbm_vectorised


def chunked_single_thread_uint32(n, num_threads, key0, key1):
    parts = []
    for th in range(num_threads):
        begin = th * n // num_threads
        end = (th + 1) * n // num_threads
        if begin < end:
            parts.append(px.uint32_batch(end - begin, 0, th, key0, key1))
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.uint32)


def test_uint32_batch_zero_returns_empty_array():
    assert px.uint32_batch(0).tolist() == []


def test_uint32_batch_dtype_and_shape():
    batch = px.uint32_batch(37)
    assert batch.dtype == np.uint32
    assert batch.shape == (37,)


def test_float_normal_dtype_and_shape():
    batch = px.float_normal_batch(37)
    assert batch.dtype == np.float32
    assert batch.shape == (37,)


def test_double_normal_dtype_and_shape():
    batch = px.double_normal_batch(37)
    assert batch.dtype == np.float64
    assert batch.shape == (37,)


def test_size_matches_request_including_non_multiples_of_4():
    for n in range(41):
        assert px.uint32_batch(n).shape == (n,)
        assert px.float_normal_batch(n).shape == (n,)
        assert px.double_normal_batch(n).shape == (n,)


def test_rejects_negative_num_rands():
    with pytest.raises(TypeError):
        px.uint32_batch(-1)


def test_rejects_zero_threads():
    with pytest.raises(ValueError):
        px.uint32_batch_mt(10, 0)
    with pytest.raises(ValueError):
        px.double_normal_batch_mt(10, 0)


def test_defaults_match_explicit_zeros():
    assert px.uint32_batch(8).tolist() == px.uint32_batch(8, 0, 0, 0, 0).tolist()


def test_deterministic_for_same_inputs():
    a = px.uint32_batch(100, counter0_offset=5, counter1_offset=9, key0=3, key1=4)
    b = px.uint32_batch(100, counter0_offset=5, counter1_offset=9, key0=3, key1=4)
    assert a.tolist() == b.tolist()


def test_key0_changes_output():
    a = px.uint32_batch(16, key0=0, key1=7)
    b = px.uint32_batch(16, key0=1, key1=7)
    assert a.tolist() != b.tolist()


def test_key1_changes_output():
    a = px.uint32_batch(16, key0=7, key1=0)
    b = px.uint32_batch(16, key0=7, key1=1)
    assert a.tolist() != b.tolist()


def test_counter0_offset_changes_output():
    a = px.uint32_batch(16, counter0_offset=0)
    b = px.uint32_batch(16, counter0_offset=1)
    assert a.tolist() != b.tolist()


def test_counter1_offset_changes_output():
    a = px.uint32_batch(16, counter1_offset=0)
    b = px.uint32_batch(16, counter1_offset=1)
    assert a.tolist() != b.tolist()


@pytest.mark.parametrize("n_threads", [1, 2, 3, 5, 8])
def test_uint32_mt_matches_chunked_single_thread(n_threads):
    n = 1001
    got = px.uint32_batch_mt(n, n_threads, key0=11, key1=22)
    assert got.tolist() == chunked_single_thread_uint32(n, n_threads, 11, 22).tolist()


def test_uint32_mt_deterministic_across_calls():
    a = px.uint32_batch_mt(5000, 4, key0=1, key1=2)
    b = px.uint32_batch_mt(5000, 4, key0=1, key1=2)
    assert a.tolist() == b.tolist()


def test_uint32_blocks_have_no_repeats():
    batch = px.uint32_batch(200_000, counter0_offset=3, counter1_offset=4, key0=5, key1=6)
    blocks = batch.reshape(-1, 4)
    assert len(np.unique(blocks, axis=0)) == len(blocks)


def test_normals_are_finite():
    assert np.all(np.isfinite(px.float_normal_batch(100_000)))
    assert np.all(np.isfinite(px.double_normal_batch(100_000)))


def check_standard_normal_moments(x):
    n = x.size
    se_mean = 1.0 / math.sqrt(n)
    assert abs(float(x.mean())) < 4.0 * se_mean
    assert abs(float(x.var()) - 1.0) < 0.02


def test_float_normal_moments():
    check_standard_normal_moments(px.float_normal_batch(1_000_000, key0=1, key1=2).astype(np.float64))


def test_double_normal_moments():
    check_standard_normal_moments(px.double_normal_batch(1_000_000, key0=1, key1=2))


def test_double_normal_mt_moments():
    check_standard_normal_moments(px.double_normal_batch_mt(1_000_000, 4, key0=1, key1=2))


@pytest.mark.parametrize("n_threads", [1, 3, 8])
def test_double_normal_mt_matches_chunked_single_thread(n_threads):
    n = 1001
    parts = []
    for th in range(n_threads):
        begin = th * n // n_threads
        end = (th + 1) * n // n_threads
        if begin < end:
            parts.append(px.double_normal_batch(end - begin, 0, th, 9, 8))
    expected = np.concatenate(parts)
    got = px.double_normal_batch_mt(n, n_threads, key0=9, key1=8)
    assert np.array_equal(got, expected)


def test_gbm_brownian_increments_from_philox_have_correct_mean_and_vol():
    S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_steps, n_paths = 50, 50_000

    z = px.double_normal_batch_mt(n_paths * n_steps, 4, key0=1, key1=2).reshape(n_paths, n_steps)
    paths = simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths, normals=z)

    ST = paths[:, -1]
    theoretical_mean = S0 * math.exp(r * T)
    se_mean = ST.std(ddof=1) / math.sqrt(n_paths)
    assert abs(ST.mean() - theoretical_mean) < 4.0 * se_mean

    log_return = np.log(ST / S0)
    assert abs(log_return.std(ddof=1) - sigma * math.sqrt(T)) < 0.02 * sigma * math.sqrt(T)


def test_gbm_brownian_is_deterministic_for_keys():
    S0, r, sigma, T = 100.0, 0.05, 0.2, 1.0
    n_steps, n_paths = 20, 1000

    def run(key0, key1):
        z = px.double_normal_batch_mt(n_paths * n_steps, 4, key0=key0, key1=key1).reshape(n_paths, n_steps)
        return simulate_gbm_vectorised(S0, r, sigma, T, n_steps, n_paths, normals=z)

    assert np.array_equal(run(1, 2), run(1, 2))
    assert not np.array_equal(run(1, 2), run(3, 4))
