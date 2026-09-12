import numpy as np
import pytest

# philox32x4_cuda_py only builds when CMake finds a CUDA toolkit (see the
# CMAKE_CUDA_COMPILER/CUDAToolkit_FOUND guard in CMakeLists.txt). Neither this
# dev machine nor the CI/nightly runners have one, so the extension won't
# exist there — skip the whole module rather than fail at import/collection.
# No CI wiring is needed for that: `pytest -q` in nightly.yml already just
# collects tests/python/**, and importorskip turns "module missing" into a
# skip instead of a collection error.
philox_cuda = pytest.importorskip("philox32x4_cuda_py")

# NOTE: these tests check API/structural behaviour only (shape, dtype,
# determinism, sensitivity to inputs). They do NOT check exact output values
# against a known-answer sequence, because the GPU kernel has never actually
# been run (no CUDA-capable machine was available while writing it) — there
# is no verified sequence to assert against yet. Bit-exact correctness
# against the CPU implementation is covered in tests/cpp/test_philox32x4_cuda.cpp,
# which should be the first thing run on a real CUDA machine before trusting
# this module. Once that's been run, consider adding a known-answer test
# here too, pinned to actual verified output.


def test_next_batch_zero_returns_empty_array():
    assert philox_cuda.next_batch(0).tolist() == []


def test_next_batch_dtype_and_shape():
    batch = philox_cuda.next_batch(37)
    assert batch.dtype == np.uint32
    assert batch.shape == (37,)


def test_next_batch_size_matches_request_including_non_multiples_of_4():
    for n in range(41):
        assert philox_cuda.next_batch(n).shape == (n,)


def test_default_offsets_are_zero():
    assert philox_cuda.next_batch(4).tolist() == philox_cuda.next_batch(4, 0, 0).tolist()


def test_deterministic_for_same_inputs():
    a = philox_cuda.next_batch(100, counter0_offset=5, counter1_offset=9)
    b = philox_cuda.next_batch(100, counter0_offset=5, counter1_offset=9)
    assert a.tolist() == b.tolist()


def test_different_counter0_offset_gives_different_output():
    a = philox_cuda.next_batch(16, counter0_offset=0)
    b = philox_cuda.next_batch(16, counter0_offset=1)
    assert a.tolist() != b.tolist()


def test_different_counter1_offset_gives_different_output():
    a = philox_cuda.next_batch(16, counter1_offset=0)
    b = philox_cuda.next_batch(16, counter1_offset=1)
    assert a.tolist() != b.tolist()


def test_rejects_negative_num_rands():
    # num_rands is uint32_t in the C++ signature, so pybind11's own argument
    # conversion rejects a negative Python int before the function body runs
    # (TypeError from the automatic cast, not a ValueError raised in C++).
    with pytest.raises(TypeError):
        philox_cuda.next_batch(-1)


def test_large_batch_has_no_repeated_values():
    # Regression guard for the block/grid indexing math in the kernel launch
    # (Philox32x4Cuda.cu): a large enough batch to span multiple CUDA blocks
    # at the kernel's launch configuration should still produce 4-word blocks
    # with no accidental overlap/duplication between them.
    batch = philox_cuda.next_batch(20000, counter0_offset=3, counter1_offset=4)
    assert len(np.unique(batch)) == len(batch)
