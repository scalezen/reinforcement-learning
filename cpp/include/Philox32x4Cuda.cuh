#pragma once

#include <cstdint>
#include <vector>

// Host-callable launcher: generates num_rands Philox4x32-10 outputs on the GPU,
// bit-for-bit identical to the CPU philox32x4_batch(num_rands, counter0_offset,
// counter1_offset) in Philox32x4.h — same algorithm, same counter scheme (block
// index -> counter[0], counter1_offset -> counter[1], key={0,0}).
//
// Implemented in Philox32x4Cuda.cu (compiled by nvcc). Declared here, in a plain
// header with no CUDA-specific types, so callers built by a regular C++
// compiler (e.g. the pybind11 module) can link against it without themselves
// needing to be compiled by nvcc.
std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands,
                                             uint32_t counter0_offset = 0,
                                             uint32_t counter1_offset = 0);
