#pragma once

#include <cstdint>
#include <vector>

// Host-callable launcher: generates num_rands Philox4x32 outputs on the GPU,
// bit-for-bit identical to the CPU philox32x4_batch(num_rands, counter0_offset,
// counter1_offset, key0, key1) in Philox32x4.h
//
// Implemented in Philox32x4Cuda.cu (compiled by nvcc). Declared in a plain
// header with no CUDA-specific types, so callers built by a regular C++
// compiler (e.g. the pybind11 module) can link against it without needing
// to be compiled by nvcc.
std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands, uint32_t counter0_offset = 0,
                                            uint32_t counter1_offset = 0, uint32_t key0 = 0, uint32_t key1 = 0);
