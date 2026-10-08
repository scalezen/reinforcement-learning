#pragma once

#include <cstdint>
#include <vector>

// Host-callable launcher: generates num_rands Philox4x32 outputs on the GPU,
// analogus to generators in Philox32x4.h for the CPU.
//
// Implemented in Philox32x4Cuda.cu. Declared in a plain header, here, with no
// CUDA-specific types, so callers can link against it without needing
// to be compiled by nvcc.
std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands, uint32_t counter0_offset = 0,
                                            uint32_t counter1_offset = 0, uint32_t key0 = 0, uint32_t key1 = 0);

// Same, but float standard normals (matches philox32x4_normal_batch<float> in Philox32x4.h).
std::vector<float> philox32x4_normal_batch_cuda(uint32_t num_rands, uint32_t counter0_offset = 0,
                                                uint32_t counter1_offset = 0, uint32_t key0 = 0, uint32_t key1 = 0);
