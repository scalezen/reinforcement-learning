#pragma once

#include <cstdint>
#include <vector>

#include "cuda_utilities.cuh"

constexpr uint32_t kNumRounds = 10;

// philox32x4_device and kNumRounds are defined inline in the header file,
// so other .cu/.cuh translation units (e.g. GbmPathModel.cuh) can call
// philox32x4_device too without needing CUDA separable compilation.
__device__ __forceinline__ void philox32x4_device(uint32_t counter[4], uint32_t key[2]) {
    constexpr uint32_t M0 = 0xD2511F53u;
    constexpr uint32_t M1 = 0xCD9E8D57u;
    constexpr uint32_t W0 = 0x9E3779B9u;
    constexpr uint32_t W1 = 0xBB67AE85u;

#pragma unroll
    for (uint32_t i = 0; i < kNumRounds; ++i) {
        uint64_t prod0 = static_cast<uint64_t>(counter[0]) * M0;
        uint64_t prod1 = static_cast<uint64_t>(counter[2]) * M1;

        uint32_t new_c0 = static_cast<uint32_t>(prod1 >> 32) ^ counter[1] ^ key[0];
        uint32_t new_c1 = static_cast<uint32_t>(prod1);
        uint32_t new_c2 = static_cast<uint32_t>(prod0 >> 32) ^ counter[3] ^ key[1];
        uint32_t new_c3 = static_cast<uint32_t>(prod0);

        counter[0] = new_c0;
        counter[1] = new_c1;
        counter[2] = new_c2;
        counter[3] = new_c3;

        key[0] += W0;
        key[1] += W1;
    }
}

std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands, uint32_t counter0_offset = 0,
                                            uint32_t counter1_offset = 0, uint32_t key0 = 0, uint32_t key1 = 0);

// Same, but float standard normals (matches philox32x4_normal_batch<float> in Philox32x4.h).
std::vector<float> philox32x4_normal_batch_cuda(uint32_t num_rands, uint32_t counter0_offset = 0,
                                                uint32_t counter1_offset = 0, uint32_t key0 = 0, uint32_t key1 = 0);
