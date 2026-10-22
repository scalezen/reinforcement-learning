#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "GilesErfinv.h"

constexpr float SQRT2 = 1.4142135623730950488f;

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__device__ __forceinline__ float to_normal(uint32_t arg) {
    int32_t t = static_cast<int32_t>(2 * (arg >> 8) + 1) - (1 << 24);
    return SQRT2 * erfinv_giles<float>(static_cast<float>(t) * 0x1.0p-24f);
}
