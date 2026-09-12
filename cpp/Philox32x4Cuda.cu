#include "Philox32x4Cuda.cuh"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace {

constexpr uint32_t kNumRounds = 10; 

// Device kernel corresponding to philox32x4() in Philox32x4.h
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

// One thread computes one 4-word Philox block, matching one iteration of the
// `loop` variable in the CPU philox32x4_batch. Thread global index `idx` maps
// to counter[0] = counter0_offset + idx; counter[1] = counter1_offset is fixed
// for the whole launch (same role as the CPU function's counter1_offset
// parameter — callers can vary it across separate calls to avoid collisions
// between independently generated batches). Guards both against launching
// more threads than blocks needed, and against writing past num_rands on the
// last (possibly partial) block, matching the CPU tail handling.
__global__ void philox32x4_kernel(uint32_t num_rands,
                                   uint32_t counter0_offset,
                                   uint32_t counter1_offset,
                                   uint32_t *out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_blocks = (num_rands + 3) / 4;
    if (idx >= num_blocks) {
        return;
    }

    uint32_t counter[4] = {counter0_offset + idx, counter1_offset, 0, 0};
    uint32_t key[2] = {0, 0};
    philox32x4_device(counter, key);

    uint32_t base = idx * 4;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (base + i < num_rands) {
            out[base + i] = counter[i];
        }
    }
}

void cuda_check(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands,
                                             uint32_t counter0_offset,
                                             uint32_t counter1_offset) {
    std::vector<uint32_t> output(num_rands);
    if (num_rands == 0) {
        return output;
    }

    uint32_t num_blocks = (num_rands + 3) / 4;

    uint32_t *device_out = nullptr;
    cuda_check(cudaMalloc(&device_out, num_rands * sizeof(uint32_t)), "cudaMalloc");

    constexpr uint32_t kThreadsPerBlock = 256;
    uint32_t grid_size = (num_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    philox32x4_kernel<<<grid_size, kThreadsPerBlock>>>(num_rands, counter0_offset, counter1_offset, device_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(device_out);
        cuda_check(launch_err, "philox32x4_kernel launch");
    }

    cudaError_t copy_err =
        cudaMemcpy(output.data(), device_out, num_rands * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    cuda_check(copy_err, "cudaMemcpy device to host");

    return output;
}
