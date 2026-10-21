#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "Philox32x4Cuda.cuh"

namespace {

// One thread computes one 4-word Philox block, matching one iteration of the
// `loop` variable in the CPU philox32x4_batch. Thread global index `idx` maps
// to counter[0] = counter0_offset + idx; counter[1] = counter1_offset is fixed
// for the whole launch (same role as the CPU function's counter1_offset
// parameter — callers can vary it across separate calls to avoid collisions
// between independently generated batches).
__global__ void philox32x4_kernel(uint32_t num_rands, uint32_t counter0_offset, uint32_t counter1_offset, uint32_t key0,
                                  uint32_t key1, uint32_t* out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_blocks = (num_rands + 3) / 4;

    // Guard against launching more threads than blocks needed.
    if (idx >= num_blocks) {
        return;
    }

    uint32_t counter[4] = {counter0_offset + idx, counter1_offset, 0, 0};
    uint32_t key[2] = {key0, key1};
    philox32x4_device(counter, key);

    uint32_t base = idx * 4;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        // Guard against writing past num_rands.
        if (base + i < num_rands) {
            out[base + i] = counter[i];
        }
    }
}

// Same block/thread mapping as philox32x4_kernel, but outputs float normals.
__global__ void philox32x4_normal_kernel(uint32_t num_rands, uint32_t counter0_offset, uint32_t counter1_offset,
                                         uint32_t key0, uint32_t key1, float* out) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t num_blocks = (num_rands + 3) / 4;
    if (idx >= num_blocks) {
        return;
    }

    uint32_t counter[4] = {counter0_offset + idx, counter1_offset, 0, 0};
    uint32_t key[2] = {key0, key1};
    philox32x4_device(counter, key);

    uint32_t base = idx * 4;
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        if (base + i < num_rands) {
            out[base + i] = to_normal(counter[i]);
        }
    }
}

}  // namespace

std::vector<uint32_t> philox32x4_batch_cuda(uint32_t num_rands, uint32_t counter0_offset, uint32_t counter1_offset,
                                            uint32_t key0, uint32_t key1) {
    std::vector<uint32_t> output(num_rands);
    if (num_rands == 0) {
        return output;
    }

    uint32_t num_blocks = (num_rands + 3) / 4;

    uint32_t* device_out = nullptr;
    cuda_check(cudaMalloc(&device_out, num_rands * sizeof(uint32_t)), "cudaMalloc");

    constexpr uint32_t kThreadsPerBlock = 256;
    uint32_t grid_size = (num_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    philox32x4_kernel<<<grid_size, kThreadsPerBlock>>>(num_rands, counter0_offset, counter1_offset, key0, key1,
                                                       device_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(device_out);
        cuda_check(launch_err, "philox32x4_kernel launch");
    }

    cudaError_t copy_err = cudaMemcpy(output.data(), device_out, num_rands * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    cuda_check(copy_err, "cudaMemcpy device to host");

    return output;
}

std::vector<float> philox32x4_normal_batch_cuda(uint32_t num_rands, uint32_t counter0_offset, uint32_t counter1_offset,
                                                uint32_t key0, uint32_t key1) {
    std::vector<float> output(num_rands);
    if (num_rands == 0) {
        return output;
    }

    uint32_t num_blocks = (num_rands + 3) / 4;

    float* device_out = nullptr;
    cuda_check(cudaMalloc(&device_out, num_rands * sizeof(float)), "cudaMalloc");

    constexpr uint32_t kThreadsPerBlock = 256;
    uint32_t grid_size = (num_blocks + kThreadsPerBlock - 1) / kThreadsPerBlock;
    philox32x4_normal_kernel<<<grid_size, kThreadsPerBlock>>>(num_rands, counter0_offset, counter1_offset, key0, key1,
                                                              device_out);

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        cudaFree(device_out);
        cuda_check(launch_err, "philox32x4_normal_kernel launch");
    }

    cudaError_t copy_err = cudaMemcpy(output.data(), device_out, num_rands * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(device_out);
    cuda_check(copy_err, "cudaMemcpy device to host");

    return output;
}
