#pragma once
#include <vector>

#include "Philox32x4Cuda.cuh"
#include "cuda_utilities.cuh"

__global__ void simulate_gbm_path_kernel(uint32_t n_steps, uint32_t n_paths, float s0, float drift, float vol,
                                         uint32_t counter0_offset, uint32_t counter1_offset, uint32_t key0,
                                         uint32_t key1, float* gpu_paths) {
    uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    // we will handle 4 paths per thread

    if (idx >= n_paths / 4) return;

    uint32_t key[2] = {key0, key1};
    uint32_t counter[4] = {counter0_offset, counter1_offset, 0, 0};

    for (size_t i = 0; i < 4; ++i) gpu_paths[4 * idx + i] = s0;

    for (size_t t = 1; t <= n_steps; ++t) {
        // for (size_t ipath = idx; ipath < n_paths_; ipath += 4) {
        counter[0] = counter0_offset + (t - 1) * n_paths + idx;
        counter[1] = counter1_offset;
        counter[2] = 0;
        counter[3] = 0;
        philox32x4_device(counter, key);
        for (size_t i = 0; i < 4; ++i) {
            float prev = gpu_paths[(t - 1) * n_paths + 4 * idx + i];
            float zt = to_normal(counter[i]);
            gpu_paths[t * n_paths + 4 * idx + i] = prev * expf(drift + vol * zt);
        }
    }
    //}
}

// intra-warp-shuffle coalesced GBM kernel.
__global__ void simulate_gbm_path_kernel_coalesced(uint32_t n_steps, uint32_t n_paths, float s0, float drift, float vol,
                                                   uint32_t counter0_offset, uint32_t counter1_offset, uint32_t key0,
                                                   uint32_t key1, float* gpu_paths) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;  // blockDim.x is 256
    if (tid >= n_paths) return;

    // Every 4 consecutive global thread ids share one Philox call (one 4-word
    // block).  Each thread owns one path instead of 4, so writes below are stride 1.
    uint32_t group_id = tid / 4;
    uint32_t lane_offset = tid % 4;

    // Which lane in this *warp* is the producer for this thread's group of 4.
    uint32_t lane = threadIdx.x % warpSize;  // which lane in this warp is the producer for this thread's 4 rns
    uint32_t producer_lane = (lane / 4) * 4;

    gpu_paths[tid] = s0;

    uint32_t counter[4] = {0, 0, 0, 0};

    for (uint32_t t = 1; t <= n_steps; ++t) {
        // Only the producer lane calls Philox; others are 0. shfl below only
        // ever reads producer_lane's copy, so this is the condition that
        // actually matters.
        if (lane == producer_lane) {
            counter[0] = counter0_offset + (t - 1) * n_paths + group_id;  // same formula as before, same RNG draws
            counter[1] = counter1_offset;
            counter[2] = 0;
            counter[3] = 0;
            uint32_t key[2] = {key0, key1};
            philox32x4_device(counter, key);
        }

        // Broadcast all 4 words from the producer lane; each lane keeps only
        // the one word matching its own lane_offset.
        uint32_t rn = 0;
#pragma unroll
        for (uint32_t w = 0; w < 4; ++w) {
            uint32_t shuffled = __shfl_sync(0xFFFFFFFFu, counter[w], producer_lane);
            if (w == lane_offset) rn = shuffled;
        }

        float prev = gpu_paths[(t - 1) * n_paths + tid];
        float zt = to_normal(rn);
        gpu_paths[t * n_paths + tid] = prev * expf(drift + vol * zt);
    }
}

// host code
inline std::vector<double> simulate_gbm_path_model_gpu_impl(int n_steps, int n_paths, float s0, float step, float r,
                                                            float sigma, uint32_t counter0_offset,
                                                            uint32_t counter1_offset, uint32_t key0, uint32_t key1) {
    double drift = (r - 0.5 * sigma * sigma) * step;
    double vol = sigma * std::sqrt(step);

    std::vector<float> cpu_paths((n_steps + 1) * n_paths);
    float* gpu_paths = nullptr;
    cuda_check(cudaMalloc(&gpu_paths, sizeof(float) * (n_steps + 1) * n_paths), "cudaMalloc");

    uint32_t threadsPerBlock = 256;

    if (n_paths % 4 != 0) throw std::runtime_error("Number of mc_paths needs to be divisible by 4.");

    // we will handle 4 paths per thread, corresponding to 4 randoms from Philox
    uint32_t numBlocks = (n_paths / 4 + threadsPerBlock - 1) / threadsPerBlock;

    simulate_gbm_path_kernel<<<numBlocks, threadsPerBlock>>>(n_steps, n_paths, s0, drift, vol, counter0_offset,
                                                             counter1_offset, key0, key1, gpu_paths);

    // if we were to run the coalesced kernel, then the launch config would change,
    // each thread would handle one path
    //   uint32_t numBlocks = (n_paths + threadsPerBlock - 1) / threadsPerBlock;  // was n_paths/4
    //
    //   simulate_gbm_path_kernel_coalesced<<<numBlocks, threadsPerBlock>>>(
    //       n_steps, n_paths, s0, drift, vol, counter0_offset, counter1_offset, key0, key1, gpu_paths);

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        cudaFree(gpu_paths);
        cuda_check(status, "Error launching kernel.");
    }

    cudaError_t copy_status =
        cudaMemcpy(cpu_paths.data(), gpu_paths, sizeof(float) * (n_steps + 1) * n_paths, cudaMemcpyDeviceToHost);
    cudaFree(gpu_paths);
    cuda_check(copy_status, "Error copying from device to host.");

    return std::vector<double>(cpu_paths.begin(), cpu_paths.end());
}