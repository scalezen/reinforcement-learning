#include <cstdint>

#include <gtest/gtest.h>

#include "Philox32x4.h"
#include "Philox32x4Cuda.cuh"

// Cross-validates the GPU kernel against the CPU implementation in
// Philox32x4.h, which is independently tested in test_philox32x4.cpp. If
// these ever disagree, the two implementations have drifted apart.

TEST(Philox32x4CudaTest, EmptyBatchReturnsEmpty) {
    EXPECT_TRUE(philox32x4_batch_cuda(0).empty());
}

TEST(Philox32x4CudaTest, SizeMatchesRequestForRangeIncludingNonMultiplesOf4) {
    for (uint32_t n = 0; n <= 40; ++n) {
        EXPECT_EQ(philox32x4_batch_cuda(n).size(), n) << "n=" << n;
    }
}

TEST(Philox32x4CudaTest, MatchesCpuImplementationForVariousSizesAndOffsets) {
    for (uint32_t c0_offset : {0u, 5u}) {
        for (uint32_t c1_offset : {0u, 9u}) {
            for (uint32_t n : {0u, 1u, 3u, 4u, 7u, 20u, 257u}) {
                auto gpu = philox32x4_batch_cuda(n, c0_offset, c1_offset);
                auto cpu = philox32x4_batch(n, c0_offset, c1_offset);
                EXPECT_EQ(gpu, cpu) << "n=" << n << " c0_offset=" << c0_offset << " c1_offset=" << c1_offset;
            }
        }
    }
}

TEST(Philox32x4CudaTest, ManyBlocksSpanningMultipleCudaBlocksMatchesCpu) {
    // Large enough to require more than one CUDA block at the kernel's
    // 256-threads-per-block launch configuration (kThreadsPerBlock in
    // Philox32x4Cuda.cu), to exercise the block/grid indexing math, not just
    // a single block's worth of threads.
    const uint32_t n = 10000;
    auto gpu = philox32x4_batch_cuda(n, 3, 4);
    auto cpu = philox32x4_batch(n, 3, 4);
    EXPECT_EQ(gpu, cpu);
}
