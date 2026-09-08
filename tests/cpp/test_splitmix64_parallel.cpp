#include <cstdint>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "SplitMix64.h"
#include "SplitMix64Parallel.h"

TEST(SplitMix64ParallelTest, RejectsZeroThreads) {
    EXPECT_THROW(SplitMix64Parallel(0, 0), std::invalid_argument);
}

TEST(SplitMix64ParallelTest, EmptyBatchReturnsEmpty) {
    SplitMix64Parallel rng(0, 4);
    EXPECT_TRUE(rng.next_batch(0).empty());
    EXPECT_TRUE(rng.next_uniform(0).empty());
}

TEST(SplitMix64ParallelTest, BatchSizeMatchesRequest) {
    SplitMix64Parallel rng(42, 4);
    EXPECT_EQ(rng.next_batch(37).size(), 37u);
    EXPECT_EQ(rng.next_uniform(37).size(), 37u);
}

std::vector<uint64_t> plain_sequence(uint64_t seed, size_t n) {
    SplitMix64 plain(seed);
    std::vector<uint64_t> expected(n);
    for (size_t i = 0; i < n; ++i) {
        expected[i] = plain.next();
    }
    return expected;
}

TEST(SplitMix64ParallelTest, SingleCallMatchesPlainSequenceForVariousThreadCounts) {
    for (unsigned n_threads : {1u, 2u, 3u, 4u, 8u, 16u}) {
        SplitMix64Parallel rng(0, n_threads);
        EXPECT_EQ(rng.next_batch(1000), plain_sequence(0, 1000)) << "n_threads=" << n_threads;
    }
}

TEST(SplitMix64ParallelTest, MultipleCallsWithUnevenSizesMatchOnePlainSequence) {
    const uint64_t seed = 7;
    SplitMix64Parallel rng(seed, 5);  // 5 threads, sizes below are not multiples of 5

    std::vector<uint64_t> got;
    for (size_t n : {3, 1, 4, 1, 5, 9, 2, 6}) {
        std::vector<uint64_t> batch = rng.next_batch(n);
        got.insert(got.end(), batch.begin(), batch.end());
    }

    EXPECT_EQ(got, plain_sequence(seed, got.size()));
}

TEST(SplitMix64ParallelTest, NextUniformMatchesPlainNextDoubleSequence) {
    const uint64_t seed = 3;
    const size_t n = 500;

    SplitMix64Parallel rng(seed, 4);
    std::vector<double> got = rng.next_uniform(n);

    SplitMix64 plain(seed);
    for (size_t i = 0; i < n; ++i) {
        EXPECT_DOUBLE_EQ(got[i], plain.next_double());
    }
}

TEST(SplitMix64ParallelTest, UniformDrawsAreInRange) {
    SplitMix64Parallel rng(99, 8);
    std::vector<double> draws = rng.next_uniform(10000);
    for (double d : draws) {
        EXPECT_GE(d, 0.0);
        EXPECT_LT(d, 1.0);
    }
}
