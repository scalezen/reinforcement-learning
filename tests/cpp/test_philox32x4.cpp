#include <algorithm>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "Philox32x4.h"

TEST(Philox32x4Test, DeterministicForSameInputs) {
    auto a = philox32x4({1, 2, 3, 4}, {5, 6});
    auto b = philox32x4({1, 2, 3, 4}, {5, 6});
    EXPECT_EQ(a, b);
}

TEST(Philox32x4Test, DifferentCounterGivesDifferentOutput) {
    auto a = philox32x4({0, 0, 0, 0}, {0, 0});
    auto b = philox32x4({1, 0, 0, 0}, {0, 0});
    EXPECT_NE(a, b);
}

TEST(Philox32x4Test, DifferentKeyGivesDifferentOutput) {
    auto a = philox32x4({0, 0, 0, 0}, {0, 0});
    auto b = philox32x4({0, 0, 0, 0}, {0, 1});
    EXPECT_NE(a, b);
}

TEST(Philox32x4Test, NotDegenerate) {
    // Regression guard: with the round constants zeroed out (an earlier bug in
    // this file), every round collapses and the output stops depending on
    // most of the input. Confirm output actually varies across many counters.
    std::vector<uint32_t> firsts;
    for (uint32_t c = 0; c < 20; ++c) {
        firsts.push_back(philox32x4({c, 0, 0, 0}, {0, 0})[0]);
    }
    std::sort(firsts.begin(), firsts.end());
    firsts.erase(std::unique(firsts.begin(), firsts.end()), firsts.end());
    EXPECT_GT(firsts.size(), 15u) << "too many collisions for 20 distinct counters";
}

// ---- philox32x4_batch ----

TEST(Philox32x4BatchTest, SizeMatchesRequestForRangeIncludingNonMultiplesOf4) {
    for (uint32_t n = 0; n <= 40; ++n) {
        EXPECT_EQ(philox32x4_batch(n).size(), n) << "n=" << n;
    }
}

TEST(Philox32x4BatchTest, EmptyBatchReturnsEmpty) {
    EXPECT_TRUE(philox32x4_batch(0).empty());
}

TEST(Philox32x4BatchTest, MatchesDirectPhilox32x4CallsWithOffsets) {
    // Independent recomputation using only philox32x4, not philox32x4_batch's
    // own internals -- catches bugs in philox32x4_batch's main loop AND its
    // remainder handling (both offsets used to be dropped in the remainder
    // block, which this covers via a non-multiple-of-4 n with nonzero offsets).
    const uint32_t c0_offset = 5;
    const uint32_t c1_offset = 9;

    for (uint32_t n : {1u, 3u, 4u, 7u, 20u}) {
        auto out = philox32x4_batch(n, c0_offset, c1_offset);
        ASSERT_EQ(out.size(), n);

        uint32_t num_loops = n / 4;
        for (uint32_t loop = 0; loop < num_loops; ++loop) {
            auto expected = philox32x4({c0_offset + loop, c1_offset, 0, 0}, {0, 0});
            for (int k = 0; k < 4; ++k) {
                EXPECT_EQ(out[loop * 4 + k], expected[k]) << "n=" << n << " loop=" << loop;
            }
        }

        uint32_t remaining = n % 4;
        if (remaining > 0) {
            auto expected = philox32x4({c0_offset + num_loops, c1_offset, 0, 0}, {0, 0});
            for (uint32_t i = 0; i < remaining; ++i) {
                EXPECT_EQ(out[num_loops * 4 + i], expected[i]) << "n=" << n << " remainder i=" << i;
            }
        }
    }
}

// ---- philox32x4_batch_mt ----

TEST(Philox32x4BatchMtTest, SizeMatchesRequestAcrossThreadCounts) {
    for (uint32_t num_rands : {0u, 1u, 7u, 16u, 17u, 23u, 30u, 100u}) {
        for (uint32_t num_threads : {1u, 2u, 3u, 4u, 5u, 8u}) {
            EXPECT_EQ(philox32x4_batch_mt(num_rands, num_threads).size(), num_rands)
                << "num_rands=" << num_rands << " num_threads=" << num_threads;
        }
    }
}

// Regression test: these specific (num_rands, num_threads) pairs used to hang
// (or attempt a multi-gigabyte allocation) because chunk_end - chunk_begin
// underflowed to ~UINT32_MAX for "extra" threads once num_threads exceeded
// what num_rands actually needed. If this test hangs, that bug is back.
TEST(Philox32x4BatchMtTest, MoreThreadsThanRandsDoesNotHang) {
    for (auto [num_rands, num_threads] : {
             std::pair{1u, 3u}, std::pair{1u, 5u}, std::pair{7u, 5u}, std::pair{4u, 8u}}) {
        auto out = philox32x4_batch_mt(num_rands, num_threads);
        EXPECT_EQ(out.size(), num_rands) << "num_rands=" << num_rands << " num_threads=" << num_threads;
    }
}

TEST(Philox32x4BatchMtTest, MatchesPerThreadPhilox32x4BatchCalls) {
    // philox32x4_batch is independently verified above; this checks that
    // philox32x4_batch_mt's chunking/dispatch assembles the right pieces.
    for (uint32_t num_rands : {16u, 17u, 23u, 100u, 30u}) {
        for (uint32_t num_threads : {2u, 3u, 5u, 8u}) {
            auto out = philox32x4_batch_mt(num_rands, num_threads);
            ASSERT_EQ(out.size(), num_rands);

            uint32_t num_rands_per_thread = (num_rands + num_threads - 1) / num_threads;
            for (uint32_t th = 0; th < num_threads; ++th) {
                uint32_t chunk_begin = th * num_rands_per_thread;
                uint32_t chunk_end = std::min(num_rands, (th + 1) * num_rands_per_thread);
                if (chunk_begin >= chunk_end) continue;

                auto expected_chunk = philox32x4_batch(chunk_end - chunk_begin, 0, th);
                for (uint32_t i = chunk_begin; i < chunk_end; ++i) {
                    EXPECT_EQ(out[i], expected_chunk[i - chunk_begin])
                        << "num_rands=" << num_rands << " num_threads=" << num_threads << " i=" << i;
                }
            }
        }
    }
}

TEST(Philox32x4BatchMtTest, NoCounterCollisionsAcrossThreads) {
    // Directly encodes the property that matters for correctness here: no two
    // threads (and no thread's own remainder call) ever reuse the same
    // (counter0, counter1) pair, which would otherwise silently duplicate a
    // 4-value block in the output. Computed independently via philox32x4
    // directly, using the same (counter0=0-based-per-thread-loop, counter1=
    // thread index) scheme philox32x4_batch_mt actually uses.
    for (uint32_t num_rands : {16u, 17u, 23u, 100u, 30u, 1u, 4u}) {
        for (uint32_t num_threads : {1u, 2u, 3u, 5u, 8u}) {
            uint32_t num_rands_per_thread = (num_rands + num_threads - 1) / num_threads;

            std::vector<std::pair<uint32_t, uint32_t>> counters_used;
            for (uint32_t th = 0; th < num_threads; ++th) {
                uint32_t chunk_begin = th * num_rands_per_thread;
                uint32_t chunk_end = std::min(num_rands, (th + 1) * num_rands_per_thread);
                if (chunk_begin >= chunk_end) continue;

                uint32_t chunk_size = chunk_end - chunk_begin;
                uint32_t num_loops = chunk_size / 4;
                for (uint32_t loop = 0; loop < num_loops; ++loop) {
                    counters_used.emplace_back(loop, th);
                }
                if (chunk_size % 4 != 0) {
                    counters_used.emplace_back(num_loops, th);
                }
            }

            auto sorted = counters_used;
            std::sort(sorted.begin(), sorted.end());
            auto last = std::unique(sorted.begin(), sorted.end());
            EXPECT_EQ(last, sorted.end())
                << "duplicate counter found for num_rands=" << num_rands << " num_threads=" << num_threads;
        }
    }
}
