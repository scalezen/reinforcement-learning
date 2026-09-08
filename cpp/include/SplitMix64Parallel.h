#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "SplitMix64.h"

// Splits SplitMix64 generation across n_threads worker threads while reproducing
// the sequence a plain sequential SplitMix64(seed) would produce.
//
// SplitMix64's state recurrence is affine: after k calls the state is exactly
// seed + k*kGoldenGamma (mod 2^64), so the k-th output can be computed directly
// in O(1) via SplitMix64::mix(seed + k*kGoldenGamma) precluding the need to step
// through the k-1 states before it. 
//
// Each thread can independently compute
// its slice of a batch with no shared mutable state (and so no locking).
//
// A single running counter (total_emitted_) tracks how many values have been
// produced across all calls so far, so splitting one logical stream across
// multiple next_batch/next_uniform calls of any sizes reproduces exactly the
// same sequence as one large sequential call.
class SplitMix64Parallel {
public:
    SplitMix64Parallel(uint64_t seed, unsigned n_threads)
        : seed_(seed), n_threads_(n_threads), total_emitted_(0) {
        if (n_threads_ == 0) {
            throw std::invalid_argument("n_threads must be positive");
        }
    }

    uint64_t seed() const { return seed_; }
    unsigned n_threads() const { return n_threads_; }

    std::vector<uint64_t> next_batch(size_t n);

    std::vector<double> next_uniform(size_t n);

private:
    // The state a plain SplitMix64(seed) would have after (global_index + 1) calls
    // to next() — i.e. the state that produces the value at 0-indexed global_index.
    uint64_t state_at(size_t global_index) const {
        return seed_ + (static_cast<uint64_t>(global_index) + 1) * SplitMix64::kGoldenGamma;
    }

    // Splits [0, n) into n_threads_ contiguous, near-equal chunks and runs `work`
    // for each non-empty chunk on its own thread. Each thread only reads seed_/
    // total_emitted_ and writes its own disjoint slice of the output with no
    // need for shared mutable state and no locking.
    //template <typename Work>
    //void run_parallel(size_t n, const std::function() work);
    template<typename T>
    void run_parallel(size_t n, std::vector<T>& out);

    uint64_t seed_;
    unsigned n_threads_;
    size_t total_emitted_;
};
