#pragma once

#include <cstdint>

// Vigna's SplitMix64. Stateful and NOT counter-based on its own.
// Conventionally used to expand a single 64-bit seed into 
// well-mixed words (e.g. Philox keys/counters).
class SplitMix64 {
public:
    // The golden-ratio increment from Vigna's reference implementation.
    // SplitMix64Parallel can jump to any position k in this sequence =>
    // mix(seed + k*kGoldenGamma), without having to step through k-1 intermediate
    // states => the state recurrence is affine.
    static constexpr uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ULL;

    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        state_ += kGoldenGamma;
        return mix(state_);
    }

    // Uniform double in [0, 1), using the top 53 bits.
    double next_double() {
        return to_double(next());
    }

    // The avalanche finalizer applied to a raw state word. Also used by
    // SplitMix64Parallel to compute mix(seed + k*kGoldenGamma)for an
    // arbitrary k calls to next().
    static uint64_t mix(uint64_t z) {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Scales a mixed 64-bit word to a uniform double in [0, 1), same as next_double().
    static double to_double(uint64_t mixed) {
        return static_cast<double>(mixed >> 11) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t state_;
};