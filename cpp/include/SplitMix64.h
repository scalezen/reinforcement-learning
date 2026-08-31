#pragma once

#include <cstdint>

// Vigna's SplitMix64. Stateful and NOT counter-based on its own — should be used to
// expand a single 64-bit seed into well-mixed words (e.g. Philox keys/counters).
class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) : state_(seed) {}

    uint64_t next() {
        uint64_t z = (state_ += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }

    // Uniform double in [0, 1), using the top 53 bits.
    double next_double() {
        return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
    }

private:
    uint64_t state_;
};
