#include <cstdint>

#include <gtest/gtest.h>

#include "SplitMix64.h"

TEST(SplitMix64Test, KnownAnswerSequenceForSeedZero) {
    SplitMix64 rng(0);

    const uint64_t expected[5] = {
        16294208416658607535ULL,
        7960286522194355700ULL,
        487617019471545679ULL,
        17909611376780542444ULL,
        1961750202426094747ULL,
    };

    for (uint64_t exp : expected) {
        EXPECT_EQ(rng.next(), exp);
    }
}
