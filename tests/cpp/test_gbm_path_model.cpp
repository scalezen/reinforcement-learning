#include <gtest/gtest.h>

#include <cmath>

#include "mc/GbmPathModel.h"

TEST(GbmPathModelTest, PathArraySizeMatchesRequest) {
    GbmPathModel model(/*S0=*/100.0, /*r=*/0.05, /*sigma=*/0.2, /*T=*/1.0, /*n_steps=*/10, /*n_paths=*/50,
                       /*key0=*/1, /*key1=*/2, /*num_threads=*/1);

    PathArray paths = model.simulate();

    EXPECT_EQ(paths.num_times(), 11u);  // n_steps + 1
    EXPECT_EQ(paths.num_paths(), 50u);
    EXPECT_EQ(paths.num_factors(), 1u);
}

TEST(GbmPathModelTest, EveryPathStartsAtS0) {
    double S0 = 100.0;
    GbmPathModel model(S0, 0.05, 0.2, 1.0, 10, 50, 1, 2, 1);

    PathArray paths = model.simulate();
    for (size_t path = 0; path < paths.num_paths(); ++path) {
        EXPECT_DOUBLE_EQ(paths(0, path), S0);
    }
}

// Numerical check, same idea as the Python test_gbm.py::test_mean:
// E[S_T] = S0*exp(rT) under the risk-neutral measure.
// Single-threaded so result is deterministic for a given key pair.
TEST(GbmPathModelTest, TerminalMeanMatchesTheory) {
    double S0 = 100.0, r = 0.05, sigma = 0.2, T = 1.0;
    size_t n_steps = 50, n_paths = 200000;

    GbmPathModel model(S0, r, sigma, T, n_steps, n_paths, /*key0=*/1, /*key1=*/2, /*num_threads=*/1);
    PathArray paths = model.simulate();

    const double* maturity = paths.time_slice(n_steps);
    double sum = 0.0, sum_sq = 0.0;
    for (size_t path = 0; path < n_paths; ++path) {
        sum += maturity[path];
        sum_sq += maturity[path] * maturity[path];
    }
    double mean = sum / static_cast<double>(n_paths);
    double variance = (sum_sq - static_cast<double>(n_paths) * mean * mean) / static_cast<double>(n_paths - 1);
    double standard_error = std::sqrt(variance / static_cast<double>(n_paths));

    double theoretical_mean = S0 * std::exp(r * T);
    EXPECT_NEAR(mean, theoretical_mean, 4.0 * standard_error);
}
