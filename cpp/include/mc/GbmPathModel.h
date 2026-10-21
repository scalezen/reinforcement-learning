#pragma once

#include <cstdint>

#include "PathModel.h"

class GbmPathModel : public PathModel {
   public:
    // num_threads controls how many threads generate the underlying normals
    // (see Philox32x4.h's philox32x4_normal_batch_mt) — defaults to 1
    // (single-threaded), not parallel by default.
    GbmPathModel(double S0, double r, double sigma, double T, size_t n_steps, size_t n_paths, uint32_t key0 = 0,
                 uint32_t key1 = 0, uint32_t num_threads = 1);

    size_t num_steps() const override { return n_steps_; }
    size_t num_paths() const override { return n_paths_; }
    size_t num_factors() const override { return 1; }
    double dt() const override { return T_ / static_cast<double>(n_steps_); }

    PathArray simulate() const override;
    PathArray simulate_gpu() const override;

   private:
    double S0_, r_, sigma_, T_;
    size_t n_steps_, n_paths_;
    uint32_t key0_, key1_;
    uint32_t num_threads_;
};
