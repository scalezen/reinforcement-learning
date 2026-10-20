#include "mc/GbmPathModel.h"

#include <cmath>

#include "Philox32x4.h"

GbmPathModel::GbmPathModel(double S0, double r, double sigma, double T, size_t n_steps, size_t n_paths,
                            uint32_t key0, uint32_t key1, uint32_t num_threads)
    : S0_(S0),
      r_(r),
      sigma_(sigma),
      T_(T),
      n_steps_(n_steps),
      n_paths_(n_paths),
      key0_(key0),
      key1_(key1),
      num_threads_(num_threads) {}

PathArray GbmPathModel::simulate() const {
    PathArray paths(n_steps_ + 1, n_paths_, 1);
    double step = dt();
    double drift = (r_ - 0.5 * sigma_ * sigma_) * step;
    double vol = sigma_ * std::sqrt(step);

    for (size_t path = 0; path < n_paths_; ++path) paths(0, path) = S0_;

    // One normal per (step, path); time-major already matches PathArray's
    // layout, so there's no reshape between Philox's output and here.
    auto z = philox32x4_normal_batch_mt<double>(static_cast<uint32_t>(n_steps_ * n_paths_), num_threads_, key0_,
                                                 key1_);

    for (size_t t = 1; t <= n_steps_; ++t) {
        for (size_t path = 0; path < n_paths_; ++path) {
            double prev = paths(t - 1, path);
            double zt = z[(t - 1) * n_paths_ + path];
            paths(t, path) = prev * std::exp(drift + vol * zt);
        }
    }
    return paths;
}
