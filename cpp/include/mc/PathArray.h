#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

// Flat (time, path, factor) buffer.
// Time-major as required by LSM regression which proceeds time-step by time-step.
// Contrast layout to gbm.py's (n_paths, n_steps) layout and per-path cumsum over time.
class PathArray {
   public:
    PathArray(size_t num_times, size_t num_paths, size_t num_factors = 1)
        : num_times_(num_times),
          num_paths_(num_paths),
          num_factors_(num_factors),
          data_(num_times * num_paths * num_factors) {}

    PathArray(size_t num_times, size_t num_paths, size_t num_factors, std::vector<double> data)
        : num_times_(num_times), num_paths_(num_paths), num_factors_(num_factors), data_(std::move(data)) {
        if (data_.size() != num_times_ * num_paths_ * num_factors_) {
            throw std::invalid_argument("PathArray: data size does not match num_times*num_paths*num_factors");
        }
    }

    size_t num_times() const { return num_times_; }
    size_t num_paths() const { return num_paths_; }
    size_t num_factors() const { return num_factors_; }

    double& operator()(size_t t, size_t path, size_t factor = 0) {
        return data_[(t * num_paths_ + path) * num_factors_ + factor];
    }
    double operator()(size_t t, size_t path, size_t factor = 0) const {
        return data_[(t * num_paths_ + path) * num_factors_ + factor];
    }

    // Contiguous view of every path's value at time t, factor 0. Only valid
    // when num_factors() == 1 (true for GBM).
    // NOTE: Heston will need a strided view.
    double* time_slice(size_t t) { return data_.data() + t * num_paths_ * num_factors_; }
    const double* time_slice(size_t t) const { return data_.data() + t * num_paths_ * num_factors_; }

   private:
    size_t num_times_;
    size_t num_paths_;
    size_t num_factors_;
    std::vector<double> data_;
};
