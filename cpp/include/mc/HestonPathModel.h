#pragma once

#include <stdexcept>

#include "PathModel.h"

// Skeleton only, unimplemented.
class HestonPathModel : public PathModel {
   public:
    HestonPathModel(double S0, double v0, double r, double kappa, double theta, double xi, double rho, double T,
                    size_t n_steps, size_t n_paths)
        : S0_(S0),
          v0_(v0),
          r_(r),
          kappa_(kappa),
          theta_(theta),
          xi_(xi),
          rho_(rho),
          T_(T),
          n_steps_(n_steps),
          n_paths_(n_paths) {}

    size_t num_steps() const override { return n_steps_; }
    size_t num_paths() const override { return n_paths_; }
    size_t num_factors() const override { return 2; }  // 0: spot, 1: variance
    double dt() const override { return T_ / static_cast<double>(n_steps_); }

    PathArray simulate() const override { throw std::logic_error("HestonPathModel::simulate not implemented yet"); }

   private:
    // Heston needs 2 correlated factors (spot, variance)
    // and variance-positivity scheme (e.g. full truncation or QE).
    double S0_, v0_, r_, kappa_, theta_, xi_, rho_, T_;
    size_t n_steps_, n_paths_;
};
