#pragma once

#include "PathArray.h"

// Owns path generation parameters (spot, rates, vol, ...).
// simulate() generates the PathArray, will be implemented on CPU and GPU.
// Generating paths is embarrassingly parallel.
class PathModel {
   public:
    virtual ~PathModel() = default;

    virtual size_t num_steps() const = 0;  // excludes t = 0
    virtual size_t num_paths() const = 0;
    virtual size_t num_factors() const = 0;  // 1 for GBM, 2 for Heston (spot, variance)
    virtual double dt() const = 0;

    // Returns a PathArray sized (num_steps()+1, num_paths(), num_factors()).
    virtual PathArray simulate() const = 0;
};
