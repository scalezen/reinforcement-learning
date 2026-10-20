#pragma once

#include <vector>

// fit() runs on a (typically small) in-the-money subset, then predict() is
// called on that same subset's x values. Both loops are the other seam
// meant to move to CUDA/cuBLAS later, once there are enough paths/basis
// terms for it to be worth it.
class Regressor {
public:
    virtual ~Regressor() = default;
    virtual void fit(const std::vector<double>& x, const std::vector<double>& y) = 0;
    virtual double predict(double x) const = 0;
};
