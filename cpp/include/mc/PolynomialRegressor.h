#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Regressor.h"

// Least squares on the monomial basis [1, x, x^2, ..., x^degree], via the
// normal equations solved by Gauss-Jordan elimination. Fine for the small
// (degree+1)x(degree+1) systems LSM uses; swap for Eigen/cuBLAS if a larger
// basis is ever needed.
class PolynomialRegressor : public Regressor {
public:
    explicit PolynomialRegressor(size_t degree) : degree_(degree), coeffs_(degree + 1, 0.0) {}

    void fit(const std::vector<double>& x, const std::vector<double>& y) override {
        size_t n_terms = degree_ + 1;
        std::vector<std::vector<double>> basis(x.size(), std::vector<double>(n_terms));
        for (size_t i = 0; i < x.size(); ++i) {
            double power = 1.0;
            for (size_t j = 0; j < n_terms; ++j) {
                basis[i][j] = power;
                power *= x[i];
            }
        }

        std::vector<std::vector<double>> xtx(n_terms, std::vector<double>(n_terms, 0.0));
        std::vector<double> xty(n_terms, 0.0);
        for (size_t i = 0; i < x.size(); ++i) {
            for (size_t a = 0; a < n_terms; ++a) {
                xty[a] += basis[i][a] * y[i];
                for (size_t b = 0; b < n_terms; ++b) xtx[a][b] += basis[i][a] * basis[i][b];
            }
        }

        coeffs_ = solve(xtx, xty);
    }

    double predict(double x) const override {
        double power = 1.0, result = 0.0;
        for (double c : coeffs_) {
            result += c * power;
            power *= x;
        }
        return result;
    }

private:
    // Gauss-Jordan with partial pivoting. n_terms is small (a handful), so
    // this isn't performance-sensitive.
    static std::vector<double> solve(std::vector<std::vector<double>> a, std::vector<double> b) {
        size_t n = b.size();
        for (size_t col = 0; col < n; ++col) {
            size_t pivot = col;
            for (size_t row = col + 1; row < n; ++row) {
                if (std::fabs(a[row][col]) > std::fabs(a[pivot][col])) pivot = row;
            }
            std::swap(a[col], a[pivot]);
            std::swap(b[col], b[pivot]);

            if (std::fabs(a[col][col]) < 1e-12) {
                throw std::runtime_error("PolynomialRegressor: singular normal equations");
            }

            for (size_t row = 0; row < n; ++row) {
                if (row == col) continue;
                double factor = a[row][col] / a[col][col];
                for (size_t k = col; k < n; ++k) a[row][k] -= factor * a[col][k];
                b[row] -= factor * b[col];
            }
        }
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) x[i] = b[i] / a[i][i];
        return x;
    }

    size_t degree_;
    std::vector<double> coeffs_;
};
