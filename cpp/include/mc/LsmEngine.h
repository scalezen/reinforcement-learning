#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "PathModel.h"
#include "Payoff.h"
#include "Regressor.h"

struct LsmResult {
    double price;
    double standard_error;
};

// Longstaff-Schwartz (2001): simulate, then walk backward from maturity,
// regressing the discounted future cashflow on the in-the-money paths'
// state to estimate continuation value, and exercising wherever immediate
// payoff beats it. model.simulate() and regressor.fit()/predict() are the
// two compute-heavy seams; this loop itself is sequential over time steps
// and stays on the CPU.
class LsmEngine {
public:
    LsmEngine(const PathModel& model, const Payoff& payoff, Regressor& regressor, double r)
        : model_(model), payoff_(payoff), regressor_(regressor), r_(r) {
        if (model_.num_factors() != 1) {
            throw std::invalid_argument("LsmEngine currently assumes a single-factor model (spot only)");
        }
    }

    LsmResult price() const {
        PathArray paths = model_.simulate();
        size_t n_steps = model_.num_steps();
        size_t n_paths = model_.num_paths();
        double discount = std::exp(-r_ * model_.dt());

        std::vector<double> cashflow(n_paths);
        const double* maturity = paths.time_slice(n_steps);
        for (size_t path = 0; path < n_paths; ++path) cashflow[path] = payoff_(maturity[path]);

        // t runs n_steps-1 down to 1: exercise dates strictly before maturity.
        // t=0 isn't an exercise opportunity; its value is just the final discount below.
        for (size_t t = n_steps - 1; t >= 1; --t) {
            for (double& cf : cashflow) cf *= discount;

            const double* spot = paths.time_slice(t);

            std::vector<double> itm_x, itm_y;
            std::vector<size_t> itm_idx;
            for (size_t path = 0; path < n_paths; ++path) {
                if (payoff_(spot[path]) > 0.0) {
                    itm_x.push_back(spot[path]);
                    itm_y.push_back(cashflow[path]);
                    itm_idx.push_back(path);
                }
            }
            if (itm_x.empty()) continue;

            regressor_.fit(itm_x, itm_y);

            for (size_t path : itm_idx) {
                double exercise = payoff_(spot[path]);
                double continuation = regressor_.predict(spot[path]);
                if (exercise > continuation) cashflow[path] = exercise;
            }
        }

        double mean = std::accumulate(cashflow.begin(), cashflow.end(), 0.0) / static_cast<double>(n_paths);
        double discounted_to_zero = mean * discount;  // cashflow is valued at t=1; one more step to t=0

        double sq = 0.0;
        for (double cf : cashflow) {
            double diff = cf * discount - discounted_to_zero;
            sq += diff * diff;
        }
        double se = std::sqrt(sq / static_cast<double>(n_paths - 1) / static_cast<double>(n_paths));

        return {discounted_to_zero, se};
    }

private:
    const PathModel& model_;
    const Payoff& payoff_;
    Regressor& regressor_;
    double r_;
};
