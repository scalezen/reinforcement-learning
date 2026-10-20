#pragma once

#include <algorithm>

class Payoff {
public:
    virtual ~Payoff() = default;
    virtual double operator()(double spot) const = 0;
};

class PutPayoff : public Payoff {
public:
    explicit PutPayoff(double strike) : strike_(strike) {}
    double operator()(double spot) const override { return std::max(strike_ - spot, 0.0); }

private:
    double strike_;
};

class CallPayoff : public Payoff {
public:
    explicit CallPayoff(double strike) : strike_(strike) {}
    double operator()(double spot) const override { return std::max(spot - strike_, 0.0); }

private:
    double strike_;
};
