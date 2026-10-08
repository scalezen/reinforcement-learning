// Float: M. B. Giles, "Approximating the erfinv function", GPU Computing Gems, 2010
// (single-precision coefficients; source at http://gpucomputing.net/?q=node/1828).
// Double: not Giles' double-precision code. It refines the float approximation
// with Newton steps on std::erf.

#pragma once

#include <cmath>
#include <limits>

#if defined(__CUDACC__)
#define PHILOX_HOST_DEVICE __host__ __device__
#else
#define PHILOX_HOST_DEVICE
#endif

template <typename T>
T erfinv_giles(T x);

template <>
PHILOX_HOST_DEVICE inline float erfinv_giles<float>(float x) {
    float w = -std::log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = std::sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

template <>
inline double erfinv_giles<double>(double x) {
    if (x >= 1.0) return std::numeric_limits<double>::infinity();
    if (x <= -1.0) return -std::numeric_limits<double>::infinity();

    constexpr double kTwoOverSqrtPi = 1.1283791670955126;
    float seed_in = static_cast<float>(x);
    if (seed_in >= 1.0f) seed_in = std::nextafter(1.0f, 0.0f);
    if (seed_in <= -1.0f) seed_in = std::nextafter(-1.0f, 0.0f);

    double y = erfinv_giles<float>(seed_in);
    for (int i = 0; i < 100; ++i) {
        double step = (std::erf(y) - x) / (kTwoOverSqrtPi * std::exp(-y * y));
        y -= step;
        if (std::fabs(step) <= 1e-15 * std::fabs(y)) break;
    }
    return y;
}
