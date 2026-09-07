#include <cstddef>
#include <stdexcept>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SplitMix64.h"
#include "SplitMix64Parallel.h"

namespace py = pybind11;

namespace {

py::array_t<uint64_t> next_batch(SplitMix64 &self, py::ssize_t n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    py::array_t<uint64_t> out(n);
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
        buf(i) = self.next();
    }
    return out;
}

py::array_t<double> next_uniform(SplitMix64 &self, py::ssize_t n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    py::array_t<double> out(n);
    auto buf = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
        buf(i) = self.next_double();
    }
    return out;
}

py::array_t<uint64_t> next_batch_parallel(SplitMix64Parallel &self, py::ssize_t n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    std::vector<uint64_t> result = self.next_batch(static_cast<std::size_t>(n));
    py::array_t<uint64_t> out(result.size());
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < result.size(); ++i) {
        buf(i) = result[i];
    }
    return out;
}

py::array_t<double> next_uniform_parallel(SplitMix64Parallel &self, py::ssize_t n) {
    if (n < 0) {
        throw std::invalid_argument("n must be non-negative");
    }
    std::vector<double> result = self.next_uniform(static_cast<std::size_t>(n));
    py::array_t<double> out(result.size());
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < result.size(); ++i) {
        buf(i) = result[i];
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(splitmix64_py, m) {
    m.doc() = "SplitMix64 PRNG — single-draw and batch calls, backed by cpp/include/SplitMix64.h";

    py::class_<SplitMix64>(m, "SplitMix64")
        .def(py::init<uint64_t>(), py::arg("seed"))
        .def("next", &SplitMix64::next, "Next raw uint64 draw; advances the stream by one.")
        .def("next_double", &SplitMix64::next_double,
             "Next uniform double in [0, 1); advances the stream by one.")
        .def("next_batch", &next_batch, py::arg("n"),
             "n raw uint64 draws as a numpy array, filled in C++ with no per-call overhead.")
        .def("next_uniform", &next_uniform, py::arg("n"),
             "n uniform doubles in [0, 1) as a numpy array, filled in C++ with no per-call overhead.");

    py::class_<SplitMix64Parallel>(m, "SplitMix64Parallel")
        .def(py::init<uint64_t, unsigned>(), py::arg("seed"), py::arg("n_threads"),
             "Reproduces the sequence of SplitMix64(seed), split across n_threads "
             "worker threads computed in parallel for speed."
        .def("next_batch", &next_batch_parallel, py::arg("n"),
             "n raw uint64 draws, matching SplitMix64(seed).next_batch(n), computed across "
             "n_threads worker threads.")
        .def("next_uniform", &next_uniform_parallel, py::arg("n"),
             "n uniform doubles in [0, 1), matching SplitMix64(seed).next_uniform(n), "
             "computed across n_threads worker threads.")
        .def_property_readonly("n_threads", &SplitMix64Parallel::n_threads);
}
