#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "SplitMix64.h"

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
}
