#include <cstdint>
#include <cstring>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Philox32x4.h"

namespace py = pybind11;

namespace {

template <typename T>
py::array_t<T> to_array(const std::vector<T>& v) {
    py::array_t<T> out(v.size());
    if (!v.empty()) {
        std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(T));
    }
    return out;
}

py::array_t<uint32_t> batch_u32(uint32_t num_rands, uint32_t c0, uint32_t c1) {
    return to_array(philox32x4_batch(num_rands, c0, c1));
}

py::array_t<uint32_t> batch_u32_mt(uint32_t num_rands, uint32_t num_threads) {
    return to_array(philox32x4_batch_mt(num_rands, num_threads));
}

template <typename T>
py::array_t<T> normal_batch(uint32_t num_rands, uint32_t c0, uint32_t c1) {
    return to_array(philox32x4_normal_batch<T>(num_rands, c0, c1));
}

template <typename T>
py::array_t<T> normal_batch_mt(uint32_t num_rands, uint32_t num_threads) {
    return to_array(philox32x4_normal_batch_mt<T>(num_rands, num_threads));
}

}  // namespace

PYBIND11_MODULE(philox32x4_py, m) {
    m.doc() = "Philox4x32-10 batch generation on the CPU, backed by cpp/include/Philox32x4.h. "
              "Single-threaded and multithreaded variants; uint32 raw draws and float/double normals.";

    m.def("uint32_batch", &batch_u32, py::arg("num_rands"), py::arg("counter0_offset") = 0,
          py::arg("counter1_offset") = 0, "num_rands raw uint32 Philox draws, single-threaded.");
    m.def("uint32_batch_mt", &batch_u32_mt, py::arg("num_rands"), py::arg("num_threads"),
          "num_rands raw uint32 Philox draws, generated across num_threads threads.");

    m.def("float_normal_batch", &normal_batch<float>, py::arg("num_rands"),
          py::arg("counter0_offset") = 0, py::arg("counter1_offset") = 0,
          "num_rands float32 standard normal draws, single-threaded.");
    m.def("float_normal_batch_mt", &normal_batch_mt<float>, py::arg("num_rands"), py::arg("num_threads"),
          "num_rands float32 standard normal draws, generated across num_threads threads.");

    m.def("double_normal_batch", &normal_batch<double>, py::arg("num_rands"),
          py::arg("counter0_offset") = 0, py::arg("counter1_offset") = 0,
          "num_rands float64 standard normal draws, single-threaded.");
    m.def("double_normal_batch_mt", &normal_batch_mt<double>, py::arg("num_rands"), py::arg("num_threads"),
          "num_rands float64 standard normal draws, generated across num_threads threads.");
}
