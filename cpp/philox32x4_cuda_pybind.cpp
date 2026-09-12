#include <cstddef>
#include <cstdint>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "Philox32x4Cuda.cuh"

namespace py = pybind11;

namespace {

py::array_t<uint32_t> next_batch(uint32_t num_rands, uint32_t counter0_offset, uint32_t counter1_offset) {
    std::vector<uint32_t> result = philox32x4_batch_cuda(num_rands, counter0_offset, counter1_offset);
    py::array_t<uint32_t> out(result.size());
    auto buf = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < result.size(); ++i) {
        buf(i) = result[i];
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(philox32x4_cuda_py, m) {
    m.doc() = "Philox4x32-10 batch generation on the GPU — backed by cpp/Philox32x4Cuda.cu, "
              "bit-for-bit identical to the CPU philox32x4_batch in cpp/include/Philox32x4.h "
              "for the same (num_rands, counter0_offset, counter1_offset).";

    m.def("next_batch", &next_batch, py::arg("num_rands"), py::arg("counter0_offset") = 0,
          py::arg("counter1_offset") = 0, "num_rands Philox4x32-10 uint32 draws, generated on the GPU.");
}
