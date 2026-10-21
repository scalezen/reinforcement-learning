#include "mc/GbmPathModel.h"

#include "GbmPathModel.cuh"

PathArray GbmPathModel::simulate_gpu() const {
    constexpr uint32_t c0_offset = 0;
    constexpr uint32_t c1_offset = 0;

    return PathArray(
        n_steps_ + 1, n_paths_, 1,
        simulate_gbm_path_model_gpu_impl(n_steps_, n_paths_, S0_, dt(), r_, sigma_, c0_offset, c1_offset, key0, key1));
}
