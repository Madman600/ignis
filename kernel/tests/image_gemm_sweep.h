// The FP8 W8A8 GEMM's tile configurations for the bench's sweep (spec image-model/02, GitHub
// #319) -- OURS. image_gemm_sweep.cu instantiates kernel/src/image/fp8_gemm_kernel.cuh at every
// configuration; it is built into ignis_image_gemm_bench and ignis_kernel_image_gemm_test only,
// so the leaf carries the default configuration alone (ignis_image_gemm_w8a8).
#ifndef IGNIS_TESTS_IMAGE_GEMM_SWEEP_H
#define IGNIS_TESTS_IMAGE_GEMM_SWEEP_H

#include <cstdint>

namespace image_gemm_sweep {

// The number of tile configurations, and the index of the one the C ABI runs.
int config_count();
int default_config();
// "BMxBN, BK k-bytes, S stages, WMxWN warps, C CTAs/SM" for a configuration.
const char *config_name(int config);

// Sets every configuration's kernel attributes on the current device. 0, or -1 with the reason in
// ignis_image_gemm_last_error().
int32_t prepare();

// ignis_image_gemm_w8a8 on configuration `config`, same arguments and checks; prepare() first.
int32_t gemm_w8a8(int config, const void *weight, uint32_t rows, uint32_t cols, const void *codes, const float *scales,
                  uint32_t tokens, void *y, uint32_t y_f32, void *stream);

}  // namespace image_gemm_sweep

#endif  // IGNIS_TESTS_IMAGE_GEMM_SWEEP_H
