// ignis kernel leaf: the FP8 W8A8 GEMM's tile configurations (kernel/src/image/fp8_gemm.cu) --
// OURS (spec image-model/02, GitHub #319). Leaf-internal: ignis_image_gemm_w8a8 runs
// kGemmDefaultConfig; the others exist for kernel/tests/bench_image_gemm.cu's sweep and are held
// to the same mechanics test (kernel/tests/test_image_gemm.cu) at the edge shapes.
#ifndef IGNIS_IMAGE_FP8_GEMM_INTERNAL_H
#define IGNIS_IMAGE_FP8_GEMM_INTERNAL_H

#include <cstdint>

namespace ignis_image {

// The number of tile configurations, and the one the public entry point takes.
int gemm_config_count();
int gemm_default_config();
// "BMxBN, BK k-bytes, S stages, WMxWN warps, C CTAs/SM" for a configuration.
const char *gemm_config_name(int config);

// ignis_image_gemm_w8a8 on configuration `config`, same arguments and checks.
int32_t gemm_w8a8_config(int config, const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream);

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_FP8_GEMM_INTERNAL_H
