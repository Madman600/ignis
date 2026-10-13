// The FP8 W8A8 GEMM at every tile configuration of the bench's sweep (image_gemm_sweep.h) --
// OURS (spec image-model/02, GitHub #319).

#include "image_gemm_sweep.h"

#include "image/fp8_gemm_kernel.cuh"

namespace image_gemm_sweep {
namespace {

using ignis_image::GemmDefaultConfig;
using ignis_image::TileConfig;

// The configurations: <BM, BN, BK, STAGES, WM, WN, CTAs per SM>. 6 is the leaf's.
using Config0 = TileConfig<128, 128, 64, 4, 2, 4, 1>;
using Config1 = TileConfig<128, 128, 64, 3, 2, 4, 2>;
using Config2 = TileConfig<128, 128, 128, 3, 2, 4, 1>;
using Config3 = TileConfig<128, 256, 64, 3, 2, 4, 1>;
using Config4 = TileConfig<256, 128, 64, 3, 4, 2, 1>;
using Config5 = TileConfig<128, 128, 64, 3, 2, 2, 2>;
using Config6 = GemmDefaultConfig;
constexpr int kConfigs = 7;
constexpr int kDefaultConfig = 6;
const char *const kConfigNames[kConfigs] = {
    "128x128, BK 64, 4 stages, 2x4 warps, 1 CTA/SM",  "128x128, BK 64, 3 stages, 2x4 warps, 2 CTAs/SM",
    "128x128, BK 128, 3 stages, 2x4 warps, 1 CTA/SM", "128x256, BK 64, 3 stages, 2x4 warps, 1 CTA/SM",
    "256x128, BK 64, 3 stages, 4x2 warps, 1 CTA/SM",  "128x128, BK 64, 3 stages, 2x2 warps, 2 CTAs/SM",
    "128x256, BK 64, 4 stages, 2x4 warps, 1 CTA/SM",
};

// Runs f(Config) for configuration `config`.
template <class F> void with_config(int config, F &&f) {
  switch (config) {
  case 0: f(Config0{}); break;
  case 1: f(Config1{}); break;
  case 2: f(Config2{}); break;
  case 3: f(Config3{}); break;
  case 4: f(Config4{}); break;
  case 5: f(Config5{}); break;
  default: f(Config6{}); break;
  }
}

}  // namespace

int config_count() { return kConfigs; }
int default_config() { return kDefaultConfig; }
const char *config_name(int config) {
  return config >= 0 && config < kConfigs ? kConfigNames[config] : "no such configuration";
}

int32_t prepare() {
  cudaError_t err = cudaSuccess;
  for (int config = 0; config < kConfigs && err == cudaSuccess; ++config) {
    with_config(config, [&](auto c) { err = ignis_image::gemm_w8a8_set_attributes<decltype(c)>(); });
  }
  if (err != cudaSuccess) return ignis_image::fail(std::string("image_gemm_sweep::prepare: ") + cudaGetErrorString(err));
  return 0;
}

int32_t gemm_w8a8(int config, const void *weight, uint32_t rows, uint32_t cols, const void *codes, const float *scales,
                  uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  if (config < 0 || config >= kConfigs) return ignis_image::fail("ignis_image_gemm_w8a8: no such tile configuration");
  int32_t rc = 0;
  with_config(config, [&](auto c) {
    rc = ignis_image::gemm_w8a8_launch<decltype(c)>(weight, rows, cols, codes, scales, tokens, y, y_f32, stream);
  });
  return rc;
}

}  // namespace image_gemm_sweep
