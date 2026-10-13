// ignis kernel leaf: the per-token E4M3 activation quantization's device steps (spec
// image-model/02 acceptance 3, GitHub #319) -- OURS. The definition is kernel/include/
// ignis_image_gemm.h's; a producer that fuses the quantization into its epilogue (a norm or
// modulation, SwiGLU, the attention output) computes its row's amax, then calls these, and emits
// the same codes and scale as ignis_image_quantize_tokens would.
#ifndef IGNIS_IMAGE_FP8_QUANT_CUH
#define IGNIS_IMAGE_FP8_QUANT_CUH

#include <cuda_fp8.h>

#include <cstdint>

namespace ignis_image {

// E4M3FN's largest finite magnitude: a token's amax maps to it.
constexpr float kE4m3Max = 448.0f;

// scales[t]: amax / 448, or 1 for an all-zero row.
__device__ __forceinline__ float e4m3_token_scale(float amax) {
  return amax > 0.0f ? __fdiv_rn(amax, kE4m3Max) : 1.0f;
}

// The multiplier the codes are taken with: 448 / amax, or 0 for an all-zero row.
__device__ __forceinline__ float e4m3_token_inverse(float amax) {
  return amax > 0.0f ? __fdiv_rn(kE4m3Max, amax) : 0.0f;
}

// Two codes, `a` in the low byte: E4M3(v * inverse), the product rounded once in fp32, then to
// nearest even and saturated to +-448 (NaN stays NaN).
__device__ __forceinline__ uint16_t e4m3x2_quantize(float a, float b, float inverse) {
  return static_cast<uint16_t>(
      __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(a, inverse), __fmul_rn(b, inverse)), __NV_SATFINITE, __NV_E4M3));
}

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_FP8_QUANT_CUH
