// ignis kernel leaf: the standalone per-token E4M3 activation quantization (spec image-model/02
// acceptance 3, GitHub #319) -- OURS. The definition is kernel/include/ignis_image_gemm.h's, the
// device steps fp8_quant.cuh's.
//
// One CTA per token: the row is read once for its amax (a fixed-order block reduction), then again
// -- from L1/L2 -- to write its codes, 8 BF16 in and 8 codes out per thread per step.

#include "ignis_image_gemm.h"
#include "image_common.cuh"
#include "fp8_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace ignis_image {
namespace {

constexpr int kQuantThreads = 256;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  return v;
}

__global__ void __launch_bounds__(kQuantThreads) quantize_tokens_kernel(const __nv_bfloat16 *__restrict__ x, int cols,
                                                                        uint8_t *__restrict__ codes,
                                                                        float *__restrict__ scales) {
  __shared__ float partial[kQuantThreads / 32];
  const std::size_t row = blockIdx.x;
  const uint4 *xr = reinterpret_cast<const uint4 *>(x + row * cols);
  const int chunks = cols / 8;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kQuantThreads) {
    const uint4 v = xr[i];
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = 0.0f;
#pragma unroll
  for (int w = 0; w < kQuantThreads / 32; ++w) amax = fmaxf(amax, partial[w]);
  const float inverse = e4m3_token_inverse(amax);
  if (threadIdx.x == 0) scales[row] = e4m3_token_scale(amax);
  uint2 *cr = reinterpret_cast<uint2 *>(codes + row * cols);
  for (int i = threadIdx.x; i < chunks; i += kQuantThreads) {
    const uint4 v = xr[i];
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
    uint32_t packed[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 lo = __bfloat1622float2(h[2 * j]), hi = __bfloat1622float2(h[2 * j + 1]);
      packed[j] = static_cast<uint32_t>(e4m3x2_quantize(lo.x, lo.y, inverse)) |
                  (static_cast<uint32_t>(e4m3x2_quantize(hi.x, hi.y, inverse)) << 16);
    }
    cr[i] = make_uint2(packed[0], packed[1]);
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_quantize_tokens(const void *x, uint32_t tokens, uint32_t cols, void *codes, float *scales,
                                               void *stream) {
  const char *op = "ignis_image_quantize_tokens";
  if (x == nullptr || codes == nullptr || scales == nullptr) return fail(std::string(op) + ": NULL pointer");
  if (tokens == 0) return fail(std::string(op) + ": tokens must be at least 1");
  if (cols == 0 || cols % 16 != 0) return fail(std::string(op) + ": cols must be a positive multiple of 16");
  if (!aligned(x, 16) || !aligned(codes, 16) || !aligned(scales, 4)) {
    return fail(std::string(op) + ": x and codes must be 16-byte aligned, scales 4-byte aligned");
  }
  quantize_tokens_kernel<<<tokens, kQuantThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(x), static_cast<int>(cols), static_cast<uint8_t *>(codes), scales);
  return check_launch(op);
}
