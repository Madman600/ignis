// ignis kernel leaf: the VAE's channel RMS norm with its SiLU (spec image-model/02 acceptance 7,
// GitHub #319) -- OURS. The definition is kernel/include/ignis_image_vae.h's.
//
// One warp per pixel: lane l on the 8-channel chunks l, l + 32, ... of the pixel's NHWC row. The
// sum of squares runs in a fixed order (each lane's channels in order, then the warp butterfly,
// after which every lane holds the same sum), and every rounding step is an _rn intrinsic, as in
// dit_ops.cu. The second pass reads the row again (from L1) and writes each lane's own chunks, so
// y may be x, for BF16 output only.

#include "ignis_image_vae.h"
#include "image_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kNormThreads = 256;
constexpr float kNormEps = 1e-12f;  // F.normalize's eps: the norm is clamped to it

__global__ void __launch_bounds__(kNormThreads)
    vae_rms_norm_kernel(const __nv_bfloat16 *x, std::size_t pixels, int channels, const __nv_bfloat16 *gamma,
                        float scale, uint32_t silu, void *y, uint32_t y_f32) {
  const std::size_t p = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
  if (p >= pixels) return;  // whole warps
  const int lane = threadIdx.x & 31;
  const int chunks = channels / 8;
  const uint4 *xr = reinterpret_cast<const uint4 *>(x + p * channels);
  float ss = 0.0f;
  for (int i = lane; i < chunks; i += 32) {
    const uint4 v = xr[i];
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      ss = __fmaf_rn(f.x, f.x, ss);
      ss = __fmaf_rn(f.y, f.y, ss);
    }
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xFFFFFFFFu, ss, m));
  const float inv = __fdiv_rn(scale, fmaxf(__fsqrt_rn(ss), kNormEps));
  for (int i = lane; i < chunks; i += 32) {
    const uint4 v = xr[i], g = reinterpret_cast<const uint4 *>(gamma)[i];
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
    const __nv_bfloat162 *gh = reinterpret_cast<const __nv_bfloat162 *>(&g);
    float o[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]), w = __bfloat1622float2(gh[j]);
      o[2 * j] = __fmul_rn(__fmul_rn(f.x, inv), w.x);
      o[2 * j + 1] = __fmul_rn(__fmul_rn(f.y, inv), w.y);
    }
    if (silu) {
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] = __fdiv_rn(o[j], __fadd_rn(1.0f, expf(-o[j])));
    }
    if (y_f32) {
      float4 *yr = reinterpret_cast<float4 *>(static_cast<float *>(y) + p * channels) + 2 * i;
      yr[0] = make_float4(o[0], o[1], o[2], o[3]);
      yr[1] = make_float4(o[4], o[5], o[6], o[7]);
    } else {
      uint4 b;
      __nv_bfloat162 *bh = reinterpret_cast<__nv_bfloat162 *>(&b);
#pragma unroll
      for (int j = 0; j < 4; ++j) bh[j] = __floats2bfloat162_rn(o[2 * j], o[2 * j + 1]);
      reinterpret_cast<uint4 *>(static_cast<__nv_bfloat16 *>(y) + p * channels)[i] = b;
    }
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_vae_rms_norm(const void *x, uint32_t pixels, uint32_t channels, const void *gamma,
                                            uint32_t silu, void *y, uint32_t y_f32, void *stream) {
  const std::string op = "ignis_image_vae_rms_norm";
  if (x == nullptr || gamma == nullptr || y == nullptr) return fail(op + ": NULL pointer");
  if (pixels == 0 || pixels > (1u << 24)) return fail(op + ": pixels must be 1 .. 2^24");
  if (channels == 0 || channels % 8 != 0) return fail(op + ": channels must be a positive multiple of 8");
  if (!aligned(x, 16) || !aligned(gamma, 16) || !aligned(y, 16)) {
    return fail(op + ": x, gamma and y must be 16-byte aligned");
  }
  // diffusers' scale is the Python float channels ** 0.5, which PyTorch's BF16 multiply takes as
  // an fp32 scalar.
  const float scale = static_cast<float>(std::sqrt(static_cast<double>(channels)));
  const std::size_t threads = static_cast<std::size_t>(pixels) * 32;
  vae_rms_norm_kernel<<<static_cast<unsigned>((threads + kNormThreads - 1) / kNormThreads), kNormThreads, 0,
                        static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16 *>(x), pixels,
                                                             static_cast<int>(channels),
                                                             static_cast<const __nv_bfloat16 *>(gamma), scale, silu, y,
                                                             y_f32);
  return check_launch(op.c_str());
}
