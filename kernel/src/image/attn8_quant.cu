// ignis kernel leaf: the 8-bit DiT attention's quantizers (kernel/include/ignis_image_attn.h) --
// OURS (spec image-model/02 acceptance 4, GitHub #319; ADR 0051 tier 1). The scheme and the V
// layout are SageAttention2's (thu-ml/SageAttention d1a57a5: per-warp Q, per-block smoothed K,
// per-channel V transposed, padded and permuted in csrc/fused/fused.cu and sageattention/quant.py);
// the kernels were written here, nothing was copied.
//
//   quantize_q   one CTA per 128 query rows and head, one warp per 32-row scale block.
//   kv_stats     one CTA per head and 8 channels: the K mean (or a copy of `mean_from`'s) and
//                the V scales, each a fixed-order reduction over the run's keys.
//   kv_quantize  one CTA per 64-key block and head: K - mean to INT8 with the block's scale, V to
//                E4M3 with the channel scales, transposed and permuted through shared memory.

#include "attn8_layout.cuh"
#include "ignis_image_attn.h"
#include "image_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace ignis_image {
namespace {

__device__ __forceinline__ int8_t sat_s8(float x) {
  int32_t r;
  asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(x));
  return static_cast<int8_t>(r);
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  return v;
}

__device__ __forceinline__ void bf16x8(const uint4 &u, float (&f)[8]) {
  const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 p = __bfloat1622float2(h[i]);
    f[2 * i] = p.x;
    f[2 * i + 1] = p.y;
  }
}

// ---- Q -----------------------------------------------------------------------------------------

__global__ void __launch_bounds__(128) quantize_q_kernel(const __nv_bfloat16 *__restrict__ q, int n, int stride,
                                                         int8_t *__restrict__ codes, float *__restrict__ scales,
                                                         int n_pad) {
  const int h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row0 = blockIdx.x * kAttnQBlock + warp * kAttnQScaleRows;
  const int part = lane & 15;  // 8 channels at part * 8
  float x[16][8];
  float amax = 0.0f;
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int row = row0 + (lane >> 4) + 2 * k;
    uint4 u = make_uint4(0, 0, 0, 0);
    if (row < n) u = *reinterpret_cast<const uint4 *>(q + static_cast<std::size_t>(row) * stride + h * kAttnDim + part * 8);
    bf16x8(u, x[k]);
#pragma unroll
    for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(x[k][i]));
  }
  amax = warp_max(amax);
  const bool zero = !(amax >= kAttnInt8MinAmax);
  const float inv = zero ? 0.0f : __fdiv_rn(127.0f, amax);
  if (lane == 0) scales[static_cast<std::size_t>(h) * (n_pad / kAttnQScaleRows) + row0 / kAttnQScaleRows] = zero ? 1.0f : __fdiv_rn(amax, 127.0f);
#pragma unroll
  for (int k = 0; k < 16; ++k) {
    const int row = row0 + (lane >> 4) + 2 * k;
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t c = static_cast<uint8_t>(sat_s8(__fmul_rn(x[k][i], inv)));
      w[i >> 2] |= c << (8 * (i & 3));
    }
    *reinterpret_cast<uint2 *>(codes + (static_cast<std::size_t>(h) * n_pad + row) * kAttnDim + part * 8) = make_uint2(w[0], w[1]);
  }
}

// ---- K and V -----------------------------------------------------------------------------------

constexpr int kStatsThreads = 256;

// The K mean (unless `mean_from`) and the V scales of channels [g * 8, g * 8 + 8) of head h.
__global__ void __launch_bounds__(kStatsThreads) kv_stats_kernel(const __nv_bfloat16 *__restrict__ k,
                                                                 const __nv_bfloat16 *__restrict__ v, int n, int stride,
                                                                 const float *__restrict__ mean_from,
                                                                 float *__restrict__ mean, float *__restrict__ vscale) {
  __shared__ float red[2][8][kStatsThreads];
  const int h = blockIdx.y, g = blockIdx.x;
  const int col = h * kAttnDim + g * 8;
  float sum[8] = {}, vmax[8] = {};
  for (int t = threadIdx.x; t < n; t += kStatsThreads) {
    float f[8];
    if (mean_from == nullptr) {
      bf16x8(*reinterpret_cast<const uint4 *>(k + static_cast<std::size_t>(t) * stride + col), f);
#pragma unroll
      for (int i = 0; i < 8; ++i) sum[i] = __fadd_rn(sum[i], f[i]);
    }
    bf16x8(*reinterpret_cast<const uint4 *>(v + static_cast<std::size_t>(t) * stride + col), f);
#pragma unroll
    for (int i = 0; i < 8; ++i) vmax[i] = fmaxf(vmax[i], fabsf(f[i]));
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    red[0][i][threadIdx.x] = sum[i];
    red[1][i][threadIdx.x] = vmax[i];
  }
  __syncthreads();
  // A fixed tree: deterministic.
  for (int half = kStatsThreads / 2; half > 0; half >>= 1) {
    if (threadIdx.x < half) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        red[0][i][threadIdx.x] = __fadd_rn(red[0][i][threadIdx.x], red[0][i][threadIdx.x + half]);
        red[1][i][threadIdx.x] = fmaxf(red[1][i][threadIdx.x], red[1][i][threadIdx.x + half]);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < 8) {
    const int i = threadIdx.x;
    mean[col + i] = mean_from != nullptr ? mean_from[col + i] : __fdiv_rn(red[0][i][0], static_cast<float>(n));
    const float amax = red[1][i][0];
    vscale[col + i] = e4m3_token_scale(amax);
  }
}

constexpr int kQuantKvThreads = 128;

// Keys [b * 64, b * 64 + 64) of head h: K codes and the block's scale, V codes transposed.
__global__ void __launch_bounds__(kQuantKvThreads) kv_quantize_kernel(const __nv_bfloat16 *__restrict__ k,
                                                                      const __nv_bfloat16 *__restrict__ v, int n,
                                                                      int stride, uint8_t *__restrict__ run, Kv8Layout l) {
  __shared__ float red[kQuantKvThreads / 32];
  __shared__ __align__(16) uint8_t vt[kAttnDim][kAttnKBlock];
  const int h = blockIdx.y, b = blockIdx.x;
  const int part = threadIdx.x & 15;  // channels part * 8 .. + 8
  const int col = h * kAttnDim + part * 8;
  const float *mean = reinterpret_cast<const float *>(run + l.mean);
  const float *vscale = reinterpret_cast<const float *>(run + l.vscale);
  float mu[8], vs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    mu[i] = mean[col + i];
    vs[i] = vscale[col + i];
  }
  float kx[8][8];
  float amax = 0.0f;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int key = b * kAttnKBlock + (threadIdx.x >> 4) + 8 * r;
    if (key < n) {
      bf16x8(*reinterpret_cast<const uint4 *>(k + static_cast<std::size_t>(key) * stride + col), kx[r]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        kx[r][i] = __fsub_rn(kx[r][i], mu[i]);
        amax = fmaxf(amax, fabsf(kx[r][i]));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 8; ++i) kx[r][i] = 0.0f;
    }
  }
  amax = warp_max(amax);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  amax = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  const bool zero = !(amax >= kAttnInt8MinAmax);
  const float inv = zero ? 0.0f : __fdiv_rn(127.0f, amax);
  if (threadIdx.x == 0) {
    reinterpret_cast<float *>(run + l.kscale)[static_cast<std::size_t>(h) * (l.n_pad / kAttnKBlock) + b] =
        zero ? 1.0f : __fdiv_rn(amax, 127.0f);
  }
  int8_t *kcodes = reinterpret_cast<int8_t *>(run);
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int key = b * kAttnKBlock + (threadIdx.x >> 4) + 8 * r;
    uint32_t w[2] = {0, 0};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint32_t c = static_cast<uint8_t>(sat_s8(__fmul_rn(kx[r][i], inv)));
      w[i >> 2] |= c << (8 * (i & 3));
    }
    *reinterpret_cast<uint2 *>(kcodes + (static_cast<std::size_t>(h) * l.n_pad + key) * kAttnDim + part * 8) =
        make_uint2(w[0], w[1]);
  }
  // V: E4M3(v / scale) into the transposed tile at the permuted key position, then out by rows.
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int j = (threadIdx.x >> 4) + 8 * r;  // key within the block
    const int key = b * kAttnKBlock + j;
    float f[8] = {};
    if (key < n) bf16x8(*reinterpret_cast<const uint4 *>(v + static_cast<std::size_t>(key) * stride + col), f);
#pragma unroll
    for (int i = 0; i < 8; i += 2) {
      const float a = key < n ? __fdiv_rn(f[i], vs[i]) : 0.0f, c = key < n ? __fdiv_rn(f[i + 1], vs[i + 1]) : 0.0f;
      const uint16_t two = static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, c), __NV_SATFINITE, __NV_E4M3));
      vt[part * 8 + i][kv8_v_position(j)] = static_cast<uint8_t>(two & 0xFF);
      vt[part * 8 + i + 1][kv8_v_position(j)] = static_cast<uint8_t>(two >> 8);
    }
  }
  __syncthreads();
  uint8_t *vcodes = run + l.v;
#pragma unroll
  for (int i = 0; i < kAttnDim * (kAttnKBlock / 16) / kQuantKvThreads; ++i) {
    const int chunk = threadIdx.x + i * kQuantKvThreads;
    const int c = chunk >> 2, p = chunk & 3;
    *reinterpret_cast<uint4 *>(vcodes + (static_cast<std::size_t>(h) * kAttnDim + c) * l.n_pad + b * kAttnKBlock + p * 16) =
        *reinterpret_cast<const uint4 *>(&vt[c][p * 16]);
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" uint64_t ignis_image_attn_q8_bytes(uint32_t tokens) { return q8_layout(tokens).bytes; }

extern "C" uint64_t ignis_image_attn_kv8_bytes(uint32_t tokens) { return kv8_layout(tokens).bytes; }

extern "C" int32_t ignis_image_attn_quantize_q(const void *q, uint32_t tokens, uint32_t row_stride, void *q8,
                                               void *stream) {
  const std::string op = "ignis_image_attn_quantize_q";
  if (q == nullptr || q8 == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > (1u << 30)) return fail(op + ": tokens must be in [1, 2^30]");
  if (row_stride < kAttnHD || row_stride % 8 != 0) return fail(op + ": row_stride must be >= 4096 and a multiple of 8");
  if (!aligned(q, 16) || !aligned(q8, 16)) return fail(op + ": q and q8 must be 16-byte aligned");
  const Q8Layout l = q8_layout(tokens);
  uint8_t *base = static_cast<uint8_t *>(q8);
  quantize_q_kernel<<<dim3(l.n_pad / kAttnQBlock, kAttnHeads), 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(q), static_cast<int>(tokens), static_cast<int>(row_stride),
      reinterpret_cast<int8_t *>(base), reinterpret_cast<float *>(base + l.scales), static_cast<int>(l.n_pad));
  return check_launch(op.c_str());
}

extern "C" int32_t ignis_image_attn_store_kv(const void *k, const void *v, uint32_t tokens, uint32_t row_stride,
                                             const void *mean_from, uint32_t mean_from_tokens, void *kv8, void *stream) {
  const std::string op = "ignis_image_attn_store_kv";
  if (k == nullptr || v == nullptr || kv8 == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > (1u << 30)) return fail(op + ": tokens must be in [1, 2^30]");
  if (row_stride < kAttnHD || row_stride % 8 != 0) return fail(op + ": row_stride must be >= 4096 and a multiple of 8");
  if (!aligned(k, 16) || !aligned(v, 16) || !aligned(kv8, 16)) return fail(op + ": k, v and kv8 must be 16-byte aligned");
  if (mean_from != nullptr && (mean_from_tokens == 0 || !aligned(mean_from, 16))) {
    return fail(op + ": mean_from must be a 16-byte aligned run of at least one token");
  }
  const Kv8Layout l = kv8_layout(tokens);
  uint8_t *run = static_cast<uint8_t *>(kv8);
  const float *mean_src =
      mean_from != nullptr
          ? reinterpret_cast<const float *>(static_cast<const uint8_t *>(mean_from) + kv8_layout(mean_from_tokens).mean)
          : nullptr;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const auto *kb = static_cast<const __nv_bfloat16 *>(k);
  const auto *vb = static_cast<const __nv_bfloat16 *>(v);
  kv_stats_kernel<<<dim3(kAttnDim / 8, kAttnHeads), kStatsThreads, 0, s>>>(
      kb, vb, static_cast<int>(tokens), static_cast<int>(row_stride), mean_src, reinterpret_cast<float *>(run + l.mean),
      reinterpret_cast<float *>(run + l.vscale));
  if (check_launch(op.c_str()) != 0) return -1;
  kv_quantize_kernel<<<dim3(l.n_pad / kAttnKBlock, kAttnHeads), kQuantKvThreads, 0, s>>>(
      kb, vb, static_cast<int>(tokens), static_cast<int>(row_stride), run, l);
  return check_launch(op.c_str());
}

extern "C" const char *ignis_image_attn_last_error(void) { return last_error_cstr(); }
