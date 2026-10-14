// ignis kernel leaf: the image model's text encoder, Qwen3-VL-8B's text model (spec image-model/03,
// GitHub #320) -- OURS. The definitions are kernel/include/ignis_image_encoder.h's.
//
// The residual norm: one CTA of 256 threads per token, each thread on the row's 8-element chunks
// tid, tid + 256, ...; the fp32 row (after the add) is staged in shared memory, so the norm's passes
// read global memory once. Sums run in a fixed order: each thread's elements in order, a warp
// butterfly, then the 8 warps in order from 0 (dit_ops.cu's). Every rounding step is spelled with
// an _rn intrinsic, so nvcc's FMA contraction cannot change the sequence the tests' bound follows.
//
// The QK norm with mrope: one warp per (token, head), q's 32 heads then k's 8; lane l holds
// channels 2l, 2l + 1 and 64 + 2l, 65 + 2l, the two rotate_half pairs of slots 2l and 2l + 1.
//
// The forward: the ops in the header's order, on views of one workspace (below).

#include "ignis_image_encoder.h"
#include "ignis_image_dit.h"
#include "ignis_image_encoder_attn.h"
#include "ignis_image_gemm.h"
#include "image_common.cuh"
#include "fp8_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kNormThreads = 256;
constexpr uint32_t kMaxNormCols = 8192;  // the fp32 row in 32 KiB of shared memory
constexpr uint32_t kMaxTokens = 1u << 20;
constexpr uint32_t kHidden = IGNIS_IMAGE_ENCODER_HIDDEN;
constexpr uint32_t kHeads = IGNIS_IMAGE_ENCODER_HEADS;
constexpr uint32_t kKvHeads = IGNIS_IMAGE_ENCODER_KV_HEADS;
constexpr uint32_t kHeadDim = IGNIS_IMAGE_ENCODER_HEAD_DIM;
constexpr uint32_t kInter = IGNIS_IMAGE_ENCODER_INTERMEDIATE;
constexpr uint32_t kKv = kKvHeads * kHeadDim;
constexpr int kSlots = kHeadDim / 2;  // 64 frequency slots, rotate_half pairs (d, d + 64)
// transformers' Qwen3VLTextAttention.scaling, head_dim ** -0.5, rounded once to fp32.
constexpr float kAttnScale = static_cast<float>(0.08838834764831845);

// The mrope's per-slot frequency and axis (0 t, 1 h, 2 w), by value into the kernel.
struct Mrope {
  float freq[kSlots];
  int32_t axis[kSlots];
};

__device__ float block_sum(float v, float *red) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.0f;
#pragma unroll
  for (int w = 0; w < kNormThreads / 32; ++w) s = __fadd_rn(s, red[w]);
  return s;
}

__device__ float block_max(float v, float *red) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.0f;
#pragma unroll
  for (int w = 0; w < kNormThreads / 32; ++w) s = fmaxf(s, red[w]);
  return s;
}

__device__ __forceinline__ void unpack8(const uint4 &v, float (&f)[8]) {
  const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 p = __bfloat1622float2(h[j]);
    f[2 * j] = p.x;
    f[2 * j + 1] = p.y;
  }
}

__device__ __forceinline__ void load8(const float *p, float (&f)[8]) {
  const float4 a = reinterpret_cast<const float4 *>(p)[0], b = reinterpret_cast<const float4 *>(p)[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8(float *p, const float (&f)[8]) {
  reinterpret_cast<float4 *>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
  reinterpret_cast<float4 *>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

__global__ void __launch_bounds__(kNormThreads)
    residual_norm_kernel(float *x, const float *delta, int cols, const __nv_bfloat16 *weight, float eps, void *y,
                         float *y_scales, uint32_t y_f32) {
  extern __shared__ float4 srow[];  // the row after the add, two float4 per chunk
  __shared__ float red[kNormThreads / 32];
  const std::size_t t = blockIdx.x;
  const int chunks = cols / 8;
  float *xr = x + t * cols;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kNormThreads) {
    float v[8];
    load8(xr + 8 * i, v);
    if (delta != nullptr) {
      float d[8];
      load8(delta + t * cols + 8 * i, d);
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = __fadd_rn(v[j], d[j]);
      store8(xr + 8 * i, v);
    }
    if (weight != nullptr) {
      store8(reinterpret_cast<float *>(srow) + 8 * i, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = __fmaf_rn(v[j], v[j], acc);
    }
  }
  if (weight == nullptr) return;
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum(acc, red), static_cast<float>(cols)), eps));

  // y = (x * r) * w, computed in each pass from the staged row: the amax, then the codes.
  const std::size_t out_row = t * static_cast<std::size_t>(cols);
  float amax = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kNormThreads) {
    float v[8], w[8], o[8];
    load8(reinterpret_cast<const float *>(srow) + 8 * i, v);
    unpack8(reinterpret_cast<const uint4 *>(weight)[i], w);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = __fmul_rn(__fmul_rn(v[j], r), w[j]);
      amax = fmaxf(amax, fabsf(o[j]));
    }
    if (y_f32) store8(static_cast<float *>(y) + out_row + 8 * i, o);
  }
  if (y_f32) return;

  amax = block_max(amax, red);
  const float inverse = e4m3_token_inverse(amax);
  if (threadIdx.x == 0) y_scales[t] = e4m3_token_scale(amax);
  uint2 *codes = reinterpret_cast<uint2 *>(static_cast<uint8_t *>(y) + out_row);
  for (int i = threadIdx.x; i < chunks; i += kNormThreads) {
    float v[8], w[8], o[8];
    load8(reinterpret_cast<const float *>(srow) + 8 * i, v);
    unpack8(reinterpret_cast<const uint4 *>(weight)[i], w);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __fmul_rn(__fmul_rn(v[j], r), w[j]);
    uint32_t packed[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      packed[j] = static_cast<uint32_t>(e4m3x2_quantize(o[4 * j], o[4 * j + 1], inverse)) |
                  (static_cast<uint32_t>(e4m3x2_quantize(o[4 * j + 2], o[4 * j + 3], inverse)) << 16);
    }
    codes[i] = make_uint2(packed[0], packed[1]);
  }
}

// One warp per (token, head): warp w is token w / 40, head w % 40 (q's 32, then k's 8).
__global__ void __launch_bounds__(256)
    qk_norm_mrope_kernel(const float *q, std::size_t q_stride, const float *k, std::size_t k_stride, std::size_t tokens,
                         const __nv_bfloat16 *q_weight, const __nv_bfloat16 *k_weight, float eps,
                         const int32_t *positions, Mrope m, __nv_bfloat16 *q_out, __nv_bfloat16 *k_out) {
  constexpr int kWarpsPerToken = kHeads + kKvHeads;
  const std::size_t warp = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
  if (warp >= tokens * kWarpsPerToken) return;  // whole warps: 256 threads, 40 warps per token
  const int lane = threadIdx.x & 31;
  const std::size_t t = warp / kWarpsPerToken;
  const int head = static_cast<int>(warp % kWarpsPerToken);
  const bool is_k = head >= static_cast<int>(kHeads);
  const int h = is_k ? head - static_cast<int>(kHeads) : head;
  const float *src = (is_k ? k + t * k_stride : q + t * q_stride) + static_cast<std::size_t>(h) * kHeadDim;
  const __nv_bfloat16 *wsrc = is_k ? k_weight : q_weight;
  const float2 lo = reinterpret_cast<const float2 *>(src)[lane];
  const float2 hi = reinterpret_cast<const float2 *>(src + kSlots)[lane];
  const float2 wlo = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(wsrc)[lane]);
  const float2 whi = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162 *>(wsrc + kSlots)[lane]);
  const float x[4] = {lo.x, lo.y, hi.x, hi.y}, w[4] = {wlo.x, wlo.y, whi.x, whi.y};
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss = __fmaf_rn(x[j], x[j], ss);
#pragma unroll
  for (int s = 16; s > 0; s >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xFFFFFFFFu, ss, s));
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(kHeadDim)), eps));
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = __fmul_rn(__fmul_rn(x[j], r), w[j]);
  // slots 2 lane and 2 lane + 1: n[0], n[1] are their first channels, n[2], n[3] the second (+ 64).
  float c[2], s[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int slot = 2 * lane + j;
    const float angle = __fmul_rn(static_cast<float>(positions[m.axis[slot] * tokens + t]), m.freq[slot]);
    sincosf(angle, &s[j], &c[j]);
  }
  const float2 out_lo = make_float2(__fsub_rn(__fmul_rn(n[0], c[0]), __fmul_rn(n[2], s[0])),
                                    __fsub_rn(__fmul_rn(n[1], c[1]), __fmul_rn(n[3], s[1])));
  const float2 out_hi = make_float2(__fadd_rn(__fmul_rn(n[2], c[0]), __fmul_rn(n[0], s[0])),
                                    __fadd_rn(__fmul_rn(n[3], c[1]), __fmul_rn(n[1], s[1])));
  __nv_bfloat16 *dst = is_k ? k_out + t * kKv : q_out + t * (kHeads * kHeadDim);
  dst += static_cast<std::size_t>(h) * kHeadDim;
  reinterpret_cast<__nv_bfloat162 *>(dst)[lane] = __floats2bfloat162_rn(out_lo.x, out_lo.y);
  reinterpret_cast<__nv_bfloat162 *>(dst + kSlots)[lane] = __floats2bfloat162_rn(out_hi.x, out_hi.y);
}

// One CTA per source row.
__global__ void rows_kernel(float *x, std::size_t tokens, int cols, const int32_t *row_index,
                            const __nv_bfloat16 *src, uint32_t add) {
  const std::size_t j = blockIdx.x;
  const int32_t r = row_index[j];
  if (r < 0 || static_cast<std::size_t>(r) >= tokens) return;
  float *xr = x + static_cast<std::size_t>(r) * cols;
  const uint4 *sr = reinterpret_cast<const uint4 *>(src + j * cols);
  for (int i = threadIdx.x; i < cols / 8; i += blockDim.x) {
    float f[8];
    unpack8(sr[i], f);
    if (add) {
      float v[8];
      load8(xr + 8 * i, v);
#pragma unroll
      for (int e = 0; e < 8; ++e) f[e] = __fadd_rn(v[e], f[e]);
    }
    store8(xr + 8 * i, f);
  }
}

// The mrope's slots (ignis_image_encoder.h): f_i = fl32(1 / fl32(theta^fl32(2i / 128))), each step
// correctly rounded, and transformers' interleaved axis rule.
Mrope make_mrope(float theta, const uint32_t *section) {
  Mrope m{};
  for (int i = 0; i < kSlots; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(kHeadDim);
    const float p = static_cast<float>(std::pow(static_cast<double>(theta), static_cast<double>(e)));
    m.freq[i] = static_cast<float>(1.0 / static_cast<double>(p));
    const uint32_t u = static_cast<uint32_t>(i);
    m.axis[i] = (u % 3 == 1 && u < 3 * section[1]) ? 1 : ((u % 3 == 2 && u < 3 * section[2]) ? 2 : 0);
  }
  return m;
}

std::string mrope_checks(const char *op, float theta, const uint32_t *section) {
  const std::string o(op);
  if (section == nullptr) return o + ": NULL mrope_section";
  if (section[0] + section[1] + section[2] != static_cast<uint32_t>(kSlots) || 3 * section[1] > kSlots ||
      3 * section[2] > kSlots) {
    return o + ": mrope_section must sum to 64 with 3 section[1] and 3 section[2] at most 64";
  }
  if (!(theta > 1.0f) || !std::isfinite(theta)) return o + ": rope_theta must be finite and > 1";
  return "";
}

constexpr std::size_t align256(std::size_t n) { return (n + 255) & ~static_cast<std::size_t>(255); }

// The forward's workspace, carved from one buffer:
//   scales   fp32 [tokens], the E4M3 activations' token scales;
//   codes    E4M3 [tokens][12288], the GEMM input (the norms', the attention output's, SwiGLU's);
//   region   49,152 bytes a token, reused by phase:
//     attention  q fp32 [T][4096] | k fp32 [T][1024] | v BF16 [T][1024] | q BF16 [T][4096] |
//                k BF16 [T][1024] | the attention output BF16 [T][4096]          (40,960 B a token)
//     MLP        gate BF16 [T][12288] | up BF16 [T][12288]                       (49,152 B a token)
//     delta      the out-projections' fp32 [T][4096] at the region's start: o_proj writes it once
//                the attention output is quantized, down_proj once SwiGLU has read gate and up.
struct Workspace {
  float *scales;
  uint8_t *codes;
  float *q32, *k32, *delta;
  __nv_bfloat16 *v, *qb, *kb, *attn, *gate, *up;
};

std::size_t workspace_size(std::size_t tokens) {
  return align256(4 * tokens) + align256(static_cast<std::size_t>(kInter) * tokens) + 49152 * tokens;
}

Workspace carve(void *base, std::size_t tokens) {
  uint8_t *p = static_cast<uint8_t *>(base);
  Workspace w{};
  w.scales = reinterpret_cast<float *>(p);
  p += align256(4 * tokens);
  w.codes = p;
  p += align256(static_cast<std::size_t>(kInter) * tokens);
  w.q32 = reinterpret_cast<float *>(p);
  w.delta = w.q32;
  w.k32 = reinterpret_cast<float *>(p + 16384 * tokens);
  w.v = reinterpret_cast<__nv_bfloat16 *>(p + 20480 * tokens);
  w.qb = reinterpret_cast<__nv_bfloat16 *>(p + 22528 * tokens);
  w.kb = reinterpret_cast<__nv_bfloat16 *>(p + 30720 * tokens);
  w.attn = reinterpret_cast<__nv_bfloat16 *>(p + 32768 * tokens);
  w.gate = reinterpret_cast<__nv_bfloat16 *>(p);
  w.up = reinterpret_cast<__nv_bfloat16 *>(p + 24576 * tokens);
  return w;
}

int32_t launch_residual_norm(float *x, const float *delta, uint32_t tokens, uint32_t cols, const void *weight,
                             float eps, void *y, float *y_scales, uint32_t y_f32, void *stream) {
  const std::size_t smem = weight != nullptr ? static_cast<std::size_t>(cols) * 4 : 0;
  residual_norm_kernel<<<tokens, kNormThreads, smem, static_cast<cudaStream_t>(stream)>>>(
      x, delta, static_cast<int>(cols), static_cast<const __nv_bfloat16 *>(weight), eps, y, y_scales, y_f32);
  return check_launch("ignis_image_encoder_residual_norm");
}

int32_t launch_qk_norm_mrope(const float *q, uint32_t q_stride, const float *k, uint32_t k_stride, uint32_t tokens,
                             const void *q_weight, const void *k_weight, float eps, const int32_t *positions,
                             const Mrope &m, void *q_out, void *k_out, void *stream) {
  const std::size_t blocks = static_cast<std::size_t>(tokens) * (kHeads + kKvHeads) * 32 / 256;
  qk_norm_mrope_kernel<<<static_cast<unsigned>(blocks), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      q, q_stride, k, k_stride, tokens, static_cast<const __nv_bfloat16 *>(q_weight),
      static_cast<const __nv_bfloat16 *>(k_weight), eps, positions, m, static_cast<__nv_bfloat16 *>(q_out),
      static_cast<__nv_bfloat16 *>(k_out));
  return check_launch("ignis_image_encoder_qk_norm_mrope");
}

int32_t launch_rows(float *x, uint32_t tokens, uint32_t cols, const int32_t *row_index, uint32_t rows, const void *src,
                    uint32_t add, void *stream) {
  rows_kernel<<<rows, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      x, tokens, static_cast<int>(cols), row_index, static_cast<const __nv_bfloat16 *>(src), add);
  return check_launch("ignis_image_encoder_rows");
}

// A sub-call's failure, with where it happened in front of its own message.
int32_t failed_at(uint32_t layer, const char *what) {
  return fail("ignis_image_encoder_forward: layer " + std::to_string(layer) + ", " + what + ": " + last_error_cstr());
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" uint64_t ignis_image_encoder_workspace_bytes(uint32_t tokens) { return workspace_size(tokens); }

extern "C" int32_t ignis_image_encoder_residual_norm(float *x, const float *delta, uint32_t tokens, uint32_t cols,
                                                     const void *weight, float eps, void *y, float *y_scales,
                                                     uint32_t y_f32, void *stream) {
  const std::string op = "ignis_image_encoder_residual_norm";
  if (x == nullptr) return fail(op + ": NULL x");
  if (delta == nullptr && weight == nullptr) return fail(op + ": neither delta nor weight: nothing to do");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (cols == 0 || cols % 8 != 0 || cols > kMaxNormCols) {
    return fail(op + ": cols must be a positive multiple of 8, at most 8192");
  }
  if (weight != nullptr) {
    if (y == nullptr) return fail(op + ": a norm needs y");
    if ((y_f32 != 0) == (y_scales != nullptr)) return fail(op + ": y_scales must be given exactly for E4M3 output");
    if (!aligned(weight, 16) || !aligned(y, 16) || (y_scales != nullptr && !aligned(y_scales, 4))) {
      return fail(op + ": weight and y must be 16-byte aligned, y_scales 4-byte aligned");
    }
    if (!(eps >= 0.0f)) return fail(op + ": eps must be >= 0");
  }
  if (!aligned(x, 16) || (delta != nullptr && !aligned(delta, 16))) return fail(op + ": x and delta must be 16-byte aligned");
  return launch_residual_norm(x, delta, tokens, cols, weight, eps, y, y_scales, y_f32, stream);
}

extern "C" int32_t ignis_image_encoder_qk_norm_mrope(const float *q, uint32_t q_stride, const float *k,
                                                     uint32_t k_stride, uint32_t tokens, const void *q_weight,
                                                     const void *k_weight, float eps, const int32_t *positions,
                                                     float rope_theta, const uint32_t *mrope_section, void *q_out,
                                                     void *k_out, void *stream) {
  const char *op = "ignis_image_encoder_qk_norm_mrope";
  if (q == nullptr || k == nullptr || q_weight == nullptr || k_weight == nullptr || positions == nullptr ||
      q_out == nullptr || k_out == nullptr) {
    return fail(std::string(op) + ": NULL pointer");
  }
  if (tokens == 0 || tokens > kMaxTokens) return fail(std::string(op) + ": tokens must be 1 .. 2^20");
  if (q_stride < kHeads * kHeadDim || q_stride % 4 != 0 || k_stride < kKv || k_stride % 4 != 0) {
    return fail(std::string(op) + ": q_stride must be >= 4096 and k_stride >= 1024, each a multiple of 4");
  }
  if (!aligned(q, 16) || !aligned(k, 16) || !aligned(q_weight, 16) || !aligned(k_weight, 16) || !aligned(q_out, 16) ||
      !aligned(k_out, 16) || !aligned(positions, 4)) {
    return fail(std::string(op) + ": pointers must be 16-byte aligned (positions 4-byte)");
  }
  if (!(eps >= 0.0f)) return fail(std::string(op) + ": eps must be >= 0");
  const std::string why = mrope_checks(op, rope_theta, mrope_section);
  if (!why.empty()) return fail(why);
  return launch_qk_norm_mrope(q, q_stride, k, k_stride, tokens, q_weight, k_weight, eps, positions,
                              make_mrope(rope_theta, mrope_section), q_out, k_out, stream);
}

extern "C" int32_t ignis_image_encoder_rows(float *x, uint32_t tokens, uint32_t cols, const int32_t *row_index,
                                            uint32_t rows, const void *src, uint32_t add, void *stream) {
  const std::string op = "ignis_image_encoder_rows";
  if (x == nullptr || row_index == nullptr || src == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (rows == 0 || rows > tokens) return fail(op + ": rows must be 1 .. tokens");
  if (cols == 0 || cols % 8 != 0) return fail(op + ": cols must be a positive multiple of 8");
  if (!aligned(x, 16) || !aligned(src, 16) || !aligned(row_index, 4)) {
    return fail(op + ": x and src must be 16-byte aligned, row_index 4-byte");
  }
  return launch_rows(x, tokens, cols, row_index, rows, src, add, stream);
}

extern "C" int32_t ignis_image_encoder_forward(const struct ignis_image_encoder_weights *weights, uint32_t first_layer,
                                               uint32_t end_layer, float *x, uint32_t tokens, const int32_t *positions,
                                               const struct ignis_image_encoder_vision *vision, void *workspace,
                                               uint64_t workspace_bytes, void *stream) {
  const std::string op = "ignis_image_encoder_forward";
  if (weights == nullptr || weights->layers == nullptr || x == nullptr || positions == nullptr || workspace == nullptr) {
    return fail(op + ": NULL pointer");
  }
  if (first_layer > end_layer || end_layer > weights->layer_count) {
    return fail(op + ": the layers [" + std::to_string(first_layer) + ", " + std::to_string(end_layer) +
                ") are not within the model's " + std::to_string(weights->layer_count));
  }
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (!aligned(x, 16) || !aligned(positions, 4) || !aligned(workspace, 256)) {
    return fail(op + ": x must be 16-byte aligned, positions 4-byte, workspace 256-byte");
  }
  if (workspace_bytes < workspace_size(tokens)) {
    return fail(op + ": the workspace holds " + std::to_string(workspace_bytes) + " bytes, " + std::to_string(tokens) +
                " tokens need " + std::to_string(workspace_size(tokens)));
  }
  if (!(weights->rms_norm_eps >= 0.0f)) return fail(op + ": rms_norm_eps must be >= 0");
  const std::string why = mrope_checks(op.c_str(), weights->rope_theta, weights->mrope_section);
  if (!why.empty()) return fail(why);
  for (uint32_t l = first_layer; l < end_layer; ++l) {
    const ignis_image_encoder_layer &L = weights->layers[l];
    const void *all[] = {L.input_norm, L.q_proj, L.k_proj, L.v_proj, L.q_norm, L.k_norm,
                         L.o_proj,     L.post_norm, L.gate_proj, L.up_proj, L.down_proj};
    for (const void *p : all) {
      if (p == nullptr || !aligned(p, 16)) {
        return fail(op + ": layer " + std::to_string(l) + " has a NULL or unaligned weight");
      }
    }
  }
  const bool has_vision = vision != nullptr && vision->rows != 0;
  if (has_vision) {
    if (vision->row_index == nullptr || !aligned(vision->row_index, 4) || vision->rows > tokens) {
      return fail(op + ": the vision rows need a 4-byte aligned row_index of at most tokens rows");
    }
    const void *features[] = {vision->embeds, vision->deepstack[0], vision->deepstack[1], vision->deepstack[2]};
    for (const void *p : features) {
      if (p != nullptr && !aligned(p, 16)) return fail(op + ": the vision features must be 16-byte aligned");
    }
  }

  const Workspace w = carve(workspace, tokens);
  const Mrope mrope = make_mrope(weights->rope_theta, weights->mrope_section);
  const float eps = weights->rms_norm_eps;
  const ignis_image_encoder_layer *layers = weights->layers;
  if (has_vision && first_layer == 0 && vision->embeds != nullptr &&
      launch_rows(x, tokens, kHidden, vision->row_index, vision->rows, vision->embeds, 0, stream) != 0) {
    return failed_at(0, "the vision embeddings");
  }
  if (first_layer < end_layer &&
      launch_residual_norm(x, nullptr, tokens, kHidden, layers[first_layer].input_norm, eps, w.codes, w.scales, 0,
                           stream) != 0) {
    return failed_at(first_layer, "input_layernorm");
  }
  for (uint32_t l = first_layer; l < end_layer; ++l) {
    const ignis_image_encoder_layer &L = layers[l];
    // attention: the input norm's codes are in w.codes
    if (ignis_image_gemm_w8a8(L.q_proj, kHeads * kHeadDim, kHidden, w.codes, w.scales, tokens, w.q32, 1, stream) != 0) {
      return failed_at(l, "q_proj");
    }
    if (ignis_image_gemm_w8a8(L.k_proj, kKv, kHidden, w.codes, w.scales, tokens, w.k32, 1, stream) != 0) {
      return failed_at(l, "k_proj");
    }
    if (ignis_image_gemm_w8a8(L.v_proj, kKv, kHidden, w.codes, w.scales, tokens, w.v, 0, stream) != 0) {
      return failed_at(l, "v_proj");
    }
    if (launch_qk_norm_mrope(w.q32, kHeads * kHeadDim, w.k32, kKv, tokens, L.q_norm, L.k_norm, eps, positions, mrope,
                             w.qb, w.kb, stream) != 0) {
      return failed_at(l, "the QK norm and mrope");
    }
    if (ignis_image_encoder_attn(w.qb, w.kb, w.v, tokens, kHeads * kHeadDim, kKv, 1, kAttnScale, w.attn,
                                 kHeads * kHeadDim, 0, stream) != 0) {
      return failed_at(l, "the attention");
    }
    if (ignis_image_quantize_tokens(w.attn, tokens, kHeads * kHeadDim, w.codes, w.scales, stream) != 0) {
      return failed_at(l, "the attention output's quantization");
    }
    if (ignis_image_gemm_w8a8(L.o_proj, kHidden, kHeads * kHeadDim, w.codes, w.scales, tokens, w.delta, 1, stream) != 0) {
      return failed_at(l, "o_proj");
    }
    if (launch_residual_norm(x, w.delta, tokens, kHidden, L.post_norm, eps, w.codes, w.scales, 0, stream) != 0) {
      return failed_at(l, "post_attention_layernorm");
    }
    // MLP
    if (ignis_image_gemm_w8a8(L.gate_proj, kInter, kHidden, w.codes, w.scales, tokens, w.gate, 0, stream) != 0) {
      return failed_at(l, "gate_proj");
    }
    if (ignis_image_gemm_w8a8(L.up_proj, kInter, kHidden, w.codes, w.scales, tokens, w.up, 0, stream) != 0) {
      return failed_at(l, "up_proj");
    }
    if (ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, w.gate, w.up, tokens, kInter, kInter, w.codes, w.scales,
                            IGNIS_IMAGE_DIT_OUT_E4M3, stream) != 0) {
      return failed_at(l, "SwiGLU");
    }
    if (ignis_image_gemm_w8a8(L.down_proj, kHidden, kInter, w.codes, w.scales, tokens, w.delta, 1, stream) != 0) {
      return failed_at(l, "down_proj");
    }
    // x += the MLP's output, then the DeepStack features after layers 0-2, then the next layer's
    // input norm (fused with the add when no DeepStack add comes between them).
    const void *deepstack = has_vision && l < IGNIS_IMAGE_ENCODER_DEEPSTACK ? vision->deepstack[l] : nullptr;
    const void *next_norm = l + 1 < end_layer ? layers[l + 1].input_norm : nullptr;
    if (deepstack != nullptr) {
      if (launch_residual_norm(x, w.delta, tokens, kHidden, nullptr, eps, nullptr, nullptr, 0, stream) != 0) {
        return failed_at(l, "the residual add");
      }
      if (launch_rows(x, tokens, kHidden, vision->row_index, vision->rows, deepstack, 1, stream) != 0) {
        return failed_at(l, "the DeepStack add");
      }
      if (next_norm != nullptr &&
          launch_residual_norm(x, nullptr, tokens, kHidden, next_norm, eps, w.codes, w.scales, 0, stream) != 0) {
        return failed_at(l + 1, "input_layernorm");
      }
    } else if (launch_residual_norm(x, w.delta, tokens, kHidden, next_norm, eps, w.codes, w.scales, 0, stream) != 0) {
      return failed_at(l, next_norm != nullptr ? "the residual add and the next input_layernorm" : "the residual add");
    }
  }
  return 0;
}

extern "C" const char *ignis_image_encoder_last_error(void) { return last_error_cstr(); }
