// ignis kernel leaf: the image model's DiT elementwise ops (spec image-model/02 acceptance 7,
// GitHub #319) -- OURS. The definitions are kernel/include/ignis_image_dit.h's.
//
// Row ops (the two norms and the activations): one CTA of 256 threads per token, each thread on
// the row's 8-element chunks tid, tid + 256, ... in every pass. A norm stages its BF16 row in
// shared memory, so its passes read global memory once; an E4M3 output stages the BF16 output
// there for the quantization pass (fp8_quant.cuh's steps, as ignis_image_quantize_tokens takes
// them). Sums run in a fixed order: each thread's elements in order, a warp butterfly, then the 8
// warps in order from 0. Every rounding step is spelled with an _rn intrinsic, so nvcc's FMA
// contraction cannot change the sequence the tests' bound follows.
//
// The QK RMSNorm with RoPE: one warp per (token, head), 4 channels a lane (RoPE pairs 2l, 2l + 1).
// The gated residual: 8 elements a thread. The RoPE table: one thread per (token, pair).

#include "ignis_image_dit.h"
#include "image_common.cuh"
#include "fp8_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kRowThreads = 256;
constexpr uint32_t kMaxRowCols = 24560;  // a BF16 row beside the 32-byte reduction in 48 KiB of shared memory
constexpr uint32_t kMaxTokens = 1u << 20;
constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kRopePairs = 64;            // 8 frame, 28 height, 28 width
constexpr int kRopeMin = -1024;           // diffusers' table rows: positions [-1024, 8192)
constexpr int kRopeEnd = 8192;

// f_i for the 64 pairs, PyTorch's fp32 values (ignis_image_dit.h): frame (d = 16), height and width
// (d = 56). Each is fl32(1 / fl32(10000^fl32(2i / d))) except height/width i = 11, torch's powf one
// ulp up (0x1.b7886ep-6, where that formula gives 0x1.b7886cp-6). The CTest checks them.
__constant__ float kRopeFreq[kRopePairs] = {
    // frame
    0x1p+0f, 0x1.43d136p-2f, 0x1.99999ap-4f, 0x1.030dc6p-5f, 0x1.47ae14p-7f, 0x1.9e7c7p-9f, 0x1.0624dep-10f,
    0x1.4b96cp-12f,
    // height
    0x1p+0f, 0x1.707aa4p-1f, 0x1.093068p-1f, 0x1.7db4a2p-2f, 0x1.12b542p-2f, 0x1.8b682ap-3f, 0x1.1c9194p-3f,
    0x1.99999ap-4f, 0x1.26c88p-4f, 0x1.a84d76p-5f, 0x1.315d4ep-5f, 0x1.b7886ep-6f, 0x1.3c5354p-6f, 0x1.c74f58p-7f,
    0x1.47ae14p-7f, 0x1.d7a73ep-8f, 0x1.537124p-8f, 0x1.e89548p-9f, 0x1.5fa058p-9f, 0x1.fa1ef8p-10f, 0x1.6c3f72p-10f,
    0x1.0624dep-10f, 0x1.795298p-11f, 0x1.0f8db6p-11f, 0x1.86ddd2p-12f, 0x1.194d12p-12f, 0x1.94e59p-13f,
    0x1.2365f6p-13f,
    // width (the height's)
    0x1p+0f, 0x1.707aa4p-1f, 0x1.093068p-1f, 0x1.7db4a2p-2f, 0x1.12b542p-2f, 0x1.8b682ap-3f, 0x1.1c9194p-3f,
    0x1.99999ap-4f, 0x1.26c88p-4f, 0x1.a84d76p-5f, 0x1.315d4ep-5f, 0x1.b7886ep-6f, 0x1.3c5354p-6f, 0x1.c74f58p-7f,
    0x1.47ae14p-7f, 0x1.d7a73ep-8f, 0x1.537124p-8f, 0x1.e89548p-9f, 0x1.5fa058p-9f, 0x1.fa1ef8p-10f, 0x1.6c3f72p-10f,
    0x1.0624dep-10f, 0x1.795298p-11f, 0x1.0f8db6p-11f, 0x1.86ddd2p-12f, 0x1.194d12p-12f, 0x1.94e59p-13f,
    0x1.2365f6p-13f};

// GELU-tanh's constants, each one fp32 rounding of the definition's: 0.044715 and 2 sqrt(2 / pi).
constexpr float kGeluA = 0.044715f;
constexpr float kGeluTwoC = static_cast<float>(1.5957691216057308);

enum RowOp : int { kLayerNorm, kRmsNorm, kSilu, kGelu, kSwiglu };

struct RowArgs {
  const __nv_bfloat16 *x, *up;
  std::size_t row_stride;          // elements between rows of x (and up)
  int cols;
  const float *scale_t, *scale_0;  // the LayerNorm's modulation rows
  std::size_t first_target;
  const __nv_bfloat16 *weight;     // the RMS norm's w
  float eps;
  void *y;
  float *y_scales;
  uint32_t y_kind;
};

// The CTA's sum in a fixed order: the warp butterfly (every lane ends with the same value), then
// the warps' sums in order from 0. The leading barrier lets a second call reuse `red`.
__device__ float block_sum(float v, float *red) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.0f;
#pragma unroll
  for (int w = 0; w < kRowThreads / 32; ++w) s = __fadd_rn(s, red[w]);
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
  for (int w = 0; w < kRowThreads / 32; ++w) s = fmaxf(s, red[w]);
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

__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 v;
  __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; ++j) h[j] = __floats2bfloat162_rn(f[2 * j], f[2 * j + 1]);
  return v;
}

__device__ __forceinline__ void load8(const float *p, float (&f)[8]) {
  const float4 a = reinterpret_cast<const float4 *>(p)[0], b = reinterpret_cast<const float4 *>(p)[1];
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w, f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

__device__ __forceinline__ void store8(float *p, const float (&f)[8]) {
  reinterpret_cast<float4 *>(p)[0] = make_float4(f[0], f[1], f[2], f[3]);
  reinterpret_cast<float4 *>(p)[1] = make_float4(f[4], f[5], f[6], f[7]);
}

__device__ __forceinline__ float silu(float g) { return __fdiv_rn(g, __fadd_rn(1.0f, expf(-g))); }

__device__ __forceinline__ float gelu_tanh(float x) {
  const float x3 = __fmul_rn(__fmul_rn(x, x), x);
  const float two_z = __fmul_rn(kGeluTwoC, __fmaf_rn(kGeluA, x3, x));
  return __fdiv_rn(x, __fadd_rn(1.0f, expf(-two_z)));
}

template <int kOp>
__global__ void __launch_bounds__(kRowThreads) row_kernel(RowArgs a) {
  extern __shared__ uint4 row[];  // the row's BF16 chunks (the norm's input, then an E4M3 output's BF16)
  __shared__ float red[kRowThreads / 32];
  constexpr bool kNorm = kOp == kLayerNorm || kOp == kRmsNorm;
  const std::size_t t = blockIdx.x;
  const int chunks = a.cols / 8;
  const uint4 *x = reinterpret_cast<const uint4 *>(a.x + t * a.row_stride);

  float mean = 0.0f, r = 0.0f;
  if constexpr (kNorm) {
    float acc = 0.0f;
    for (int i = threadIdx.x; i < chunks; i += kRowThreads) {
      const uint4 v = x[i];
      row[i] = v;
      float f[8];
      unpack8(v, f);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = kOp == kLayerNorm ? __fadd_rn(acc, f[j]) : __fmaf_rn(f[j], f[j], acc);
    }
    const float total = block_sum(acc, red);
    const float n = static_cast<float>(a.cols);
    if constexpr (kOp == kLayerNorm) {
      mean = __fdiv_rn(total, n);
      float acc2 = 0.0f;
      for (int i = threadIdx.x; i < chunks; i += kRowThreads) {
        float f[8];
        unpack8(row[i], f);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const float c = __fsub_rn(f[j], mean);
          acc2 = __fmaf_rn(c, c, acc2);
        }
      }
      r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum(acc2, red), n), a.eps));
    } else {
      r = rsqrtf(__fadd_rn(__fdiv_rn(total, n), a.eps));
    }
  }

  const float *scale = t < a.first_target ? a.scale_0 : a.scale_t;
  const std::size_t out_row = t * static_cast<std::size_t>(a.cols);
  float amax = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kRowThreads) {
    float f[8], y[8];
    unpack8(kNorm ? row[i] : x[i], f);
    if constexpr (kOp == kLayerNorm) {
      float s[8];
      load8(scale + 8 * i, s);
#pragma unroll
      for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(__fsub_rn(f[j], mean), r), __fadd_rn(1.0f, s[j]));
    } else if constexpr (kOp == kRmsNorm) {
      float w[8];
      unpack8(reinterpret_cast<const uint4 *>(a.weight)[i], w);
#pragma unroll
      for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(f[j], r), __fadd_rn(w[j], 1.0f));
    } else if constexpr (kOp == kSilu) {
#pragma unroll
      for (int j = 0; j < 8; ++j) y[j] = silu(f[j]);
    } else if constexpr (kOp == kGelu) {
#pragma unroll
      for (int j = 0; j < 8; ++j) y[j] = gelu_tanh(f[j]);
    } else {
      float u[8];
      unpack8(reinterpret_cast<const uint4 *>(a.up + t * a.row_stride)[i], u);
#pragma unroll
      for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(silu(f[j]), u[j]);
    }
    if (a.y_kind == IGNIS_IMAGE_DIT_OUT_F32) {
      store8(static_cast<float *>(a.y) + out_row + 8 * i, y);
    } else {
      const uint4 b = pack8(y);
      if (a.y_kind == IGNIS_IMAGE_DIT_OUT_BF16) {
        reinterpret_cast<uint4 *>(static_cast<__nv_bfloat16 *>(a.y) + out_row)[i] = b;
      } else {
        row[i] = b;  // this thread's own chunk: no other thread reads or writes it
        float q[8];
        unpack8(b, q);
#pragma unroll
        for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(q[j]));
      }
    }
  }
  if (a.y_kind != IGNIS_IMAGE_DIT_OUT_E4M3) return;

  // ignis_image_quantize_tokens' steps on the staged BF16 row.
  amax = block_max(amax, red);
  const float inverse = e4m3_token_inverse(amax);
  if (threadIdx.x == 0) a.y_scales[t] = e4m3_token_scale(amax);
  uint2 *codes = reinterpret_cast<uint2 *>(static_cast<uint8_t *>(a.y) + out_row);
  for (int i = threadIdx.x; i < chunks; i += kRowThreads) {
    const uint4 v = row[i];
    const __nv_bfloat162 *h = reinterpret_cast<const __nv_bfloat162 *>(&v);
    uint32_t packed[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 lo = __bfloat1622float2(h[2 * j]), hi = __bfloat1622float2(h[2 * j + 1]);
      packed[j] = static_cast<uint32_t>(e4m3x2_quantize(lo.x, lo.y, inverse)) |
                  (static_cast<uint32_t>(e4m3x2_quantize(hi.x, hi.y, inverse)) << 16);
    }
    codes[i] = make_uint2(packed[0], packed[1]);
  }
}

__global__ void gated_residual_kernel(const __nv_bfloat16 *x, const __nv_bfloat16 *y, std::size_t chunks,
                                      int row_chunks, const float *gate_t, const float *gate_0,
                                      std::size_t first_target, void *out, uint32_t out_f32) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < chunks;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t t = e / row_chunks;
    const int c = static_cast<int>(e % row_chunks) * 8;
    float xv[8], yv[8], g[8], o[8];
    unpack8(reinterpret_cast<const uint4 *>(x)[e], xv);
    unpack8(reinterpret_cast<const uint4 *>(y)[e], yv);
    load8((t < first_target ? gate_0 : gate_t) + c, g);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __fmaf_rn(tanhf(g[j]), yv[j], xv[j]);
    if (out_f32) {
      store8(static_cast<float *>(out) + 8 * e, o);
    } else {
      reinterpret_cast<uint4 *>(out)[e] = pack8(o);
    }
  }
}

// One warp per (token, head): warp w is token w / 64, head w % 64 (q's 32, then k's).
__global__ void __launch_bounds__(256)
    qk_norm_rope_kernel(const __nv_bfloat16 *q, const __nv_bfloat16 *k, std::size_t tokens, std::size_t row_stride,
                        const __nv_bfloat16 *q_weight, const __nv_bfloat16 *k_weight, float eps, const float4 *rope,
                        void *q_out, void *k_out, std::size_t out_row_stride, uint32_t out_f32) {
  const std::size_t warp = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
  if (warp >= tokens * 2 * kHeads) return;  // whole warps: 256 threads and 64 warps per token
  const int lane = threadIdx.x & 31;
  const std::size_t t = warp / (2 * kHeads);
  const int head = static_cast<int>(warp % (2 * kHeads));
  const bool is_k = head >= kHeads;
  const std::size_t col = static_cast<std::size_t>(head % kHeads) * kHeadDim + 4 * lane;
  const uint2 raw = *reinterpret_cast<const uint2 *>((is_k ? k : q) + t * row_stride + col);
  const uint2 wraw = *reinterpret_cast<const uint2 *>((is_k ? k_weight : q_weight) + 4 * lane);
  const __nv_bfloat162 *xh = reinterpret_cast<const __nv_bfloat162 *>(&raw);
  const __nv_bfloat162 *wh = reinterpret_cast<const __nv_bfloat162 *>(&wraw);
  const float2 x01 = __bfloat1622float2(xh[0]), x23 = __bfloat1622float2(xh[1]);
  const float2 w01 = __bfloat1622float2(wh[0]), w23 = __bfloat1622float2(wh[1]);
  const float x[4] = {x01.x, x01.y, x23.x, x23.y}, w[4] = {w01.x, w01.y, w23.x, w23.y};
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) ss = __fmaf_rn(x[j], x[j], ss);
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xFFFFFFFFu, ss, m));
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(ss, static_cast<float>(kHeadDim)), eps));
  float n[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) n[j] = __fmul_rn(__fmul_rn(x[j], r), w[j]);
  const float4 cs = rope[t * (kRopePairs / 2) + lane];  // (cos, sin) of pairs 2 lane and 2 lane + 1
  const float o[4] = {__fmaf_rn(n[0], cs.x, -__fmul_rn(n[1], cs.y)), __fmaf_rn(n[0], cs.y, __fmul_rn(n[1], cs.x)),
                      __fmaf_rn(n[2], cs.z, -__fmul_rn(n[3], cs.w)), __fmaf_rn(n[2], cs.w, __fmul_rn(n[3], cs.z))};
  const std::size_t at = t * out_row_stride + col;
  if (out_f32) {
    reinterpret_cast<float4 *>(static_cast<float *>(is_k ? k_out : q_out) + at)[0] = make_float4(o[0], o[1], o[2], o[3]);
  } else {
    uint2 v;
    reinterpret_cast<__nv_bfloat162 *>(&v)[0] = __floats2bfloat162_rn(o[0], o[1]);
    reinterpret_cast<__nv_bfloat162 *>(&v)[1] = __floats2bfloat162_rn(o[2], o[3]);
    *reinterpret_cast<uint2 *>(static_cast<__nv_bfloat16 *>(is_k ? k_out : q_out) + at) = v;
  }
}

__global__ void rope_table_kernel(const int32_t *positions, std::size_t tokens, float2 *table) {
  __shared__ float freq[kRopePairs];
  if (threadIdx.x < kRopePairs) freq[threadIdx.x] = kRopeFreq[threadIdx.x];
  __syncthreads();
  const std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (e >= tokens * kRopePairs) return;
  const std::size_t t = e / kRopePairs;
  const int i = static_cast<int>(e % kRopePairs);
  const int axis = i < 8 ? 0 : (i < 36 ? 1 : 2);
  const float angle = __fmul_rn(static_cast<float>(positions[t * 3 + axis]), freq[i]);
  double s, c;
  sincos(static_cast<double>(angle), &s, &c);
  table[e] = make_float2(__double2float_rn(c), __double2float_rn(s));
}

std::string row_checks(const char *op, const void *x, const void *up, uint32_t tokens, uint32_t cols,
                       uint32_t row_stride, const void *y, const float *y_scales, uint32_t y_kind) {
  const std::string o(op);
  if (x == nullptr || y == nullptr) return o + ": NULL pointer";
  if (tokens == 0 || tokens > kMaxTokens) return o + ": tokens must be 1 .. 2^20";
  if (cols == 0 || cols % 16 != 0 || cols > kMaxRowCols) return o + ": cols must be a positive multiple of 16, at most 24560";
  if (row_stride < cols || row_stride % 8 != 0) return o + ": row_stride must be >= cols and a multiple of 8";
  if (y_kind > IGNIS_IMAGE_DIT_OUT_E4M3) return o + ": unknown y_kind";
  if ((y_kind == IGNIS_IMAGE_DIT_OUT_E4M3) != (y_scales != nullptr)) {
    return o + ": y_scales must be given exactly when y_kind is E4M3";
  }
  if (!aligned(x, 16) || !aligned(y, 16) || (up != nullptr && !aligned(up, 16)) ||
      (y_scales != nullptr && !aligned(y_scales, 4))) {
    return o + ": x, up and y must be 16-byte aligned, y_scales 4-byte aligned";
  }
  return "";
}

int32_t launch_row(const char *op, int kind, const RowArgs &a, uint32_t tokens, void *stream) {
  const bool norm = kind == kLayerNorm || kind == kRmsNorm;
  const std::size_t smem = norm || a.y_kind == IGNIS_IMAGE_DIT_OUT_E4M3 ? static_cast<std::size_t>(a.cols) * 2 : 0;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (kind) {
    case kLayerNorm: row_kernel<kLayerNorm><<<tokens, kRowThreads, smem, s>>>(a); break;
    case kRmsNorm: row_kernel<kRmsNorm><<<tokens, kRowThreads, smem, s>>>(a); break;
    case kSilu: row_kernel<kSilu><<<tokens, kRowThreads, smem, s>>>(a); break;
    case kGelu: row_kernel<kGelu><<<tokens, kRowThreads, smem, s>>>(a); break;
    default: row_kernel<kSwiglu><<<tokens, kRowThreads, smem, s>>>(a); break;
  }
  return check_launch(op);
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_dit_rope_positions(uint32_t tokens, uint32_t blocks, const uint32_t *block_start,
                                                  const uint32_t *block_height, const uint32_t *block_width,
                                                  int32_t *positions) {
  const std::string op = "ignis_image_dit_rope_positions";
  if (positions == nullptr || (blocks != 0 && (block_start == nullptr || block_height == nullptr || block_width == nullptr))) {
    return fail(op + ": NULL pointer");
  }
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  // diffusers' QwenImage21Rope.forward, with the block starts given instead of found in a mask.
  int64_t cursor = 0, position = 0;
  auto fits = [](int64_t p) { return p >= kRopeMin && p < kRopeEnd; };
  auto text = [&](int64_t end) {
    for (; cursor < end; ++cursor, ++position) {
      if (!fits(position)) return false;
      positions[3 * cursor] = positions[3 * cursor + 1] = positions[3 * cursor + 2] = static_cast<int32_t>(position);
    }
    return true;
  };
  for (uint32_t b = 0; b < blocks; ++b) {
    const int64_t h = block_height[b], w = block_width[b], start = block_start[b];
    if (h == 0 || w == 0) return fail(op + ": block " + std::to_string(b) + " is empty");
    // h, w and start are caller-sized u32: bound each by tokens before the product can overflow.
    if (h > tokens || w > tokens || start > tokens || start < cursor || start + h * w > tokens) {
      return fail(op + ": block " + std::to_string(b) + " overlaps the one before it or runs past the sequence");
    }
    if (!text(start)) return fail(op + ": a text position falls outside [-1024, 8192)");
    const int64_t h0 = -(h - h / 2), w0 = -(w - w / 2);
    if (!fits(position) || !fits(h0) || !fits(w0)) {
      return fail(op + ": block " + std::to_string(b) + " has a position outside [-1024, 8192)");
    }
    for (int64_t i = 0; i < h; ++i) {
      for (int64_t j = 0; j < w; ++j, ++cursor) {
        positions[3 * cursor] = static_cast<int32_t>(position);
        positions[3 * cursor + 1] = static_cast<int32_t>(h0 + i);
        positions[3 * cursor + 2] = static_cast<int32_t>(w0 + j);
      }
    }
    position += std::max(h, w);
  }
  if (!text(tokens)) return fail(op + ": a text position falls outside [-1024, 8192)");
  return 0;
}

extern "C" int32_t ignis_image_dit_rope_table(const int32_t *positions, uint32_t tokens, float *table, void *stream) {
  const std::string op = "ignis_image_dit_rope_table";
  if (positions == nullptr || table == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (!aligned(positions, 4) || !aligned(table, 16)) return fail(op + ": positions must be 4-byte aligned, table 16-byte");
  const std::size_t n = static_cast<std::size_t>(tokens) * kRopePairs;
  rope_table_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      positions, tokens, reinterpret_cast<float2 *>(table));
  return check_launch(op.c_str());
}

extern "C" int32_t ignis_image_dit_modulated_norm(const void *x, uint32_t tokens, uint32_t cols, const float *scale_t,
                                                  const float *scale_0, uint32_t first_target, float eps, void *y,
                                                  float *y_scales, uint32_t y_kind, void *stream) {
  const char *op = "ignis_image_dit_modulated_norm";
  std::string why = row_checks(op, x, nullptr, tokens, cols, cols, y, y_scales, y_kind);
  if (why.empty() && first_target > tokens) why = std::string(op) + ": first_target must be <= tokens";
  if (why.empty() && ((first_target < tokens && scale_t == nullptr) || (first_target > 0 && scale_0 == nullptr))) {
    why = std::string(op) + ": a modulation row the tokens read is NULL";
  }
  if (why.empty() && ((scale_t != nullptr && !aligned(scale_t, 16)) || (scale_0 != nullptr && !aligned(scale_0, 16)))) {
    why = std::string(op) + ": the modulation rows must be 16-byte aligned";
  }
  if (why.empty() && !(eps >= 0.0f)) why = std::string(op) + ": eps must be >= 0";
  if (!why.empty()) return fail(why);
  RowArgs a{static_cast<const __nv_bfloat16 *>(x), nullptr, cols, static_cast<int>(cols), scale_t, scale_0,
            first_target, nullptr, eps, y, y_scales, y_kind};
  return launch_row(op, kLayerNorm, a, tokens, stream);
}

extern "C" int32_t ignis_image_dit_rms_norm(const void *x, uint32_t tokens, uint32_t cols, const void *weight, float eps,
                                            void *y, float *y_scales, uint32_t y_kind, void *stream) {
  const char *op = "ignis_image_dit_rms_norm";
  std::string why = row_checks(op, x, nullptr, tokens, cols, cols, y, y_scales, y_kind);
  if (why.empty() && (weight == nullptr || !aligned(weight, 16))) {
    why = std::string(op) + ": weight must be a 16-byte aligned pointer";
  }
  if (why.empty() && !(eps >= 0.0f)) why = std::string(op) + ": eps must be >= 0";
  if (!why.empty()) return fail(why);
  RowArgs a{static_cast<const __nv_bfloat16 *>(x), nullptr, cols, static_cast<int>(cols), nullptr, nullptr, 0,
            static_cast<const __nv_bfloat16 *>(weight), eps, y, y_scales, y_kind};
  return launch_row(op, kRmsNorm, a, tokens, stream);
}

extern "C" int32_t ignis_image_dit_act(uint32_t kind, const void *x, const void *up, uint32_t tokens, uint32_t cols,
                                       uint32_t row_stride, void *y, float *y_scales, uint32_t y_kind, void *stream) {
  const char *op = "ignis_image_dit_act";
  std::string why = row_checks(op, x, up, tokens, cols, row_stride, y, y_scales, y_kind);
  if (why.empty() && kind > IGNIS_IMAGE_DIT_SWIGLU) why = std::string(op) + ": unknown kind";
  if (why.empty() && (kind == IGNIS_IMAGE_DIT_SWIGLU) != (up != nullptr)) {
    why = std::string(op) + ": up must be given exactly for SwiGLU";
  }
  if (!why.empty()) return fail(why);
  RowArgs a{static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(up), row_stride,
            static_cast<int>(cols), nullptr, nullptr, 0, nullptr, 0.0f, y, y_scales, y_kind};
  const int k = kind == IGNIS_IMAGE_DIT_SILU ? kSilu : (kind == IGNIS_IMAGE_DIT_GELU_TANH ? kGelu : kSwiglu);
  return launch_row(op, k, a, tokens, stream);
}

extern "C" int32_t ignis_image_dit_gated_residual(const void *x, const void *y, uint32_t tokens, uint32_t cols,
                                                  const float *gate_t, const float *gate_0, uint32_t first_target,
                                                  void *out, uint32_t out_f32, void *stream) {
  const std::string op = "ignis_image_dit_gated_residual";
  if (x == nullptr || y == nullptr || out == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (cols == 0 || cols % 8 != 0) return fail(op + ": cols must be a positive multiple of 8");
  if (first_target > tokens) return fail(op + ": first_target must be <= tokens");
  if ((first_target < tokens && gate_t == nullptr) || (first_target > 0 && gate_0 == nullptr)) {
    return fail(op + ": a gate row the tokens read is NULL");
  }
  if (!aligned(x, 16) || !aligned(y, 16) || !aligned(out, 16) || (gate_t != nullptr && !aligned(gate_t, 16)) ||
      (gate_0 != nullptr && !aligned(gate_0, 16))) {
    return fail(op + ": x, y, out and the gate rows must be 16-byte aligned");
  }
  const std::size_t chunks = static_cast<std::size_t>(tokens) * (cols / 8);
  const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((chunks + 255) / 256, 170 * 32));
  gated_residual_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(x), static_cast<const __nv_bfloat16 *>(y), chunks, static_cast<int>(cols / 8),
      gate_t, gate_0, first_target, out, out_f32);
  return check_launch(op.c_str());
}

extern "C" int32_t ignis_image_dit_qk_norm_rope(const void *q, const void *k, uint32_t tokens, uint32_t row_stride,
                                                const void *q_weight, const void *k_weight, float eps,
                                                const float *rope, void *q_out, void *k_out, uint32_t out_row_stride,
                                                uint32_t out_f32, void *stream) {
  const std::string op = "ignis_image_dit_qk_norm_rope";
  if (q == nullptr || k == nullptr || q_weight == nullptr || k_weight == nullptr || rope == nullptr ||
      q_out == nullptr || k_out == nullptr) {
    return fail(op + ": NULL pointer");
  }
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (row_stride < kHeads * kHeadDim || row_stride % 8 != 0 || out_row_stride < kHeads * kHeadDim ||
      out_row_stride % 8 != 0) {
    return fail(op + ": row_stride and out_row_stride must be >= 4096 and multiples of 8");
  }
  if (!aligned(q, 16) || !aligned(k, 16) || !aligned(q_weight, 16) || !aligned(k_weight, 16) || !aligned(rope, 16) ||
      !aligned(q_out, 16) || !aligned(k_out, 16)) {
    return fail(op + ": every pointer must be 16-byte aligned");
  }
  if (!(eps >= 0.0f)) return fail(op + ": eps must be >= 0");
  const std::size_t threads = static_cast<std::size_t>(tokens) * 2 * kHeads * 32;
  qk_norm_rope_kernel<<<static_cast<unsigned>(threads / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(q), static_cast<const __nv_bfloat16 *>(k), tokens, row_stride,
      static_cast<const __nv_bfloat16 *>(q_weight), static_cast<const __nv_bfloat16 *>(k_weight), eps,
      reinterpret_cast<const float4 *>(rope), q_out, k_out, out_row_stride, out_f32);
  return check_launch(op.c_str());
}

extern "C" const char *ignis_image_dit_last_error(void) { return last_error_cstr(); }
