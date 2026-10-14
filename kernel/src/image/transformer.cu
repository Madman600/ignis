// ignis kernel leaf: the image model's diffusion transformer, Qwen-Image-2.1's DiT (spec
// image-model/04, GitHub #321) -- OURS. The definitions are kernel/include/ignis_image_transformer.h's.
//
// The residual norm: one CTA of 256 threads per token, each thread on the row's 8-element chunks
// tid, tid + 256, ...; the fp32 row (after the add) is staged in shared memory, so the LayerNorm's
// passes read global memory once. Sums run in a fixed order: each thread's elements in order, a
// warp butterfly, then the 8 warps in order from 0 (dit_ops.cu's). Every rounding step is spelled
// with an _rn intrinsic, so nvcc's FMA contraction cannot change the sequence the tests' bound
// follows. The gate arrives as tanh(gate): the program takes the tanh once per call, not once per
// token.
//
// The BF16 linear (the timestep embedding's): one warp per output, each lane on 8-element chunks
// lane, lane + 32, ... in order, then a warp butterfly.
//
// The program: the ops in the header's order, on views of one workspace (below).

#include "ignis_image_transformer.h"
#include "ignis_image_attn.h"
#include "ignis_image_dit.h"
#include "ignis_image_gemm.h"
#include "image_common.cuh"
#include "fp8_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace ignis_image {
namespace {

constexpr int kThreads = 256;
constexpr uint32_t kMaxNormCols = 8192;  // the fp32 row in 32 KiB of shared memory
constexpr uint32_t kMaxTokens = 1u << 20;
constexpr uint32_t kHidden = IGNIS_IMAGE_TRANSFORMER_HIDDEN;
constexpr uint32_t kMlp = IGNIS_IMAGE_TRANSFORMER_MLP;
constexpr uint32_t kLatent = IGNIS_IMAGE_TRANSFORMER_LATENT;
constexpr uint32_t kTimestep = IGNIS_IMAGE_TRANSFORMER_TIMESTEP;
constexpr uint32_t kMod = IGNIS_IMAGE_TRANSFORMER_MODULATION;
constexpr int kHalf = static_cast<int>(kTimestep / 2);
// SDPA's default scale, head_dim ** -0.5, rounded once to fp32.
constexpr float kAttnScale = static_cast<float>(0.08838834764831845);

// The sinusoid's frequencies, by value into the kernel.
struct Freqs {
  float f[kHalf];
};

__device__ float block_sum(float v, float *red) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = __fadd_rn(v, __shfl_xor_sync(0xFFFFFFFFu, v, m));
  __syncthreads();
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float s = 0.0f;
#pragma unroll
  for (int w = 0; w < kThreads / 32; ++w) s = __fadd_rn(s, red[w]);
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
  for (int w = 0; w < kThreads / 32; ++w) s = fmaxf(s, red[w]);
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

// Eight codes from fp32 values and the token's multiplier, as two 32-bit words.
__device__ __forceinline__ uint2 codes8(const float (&o)[8], float inverse) {
  uint32_t packed[2];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    packed[j] = static_cast<uint32_t>(e4m3x2_quantize(o[4 * j], o[4 * j + 1], inverse)) |
                (static_cast<uint32_t>(e4m3x2_quantize(o[4 * j + 2], o[4 * j + 3], inverse)) << 16);
  }
  return make_uint2(packed[0], packed[1]);
}

__global__ void __launch_bounds__(kThreads)
    residual_norm_kernel(float *x, const float *delta, const float *tanh_gate, int cols, const float *scale, float eps,
                         void *y, float *y_scales, uint32_t y_f32) {
  extern __shared__ float4 srow[];  // the row after the add, two float4 per chunk
  __shared__ float red[kThreads / 32];
  const std::size_t t = blockIdx.x;
  const int chunks = cols / 8;
  float *xr = x + t * cols;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8];
    load8(xr + 8 * i, v);
    if (delta != nullptr) {
      float d[8], g[8];
      load8(delta + t * cols + 8 * i, d);
      load8(tanh_gate + 8 * i, g);
#pragma unroll
      for (int j = 0; j < 8; ++j) v[j] = __fmaf_rn(g[j], d[j], v[j]);
      store8(xr + 8 * i, v);
    }
    if (scale != nullptr) {
      store8(reinterpret_cast<float *>(srow) + 8 * i, v);
#pragma unroll
      for (int j = 0; j < 8; ++j) acc = __fadd_rn(acc, v[j]);
    }
  }
  if (scale == nullptr) return;
  const float n = static_cast<float>(cols);
  const float mean = __fdiv_rn(block_sum(acc, red), n);
  float acc2 = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8];
    load8(reinterpret_cast<const float *>(srow) + 8 * i, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float c = __fsub_rn(v[j], mean);
      acc2 = __fmaf_rn(c, c, acc2);
    }
  }
  const float r = rsqrtf(__fadd_rn(__fdiv_rn(block_sum(acc2, red), n), eps));

  // y = ((x - mean) * r) * (1 + s), computed in each pass from the staged row: the amax, then the
  // codes.
  const std::size_t out_row = t * static_cast<std::size_t>(cols);
  float amax = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8], s[8], o[8];
    load8(reinterpret_cast<const float *>(srow) + 8 * i, v);
    load8(scale + 8 * i, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      o[j] = __fmul_rn(__fmul_rn(__fsub_rn(v[j], mean), r), __fadd_rn(1.0f, s[j]));
      amax = fmaxf(amax, fabsf(o[j]));
    }
    if (y_f32) store8(static_cast<float *>(y) + out_row + 8 * i, o);
  }
  if (y_f32) return;

  amax = block_max(amax, red);
  const float inverse = e4m3_token_inverse(amax);
  if (threadIdx.x == 0) y_scales[t] = e4m3_token_scale(amax);
  uint2 *codes = reinterpret_cast<uint2 *>(static_cast<uint8_t *>(y) + out_row);
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8], s[8], o[8];
    load8(reinterpret_cast<const float *>(srow) + 8 * i, v);
    load8(scale + 8 * i, s);
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __fmul_rn(__fmul_rn(__fsub_rn(v[j], mean), r), __fadd_rn(1.0f, s[j]));
    codes[i] = codes8(o, inverse);
  }
}

// One CTA per token: the amax, then the codes, from the row in global memory (L1/L2 the second time).
__global__ void __launch_bounds__(kThreads)
    quantize_f32_kernel(const float *x, int cols, uint8_t *codes, float *scales) {
  __shared__ float red[kThreads / 32];
  const std::size_t t = blockIdx.x;
  const float *xr = x + t * cols;
  const int chunks = cols / 8;
  float amax = 0.0f;
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8];
    load8(xr + 8 * i, v);
#pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
  }
  amax = block_max(amax, red);
  const float inverse = e4m3_token_inverse(amax);
  if (threadIdx.x == 0) scales[t] = e4m3_token_scale(amax);
  uint2 *cr = reinterpret_cast<uint2 *>(codes + t * cols);
  for (int i = threadIdx.x; i < chunks; i += kThreads) {
    float v[8];
    load8(xr + 8 * i, v);
    cr[i] = codes8(v, inverse);
  }
}

// One warp per output: warp w is input w / rows, output row w % rows.
__global__ void __launch_bounds__(kThreads)
    linear_bf16_kernel(const __nv_bfloat16 *w, std::size_t rows, int cols, const float *x, std::size_t outputs,
                       float *y) {
  const std::size_t warp = (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 5;
  if (warp >= outputs) return;  // whole warps: 256 threads, 8 outputs a CTA
  const int lane = threadIdx.x & 31;
  const std::size_t r = warp % rows, s = warp / rows;
  const uint4 *wr = reinterpret_cast<const uint4 *>(w + r * cols);
  const float *xs = x + s * cols;
  float acc = 0.0f;
  for (int i = lane; i < cols / 8; i += 32) {
    float wf[8], xf[8];
    unpack8(wr[i], wf);
    load8(xs + 8 * i, xf);
#pragma unroll
    for (int j = 0; j < 8; ++j) acc = __fmaf_rn(wf[j], xf[j], acc);
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) acc = __fadd_rn(acc, __shfl_xor_sync(0xFFFFFFFFu, acc, m));
  if (lane == 0) y[s * rows + r] = acc;
}

// features [256]: cos of the 128 angles, then their sin (QwenImage21TemporalTimesteps).
__global__ void timestep_features_kernel(float sigma, Freqs freqs, float *features) {
  const int i = threadIdx.x;
  if (i >= static_cast<int>(kTimestep)) return;
  const float t = __fmul_rn(1000.0f, sigma);
  const float a = __fmul_rn(t, freqs.f[i % kHalf]);
  features[i] = i < kHalf ? cosf(a) : sinf(a);
}

__global__ void silu_kernel(const float *x, int n, float *y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __fdiv_rn(x[i], __fadd_rn(1.0f, expf(-x[i])));
}

// The gates of a modulation row, tanh'd: out[0, 4096) = tanh(gate1), out[4096, 8192) = tanh(gate2).
__global__ void tanh_gates_kernel(const float *mod, float *out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= static_cast<int>(2 * kHidden)) return;
  const int at = i < static_cast<int>(kHidden) ? static_cast<int>(kHidden) + i : 3 * static_cast<int>(kHidden) + (i - static_cast<int>(kHidden));
  out[i] = tanhf(mod[at]);
}

__global__ void to_bf16_kernel(const float *x, std::size_t n, __nv_bfloat16 *y) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    y[e] = __float2bfloat16_rn(x[e]);
  }
}

__global__ void euler_kernel(float *latents, const float *velocity, std::size_t n, float dt) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    latents[e] = __fadd_rn(latents[e], __fmul_rn(dt, velocity[e]));
  }
}

// PyTorch's fp32 freqs (QwenImage21TemporalTimesteps, computed on the host at init): exp of
// fl32(fl32(-ln 10000 * i) / 128), the scalar rounded to fp32 first, exp correctly rounded.
Freqs make_freqs() {
  Freqs f{};
  const float c = static_cast<float>(-std::log(10000.0));
  for (int i = 0; i < kHalf; ++i) {
    const float e = (c * static_cast<float>(i)) / static_cast<float>(kHalf);
    f.f[i] = static_cast<float>(std::exp(static_cast<double>(e)));
  }
  return f;
}

constexpr std::size_t align256(std::size_t n) { return (n + 255) & ~static_cast<std::size_t>(255); }

// The workspace, carved from one buffer:
//   gates    fp32 [2][4096], the modulation row's tanh'd gates (the modulation call's scratch too);
//   scales   fp32 [tokens], the E4M3 activations' token scales;
//   codes    E4M3 [tokens][12288], the GEMM input (the norms', the attention output's, SwiGLU's);
//   region   reused by phase:
//     attention  q | k | v | o, BF16 [T][4096] each, then the q8 buffer and the target's K/V run
//     MLP        gate | up, BF16 [T][12288] each
//     delta      the out-projections' fp32 [T][4096] at the region's start: to_out writes it once
//                the attention output is quantized, down once SwiGLU has read gate and up
//     txt_in     the rows in BF16 | in_layer's output BF16, [T][4096] each
constexpr std::size_t kGateBytes = 2 * kHidden * 4;
constexpr std::size_t kModulationScratch = (kTimestep + 4 * kHidden) * 4;  // features, h, silu(h), temb, silu(temb)

struct Workspace {
  float *gates;
  float *scales;
  uint8_t *codes;
  __nv_bfloat16 *q, *k, *v, *o, *gate, *up, *rows, *hidden;
  float *delta;
  void *q8, *kv8;
};

std::size_t region_size(std::size_t tokens) {
  const std::size_t attention = 4 * 8192 * tokens + align256(ignis_image_attn_q8_bytes(static_cast<uint32_t>(tokens))) +
                                align256(ignis_image_attn_kv8_bytes(static_cast<uint32_t>(tokens)));
  return std::max(attention, static_cast<std::size_t>(2 * kMlp * 2) * tokens);
}

std::size_t workspace_size(std::size_t tokens) {
  return std::max(align256(kGateBytes), align256(kModulationScratch)) + align256(4 * tokens) +
         align256(static_cast<std::size_t>(kMlp) * tokens) + region_size(tokens);
}

Workspace carve(void *base, std::size_t tokens) {
  uint8_t *p = static_cast<uint8_t *>(base);
  Workspace w{};
  w.gates = reinterpret_cast<float *>(p);
  p += std::max(align256(kGateBytes), align256(kModulationScratch));
  w.scales = reinterpret_cast<float *>(p);
  p += align256(4 * tokens);
  w.codes = p;
  p += align256(static_cast<std::size_t>(kMlp) * tokens);
  w.q = reinterpret_cast<__nv_bfloat16 *>(p);
  w.delta = reinterpret_cast<float *>(p);
  w.gate = w.q;
  w.rows = w.q;
  w.k = reinterpret_cast<__nv_bfloat16 *>(p + 8192 * tokens);
  w.hidden = w.k;
  w.v = reinterpret_cast<__nv_bfloat16 *>(p + 2 * 8192 * tokens);
  w.o = reinterpret_cast<__nv_bfloat16 *>(p + 3 * 8192 * tokens);
  w.up = reinterpret_cast<__nv_bfloat16 *>(p + kMlp * 2 * tokens);
  w.q8 = p + 4 * 8192 * tokens;
  w.kv8 = static_cast<uint8_t *>(w.q8) + align256(ignis_image_attn_q8_bytes(static_cast<uint32_t>(tokens)));
  return w;
}

int32_t launch_residual_norm(float *x, const float *delta, const float *tanh_gate, uint32_t tokens, uint32_t cols,
                             const float *scale, float eps, void *y, float *y_scales, uint32_t y_f32, void *stream) {
  const std::size_t smem = scale != nullptr ? static_cast<std::size_t>(cols) * 4 : 0;
  residual_norm_kernel<<<tokens, kThreads, smem, static_cast<cudaStream_t>(stream)>>>(
      x, delta, tanh_gate, static_cast<int>(cols), scale, eps, y, y_scales, y_f32);
  return check_launch("ignis_image_transformer_residual_norm");
}

int32_t launch_quantize_f32(const float *x, uint32_t tokens, uint32_t cols, void *codes, float *scales, void *stream) {
  quantize_f32_kernel<<<tokens, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(x, static_cast<int>(cols),
                                                                                 static_cast<uint8_t *>(codes), scales);
  return check_launch("ignis_image_transformer_quantize_f32");
}

int32_t launch_linear(const void *weight, uint32_t rows, uint32_t cols, const float *x, uint32_t n, float *y,
                      void *stream) {
  const std::size_t outputs = static_cast<std::size_t>(rows) * n;
  const std::size_t blocks = (outputs + kThreads / 32 - 1) / (kThreads / 32);
  linear_bf16_kernel<<<static_cast<unsigned>(blocks), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(weight), rows, static_cast<int>(cols), x, outputs, y);
  return check_launch("ignis_image_transformer_linear_bf16");
}

int32_t launch_silu(const float *x, uint32_t n, float *y, void *stream) {
  silu_kernel<<<(n + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(x, static_cast<int>(n), y);
  return check_launch("silu");
}

unsigned grid_for(std::size_t n) { return static_cast<unsigned>(std::min<std::size_t>((n + 255) / 256, 4096)); }

// A sub-call's failure, with where it happened in front of its own message.
int32_t failed(const std::string &op, const std::string &where, const char *message) {
  return fail(op + ": " + where + ": " + message);
}

std::string check_weights(const ignis_image_transformer_weights *w) {
  if (w == nullptr) return "NULL weights";
  const void *model[] = {w->img_in, w->txt_norm, w->txt_in, w->txt_out, w->time_linear_1,
                         w->time_linear_2, w->modulation, w->norm_out, w->proj_out};
  for (const void *p : model) {
    if (p == nullptr || !aligned(p, 16)) return "a NULL or unaligned model weight";
  }
  if (w->block_count != 0 && w->blocks == nullptr) return "NULL blocks";
  if (!(w->eps >= 0.0f)) return "eps must be >= 0";
  return "";
}

// The prefix pass's segments: in order, each non-empty, covering [0, prefix->tokens).
std::string check_segments(const ignis_image_transformer_prefix *prefix) {
  if (prefix->segments == nullptr || prefix->segment_count == 0) return "the prefix pass needs the prefix's segments";
  uint32_t at = 0;
  for (uint32_t i = 0; i < prefix->segment_count; ++i) {
    const ignis_image_transformer_segment &g = prefix->segments[i];
    if (g.start != at || g.end <= g.start) {
      return "segment " + std::to_string(i) + " [" + std::to_string(g.start) + ", " + std::to_string(g.end) +
             ") does not continue the prefix at " + std::to_string(at);
    }
    at = g.end;
  }
  if (at != prefix->tokens) return "the segments do not cover the prefix";
  return "";
}

// Blocks [first, end): every weight present and aligned, and the prefix's run of each block.
std::string check_blocks(const ignis_image_transformer_weights *weights, uint32_t first, uint32_t end,
                         const ignis_image_transformer_prefix *prefix) {
  for (uint32_t b = first; b < end; ++b) {
    const ignis_image_transformer_block &B = weights->blocks[b];
    const void *all[] = {B.to_q, B.to_k, B.to_v, B.to_out, B.norm_q, B.norm_k, B.mlp_gate, B.mlp_up, B.mlp_down};
    for (const void *p : all) {
      if (p == nullptr || !aligned(p, 16)) return "block " + std::to_string(b) + " has a NULL or unaligned weight";
    }
    if (prefix->runs[b] == nullptr || !aligned(prefix->runs[b], 16)) {
      return "the prefix's run of block " + std::to_string(b) + " is NULL or unaligned";
    }
  }
  return "";
}

std::string check_workspace(void *workspace, uint64_t bytes, std::size_t need) {
  if (workspace == nullptr || !aligned(workspace, 256)) return "the workspace must be 256-byte aligned";
  if (bytes < need) {
    return "the workspace holds " + std::to_string(bytes) + " bytes, the call needs " + std::to_string(need);
  }
  return "";
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" uint64_t ignis_image_transformer_workspace_bytes(uint32_t tokens) {
  if (tokens == 0 || tokens > kMaxTokens) return 0;
  return workspace_size(tokens);
}

extern "C" int32_t ignis_image_transformer_residual_norm(float *x, const float *delta, const float *gate,
                                                         uint32_t tokens, uint32_t cols, const float *scale, float eps,
                                                         void *y, float *y_scales, uint32_t y_f32, void *stream) {
  const std::string op = "ignis_image_transformer_residual_norm";
  if (x == nullptr) return fail(op + ": NULL x");
  if (delta == nullptr && scale == nullptr) return fail(op + ": neither delta nor scale: nothing to do");
  if ((delta == nullptr) != (gate == nullptr)) return fail(op + ": delta and gate come together");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (cols == 0 || cols % 8 != 0 || cols > kMaxNormCols) {
    return fail(op + ": cols must be a positive multiple of 8, at most 8192");
  }
  if (scale != nullptr) {
    if (y == nullptr) return fail(op + ": a norm needs y");
    if ((y_f32 != 0) == (y_scales != nullptr)) return fail(op + ": y_scales must be given exactly for E4M3 output");
    if (!aligned(scale, 16) || !aligned(y, 16) || (y_scales != nullptr && !aligned(y_scales, 4))) {
      return fail(op + ": scale and y must be 16-byte aligned, y_scales 4-byte aligned");
    }
    if (!(eps >= 0.0f)) return fail(op + ": eps must be >= 0");
  }
  if (!aligned(x, 16) || (delta != nullptr && (!aligned(delta, 16) || !aligned(gate, 16)))) {
    return fail(op + ": x, delta and gate must be 16-byte aligned");
  }
  return launch_residual_norm(x, delta, gate, tokens, cols, scale, eps, y, y_scales, y_f32, stream);
}

extern "C" int32_t ignis_image_transformer_linear_bf16(const void *weight, uint32_t rows, uint32_t cols, const float *x,
                                                       uint32_t n, float *y, void *stream) {
  const std::string op = "ignis_image_transformer_linear_bf16";
  if (weight == nullptr || x == nullptr || y == nullptr) return fail(op + ": NULL pointer");
  if (rows == 0 || rows % 8 != 0 || cols == 0 || cols % 8 != 0) {
    return fail(op + ": rows and cols must be positive multiples of 8");
  }
  if (n == 0 || static_cast<uint64_t>(rows) * n > 0x7FFFFFFFull * 8) return fail(op + ": n must be 1 .. a launch");
  if (!aligned(weight, 16) || !aligned(x, 16) || !aligned(y, 16)) return fail(op + ": pointers must be 16-byte aligned");
  return launch_linear(weight, rows, cols, x, n, y, stream);
}

extern "C" int32_t ignis_image_transformer_quantize_f32(const float *x, uint32_t tokens, uint32_t cols, void *codes,
                                                        float *scales, void *stream) {
  const std::string op = "ignis_image_transformer_quantize_f32";
  if (x == nullptr || codes == nullptr || scales == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > kMaxTokens) return fail(op + ": tokens must be 1 .. 2^20");
  if (cols == 0 || cols % 8 != 0) return fail(op + ": cols must be a positive multiple of 8");
  if (!aligned(x, 16) || !aligned(codes, 16) || !aligned(scales, 4)) {
    return fail(op + ": x and codes must be 16-byte aligned, scales 4-byte");
  }
  return launch_quantize_f32(x, tokens, cols, codes, scales, stream);
}

extern "C" int32_t ignis_image_transformer_modulation(const struct ignis_image_transformer_weights *weights,
                                                      float sigma, float *mod, float *out_scale, void *workspace,
                                                      uint64_t workspace_bytes, void *stream) {
  const std::string op = "ignis_image_transformer_modulation";
  std::string why = check_weights(weights);
  if (why.empty() && (mod == nullptr || out_scale == nullptr || !aligned(mod, 16) || !aligned(out_scale, 16))) {
    why = "mod and out_scale must be 16-byte aligned";
  }
  if (why.empty() && !(sigma >= 0.0f && std::isfinite(sigma))) why = "sigma must be finite and >= 0";
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, kModulationScratch);
  if (!why.empty()) return fail(op + ": " + why);
  float *features = static_cast<float *>(workspace);
  float *h = features + kTimestep, *a = h + kHidden, *temb = a + kHidden, *st = temb + kHidden;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  timestep_features_kernel<<<1, kTimestep, 0, s>>>(sigma, make_freqs(), features);
  if (check_launch("the timestep features") != 0) return failed(op, "the timestep features", last_error_cstr());
  if (launch_linear(weights->time_linear_1, kHidden, kTimestep, features, 1, h, stream) != 0 ||
      launch_silu(h, kHidden, a, stream) != 0 ||
      launch_linear(weights->time_linear_2, kHidden, kHidden, a, 1, temb, stream) != 0) {
    return failed(op, "the timestep embedder", last_error_cstr());
  }
  if (launch_silu(temb, kHidden, st, stream) != 0 || launch_linear(weights->modulation, kMod, kHidden, st, 1, mod, stream) != 0 ||
      launch_linear(weights->norm_out, kHidden, kHidden, st, 1, out_scale, stream) != 0) {
    return failed(op, "the modulation", last_error_cstr());
  }
  return 0;
}

extern "C" int32_t ignis_image_transformer_txt_in(const struct ignis_image_transformer_weights *weights, const float *x,
                                                  uint32_t rows, float *y, void *workspace, uint64_t workspace_bytes,
                                                  void *stream) {
  const std::string op = "ignis_image_transformer_txt_in";
  std::string why = check_weights(weights);
  if (why.empty() && (x == nullptr || y == nullptr || !aligned(x, 16) || !aligned(y, 16))) {
    why = "x and y must be 16-byte aligned";
  }
  if (why.empty() && (rows == 0 || rows > kMaxTokens)) why = "rows must be 1 .. 2^20";
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, workspace_size(rows));
  if (!why.empty()) return fail(op + ": " + why);
  const Workspace w = carve(workspace, rows);
  const std::size_t n = static_cast<std::size_t>(rows) * kHidden;
  to_bf16_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(x, n, w.rows);
  if (check_launch("the BF16 rows") != 0) return failed(op, "the BF16 rows", last_error_cstr());
  if (ignis_image_dit_rms_norm(w.rows, rows, kHidden, weights->txt_norm, weights->eps, w.codes, w.scales,
                               IGNIS_IMAGE_DIT_OUT_E4M3, stream) != 0) {
    return failed(op, "text_norm", ignis_image_dit_last_error());
  }
  if (ignis_image_gemm_w8a8(weights->txt_in, kHidden, kHidden, w.codes, w.scales, rows, w.hidden, 0, stream) != 0) {
    return failed(op, "in_layer", ignis_image_gemm_last_error());
  }
  if (ignis_image_dit_act(IGNIS_IMAGE_DIT_GELU_TANH, w.hidden, nullptr, rows, kHidden, kHidden, w.codes, w.scales,
                          IGNIS_IMAGE_DIT_OUT_E4M3, stream) != 0) {
    return failed(op, "the GELU", ignis_image_dit_last_error());
  }
  if (ignis_image_gemm_w8a8(weights->txt_out, kHidden, kHidden, w.codes, w.scales, rows, y, 1, stream) != 0) {
    return failed(op, "out_layer", ignis_image_gemm_last_error());
  }
  return 0;
}

extern "C" int32_t ignis_image_transformer_img_in(const struct ignis_image_transformer_weights *weights,
                                                  const float *latents, uint32_t tokens, float *y, void *workspace,
                                                  uint64_t workspace_bytes, void *stream) {
  const std::string op = "ignis_image_transformer_img_in";
  std::string why = check_weights(weights);
  if (why.empty() && (latents == nullptr || y == nullptr || !aligned(latents, 16) || !aligned(y, 16))) {
    why = "latents and y must be 16-byte aligned";
  }
  if (why.empty() && (tokens == 0 || tokens > kMaxTokens)) why = "tokens must be 1 .. 2^20";
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, workspace_size(tokens));
  if (!why.empty()) return fail(op + ": " + why);
  const Workspace w = carve(workspace, tokens);
  if (launch_quantize_f32(latents, tokens, kLatent, w.codes, w.scales, stream) != 0) {
    return failed(op, "the latents' quantization", last_error_cstr());
  }
  if (ignis_image_gemm_w8a8(weights->img_in, kHidden, kLatent, w.codes, w.scales, tokens, y, 1, stream) != 0) {
    return failed(op, "img_in", ignis_image_gemm_last_error());
  }
  return 0;
}

extern "C" int32_t ignis_image_transformer_blocks(const struct ignis_image_transformer_weights *weights,
                                                  uint32_t first_block, uint32_t end_block, float *x, uint32_t tokens,
                                                  uint32_t target, const float *rope, const float *mod,
                                                  const struct ignis_image_transformer_prefix *prefix, void *workspace,
                                                  uint64_t workspace_bytes, void *stream) {
  const std::string op = "ignis_image_transformer_blocks";
  std::string why = check_weights(weights);
  if (why.empty() && (first_block > end_block || end_block > weights->block_count)) {
    why = "the blocks [" + std::to_string(first_block) + ", " + std::to_string(end_block) +
          ") are not within the model's " + std::to_string(weights->block_count);
  }
  if (why.empty() && (x == nullptr || rope == nullptr || mod == nullptr || !aligned(x, 16) || !aligned(rope, 16) ||
                      !aligned(mod, 16))) {
    why = "x, rope and mod must be 16-byte aligned";
  }
  if (why.empty() && (tokens == 0 || tokens > kMaxTokens)) why = "tokens must be 1 .. 2^20";
  if (why.empty() && (prefix == nullptr || prefix->tokens == 0 || prefix->runs == nullptr)) {
    why = "a prefix of at least one token, with its runs";
  }
  if (why.empty() && !target) {
    if (tokens != prefix->tokens) {
      why = "the prefix pass runs the prefix's " + std::to_string(prefix->tokens) + " tokens, not " +
            std::to_string(tokens);
    } else {
      why = check_segments(prefix);
    }
  }
  if (why.empty()) why = check_blocks(weights, first_block, end_block, prefix);
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, workspace_size(tokens));
  if (!why.empty()) return fail(op + ": " + why);
  if (first_block == end_block) return 0;

  const Workspace w = carve(workspace, tokens);
  const float eps = weights->eps;
  const float *scale1 = mod, *scale2 = mod + 2 * kHidden;
  const float *tanh_gate1 = w.gates, *tanh_gate2 = w.gates + kHidden;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  const uint32_t P = prefix->tokens;
  tanh_gates_kernel<<<(2 * kHidden + 255) / 256, 256, 0, s>>>(mod, w.gates);
  if (check_launch("the gates") != 0) return failed(op, "the gates", last_error_cstr());
  if (launch_residual_norm(x, nullptr, nullptr, tokens, kHidden, scale1, eps, w.codes, w.scales, 0, stream) != 0) {
    return failed(op, "block " + std::to_string(first_block) + ", img_norm1", last_error_cstr());
  }
  for (uint32_t b = first_block; b < end_block; ++b) {
    const ignis_image_transformer_block &B = weights->blocks[b];
    const std::string at = "block " + std::to_string(b) + ", ";
    void *run = prefix->runs[b];
    // attention: img_norm1's codes are in w.codes
    if (ignis_image_gemm_w8a8(B.to_q, kHidden, kHidden, w.codes, w.scales, tokens, w.q, 0, stream) != 0 ||
        ignis_image_gemm_w8a8(B.to_k, kHidden, kHidden, w.codes, w.scales, tokens, w.k, 0, stream) != 0 ||
        ignis_image_gemm_w8a8(B.to_v, kHidden, kHidden, w.codes, w.scales, tokens, w.v, 0, stream) != 0) {
      return failed(op, at + "to_q / to_k / to_v", ignis_image_gemm_last_error());
    }
    if (ignis_image_dit_qk_norm_rope(w.q, w.k, tokens, kHidden, B.norm_q, B.norm_k, eps, rope, w.q, w.k, kHidden, 0,
                                     stream) != 0) {
      return failed(op, at + "the QK norm and RoPE", ignis_image_dit_last_error());
    }
    if (!target) {
      if (ignis_image_attn_store_kv(w.k, w.v, tokens, kHidden, nullptr, 0, run, stream) != 0) {
        return failed(op, at + "the prefix's K/V run", ignis_image_attn_last_error());
      }
      for (uint32_t i = 0; i < prefix->segment_count; ++i) {
        const ignis_image_transformer_segment &g = prefix->segments[i];
        const uint32_t len = g.end - g.start;
        const std::size_t row = static_cast<std::size_t>(g.start) * kHidden;
        if (ignis_image_attn_quantize_q(w.q + row, len, kHidden, w.q8, stream) != 0 ||
            ignis_image_attn(w.q8, len, run, P, g.end, nullptr, 0, g.text != 0, kAttnScale, w.o + row, kHidden, 0,
                             stream) != 0) {
          return failed(op, at + "the attention of segment " + std::to_string(i), ignis_image_attn_last_error());
        }
      }
    } else {
      if (ignis_image_attn_store_kv(w.k, w.v, tokens, kHidden, run, P, w.kv8, stream) != 0 ||
          ignis_image_attn_quantize_q(w.q, tokens, kHidden, w.q8, stream) != 0 ||
          ignis_image_attn(w.q8, tokens, run, P, P, w.kv8, tokens, 0, kAttnScale, w.o, kHidden, 0, stream) != 0) {
        return failed(op, at + "the attention", ignis_image_attn_last_error());
      }
    }
    if (ignis_image_quantize_tokens(w.o, tokens, kHidden, w.codes, w.scales, stream) != 0) {
      return failed(op, at + "the attention output's quantization", ignis_image_gemm_last_error());
    }
    if (ignis_image_gemm_w8a8(B.to_out, kHidden, kHidden, w.codes, w.scales, tokens, w.delta, 1, stream) != 0) {
      return failed(op, at + "to_out", ignis_image_gemm_last_error());
    }
    if (launch_residual_norm(x, w.delta, tanh_gate1, tokens, kHidden, scale2, eps, w.codes, w.scales, 0, stream) != 0) {
      return failed(op, at + "the attention's residual and img_norm2", last_error_cstr());
    }
    // MLP
    if (ignis_image_gemm_w8a8(B.mlp_gate, kMlp, kHidden, w.codes, w.scales, tokens, w.gate, 0, stream) != 0 ||
        ignis_image_gemm_w8a8(B.mlp_up, kMlp, kHidden, w.codes, w.scales, tokens, w.up, 0, stream) != 0) {
      return failed(op, at + "gate_layer / proj", ignis_image_gemm_last_error());
    }
    if (ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, w.gate, w.up, tokens, kMlp, kMlp, w.codes, w.scales,
                            IGNIS_IMAGE_DIT_OUT_E4M3, stream) != 0) {
      return failed(op, at + "SwiGLU", ignis_image_dit_last_error());
    }
    if (ignis_image_gemm_w8a8(B.mlp_down, kHidden, kMlp, w.codes, w.scales, tokens, w.delta, 1, stream) != 0) {
      return failed(op, at + "out", ignis_image_gemm_last_error());
    }
    // x += the MLP's output, then the next block's img_norm1 (fused with the add).
    const float *next = b + 1 < end_block ? scale1 : nullptr;
    if (launch_residual_norm(x, w.delta, tanh_gate2, tokens, kHidden, next, eps, w.codes, w.scales, 0, stream) != 0) {
      return failed(op, at + (next != nullptr ? "the MLP's residual and the next img_norm1" : "the MLP's residual"),
                    last_error_cstr());
    }
  }
  return 0;
}

extern "C" uint64_t ignis_image_transformer_prefix_kv_bytes(uint32_t tokens) {
  if (tokens == 0 || tokens > kMaxTokens) return 0;
  return 2 * align256(static_cast<std::size_t>(tokens) * kHidden * 2);
}

extern "C" int32_t ignis_image_transformer_prefix_blocks(const struct ignis_image_transformer_weights *weights,
                                                         uint32_t first_block, uint32_t end_block, float *x,
                                                         const float *rope, const float *mod,
                                                         const struct ignis_image_transformer_prefix *prefix, void *kv,
                                                         uint32_t chunk, void *workspace, uint64_t workspace_bytes,
                                                         void *stream) {
  const std::string op = "ignis_image_transformer_prefix_blocks";
  std::string why = check_weights(weights);
  if (why.empty() && (first_block > end_block || end_block > weights->block_count)) {
    why = "the blocks [" + std::to_string(first_block) + ", " + std::to_string(end_block) +
          ") are not within the model's " + std::to_string(weights->block_count);
  }
  if (why.empty() && (x == nullptr || rope == nullptr || mod == nullptr || !aligned(x, 16) || !aligned(rope, 16) ||
                      !aligned(mod, 16))) {
    why = "x, rope and mod must be 16-byte aligned";
  }
  if (why.empty() && (prefix == nullptr || prefix->tokens == 0 || prefix->tokens > kMaxTokens || prefix->runs == nullptr)) {
    why = "a prefix of 1 .. 2^20 tokens, with its runs";
  }
  if (why.empty()) why = check_segments(prefix);
  if (why.empty() && (chunk == 0 || chunk % 128 != 0 || chunk > kMaxTokens)) {
    why = "chunk must be a positive multiple of 128, at most 2^20";
  }
  if (why.empty() && (kv == nullptr || !aligned(kv, 256))) why = "kv must be 256-byte aligned";
  if (why.empty()) why = check_blocks(weights, first_block, end_block, prefix);
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, workspace_size(chunk));
  if (!why.empty()) return fail(op + ": " + why);
  if (first_block == end_block) return 0;

  const uint32_t P = prefix->tokens;
  // The pieces the attention runs: each segment, or a segment longer than `chunk` cut every `chunk`
  // queries (a multiple of 128 from its start, so the quantized queries' row blocks and the
  // attention's query tiles fall as in one call; a causal text piece ends its keys at its own end).
  // Phase B's chunks: consecutive pieces of at most `chunk` tokens in all, a cut segment's pieces
  // in chunks of their own.
  struct Piece {
    uint32_t start, end, keys, causal;
  };
  struct Chunk {
    uint32_t start, end;
    std::size_t first, last;
  };
  std::vector<Piece> pieces;
  std::vector<Chunk> chunks;
  uint32_t open = 0;
  auto close = [&](uint32_t at) {
    if (at > open) chunks.push_back({open, at, chunks.empty() ? 0 : chunks.back().last, pieces.size()});
    open = at;
  };
  for (uint32_t i = 0; i < prefix->segment_count; ++i) {
    const ignis_image_transformer_segment &g = prefix->segments[i];
    if (g.end - open > chunk) close(g.start);
    for (uint32_t s = g.start; s < g.end; s += chunk) {
      const uint32_t e = std::min(g.end, s + chunk);
      if (s > g.start) close(s);
      pieces.push_back({s, e, g.text != 0 ? e : g.end, g.text != 0 ? 1u : 0u});
    }
  }
  close(P);

  const Workspace w = carve(workspace, chunk);
  const float eps = weights->eps;
  const float *scale1 = mod, *scale2 = mod + 2 * kHidden;
  const float *tanh_gate1 = w.gates, *tanh_gate2 = w.gates + kHidden;
  const cudaStream_t s = static_cast<cudaStream_t>(stream);
  __nv_bfloat16 *kv_k = static_cast<__nv_bfloat16 *>(kv);
  __nv_bfloat16 *kv_v = reinterpret_cast<__nv_bfloat16 *>(static_cast<uint8_t *>(kv) +
                                                          align256(static_cast<std::size_t>(P) * kHidden * 2));
  tanh_gates_kernel<<<(2 * kHidden + 255) / 256, 256, 0, s>>>(mod, w.gates);
  if (check_launch("the gates") != 0) return failed(op, "the gates", last_error_cstr());
  for (uint32_t b = first_block; b < end_block; ++b) {
    const ignis_image_transformer_block &B = weights->blocks[b];
    const std::string at = "block " + std::to_string(b) + ", ";
    void *run = prefix->runs[b];
    // Phase A: every token's K (normed and turned) and V into kv, then the block's run from all.
    for (uint32_t c0 = 0; c0 < P; c0 += chunk) {
      const uint32_t n = std::min(chunk, P - c0);
      const std::size_t row = static_cast<std::size_t>(c0) * kHidden;
      if (launch_residual_norm(x + row, nullptr, nullptr, n, kHidden, scale1, eps, w.codes, w.scales, 0, stream) != 0) {
        return failed(op, at + "img_norm1", last_error_cstr());
      }
      if (ignis_image_gemm_w8a8(B.to_k, kHidden, kHidden, w.codes, w.scales, n, w.k, 0, stream) != 0 ||
          ignis_image_gemm_w8a8(B.to_v, kHidden, kHidden, w.codes, w.scales, n, kv_v + row, 0, stream) != 0) {
        return failed(op, at + "to_k / to_v", ignis_image_gemm_last_error());
      }
      // The k heads' norm and RoPE into kv; the call's q side reads the same keys into scratch.
      if (ignis_image_dit_qk_norm_rope(w.k, w.k, n, kHidden, B.norm_k, B.norm_k, eps,
                                       rope + static_cast<std::size_t>(c0) * 128, w.q, kv_k + row, kHidden, 0,
                                       stream) != 0) {
        return failed(op, at + "the K norm and RoPE", ignis_image_dit_last_error());
      }
    }
    if (ignis_image_attn_store_kv(kv_k, kv_v, P, kHidden, nullptr, 0, run, stream) != 0) {
      return failed(op, at + "the prefix's K/V run", ignis_image_attn_last_error());
    }
    // Phase B, chunk by chunk: Q, the attention of its pieces over the run, to_out and the MLP.
    for (const Chunk &c : chunks) {
      const uint32_t n = c.end - c.start;
      float *xc = x + static_cast<std::size_t>(c.start) * kHidden;
      if (launch_residual_norm(xc, nullptr, nullptr, n, kHidden, scale1, eps, w.codes, w.scales, 0, stream) != 0) {
        return failed(op, at + "img_norm1", last_error_cstr());
      }
      if (ignis_image_gemm_w8a8(B.to_q, kHidden, kHidden, w.codes, w.scales, n, w.q, 0, stream) != 0) {
        return failed(op, at + "to_q", ignis_image_gemm_last_error());
      }
      // The q heads' norm and RoPE into w.k; the call's k side writes scratch.
      if (ignis_image_dit_qk_norm_rope(w.q, w.q, n, kHidden, B.norm_q, B.norm_q, eps,
                                       rope + static_cast<std::size_t>(c.start) * 128, w.k, w.v, kHidden, 0,
                                       stream) != 0) {
        return failed(op, at + "the Q norm and RoPE", ignis_image_dit_last_error());
      }
      for (std::size_t i = c.first; i < c.last; ++i) {
        const Piece &g = pieces[i];
        const std::size_t off = static_cast<std::size_t>(g.start - c.start) * kHidden;
        if (ignis_image_attn_quantize_q(w.k + off, g.end - g.start, kHidden, w.q8, stream) != 0 ||
            ignis_image_attn(w.q8, g.end - g.start, run, P, g.keys, nullptr, 0, g.causal, kAttnScale, w.o + off,
                             kHidden, 0, stream) != 0) {
          return failed(op, at + "the attention of tokens [" + std::to_string(g.start) + ", " + std::to_string(g.end) + ")",
                        ignis_image_attn_last_error());
        }
      }
      if (ignis_image_quantize_tokens(w.o, n, kHidden, w.codes, w.scales, stream) != 0) {
        return failed(op, at + "the attention output's quantization", ignis_image_gemm_last_error());
      }
      if (ignis_image_gemm_w8a8(B.to_out, kHidden, kHidden, w.codes, w.scales, n, w.delta, 1, stream) != 0) {
        return failed(op, at + "to_out", ignis_image_gemm_last_error());
      }
      if (launch_residual_norm(xc, w.delta, tanh_gate1, n, kHidden, scale2, eps, w.codes, w.scales, 0, stream) != 0) {
        return failed(op, at + "the attention's residual and img_norm2", last_error_cstr());
      }
      if (ignis_image_gemm_w8a8(B.mlp_gate, kMlp, kHidden, w.codes, w.scales, n, w.gate, 0, stream) != 0 ||
          ignis_image_gemm_w8a8(B.mlp_up, kMlp, kHidden, w.codes, w.scales, n, w.up, 0, stream) != 0) {
        return failed(op, at + "gate_layer / proj", ignis_image_gemm_last_error());
      }
      if (ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, w.gate, w.up, n, kMlp, kMlp, w.codes, w.scales,
                              IGNIS_IMAGE_DIT_OUT_E4M3, stream) != 0) {
        return failed(op, at + "SwiGLU", ignis_image_dit_last_error());
      }
      if (ignis_image_gemm_w8a8(B.mlp_down, kHidden, kMlp, w.codes, w.scales, n, w.delta, 1, stream) != 0) {
        return failed(op, at + "out", ignis_image_gemm_last_error());
      }
      // The MLP's residual alone: the next block's img_norm1 runs in its phase A.
      if (launch_residual_norm(xc, w.delta, tanh_gate2, n, kHidden, nullptr, eps, nullptr, nullptr, 0, stream) != 0) {
        return failed(op, at + "the MLP's residual", last_error_cstr());
      }
    }
  }
  return 0;
}

extern "C" int32_t ignis_image_transformer_head(const struct ignis_image_transformer_weights *weights, const float *x,
                                                uint32_t tokens, const float *out_scale, float *velocity,
                                                float *latents, float sigma, float sigma_next, void *workspace,
                                                uint64_t workspace_bytes, void *stream) {
  const std::string op = "ignis_image_transformer_head";
  std::string why = check_weights(weights);
  if (why.empty() && (x == nullptr || out_scale == nullptr || velocity == nullptr || !aligned(x, 16) ||
                      !aligned(out_scale, 16) || !aligned(velocity, 16) || (latents != nullptr && !aligned(latents, 16)))) {
    why = "x, out_scale, velocity and latents must be 16-byte aligned";
  }
  if (why.empty() && (tokens == 0 || tokens > kMaxTokens)) why = "tokens must be 1 .. 2^20";
  if (why.empty() && latents != nullptr && !(std::isfinite(sigma) && std::isfinite(sigma_next))) {
    why = "sigma and sigma_next must be finite";
  }
  if (why.empty()) why = check_workspace(workspace, workspace_bytes, workspace_size(tokens));
  if (!why.empty()) return fail(op + ": " + why);
  const Workspace w = carve(workspace, tokens);
  // The norm reads x and never writes it (no delta).
  if (launch_residual_norm(const_cast<float *>(x), nullptr, nullptr, tokens, kHidden, out_scale, weights->eps, w.codes,
                           w.scales, 0, stream) != 0) {
    return failed(op, "norm_out", last_error_cstr());
  }
  if (ignis_image_gemm_w8a8(weights->proj_out, kLatent, kHidden, w.codes, w.scales, tokens, velocity, 1, stream) != 0) {
    return failed(op, "proj_out", ignis_image_gemm_last_error());
  }
  if (latents != nullptr) {
    const std::size_t n = static_cast<std::size_t>(tokens) * kLatent;
    euler_kernel<<<grid_for(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(latents, velocity, n, sigma_next - sigma);
    if (check_launch("the Euler update") != 0) return failed(op, "the Euler update", last_error_cstr());
  }
  return 0;
}

extern "C" const char *ignis_image_transformer_last_error(void) { return last_error_cstr(); }
