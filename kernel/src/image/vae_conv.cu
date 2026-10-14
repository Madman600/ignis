// ignis kernel leaf: the image model's VAE convolution (kernel/include/ignis_image_vae.h; spec
// image-model/02 acceptance 6, GitHub #319) -- OURS (ADR 0051 tier 1). An implicit GEMM written
// from our FP8 GEMM (fp8_gemm_kernel.cuh): the same cp.async multistage loop and ldmatrix
// fragments, with BF16 operands on mma.sync m16n8k16 (fp32 accumulate) and the A tile gathered from
// the input's pixels on the fly instead of read as rows. No convolution source was read or copied.
// The shared-memory helpers are attn_smem.cuh's (adapted from FlashInfer through SageAttention,
// Apache-2.0; kernel/NOTICE).
//
//   operands  A = the window's output pixels x the reduction (the taps in order, the input
//             channels within a tap), gathered from x; B = the packed weight rows [out][ky][kx]
//             [in]. Both k-contiguous ("TN"), as mma.sync's row.col form takes them.
//   tile      128 pixels x 96 output channels per CTA, 8 warps of 32 x 48 (4 x 2), 32 BF16 (64
//             bytes) of the reduction per stage, 3 cp.async stages: 42 KiB, within the default
//             limit, so no prepare call and two CTAs an SM. 96 divides every VAE channel count
//             but 4, 64, 128 and 144. Shared rows are swizzled by swz64.
//   gather    a thread loads the same piece position (8 channels, 16 bytes) of 2 pixel rows per
//             stage: it keeps its pixels' receptive-field corners and walks its (tap, channel)
//             position one stage at a time. A piece never straddles two taps (in_ch % 8 == 0);
//             in_ch = 4 (RGBA, the encoder's conv_in) takes 8-byte pieces of one tap each.
//             Pieces in the padding (the stride-2 kind's right column and bottom row included)
//             or past the reduction are zero-filled, and x is never read there. With the upsample
//             the gather reads pixel (iy >> 1, ix >> 1) for position (iy, ix) of the 2x grid:
//             nearest-exact 2x, never materialized.
//   epilogue  y = ((acc * row_scale) + bias) + residual, each an fp32 operation, then one rounding
//             to the output type; two outputs per store.
//   order     a 1D grid, the output-channel tiles of one pixel tile adjacent, so the CTAs that
//             share a gathered pixel tile run together.
//
// Deterministic: each output is one thread's fixed sequence of instructions, no atomics, and its
// reduction order does not depend on the tile or the window, so a band computes exactly what the
// whole image does.

#include "attn_smem.cuh"
#include "ignis_image_vae.h"
#include "image_common.cuh"
#include "image_mma.cuh"
#include "vae_conv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kBM = 128, kBN = 96, kBK = 32, kStages = 3;
constexpr int kWarpsM = 4, kWarpsN = 2, kThreads = kWarpsM * kWarpsN * 32;
constexpr int kWarpM = kBM / kWarpsM, kWarpN = kBN / kWarpsN;  // 32 x 48
constexpr int kMI = kWarpM / 16, kNI = kWarpN / 8;             // 2 m16 tiles, 6 n8 tiles per warp
constexpr int kRowBytes = kBK * 2;                              // a shared row: 64 bytes
constexpr int kStageBytes = (kBM + kBN) * kRowBytes;
constexpr int kSmem = kStages * kStageBytes;
static_assert(kNI % 2 == 0, "B fragments load in n16 pairs");
// The launch raises no shared-memory limit (no prepare call): the stages stay within the default.
static_assert(kSmem <= 48 * 1024, "the stages must fit the default 48 KiB of dynamic shared memory");

// The same as cp_async16_zfill for 8 bytes (cp.async.ca: .cg takes 16 bytes only).
__device__ __forceinline__ void cp_async8_zfill(void *smem, const void *gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 8 : 0));
}

// VEC: the channels one load moves, 8 (16 bytes) or 4 (8 bytes).
template <int VEC> __global__ void __launch_bounds__(kThreads, 2) vae_conv_kernel(VaeConvArgs p, int n_tiles, int k_total) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int kPieces = kBK / VEC;               // pieces per shared row
  constexpr int kRowsPerPass = kThreads / kPieces;  // rows one pass of the CTA covers
  constexpr int kRowsA = kBM / kRowsPerPass;        // A rows per thread
  static_assert(kThreads % kPieces == 0 && kBM % kRowsPerPass == 0, "every thread loads whole rows' pieces");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int warp_m = warp / kWarpsN, warp_n = warp % kWarpsN;
  const int m0 = (static_cast<int>(blockIdx.x) / n_tiles) * kBM;
  const int n0 = (static_cast<int>(blockIdx.x) % n_tiles) * kBN;
  const int m_total = p.y_rows * p.out_w;
  const int taps = p.ksize * p.ksize;
  const int hl = p.height << p.upsample, wl = p.width << p.upsample;  // the grid the taps index

  // This thread's A rows: each pixel's receptive-field corner in the (upsampled) input grid; a
  // row past the window gets a corner no tap reaches, so its pieces are zeros.
  const int part = tid % kPieces;
  int iy0[kRowsA], ix0[kRowsA];
#pragma unroll
  for (int i = 0; i < kRowsA; ++i) {
    const int m = m0 + tid / kPieces + i * kRowsPerPass;
    const int oy = p.y_row0 + m / p.out_w, ox = m % p.out_w;
    iy0[i] = m < m_total ? oy * p.stride - p.pad : -(1 << 29);
    ix0[i] = ox * p.stride - p.pad;
  }
  // The (tap, channel) of this thread's piece in the stage loaded next.
  int tap = (part * VEC) / p.in_ch, ci = (part * VEC) % p.in_ch, k_next = 0;

  auto load_stage = [&](int stage) {
    uint8_t *sa = smem + stage * kStageBytes;
    uint8_t *sb = sa + kBM * kRowBytes;
    const int ky = tap / p.ksize, kx = tap - ky * p.ksize;
    const bool k_ok = tap < taps;
#pragma unroll
    for (int i = 0; i < kRowsA; ++i) {
      const int row = tid / kPieces + i * kRowsPerPass;
      const int iy = iy0[i] + ky, ix = ix0[i] + kx;
      const bool ok = k_ok && iy >= 0 && iy < hl && ix >= 0 && ix < wl;
      const std::size_t at =
          ok ? (static_cast<std::size_t>((iy >> p.upsample) - p.x_row0) * p.width + (ix >> p.upsample)) * p.in_stride + ci
             : 0;
      if constexpr (VEC == 8) {
        cp_async16_zfill(sa + swz64(row, part), p.x + at, ok);
      } else {
        cp_async8_zfill(sa + swz64(row, part >> 1) + (part & 1) * 8, p.x + at, ok);
      }
    }
    for (int pc = tid; pc < kBN * kPieces; pc += kThreads) {
      const int row = pc / kPieces, pp = pc % kPieces;
      const int k = k_next + pp * VEC;
      const bool ok = n0 + row < p.out_ch && k < k_total;
      const std::size_t at = ok ? static_cast<std::size_t>(n0 + row) * p.w_stride + k : 0;
      if constexpr (VEC == 8) {
        cp_async16_zfill(sb + swz64(row, pp), p.w + at, ok);
      } else {
        cp_async8_zfill(sb + swz64(row, pp >> 1) + (pp & 1) * 8, p.w + at, ok);
      }
    }
    // Advance to the next stage's position.
    k_next += kBK;
    ci += kBK;
    while (ci >= p.in_ch) {
      ci -= p.in_ch;
      ++tap;
    }
  };

  // ldmatrix addresses (image_mma.cuh's fragment tables): A rows lane & 15, chunk half lane >> 4;
  // B rows (lane & 7) plus 8 for lanes 16..31, chunk half (lane >> 3) & 1.
  const int a_row = warp_m * kWarpM + (lane & 15), a_half = lane >> 4;
  const int b_row = warp_n * kWarpN + (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
  auto load_frags = [&](const uint8_t *sa, const uint8_t *sb, int ks, uint32_t (&af)[kMI][4], uint32_t (&bf)[kNI][2]) {
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi) ldmatrix_x4(af[mi], sa + swz64(a_row + mi * 16, 2 * ks + a_half));
#pragma unroll
    for (int nj = 0; nj < kNI / 2; ++nj) {
      uint32_t r[4];
      ldmatrix_x4(r, sb + swz64(b_row + nj * 16, 2 * ks + b_half));
      bf[2 * nj][0] = r[0];
      bf[2 * nj][1] = r[1];
      bf[2 * nj + 1][0] = r[2];
      bf[2 * nj + 1][1] = r[3];
    }
  };

  float acc[kMI][kNI][4] = {};
  const int k_tiles = (k_total + kBK - 1) / kBK;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) load_stage(s);
    cp_async_commit();
  }
  for (int kt = 0; kt < k_tiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const int next = kt + kStages - 1;
    if (next < k_tiles) load_stage(next % kStages);
    cp_async_commit();
    const uint8_t *sa = smem + (kt % kStages) * kStageBytes;
    const uint8_t *sb = sa + kBM * kRowBytes;
    uint32_t af[2][kMI][4];
    uint32_t bf[2][kNI][2];
    load_frags(sa, sb, 0, af[0], bf[0]);
    load_frags(sa, sb, 1, af[1], bf[1]);
#pragma unroll
    for (int ks = 0; ks < kBK / 16; ++ks) {
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kNI; ++ni) mma_bf16_f32(acc[mi][ni], af[ks][mi], bf[ks][ni]);
      }
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int ni = 0; ni < kNI; ++ni) {
    const int col = n0 + warp_n * kWarpN + ni * 8 + 2 * t;
    if (col >= p.out_ch) continue;
    const bool pair = col + 1 < p.out_ch;
    const float b0 = p.bias != nullptr ? __bfloat162float(p.bias[col]) : 0.0f;
    const float b1 = p.bias != nullptr && pair ? __bfloat162float(p.bias[col + 1]) : 0.0f;
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int m = m0 + warp_m * kWarpM + mi * 16 + g + hh * 8;
        if (m >= m_total) continue;
        float v0 = acc[mi][ni][2 * hh], v1 = acc[mi][ni][2 * hh + 1];
        if (p.row_scale != nullptr) {
          const float s = p.row_scale[m];
          v0 = __fmul_rn(v0, s);
          v1 = __fmul_rn(v1, s);
        }
        if (p.bias != nullptr) {
          v0 = __fadd_rn(v0, b0);
          v1 = __fadd_rn(v1, b1);
        }
        const std::size_t at = static_cast<std::size_t>(m) * p.out_stride + col;
        if (p.residual != nullptr) {
          v0 = __fadd_rn(v0, __bfloat162float(p.residual[at]));
          if (pair) v1 = __fadd_rn(v1, __bfloat162float(p.residual[at + 1]));
        }
        if (p.y_f32) {
          float *y = static_cast<float *>(p.y) + at;
          if (pair) {
            *reinterpret_cast<float2 *>(y) = make_float2(v0, v1);
          } else {
            *y = v0;
          }
        } else {
          __nv_bfloat16 *y = static_cast<__nv_bfloat16 *>(p.y) + at;
          if (pair) {
            *reinterpret_cast<__nv_bfloat162 *>(y) = __floats2bfloat162_rn(v0, v1);
          } else {
            *y = __float2bfloat16_rn(v0);
          }
        }
      }
    }
  }
}

// w [out][in][k][k] -> w_packed [out][k][k][in], one element per thread.
__global__ void pack_weight_kernel(const __nv_bfloat16 *w, int out_ch, int in_ch, int k, __nv_bfloat16 *packed) {
  const std::size_t total = static_cast<std::size_t>(out_ch) * in_ch * k * k;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < total;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t i = e % in_ch, tap = (e / in_ch) % (k * k), o = e / (static_cast<std::size_t>(in_ch) * k * k);
    packed[e] = w[(o * in_ch + i) * k * k + tap];
  }
}

}  // namespace

int32_t vae_conv_launch(const VaeConvArgs &a, void *stream, const char *op) {
  const int n_tiles = (a.out_ch + kBN - 1) / kBN;
  const int m_tiles = (a.y_rows * a.out_w + kBM - 1) / kBM;
  const int k_total = a.ksize * a.ksize * a.in_ch;
  const dim3 grid(static_cast<unsigned>(m_tiles) * static_cast<unsigned>(n_tiles));
  if (a.in_ch % 8 == 0) {
    vae_conv_kernel<8><<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(a, n_tiles, k_total);
  } else {
    vae_conv_kernel<4><<<grid, kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(a, n_tiles, k_total);
  }
  return check_launch(op);
}

void vae_conv_input_rows(int ksize, int stride, int pad, int upsample, int height, int y_row0, int y_rows, int &lo,
                         int &hi) {
  const int hl = height << upsample;
  const int first = std::max(0, y_row0 * stride - pad);
  const int last = std::min(hl - 1, (y_row0 + y_rows - 1) * stride - pad + ksize - 1);
  lo = first >> upsample;
  hi = last < first ? lo : (last >> upsample) + 1;
}

}  // namespace ignis_image

using namespace ignis_image;

extern "C" const char *ignis_image_vae_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_vae_pack_conv_weight(const void *w, uint32_t out_channels, uint32_t in_channels,
                                                    uint32_t kernel, void *w_packed, void *stream) {
  const std::string op = "ignis_image_vae_pack_conv_weight";
  if (w == nullptr || w_packed == nullptr) return fail(op + ": NULL pointer");
  if (kernel != 1 && kernel != 3) return fail(op + ": kernel must be 1 or 3");
  if (out_channels == 0 || in_channels == 0 || out_channels > 65536 || in_channels > 65536) {
    return fail(op + ": channel counts must be in [1, 65536]");
  }
  const std::size_t total = static_cast<std::size_t>(out_channels) * in_channels * kernel * kernel;
  const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((total + 255) / 256, 4096));
  pack_weight_kernel<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16 *>(w), static_cast<int>(out_channels), static_cast<int>(in_channels),
      static_cast<int>(kernel), static_cast<__nv_bfloat16 *>(w_packed));
  return check_launch(op.c_str());
}

extern "C" int32_t ignis_image_vae_conv(uint32_t kind, uint32_t upsample, const void *x, uint32_t height,
                                        uint32_t width, uint32_t in_channels, uint32_t x_row0, uint32_t x_rows,
                                        const void *weight, const void *bias, uint32_t out_channels,
                                        const void *residual, void *y, uint32_t y_row0, uint32_t y_rows, uint32_t y_f32,
                                        void *stream) {
  const std::string op = "ignis_image_vae_conv";
  if (x == nullptr || weight == nullptr || y == nullptr) return fail(op + ": NULL pointer");
  if (kind > IGNIS_IMAGE_VAE_CONV_1X1) return fail(op + ": unknown kind");
  if (upsample > 1 || (upsample == 1 && kind != IGNIS_IMAGE_VAE_CONV_3X3)) {
    return fail(op + ": upsample is 0 or 1, and 1 only with the 3 x 3 stride-1 kind");
  }
  if (height == 0 || width == 0 || height > 32768 || width > 32768) return fail(op + ": height and width must be in [1, 32768]");
  if (kind == IGNIS_IMAGE_VAE_CONV_3X3_S2 && (height % 2 != 0 || width % 2 != 0)) {
    return fail(op + ": the stride-2 kind takes an even height and width");
  }
  if (in_channels == 0 || in_channels > 65536 || (in_channels % 8 != 0 && in_channels != 4)) {
    return fail(op + ": in_channels must be a multiple of 8 (at most 65536), or 4");
  }
  if (out_channels == 0 || out_channels > 65536 || out_channels % 2 != 0) {
    return fail(op + ": out_channels must be even and in [2, 65536]");
  }
  const uint32_t vec_bytes = in_channels % 8 == 0 ? 16 : 8;
  if (!aligned(x, vec_bytes) || !aligned(weight, vec_bytes)) {
    return fail(op + ": x and weight must be " + std::to_string(vec_bytes) + "-byte aligned");
  }
  if (!aligned(y, 8) || (bias != nullptr && !aligned(bias, 4)) || (residual != nullptr && !aligned(residual, 4))) {
    return fail(op + ": y must be 8-byte aligned, bias and residual 4-byte aligned");
  }
  VaeConvArgs a{};
  a.ksize = kind == IGNIS_IMAGE_VAE_CONV_1X1 ? 1 : 3;
  a.stride = kind == IGNIS_IMAGE_VAE_CONV_3X3_S2 ? 2 : 1;
  a.pad = kind == IGNIS_IMAGE_VAE_CONV_3X3 ? 1 : 0;
  a.upsample = static_cast<int>(upsample);
  const uint32_t out_h = a.stride == 2 ? height / 2 : height << upsample;
  const uint32_t out_w = a.stride == 2 ? width / 2 : width << upsample;
  if (y_rows == 0 || y_row0 >= out_h || y_rows > out_h - y_row0) {
    return fail(op + ": the output rows [y_row0, y_row0 + y_rows) must be a non-empty range of the " +
                std::to_string(out_h) + " output rows");
  }
  const uint64_t pixels = static_cast<uint64_t>(y_rows) * out_w;
  if (pixels > 0x7FFFFFFFull - kBM || (pixels + kBM - 1) / kBM * ((out_channels + kBN - 1) / kBN) > 0x7FFFFFFFull) {
    return fail(op + ": the window is too large for one launch");
  }
  int lo = 0, hi = 0;
  vae_conv_input_rows(a.ksize, a.stride, a.pad, a.upsample, static_cast<int>(height), static_cast<int>(y_row0),
                      static_cast<int>(y_rows), lo, hi);
  if (x_rows == 0 || x_row0 >= height || x_rows > height - x_row0 || (hi > lo && (static_cast<uint32_t>(lo) < x_row0 ||
                                                                                    static_cast<uint32_t>(hi) > x_row0 + x_rows))) {
    return fail(op + ": x holds input rows [" + std::to_string(x_row0) + ", " + std::to_string(x_row0 + x_rows) +
                "), and the output rows read [" + std::to_string(lo) + ", " + std::to_string(hi) + ")");
  }
  a.x = static_cast<const __nv_bfloat16 *>(x);
  a.height = static_cast<int>(height);
  a.width = static_cast<int>(width);
  a.in_ch = static_cast<int>(in_channels);
  a.in_stride = static_cast<int>(in_channels);
  a.x_row0 = static_cast<int>(x_row0);
  a.w = static_cast<const __nv_bfloat16 *>(weight);
  a.w_stride = a.ksize * a.ksize * static_cast<int>(in_channels);
  a.bias = static_cast<const __nv_bfloat16 *>(bias);
  a.residual = static_cast<const __nv_bfloat16 *>(residual);
  a.y = y;
  a.out_ch = static_cast<int>(out_channels);
  a.out_stride = static_cast<int>(out_channels);
  a.out_w = static_cast<int>(out_w);
  a.y_row0 = static_cast<int>(y_row0);
  a.y_rows = static_cast<int>(y_rows);
  a.y_f32 = y_f32 != 0 ? 1 : 0;
  return vae_conv_launch(a, stream, op.c_str());
}
