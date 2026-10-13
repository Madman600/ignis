// ignis kernel leaf: the image model's FP8 W8A8 GEMM (spec image-model/02 acceptance 2, GitHub
// #319) -- OURS (kernel/include/ignis_image_gemm.h; ADR 0051 tier 1). It extends ignis_fp8_linear's
// tensor-core route (kernel/src/fp8_linear.cu) to FP8 activations: the same weight payload and
// cp.async multistage shape, with both operands fed to the tensor cores as E4M3.
//
//   operands  A = the activation codes [tokens][cols], B = the weight codes [rows][cols]: both
//             k-contiguous ("TN"), as mma.sync's row.col form takes them. ldmatrix.b16 delivers
//             the m16n8k32 E4M3 fragments directly (image_mma.cuh).
//   tile      BM tokens x BN output rows per CTA, WM x WN warps, BK code bytes per stage, STAGES
//             cp.async stages (TileConfig; the configurations are listed below, the default is
//             128 x 256, 8 warps of 64 x 64, 64 bytes, 4 stages = 96 KiB). A shared row is
//             BK bytes; its 16-byte chunks are XOR-swizzled by the row so ldmatrix's eight rows
//             hit eight distinct 16-byte bank groups. A warp loads its next k32 step's
//             fragments while its current one's MMAs run.
//   mma       kind::mxf8f6f4 with unit block scales: the FP8 rate the 5090 runs FP32
//             accumulation at (spec image-model/02 "As built").
//   epilogue  y = (acc * scales[t]) * wscale[r], fp32 or BF16, two outputs per store.
//   order     a 1D grid of tiles walked in groups of kGroupM token tiles, so a group's weight
//             tiles are read from L2 rather than DRAM once per token tile.
//
// Deterministic: each output is one thread's fixed sequence of instructions, no atomics.

#include "fp8_gemm_internal.h"
#include "ignis_image_gemm.h"
#include "image_common.cuh"
#include "image_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>

namespace ignis_image {
namespace {

constexpr int kGroupM = 8;  // token tiles per raster group

template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_, int CTAS_> struct TileConfig {
  static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_, CTAS = CTAS_;
  static constexpr int THREADS = WM * WN * 32;
  static constexpr int WARP_M = BM / WM, WARP_N = BN / WN;
  static constexpr int MI = WARP_M / 16;  // m16 tiles per warp
  static constexpr int NI = WARP_N / 8;   // n8 tiles per warp
  static constexpr int CHUNKS = BK / 16;  // 16-byte chunks per shared row
  static constexpr int KSTEPS = BK / 32;  // k32 MMA steps per stage
  static constexpr int STAGE_BYTES = (BM + BN) * BK;
  static constexpr int SMEM = STAGES * STAGE_BYTES;
  static_assert(BK == 64 || BK == 128, "the swizzle covers 64- and 128-byte rows");
  static_assert(MI >= 1 && NI % 2 == 0 && WARP_M % 16 == 0, "a warp tile is m16 x (2 x n8) tiles");
  static_assert(BM * CHUNKS % THREADS == 0 && BN * CHUNKS % THREADS == 0, "every thread loads whole chunks");
};

// Byte offset of (row, 16-byte chunk) in a swizzled tile of BK-byte rows: the chunk XOR the row's
// position within its eight-row ldmatrix group, folded onto the row's chunks.
template <int BK> __device__ __forceinline__ int swz(int row, int chunk) {
  constexpr int chunks = BK / 16;
  return row * BK + ((chunk ^ ((row / (8 / chunks)) % chunks)) << 4);
}

__device__ __forceinline__ void cp_async16_zfill(void *smem, const void *gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

struct GemmArgs {
  const uint8_t *a;              // activation codes [tokens][cols]
  const float *a_scale;          // [tokens]
  const uint8_t *b;              // weight codes [rows][cols]
  const __nv_bfloat16 *b_scale;  // [rows]
  void *y;
  int tokens, rows, cols;
  int m_tiles, n_tiles;
  int y_f32;
};

template <class C> __global__ void __launch_bounds__(C::THREADS, C::CTAS) gemm_w8a8_kernel(GemmArgs p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp / C::WN;
  const int warp_n = warp % C::WN;

  // Grouped raster: kGroupM token tiles share each weight tile while it is in L2.
  const int per_group = kGroupM * p.n_tiles;
  const int group = blockIdx.x / per_group;
  const int first_m = group * kGroupM;
  const int group_m = min(p.m_tiles - first_m, kGroupM);
  const int in_group = blockIdx.x % per_group;
  const int m0 = (first_m + in_group % group_m) * C::BM;
  const int n0 = (in_group / group_m) * C::BN;

  auto stage_a = [&](int stage) { return smem_raw + stage * C::STAGE_BYTES; };
  auto stage_b = [&](int stage) { return smem_raw + stage * C::STAGE_BYTES + C::BM * C::BK; };

  auto load_stage = [&](int stage, int k0) {
    uint8_t *sa = stage_a(stage);
    uint8_t *sb = stage_b(stage);
#pragma unroll
    for (int i = 0; i < C::BM * C::CHUNKS / C::THREADS; ++i) {
      const int chunk = tid + i * C::THREADS;
      const int row = chunk / C::CHUNKS;
      const int part = chunk % C::CHUNKS;
      const bool ok = m0 + row < p.tokens;
      const uint8_t *src = p.a + static_cast<std::size_t>(ok ? m0 + row : 0) * p.cols + k0 + part * 16;
      cp_async16_zfill(sa + swz<C::BK>(row, part), src, ok);
    }
#pragma unroll
    for (int i = 0; i < C::BN * C::CHUNKS / C::THREADS; ++i) {
      const int chunk = tid + i * C::THREADS;
      const int row = chunk / C::CHUNKS;
      const int part = chunk % C::CHUNKS;
      const bool ok = n0 + row < p.rows;
      const uint8_t *src = p.b + static_cast<std::size_t>(ok ? n0 + row : 0) * p.cols + k0 + part * 16;
      cp_async16_zfill(sb + swz<C::BK>(row, part), src, ok);
    }
  };

  // ldmatrix addresses: A rows (lane & 15), chunk half (lane >> 4); B rows (lane & 7) plus 8 for
  // lanes 16..31, chunk half (lane >> 3) & 1 -- see image_mma.cuh's fragment table.
  const int a_row = warp_m * C::WARP_M + (lane & 15);
  const int a_half = lane >> 4;
  const int b_row = warp_n * C::WARP_N + (lane & 7) + ((lane >> 4) << 3);
  const int b_half = (lane >> 3) & 1;
  auto load_frags = [&](const uint8_t *sa, const uint8_t *sb, int ks, uint32_t (&af)[C::MI][4],
                        uint32_t (&bf)[C::NI][2]) {
#pragma unroll
    for (int mi = 0; mi < C::MI; ++mi) ldmatrix_x4(af[mi], sa + swz<C::BK>(a_row + mi * 16, 2 * ks + a_half));
#pragma unroll
    for (int nj = 0; nj < C::NI / 2; ++nj) {
      uint32_t r[4];
      ldmatrix_x4(r, sb + swz<C::BK>(b_row + nj * 16, 2 * ks + b_half));
      bf[2 * nj][0] = r[0];
      bf[2 * nj][1] = r[1];
      bf[2 * nj + 1][0] = r[2];
      bf[2 * nj + 1][1] = r[3];
    }
  };

  float acc[C::MI][C::NI][4] = {};
  const int k_tiles = p.cols / C::BK;
#pragma unroll
  for (int s = 0; s < C::STAGES - 1; ++s) {
    if (s < k_tiles) load_stage(s, s * C::BK);
    cp_async_commit();
  }

  for (int kt = 0; kt < k_tiles; ++kt) {
    cp_async_wait<C::STAGES - 2>();
    __syncthreads();
    const int next = kt + C::STAGES - 1;
    if (next < k_tiles) load_stage(next % C::STAGES, next * C::BK);
    cp_async_commit();
    const uint8_t *sa = stage_a(kt % C::STAGES);
    const uint8_t *sb = stage_b(kt % C::STAGES);
    uint32_t af[2][C::MI][4];
    uint32_t bf[2][C::NI][2];
    load_frags(sa, sb, 0, af[0], bf[0]);
#pragma unroll
    for (int ks = 0; ks < C::KSTEPS; ++ks) {
      if (ks + 1 < C::KSTEPS) load_frags(sa, sb, ks + 1, af[(ks + 1) & 1], bf[(ks + 1) & 1]);
#pragma unroll
      for (int mi = 0; mi < C::MI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < C::NI; ++ni) mma_e4m3_mx_unit_f32(acc[mi][ni], af[ks & 1][mi], bf[ks & 1][ni]);
      }
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2;
  const int t = lane & 3;
#pragma unroll
  for (int ni = 0; ni < C::NI; ++ni) {
    const int col = n0 + warp_n * C::WARP_N + ni * 8 + 2 * t;
    if (col >= p.rows) continue;
    const float w0 = __bfloat162float(p.b_scale[col]);
    const float w1 = __bfloat162float(p.b_scale[col + 1]);
#pragma unroll
    for (int mi = 0; mi < C::MI; ++mi) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + warp_m * C::WARP_M + mi * 16 + g + hh * 8;
        if (row >= p.tokens) continue;
        const float s = p.a_scale[row];
        const float v0 = __fmul_rn(__fmul_rn(acc[mi][ni][2 * hh], s), w0);
        const float v1 = __fmul_rn(__fmul_rn(acc[mi][ni][2 * hh + 1], s), w1);
        const std::size_t at = static_cast<std::size_t>(row) * p.rows + col;
        if (p.y_f32) {
          *reinterpret_cast<float2 *>(static_cast<float *>(p.y) + at) = make_float2(v0, v1);
        } else {
          *reinterpret_cast<__nv_bfloat162 *>(static_cast<__nv_bfloat16 *>(p.y) + at) = __floats2bfloat162_rn(v0, v1);
        }
      }
    }
  }
}

// The configurations: <BM, BN, BK, STAGES, WM, WN, CTAs per SM>.
using Config0 = TileConfig<128, 128, 64, 4, 2, 4, 1>;
using Config1 = TileConfig<128, 128, 64, 3, 2, 4, 2>;
using Config2 = TileConfig<128, 128, 128, 3, 2, 4, 1>;
using Config3 = TileConfig<128, 256, 64, 3, 2, 4, 1>;
using Config4 = TileConfig<256, 128, 64, 3, 4, 2, 1>;
using Config5 = TileConfig<128, 128, 64, 3, 2, 2, 2>;
using Config6 = TileConfig<128, 256, 64, 4, 2, 4, 1>;
constexpr int kConfigs = 7;
// The sweep of 2026-10-13 (spec image-model/02 "As built"): every configuration within ~6% of
// the others on a DiT block at 1024²; 6 was the fastest on the block and on the wide shapes.
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

constexpr int kMaxDevices = 64;
std::atomic<bool> g_prepared[kMaxDevices];

int32_t require_prepared(const char *op) {
  int device = 0;
  const cudaError_t err = cudaGetDevice(&device);
  if (err != cudaSuccess) return fail(std::string(op) + ": " + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices || !g_prepared[device].load()) {
    return fail(std::string(op) + ": device " + std::to_string(device) +
                " is not prepared; call ignis_image_gemm_prepare at load");
  }
  return 0;
}

}  // namespace

int gemm_config_count() { return kConfigs; }
int gemm_default_config() { return kDefaultConfig; }
const char *gemm_config_name(int config) {
  return config >= 0 && config < kConfigs ? kConfigNames[config] : "no such configuration";
}

int32_t gemm_w8a8_config(int config, const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  const char *op = "ignis_image_gemm_w8a8";
  if (config < 0 || config >= kConfigs) return fail(std::string(op) + ": no such tile configuration");
  if (weight == nullptr || codes == nullptr || scales == nullptr || y == nullptr) {
    return fail(std::string(op) + ": NULL pointer");
  }
  if (tokens == 0) return fail(std::string(op) + ": tokens must be at least 1");
  if (rows == 0 || rows % 16 != 0) return fail(std::string(op) + ": rows must be a positive multiple of 16");
  if (cols == 0 || cols % 64 != 0) return fail(std::string(op) + ": cols must be a positive multiple of 64");
  if (!aligned(weight, 16) || !aligned(codes, 16) || !aligned(scales, 4) || !aligned(y, 8)) {
    return fail(std::string(op) + ": weight and codes must be 16-byte aligned, scales 4-byte and y 8-byte aligned");
  }
  if (tokens > 0x7FFFFFFFu || rows > 0x7FFFFFFFu || cols > 0x7FFFFFFFu) {
    return fail(std::string(op) + ": the shape is too large for one launch");
  }
  if (require_prepared(op) != 0) return -1;
  const uint8_t *w = static_cast<const uint8_t *>(weight);
  const std::size_t scale_at = (static_cast<std::size_t>(rows) * cols + 255) / 256 * 256;
  GemmArgs p{};
  p.a = static_cast<const uint8_t *>(codes);
  p.a_scale = scales;
  p.b = w;
  p.b_scale = reinterpret_cast<const __nv_bfloat16 *>(w + scale_at);
  p.y = y;
  p.tokens = static_cast<int>(tokens);
  p.rows = static_cast<int>(rows);
  p.cols = static_cast<int>(cols);
  p.y_f32 = y_f32 != 0 ? 1 : 0;
  int32_t rc = 0;
  with_config(config, [&](auto c) {
    using C = decltype(c);
    if (cols % C::BK != 0) {
      rc = fail(std::string(op) + ": cols must be a multiple of the configuration's " + std::to_string(C::BK));
      return;
    }
    const uint64_t m_tiles = (static_cast<uint64_t>(tokens) + C::BM - 1) / C::BM;
    const uint64_t n_tiles = (static_cast<uint64_t>(rows) + C::BN - 1) / C::BN;
    if (m_tiles * n_tiles > 0x7FFFFFFFull) {
      rc = fail(std::string(op) + ": the shape is too large for one launch");
      return;
    }
    p.m_tiles = static_cast<int>(m_tiles);
    p.n_tiles = static_cast<int>(n_tiles);
    gemm_w8a8_kernel<C><<<static_cast<unsigned>(m_tiles * n_tiles), C::THREADS, C::SMEM, static_cast<cudaStream_t>(stream)>>>(p);
    rc = check_launch(op);
  });
  return rc;
}

}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_gemm_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  for (int config = 0; config < kConfigs && err == cudaSuccess; ++config) {
    with_config(config, [&](auto c) {
      using C = decltype(c);
      err = cudaFuncSetAttribute(gemm_w8a8_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
    });
  }
  if (err != cudaSuccess) return fail(std::string("ignis_image_gemm_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_image_gemm_prepare: device id out of range");
  g_prepared[device].store(true);
  return 0;
}

extern "C" const char *ignis_image_gemm_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_gemm_w8a8(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  return gemm_w8a8_config(kDefaultConfig, weight, rows, cols, codes, scales, tokens, y, y_f32, stream);
}
