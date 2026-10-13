// ignis kernel leaf: the image model's FP8 W8A8 GEMM (spec image-model/02 acceptance 2, GitHub
// #319) -- OURS (kernel/include/ignis_image_gemm.h; ADR 0051 tier 1). It extends ignis_fp8_linear's
// tensor-core route (kernel/src/fp8_linear.cu) to FP8 activations: the same weight payload and
// cp.async multistage shape, with both operands fed to the tensor cores as E4M3.
//
//   operands  A = the activation codes [tokens][cols], B = the weight codes [rows][cols]: both
//             k-contiguous ("TN"), as mma.sync's row.col form takes them. ldmatrix.b16 delivers
//             the m16n8k32 E4M3 fragments directly (image_mma.cuh).
//   tile      128 tokens x 128 output rows per CTA, 8 warps of 64 x 32, 64 code bytes per stage,
//             kStages cp.async stages. Shared rows are 64 bytes, their four 16-byte chunks
//             XOR-swizzled by (row >> 1) & 3, so ldmatrix's eight rows hit eight distinct
//             16-byte bank groups.
//   mma       kind::mxf8f6f4 with unit block scales: the FP8 rate the 5090 runs FP32
//             accumulation at (spec image-model/02 "As built").
//   epilogue  y = (acc * scales[t]) * wscale[r], fp32 or BF16, two outputs per store.
//   order     a 1D grid of tiles walked in groups of kGroupM token tiles, so a group's weight
//             tiles are read from L2 rather than DRAM once per token tile.
//
// Deterministic: each output is one thread's fixed sequence of instructions, no atomics.

#include "ignis_image_gemm.h"
#include "image_common.cuh"
#include "image_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>

namespace ignis_image {
namespace {

constexpr int kBM = 128;       // tokens per CTA
constexpr int kBN = 128;       // output rows per CTA
constexpr int kBK = 64;        // code bytes (= k) per stage
constexpr int kStages = 4;
constexpr int kThreads = 256;  // 8 warps: 2 along tokens x 4 along output rows
constexpr int kWarpM = 64;
constexpr int kWarpN = 32;
constexpr int kMI = kWarpM / 16;  // m16 tiles per warp
constexpr int kNI = kWarpN / 8;   // n8 tiles per warp
constexpr int kGroupM = 8;        // token tiles per raster group

struct Smem {
  uint8_t a[kStages][kBM * kBK];
  uint8_t b[kStages][kBN * kBK];
};
constexpr std::size_t kSmemBytes = sizeof(Smem);

// Byte offset of (row, 16-byte chunk) in a swizzled 64-byte-row tile.
__device__ __forceinline__ int swz(int row, int chunk) { return row * kBK + ((chunk ^ ((row >> 1) & 3)) << 4); }

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
  const uint8_t *a;          // activation codes [tokens][cols]
  const float *a_scale;      // [tokens]
  const uint8_t *b;          // weight codes [rows][cols]
  const __nv_bfloat16 *b_scale;  // [rows]
  void *y;
  int tokens, rows, cols;
  int m_tiles, n_tiles;
  int y_f32;
};

__global__ void __launch_bounds__(kThreads, 1) gemm_w8a8_kernel(GemmArgs p) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem &sm = *reinterpret_cast<Smem *>(smem_raw);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int warp_m = warp >> 2;  // 0..1
  const int warp_n = warp & 3;   // 0..3

  // Grouped raster: kGroupM token tiles share each weight tile while it is in L2.
  const int per_group = kGroupM * p.n_tiles;
  const int group = blockIdx.x / per_group;
  const int first_m = group * kGroupM;
  const int group_m = min(p.m_tiles - first_m, kGroupM);
  const int in_group = blockIdx.x % per_group;
  const int m0 = (first_m + in_group % group_m) * kBM;
  const int n0 = (in_group / group_m) * kBN;

  // Each stage: A and B are 128 rows x 4 chunks = 512 chunks each, two per thread per operand.
  auto load_stage = [&](int stage, int k0) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int chunk = tid + i * kThreads;
      const int row = chunk >> 2;
      const int part = chunk & 3;
      const bool a_ok = m0 + row < p.tokens;
      const uint8_t *a_src = p.a + static_cast<std::size_t>(a_ok ? m0 + row : 0) * p.cols + k0 + part * 16;
      cp_async16_zfill(&sm.a[stage][swz(row, part)], a_src, a_ok);
      const bool b_ok = n0 + row < p.rows;
      const uint8_t *b_src = p.b + static_cast<std::size_t>(b_ok ? n0 + row : 0) * p.cols + k0 + part * 16;
      cp_async16_zfill(&sm.b[stage][swz(row, part)], b_src, b_ok);
    }
  };

  float acc[kMI][kNI][4] = {};
  const int k_tiles = p.cols / kBK;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) load_stage(s, s * kBK);
    cp_async_commit();
  }

  // ldmatrix addresses: A rows (lane & 15), chunk half (lane >> 4); B rows (lane & 7) plus 8 for
  // lanes 16..31, chunk half (lane >> 3) & 1 -- see image_mma.cuh's fragment table.
  const int a_row = warp_m * kWarpM + (lane & 15);
  const int a_half = lane >> 4;
  const int b_row = warp_n * kWarpN + (lane & 7) + ((lane >> 4) << 3);
  const int b_half = (lane >> 3) & 1;

  for (int kt = 0; kt < k_tiles; ++kt) {
    cp_async_wait<kStages - 2>();
    __syncthreads();
    const int next = kt + kStages - 1;
    if (next < k_tiles) load_stage(next % kStages, next * kBK);
    cp_async_commit();
    const uint8_t *sa = sm.a[kt % kStages];
    const uint8_t *sb = sm.b[kt % kStages];
#pragma unroll
    for (int ks = 0; ks < kBK / 32; ++ks) {
      uint32_t af[kMI][4];
      uint32_t bf[kNI][2];
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi) ldmatrix_x4(af[mi], sa + swz(a_row + mi * 16, 2 * ks + a_half));
#pragma unroll
      for (int nj = 0; nj < kNI / 2; ++nj) {
        uint32_t r[4];
        ldmatrix_x4(r, sb + swz(b_row + nj * 16, 2 * ks + b_half));
        bf[2 * nj][0] = r[0];
        bf[2 * nj][1] = r[1];
        bf[2 * nj + 1][0] = r[2];
        bf[2 * nj + 1][1] = r[3];
      }
#pragma unroll
      for (int mi = 0; mi < kMI; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kNI; ++ni) mma_e4m3_mx_unit_f32(acc[mi][ni], af[mi], bf[ni]);
      }
    }
  }
  cp_async_wait<0>();

  const int g = lane >> 2;
  const int t = lane & 3;
#pragma unroll
  for (int ni = 0; ni < kNI; ++ni) {
    const int col = n0 + warp_n * kWarpN + ni * 8 + 2 * t;
    if (col >= p.rows) continue;
    const float w0 = __bfloat162float(p.b_scale[col]);
    const float w1 = __bfloat162float(p.b_scale[col + 1]);
#pragma unroll
    for (int mi = 0; mi < kMI; ++mi) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int row = m0 + warp_m * kWarpM + mi * 16 + g + hh * 8;
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
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_gemm_prepare(void) {
  int device = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaFuncSetAttribute(gemm_w8a8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
  }
  if (err != cudaSuccess) return fail(std::string("ignis_image_gemm_prepare: ") + cudaGetErrorString(err));
  if (device < 0 || device >= kMaxDevices) return fail("ignis_image_gemm_prepare: device id out of range");
  g_prepared[device].store(true);
  return 0;
}

extern "C" const char *ignis_image_gemm_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_gemm_w8a8(const void *weight, uint32_t rows, uint32_t cols, const void *codes,
                                         const float *scales, uint32_t tokens, void *y, uint32_t y_f32, void *stream) {
  const char *op = "ignis_image_gemm_w8a8";
  if (weight == nullptr || codes == nullptr || scales == nullptr || y == nullptr) {
    return fail(std::string(op) + ": NULL pointer");
  }
  if (tokens == 0) return fail(std::string(op) + ": tokens must be at least 1");
  if (rows == 0 || rows % 16 != 0) return fail(std::string(op) + ": rows must be a positive multiple of 16");
  if (cols == 0 || cols % kBK != 0) return fail(std::string(op) + ": cols must be a positive multiple of 64");
  if (!aligned(weight, 16) || !aligned(codes, 16) || !aligned(scales, 4) || !aligned(y, 8)) {
    return fail(std::string(op) + ": weight and codes must be 16-byte aligned, scales 4-byte and y 8-byte aligned");
  }
  const uint64_t m_tiles = (static_cast<uint64_t>(tokens) + kBM - 1) / kBM;
  const uint64_t n_tiles = (static_cast<uint64_t>(rows) + kBN - 1) / kBN;
  if (m_tiles * n_tiles > 0x7FFFFFFFull || tokens > 0x7FFFFFFFu || rows > 0x7FFFFFFFu || cols > 0x7FFFFFFFu) {
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
  p.m_tiles = static_cast<int>(m_tiles);
  p.n_tiles = static_cast<int>(n_tiles);
  p.y_f32 = y_f32 != 0 ? 1 : 0;
  gemm_w8a8_kernel<<<static_cast<unsigned>(m_tiles * n_tiles), kThreads, kSmemBytes, static_cast<cudaStream_t>(stream)>>>(p);
  return check_launch(op);
}
