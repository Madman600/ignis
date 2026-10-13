// ignis kernel leaf: the VAE's mid-block attention (kernel/include/ignis_image_vae.h; spec
// image-model/02 acceptance 6, GitHub #319) -- OURS (ADR 0051 tier 1). One head of 768 (encoder)
// or 1152 (decoder) channels over the latent grid, too wide for FlashAttention's registers, so the
// header's three passes per tile of query rows, on a fixed workspace:
//   S = Q K^T   the convolution's GEMM (vae_conv.cuh) as a 1 x 1 convolution: the tile's q rows
//               (stride 3 * dim) against the k rows as weight rows, fp32 out;
//   softmax     one CTA per row: the row max, P = BF16(2^(S * c - fl(max * c))) (one FMA,
//               ex2.approx), zeros past the tokens, and 1 / D with D the fp32 sum of the BF16 P in
//               a fixed order: each thread adds its columns j = t, t + 256, ... in order, a warp
//               butterfly (xor 16, 8, 4, 2, 1), then thread 0 adds the 8 warps in order;
//   O = P V     the GEMM again, P against V^T (transposed once per call, zero-padded to n8
//               columns), each row times its 1 / D in the epilogue.
// Every output is one fixed sequence of operations, and a row's does not depend on its tile.

#include "attn_smem.cuh"
#include "ignis_image_vae.h"
#include "image_common.cuh"
#include "vae_conv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kSoftmaxThreads = 256;

std::size_t align256(std::size_t n) { return (n + 255) / 256 * 256; }

// The workspace's regions (byte offsets) and its size; n8 = tokens rounded up to 8.
struct AttnLayout {
  std::size_t n8, vt, s, p, inv, total;
  AttnLayout(uint32_t tokens, uint32_t dim, uint32_t row_tile) {
    n8 = (static_cast<std::size_t>(tokens) + 7) / 8 * 8;
    vt = 0;
    s = align256(vt + static_cast<std::size_t>(dim) * n8 * 2);
    p = align256(s + static_cast<std::size_t>(row_tile) * n8 * 4);
    inv = align256(p + static_cast<std::size_t>(row_tile) * n8 * 2);
    total = align256(inv + static_cast<std::size_t>(row_tile) * 4);
  }
};

// vt[d][j] = v[j][d] for j < tokens, 0 for tokens <= j < n8; 32 x 32 tiles through shared memory.
__global__ void transpose_v_kernel(const __nv_bfloat16 *v, int tokens, int dim, int v_stride, int n8,
                                   __nv_bfloat16 *vt) {
  __shared__ __nv_bfloat16 tile[32][33];
  const int j0 = static_cast<int>(blockIdx.x) * 32, d0 = static_cast<int>(blockIdx.y) * 32;
  for (int r = threadIdx.y; r < 32; r += blockDim.y) {
    const int j = j0 + r, d = d0 + static_cast<int>(threadIdx.x);
    tile[r][threadIdx.x] = j < tokens && d < dim ? v[static_cast<std::size_t>(j) * v_stride + d] : __float2bfloat16_rn(0.0f);
  }
  __syncthreads();
  for (int r = threadIdx.y; r < 32; r += blockDim.y) {
    const int d = d0 + r, j = j0 + static_cast<int>(threadIdx.x);
    if (d < dim && j < n8) vt[static_cast<std::size_t>(d) * n8 + j] = tile[threadIdx.x][r];
  }
}

// One row of S per CTA: P (BF16, n8 columns) and 1 / D, as the file header says.
__global__ void __launch_bounds__(kSoftmaxThreads) softmax_rows_kernel(const float *s, int tokens, int n8, float c,
                                                                       __nv_bfloat16 *p, float *inv) {
  __shared__ float red[kSoftmaxThreads / 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const float *row = s + static_cast<std::size_t>(blockIdx.x) * n8;
  __nv_bfloat16 *prow = p + static_cast<std::size_t>(blockIdx.x) * n8;
  float mx = neg_inf();
  for (int j = tid; j < tokens; j += kSoftmaxThreads) mx = fmaxf(mx, row[j]);
  for (int off = 16; off >= 1; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, off));
  if (lane == 0) red[warp] = mx;
  __syncthreads();
  mx = red[0];
  for (int w = 1; w < kSoftmaxThreads / 32; ++w) mx = fmaxf(mx, red[w]);
  __syncthreads();
  const float ms = __fmul_rn(mx, c);
  float sum = 0.0f;
  for (int j = tid; j < n8; j += kSoftmaxThreads) {
    const __nv_bfloat16 b = __float2bfloat16_rn(j < tokens ? ex2(__fmaf_rn(row[j], c, -ms)) : 0.0f);
    prow[j] = b;
    sum = __fadd_rn(sum, __bfloat162float(b));
  }
  for (int off = 16; off >= 1; off >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, off));
  if (lane == 0) red[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    float d = red[0];
    for (int w = 1; w < kSoftmaxThreads / 32; ++w) d = __fadd_rn(d, red[w]);
    inv[blockIdx.x] = __frcp_rn(d);
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" uint64_t ignis_image_vae_attn_workspace_bytes(uint32_t tokens, uint32_t dim, uint32_t row_tile) {
  if (tokens == 0 || tokens > (1u << 20) || dim == 0 || dim % 8 != 0 || dim > 65536 || row_tile == 0) return 0;
  return AttnLayout(tokens, dim, row_tile).total;
}

extern "C" int32_t ignis_image_vae_attn(const void *qkv, uint32_t tokens, uint32_t dim, float scale,
                                        uint32_t row_tile, void *workspace, uint64_t workspace_bytes, void *o,
                                        uint32_t o_f32, void *stream) {
  const std::string op = "ignis_image_vae_attn";
  if (qkv == nullptr || workspace == nullptr || o == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > (1u << 20)) return fail(op + ": tokens must be in [1, 2^20]");
  if (dim == 0 || dim % 8 != 0 || dim > 65536) return fail(op + ": dim must be a multiple of 8 in [8, 65536]");
  if (row_tile == 0) return fail(op + ": row_tile must be at least 1");
  if (!std::isfinite(scale) || !(scale > 0.0f)) return fail(op + ": scale must be finite and > 0");
  if (!aligned(qkv, 16) || !aligned(o, 16) || !aligned(workspace, 256)) {
    return fail(op + ": qkv and o must be 16-byte aligned, the workspace 256-byte aligned");
  }
  const uint32_t tile = std::min(row_tile, tokens);
  const AttnLayout l(tokens, dim, tile);
  if (workspace_bytes < l.total) {
    return fail(op + ": the workspace holds " + std::to_string(workspace_bytes) + " bytes, the call needs " +
                std::to_string(l.total));
  }
  if (static_cast<uint64_t>(tile) * l.n8 > 0x7FFFFFFFull) return fail(op + ": row_tile * tokens must be below 2^31");
  const cudaStream_t st = static_cast<cudaStream_t>(stream);
  uint8_t *ws = static_cast<uint8_t *>(workspace);
  const __nv_bfloat16 *q = static_cast<const __nv_bfloat16 *>(qkv);
  __nv_bfloat16 *vt = reinterpret_cast<__nv_bfloat16 *>(ws + l.vt);
  float *s = reinterpret_cast<float *>(ws + l.s);
  __nv_bfloat16 *p = reinterpret_cast<__nv_bfloat16 *>(ws + l.p);
  float *inv = reinterpret_cast<float *>(ws + l.inv);
  const int n = static_cast<int>(tokens), d = static_cast<int>(dim), n8 = static_cast<int>(l.n8);
  const float c = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634);

  transpose_v_kernel<<<dim3((n8 + 31) / 32, (d + 31) / 32), dim3(32, 8), 0, st>>>(q + 2 * d, n, d, 3 * d, n8, vt);
  if (check_launch(op.c_str()) != 0) return -1;
  for (int r0 = 0; r0 < n; r0 += static_cast<int>(tile)) {
    const int rows = std::min(static_cast<int>(tile), n - r0);
    // S = Q K^T: the tile's q rows over every k row as a 1 x 1 convolution, fp32 [rows][n8].
    VaeConvArgs qk{};
    qk.x = q + static_cast<std::size_t>(r0) * 3 * d;
    qk.height = rows;
    qk.width = 1;
    qk.in_ch = d;
    qk.in_stride = 3 * d;
    qk.ksize = qk.stride = 1;
    qk.w = q + d;
    qk.w_stride = 3 * d;
    qk.y = s;
    qk.out_ch = n;
    qk.out_stride = n8;
    qk.out_w = 1;
    qk.y_rows = rows;
    qk.y_f32 = 1;
    if (vae_conv_launch(qk, stream, op.c_str()) != 0) return -1;
    softmax_rows_kernel<<<rows, kSoftmaxThreads, 0, st>>>(s, n, n8, c, p, inv);
    if (check_launch(op.c_str()) != 0) return -1;
    // O = (P V) * (1 / D): P [rows][n8] against V^T [dim][n8].
    VaeConvArgs pv{};
    pv.x = p;
    pv.height = rows;
    pv.width = 1;
    pv.in_ch = n8;
    pv.in_stride = n8;
    pv.ksize = pv.stride = 1;
    pv.w = vt;
    pv.w_stride = n8;
    pv.row_scale = inv;
    pv.y = o_f32 != 0 ? static_cast<void *>(static_cast<float *>(o) + static_cast<std::size_t>(r0) * d)
                       : static_cast<void *>(static_cast<__nv_bfloat16 *>(o) + static_cast<std::size_t>(r0) * d);
    pv.out_ch = d;
    pv.out_stride = d;
    pv.out_w = 1;
    pv.y_rows = rows;
    pv.y_f32 = o_f32 != 0 ? 1 : 0;
    if (vae_conv_launch(pv, stream, op.c_str()) != 0) return -1;
  }
  return 0;
}
