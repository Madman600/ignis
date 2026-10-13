// ignis kernel leaf: the image model's encoder attention (kernel/include/ignis_image_encoder_attn.h;
// spec image-model/02 acceptance 5, GitHub #319) -- an ADAPTED op (ADR 0051 tier 2).
//
// Upstream: Dao-AILab/flash-attention v2.8.3, commit 060c9188beec3a8b62b33a3bfa6d5d2d44975fab,
// BSD-3-Clause ("Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file. All
// rights reserved."; the files say "Copyright (c) 2024, Tri Dao", utils.h and
// flash_fwd_launch_template.h "Copyright (c) 2023, Tri Dao"; the licence text is in kernel/NOTICE):
//   csrc/flash_attn/src/flash_fwd_kernel.h  compute_attn_1rowblock: one CTA per (query block,
//                                           head), the key blocks walked from the last one down,
//                                           the load order (V of block n during S = Q K^T, K of
//                                           block n - 1 during the softmax and P V) behind two
//                                           barriers per block, single K and V buffers, the
//                                           causal key range min(q0 + BM, tokens);
//   csrc/flash_attn/src/softmax.h           Softmax::softmax_rescale_o, scale_apply_exp2 and
//                                           normalize_softmax_lse: the running max kept in raw
//                                           scores, 2^(s * c - m * c) as one FMA, a row max of
//                                           -inf taken as 0, the per-lane row sum reduced over
//                                           the quad at the end, 1 / sum with 0 or NaN taken as 1;
//   csrc/flash_attn/src/mask.h              apply_mask_local's causal column limit;
//   csrc/flash_attn/src/utils.h             convert_layout_acc_Aregs: the S accumulator, rounded
//                                           to BF16, is P V's A fragment as it is;
//   csrc/flash_attn/src/flash_fwd_launch_template.h
//                                           run_mha_fwd_hdim128's tile for consumer SMs (sm86 /
//                                           sm89): 64 x 64 causal, 4 warps, 48 KiB, two CTAs an SM.
// FlashAttention-2 is written with CuTe, which is CUTLASS 3.x, and nothing of CUTLASS 3.x is taken
// here (spec image-model/02, ADR 0051): none of its code is compiled. Its tensors, tiled copies
// and tiled MMAs are replaced by explicit mma.sync, ldmatrix and cp.async (image_mma.cuh,
// attn_smem.cuh, the latter adapted from FlashInfer through SageAttention, Apache-2.0).
//
// Changes from upstream (ours to change, no parity claim):
//   - one geometry: BF16, 32 query heads over 8 KV heads, D = 128, one sequence; no dropout,
//     ALiBi, softcap, local window, paging, split-KV, LSE output, rotary or c10.
//   - the row sum is the fp32 sum of the BF16 P that multiply V, not of P before its rounding: O
//     is a convex combination of V's rows whatever P's rounding (as the 8-bit attention's).
//   - a row whose max did not move is rescaled by exactly 1 (no ex2(0)).
//   - ex2.approx.ftz written out, every fp32 step an explicit _rn intrinsic (no contraction), 1 /
//     sum __frcp_rn; O stored from registers, BF16 or fp32 (FA2 stages it through shared memory).
//   - Q, K and V tiles zero-filled past the sequence (FA2 clears only V's first block).
//   - causal CTAs launch heaviest first.

#include "attn_smem.cuh"
#include "ignis_image_encoder_attn.h"
#include "image_common.cuh"
#include "image_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

constexpr int kHeadsQ = 32, kHeadsKv = 8, kDim = 128;
constexpr int kWarps = 4, kThreads = kWarps * 32;

// The tile: kBM query rows per CTA (kBM / 4 per warp) and kBN keys per block, the header's key
// block of 64 (the CTest replays it). Shared memory holds Q, K and V, each row of 128 BF16 as two
// 128-byte halves (FA2's kBlockKSmem = 64): half hf of row r at (hf * rows + r) * 128, its 16-byte
// chunks swizzled by swz128.
constexpr int kBM = 64, kBN = 64;
constexpr int kRowsW = kBM / kWarps;   // query rows per warp
constexpr int kTilesM = kRowsW / 16;   // m16 tiles per warp
constexpr int kTilesN = kBN / 8;       // n8 key tiles of S
constexpr int kSmemQ = 0, kSmemK = kBM * 256, kSmemV = kSmemK + kBN * 256, kSmem = kSmemV + kBN * 256;
static_assert(kRowsW % 16 == 0 && kBN % 16 == 0, "whole m16 tiles per warp, whole k16 steps per block");
// The launch raises no shared-memory limit (no prepare call): the tile stays within the default.
static_assert(kSmem <= 48 * 1024, "the tile must fit the default 48 KiB of dynamic shared memory");

// Byte offset of 16-byte chunk `chunk` (0..15) of `row` in a tile of `rows` rows.
__device__ __forceinline__ int tile_at(int rows, int row, int chunk) {
  return (chunk >> 3) * rows * 128 + swz128(row, chunk & 7);
}

struct Args {
  const __nv_bfloat16 *q, *k, *v;
  int tokens, q_stride, kv_stride, causal;
  float c;  // fl32(scale * log2 e)
  void *o;
  int o_stride, o_f32, m_blocks;
};

// Rows [first, first + R) of a [tokens][stride] source (already at its head) into a tile; rows
// past the sequence become zeros and are not read.
template <int R>
__device__ __forceinline__ void load_tile(uint8_t *tile, const __nv_bfloat16 *src, int stride, int first, int tokens,
                                          int tid) {
#pragma unroll
  for (int i = 0; i < R * 16 / kThreads; ++i) {
    const int chunk = tid + i * kThreads, row = chunk >> 4, part = chunk & 15;
    const bool valid = first + row < tokens;
    const __nv_bfloat16 *s = src + static_cast<std::size_t>(valid ? first + row : 0) * stride + part * 8;
    cp_async16_zfill(tile + tile_at(R, row, part), s, valid);
  }
}

// Two P values rounded to BF16 (lo in the low half), added to the row sum as rounded.
__device__ __forceinline__ uint32_t round_p(float lo, float hi, float &sum) {
  const __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  sum = __fadd_rn(__fadd_rn(sum, __low2float(b)), __high2float(b));
  return *reinterpret_cast<const uint32_t *>(&b);
}

__global__ void __launch_bounds__(kThreads, 2) encoder_attn_kernel(Args p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int BM = kBM, BN = kBN, TM = kTilesM, TN = kTilesN;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int h = blockIdx.x, hk = h / (kHeadsQ / kHeadsKv);
  const int mb = p.causal ? p.m_blocks - 1 - static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.y);
  const int q0 = mb * BM, row0 = q0 + warp * kRowsW;
  const int keys = p.causal ? min(q0 + BM, p.tokens) : p.tokens;
  const int n_blocks = (keys + BN - 1) / BN;
  const __nv_bfloat16 *kh = p.k + hk * kDim, *vh = p.v + hk * kDim;

  // Prologue: Q, then the last block's K.
  load_tile<BM>(smem + kSmemQ, p.q + h * kDim, p.q_stride, q0, p.tokens, tid);
  load_tile<BN>(smem + kSmemK, kh, p.kv_stride, (n_blocks - 1) * BN, p.tokens, tid);
  cp_async_commit();

  float acc[TM][kDim / 8][4];
#pragma unroll
  for (int fq = 0; fq < TM; ++fq)
#pragma unroll
    for (int fo = 0; fo < kDim / 8; ++fo)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[fq][fo][e] = 0.0f;
  float m[TM][2], l[TM][2];
#pragma unroll
  for (int fq = 0; fq < TM; ++fq) {
    m[fq][0] = m[fq][1] = neg_inf();
    l[fq][0] = l[fq][1] = 0.0f;
  }

  // ldmatrix lanes (image_mma.cuh's fragment tables): A (Q) rows lane & 15, chunk half lane >> 4;
  // B (K) rows (lane & 7) plus 8 for lanes 16..31, chunk half (lane >> 3) & 1; B (V, transposed)
  // keys lane & 15, chunk half lane >> 4.
  const int a_row = warp * kRowsW + (lane & 15), a_half = lane >> 4;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;
  const int v_row = lane & 15, v_half = lane >> 4;

  for (int nb = n_blocks - 1; nb >= 0; --nb) {
    const int key0 = nb * BN;
    // Block nb's K (and Q) have landed; every warp is done with the previous block's V.
    cp_async_wait<0>();
    __syncthreads();
    load_tile<BN>(smem + kSmemV, vh, p.kv_stride, key0, p.tokens, tid);
    cp_async_commit();

    // S = Q K^T in fp32.
    float s[TM][TN][4];
#pragma unroll
    for (int fq = 0; fq < TM; ++fq)
#pragma unroll
      for (int nt = 0; nt < TN; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) s[fq][nt][e] = 0.0f;
#pragma unroll
    for (int kk = 0; kk < kDim / 16; ++kk) {
      uint32_t qa[TM][4];
#pragma unroll
      for (int fq = 0; fq < TM; ++fq) ldmatrix_x4(qa[fq], smem + kSmemQ + tile_at(BM, a_row + fq * 16, 2 * kk + a_half));
#pragma unroll
      for (int fk = 0; fk < BN / 16; ++fk) {
        uint32_t kb[4];
        ldmatrix_x4(kb, smem + kSmemK + tile_at(BN, fk * 16 + b_row, 2 * kk + b_half));
        const uint32_t b0[2] = {kb[0], kb[1]}, b1[2] = {kb[2], kb[3]};
#pragma unroll
        for (int fq = 0; fq < TM; ++fq) {
          mma_bf16_f32(s[fq][2 * fk], qa[fq], b0);
          mma_bf16_f32(s[fq][2 * fk + 1], qa[fq], b1);
        }
      }
    }

    // The mask, on the blocks that need it: keys past the sequence, keys after the row (causal).
    // One warp-uniform branch; the per-element test is a select.
    if (key0 + BN > p.tokens || (p.causal && key0 + BN - 1 > row0)) {
#pragma unroll
      for (int fq = 0; fq < TM; ++fq)
#pragma unroll
        for (int nt = 0; nt < TN; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const int key = key0 + nt * 8 + 2 * t + (e & 1);
            const int row = row0 + fq * 16 + g + 8 * (e >> 1);
            const bool masked = key >= p.tokens || (p.causal && key > row);
            s[fq][nt][e] = masked ? neg_inf() : s[fq][nt][e];
          }
    }

    // Block nb's V has landed; every warp is done with K, so the next block's K may load.
    cp_async_wait<0>();
    __syncthreads();
    if (nb > 0) load_tile<BN>(smem + kSmemK, kh, p.kv_stride, key0 - BN, p.tokens, tid);
    cp_async_commit();

    // softmax_rescale_o: the running max, the rescale, P in BF16 and its row sum.
    uint32_t pa[TM][TN][2];
#pragma unroll
    for (int fq = 0; fq < TM; ++fq) {
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        float mx = neg_inf();
#pragma unroll
        for (int nt = 0; nt < TN; ++nt) mx = fmaxf(mx, fmaxf(s[fq][nt][2 * rr], s[fq][nt][2 * rr + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        const float m_old = m[fq][rr], m_new = fmaxf(m_old, mx);
        // -inf - m_new = -inf rescales the empty accumulators by 0; -inf to -inf is no move.
        const float r = m_new == m_old ? 1.0f : ex2(__fmul_rn(__fsub_rn(m_old, m_new), p.c));
        m[fq][rr] = m_new;
        l[fq][rr] = __fmul_rn(l[fq][rr], r);
#pragma unroll
        for (int fo = 0; fo < kDim / 8; ++fo) {
          acc[fq][fo][2 * rr] = __fmul_rn(acc[fq][fo][2 * rr], r);
          acc[fq][fo][2 * rr + 1] = __fmul_rn(acc[fq][fo][2 * rr + 1], r);
        }
        const float ms = m_new == neg_inf() ? 0.0f : __fmul_rn(m_new, p.c);
#pragma unroll
        for (int nt = 0; nt < TN; ++nt) {
          pa[fq][nt][rr] = round_p(ex2(__fmaf_rn(s[fq][nt][2 * rr], p.c, -ms)),
                                   ex2(__fmaf_rn(s[fq][nt][2 * rr + 1], p.c, -ms)), l[fq][rr]);
        }
      }
    }

    // O += P V: P's accumulator fragments are the A fragments of keys 16 kk .. 16 kk + 15.
#pragma unroll
    for (int kk = 0; kk < BN / 16; ++kk) {
#pragma unroll
      for (int fd = 0; fd < kDim / 16; ++fd) {
        uint32_t vb[4];
        ldmatrix_x4_trans(vb, smem + kSmemV + tile_at(BN, kk * 16 + v_row, 2 * fd + v_half));
        const uint32_t b0[2] = {vb[0], vb[1]}, b1[2] = {vb[2], vb[3]};
#pragma unroll
        for (int fq = 0; fq < TM; ++fq) {
          const uint32_t a[4] = {pa[fq][2 * kk][0], pa[fq][2 * kk][1], pa[fq][2 * kk + 1][0], pa[fq][2 * kk + 1][1]};
          mma_bf16_f32(acc[fq][2 * fd], a, b0);
          mma_bf16_f32(acc[fq][2 * fd + 1], a, b1);
        }
      }
    }
  }
  cp_async_wait<0>();

  // normalize_softmax_lse: the row sum over the quad, O * (1 / sum).
#pragma unroll
  for (int fq = 0; fq < TM; ++fq) {
#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
      float sum = l[fq][rr];
      sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, 1));
      sum = __fadd_rn(sum, __shfl_xor_sync(0xFFFFFFFFu, sum, 2));
      const float inv = (sum == 0.0f || sum != sum) ? 1.0f : __frcp_rn(sum);
      const int row = row0 + fq * 16 + g + 8 * rr;
      if (row >= p.tokens) continue;
#pragma unroll
      for (int fo = 0; fo < kDim / 8; ++fo) {
        const float y0 = __fmul_rn(acc[fq][fo][2 * rr], inv), y1 = __fmul_rn(acc[fq][fo][2 * rr + 1], inv);
        const std::size_t at = static_cast<std::size_t>(row) * p.o_stride + h * kDim + fo * 8 + 2 * t;
        if (p.o_f32) {
          *reinterpret_cast<float2 *>(static_cast<float *>(p.o) + at) = make_float2(y0, y1);
        } else {
          *reinterpret_cast<__nv_bfloat162 *>(static_cast<__nv_bfloat16 *>(p.o) + at) = __floats2bfloat162_rn(y0, y1);
        }
      }
    }
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" const char *ignis_image_encoder_attn_last_error(void) { return last_error_cstr(); }

extern "C" int32_t ignis_image_encoder_attn(const void *q, const void *k, const void *v, uint32_t tokens,
                                            uint32_t q_row_stride, uint32_t kv_row_stride, uint32_t causal,
                                            float scale, void *o, uint32_t o_row_stride, uint32_t o_f32, void *stream) {
  const std::string op = "ignis_image_encoder_attn";
  if (q == nullptr || k == nullptr || v == nullptr || o == nullptr) return fail(op + ": NULL pointer");
  if (tokens == 0 || tokens > (1u << 20)) return fail(op + ": tokens must be in [1, 2^20]");
  if (q_row_stride < 4096 || q_row_stride % 8 != 0) return fail(op + ": q_row_stride must be >= 4096 and a multiple of 8");
  if (kv_row_stride < 1024 || kv_row_stride % 8 != 0) {
    return fail(op + ": kv_row_stride must be >= 1024 and a multiple of 8");
  }
  if (o_row_stride < 4096 || o_row_stride % 8 != 0) return fail(op + ": o_row_stride must be >= 4096 and a multiple of 8");
  if (q_row_stride > 0x7FFFFFFFu || kv_row_stride > 0x7FFFFFFFu || o_row_stride > 0x7FFFFFFFu) {
    return fail(op + ": row strides must be at most 2^31 - 1");
  }
  if (!aligned(q, 16) || !aligned(k, 16) || !aligned(v, 16) || !aligned(o, 16)) {
    return fail(op + ": q, k, v and o must be 16-byte aligned");
  }
  if (!std::isfinite(scale) || !(scale > 0.0f)) return fail(op + ": scale must be finite and > 0");
  Args a{};
  a.q = static_cast<const __nv_bfloat16 *>(q);
  a.k = static_cast<const __nv_bfloat16 *>(k);
  a.v = static_cast<const __nv_bfloat16 *>(v);
  a.tokens = static_cast<int>(tokens);
  a.q_stride = static_cast<int>(q_row_stride);
  a.kv_stride = static_cast<int>(kv_row_stride);
  a.causal = causal != 0 ? 1 : 0;
  a.c = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634);
  a.o = o;
  a.o_stride = static_cast<int>(o_row_stride);
  a.o_f32 = o_f32 != 0 ? 1 : 0;
  a.m_blocks = static_cast<int>((tokens + kBM - 1) / kBM);
  encoder_attn_kernel<<<dim3(kHeadsQ, a.m_blocks), kThreads, kSmem, static_cast<cudaStream_t>(stream)>>>(a);
  return check_launch(op.c_str());
}
