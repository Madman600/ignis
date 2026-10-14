// ignis kernel leaf: the image model's 8-bit DiT attention (kernel/include/ignis_image_attn.h;
// spec image-model/02 acceptance 4, GitHub #319) -- an ADAPTED op (ADR 0051 tier 2).
//
// Upstream: thu-ml/SageAttention, commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5, Apache-2.0
// (Copyright (c) 2024 by SageAttention team):
//   csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh  qk_int_sv_f8_attn_kernel: the CTA shape (128 queries,
//                                          4 warps of 32, 64-key blocks), Q, K and V staged in
//                                          shared memory by cp.async, the per-warp Q and
//                                          per-block K dequantization folded into the exponent's
//                                          FMA, the per-channel V scale at the end;
//   csrc/qattn/attn_utils.cuh              compute_int_qk, update_mdo (the online max offset by
//                                          8.807 so P fits E4M3), RS_32_to_8 (P's accumulator
//                                          fragment packed as PV's A fragment, which is why V is
//                                          stored permuted), apply_causal_mask /
//                                          apply_out_of_bound_mask, rowsum_f8f8f32 (the row sum
//                                          as an MMA against ones), normalize_d;
//   csrc/permuted_smem.cuh, csrc/cp_async.cuh, csrc/mma.cuh
//                                          the 128-byte / 64-byte shared-memory swizzles, cp.async
//                                          and ldmatrix: "Adapted from FlashInfer"
//                                          (flashinfer-ai/flashinfer v0.1.5 include/flashinfer/
//                                          {permuted_smem,cp_async,mma}.cuh, Copyright (c) 2023 by
//                                          FlashInfer team, Apache-2.0); now in attn_smem.cuh,
//                                          shared with the encoder attention.
//
// Changes from upstream (ours to change, no parity claim):
//   - torch removed; D = 128, 32 heads, one layout, no templates over granularity or dtype.
//   - PV accumulates in fp32 on block-scaled FP8 tensor cores (kind::mxf8f6f4, unit scales),
//     which on the 5090 run at the FP16-accumulate rate: no FP16 accumulator, so V's scale is
//     amax / 448 rather than SA2++'s amax / 2.25 (spec image-model/02 "As built").
//   - The denominator is the tensor-core row sum of the same E4M3 P that multiply V, not the
//     fp32 sum of P before rounding: O is a convex combination of the V rows whatever P's
//     rounding, so V's per-channel offset does not amplify P's rounding error. The offset still
//     costs V's own E4M3 resolution: V is not smoothed.
//   - Two K/V runs per call (the stored condition prefix, then the target), with the fp32
//     accumulator rescaled per channel by vscale_A / vscale_B when the key loop crosses them.
//   - Causal aligned bottom-right (query i sees keys 0 .. i + keys - queries), masked on every
//     block the diagonal crosses, at any offset; masked and plain blocks take separate score
//     conversions behind one warp-uniform branch.
//   - K and V in two shared-memory stages (48 KiB a CTA, still two CTAs an SM): the next block
//     loads during this one's QK, softmax and PV, behind one barrier per block where SA2's single
//     K and V buffers need three.
//   - The int32 scores reach fp32 by the 1.5 * 2^23 bias (exact, |S| < 2^22) instead of the
//     quarter-rate I2F, and a block that raises no row's max skips the rescale by 1.
//   - Rows the online max has not yet seen stay -inf with an exact 0 / 1 rescale (no -5e6
//     sentinel); the 1 / sum is __frcp_rn; output BF16 or fp32 stored from registers.

#include "attn8_layout.cuh"
#include "attn_smem.cuh"
#include "ignis_image_attn.h"
#include "image_common.cuh"
#include "image_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <type_traits>

namespace ignis_image {
namespace {

constexpr int kCtaQ = 128, kCtaK = 64, kWarps = 4, kThreads = kWarps * 32;
constexpr int kTilesQ = 2;   // m16 tiles per warp (32 query rows)
constexpr int kTilesK = 4;   // n16 key tiles per block (64 keys)
constexpr int kTilesV = 8;   // n16 channel tiles (128 channels)
constexpr int kInnerQK = 4;  // k32 steps over D = 128 INT8
constexpr float kExpOffset = 8.807f;  // ~log2 448, SageAttention's: P's largest value fills E4M3 (derived, not tuned)
// Shared memory: Q, then two K/V stages (the next block's K and V load while this one computes).
constexpr int kSmemQ = 0, kSmemStage0 = kCtaQ * kAttnDim, kStageBytes = 2 * kCtaK * kAttnDim;
constexpr int kSmemK = 0, kSmemV = kCtaK * kAttnDim;  // within a stage
constexpr int kSmemBytes = kSmemStage0 + 2 * kStageBytes;  // 16 + 2 x (8 + 8) KiB = 48 KiB
// The launch sets no cudaFuncAttributeMaxDynamicSharedMemorySize: more than 48 KiB would need a
// prepare call, as the GEMM has.
static_assert(kSmemBytes <= 48 * 1024, "attn8 runs without a prepare call: dynamic shared memory <= 48 KiB");
constexpr uint32_t kOnesE4m3 = 0x38383838u;  // four E4M3 1.0
// Q and K tiles use attn_smem.cuh's 128-byte swizzle, V transposed its 64-byte one.

__device__ __forceinline__ uint32_t e4m3x4(float a, float b, float c, float d) {  // a in the low byte
  const uint32_t lo = static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3));
  const uint32_t hi = static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(make_float2(c, d), __NV_SATFINITE, __NV_E4M3));
  return lo | (hi << 16);
}

struct Span {
  const int8_t *k;         // [H][n_pad][D]
  const uint8_t *v;        // [H][D][n_pad]
  const float *kscale;     // [H][n_pad / 64]
  const float *vscale;     // [H][D]
  int keys, n_pad, first;  // keys read, the run's padded length, the first key's index in the call
};

struct AttnArgs {
  const int8_t *q;
  const float *qscale;
  int nq, nq_pad;
  Span span[2];
  int spans;
  int causal_offset;  // keys - nq when causal, else -1
  float c_sm;         // fl32(scale * log2 e)
  void *o;
  int o_stride, o_f32;
};

__global__ void __launch_bounds__(kThreads, 2) attn8_kernel(AttnArgs p) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * kCtaQ;
  const bool causal = p.causal_offset >= 0;

  // Blocks per run; a causal CTA stops at the last key its last query sees.
  int nb[2] = {0, 0};
  const int q_last = min(q0 + kCtaQ, p.nq) - 1;
  for (int s = 0; s < p.spans; ++s) {
    int keys = p.span[s].keys;
    if (causal) keys = max(0, min(keys, q_last + p.causal_offset + 1 - p.span[s].first));
    nb[s] = (keys + kCtaK - 1) / kCtaK;
  }
  const int blocks = nb[0] + nb[1];

  auto block_span = [&](int it) { return it < nb[0] ? 0 : 1; };
  // Block it's K (64 rows of 128 bytes) and V^T (128 rows of 64 bytes) into stage it % 2.
  auto load_kv = [&](int it) {
    const int s = block_span(it), lb = s == 0 ? it : it - nb[0];
    const Span &sp = p.span[s];
    uint8_t *stage = smem + kSmemStage0 + (it & 1) * kStageBytes;
    const int8_t *ksrc = sp.k + (static_cast<std::size_t>(h) * sp.n_pad + lb * kCtaK) * kAttnDim;
#pragma unroll
    for (int i = 0; i < kCtaK * 8 / kThreads; ++i) {
      const int chunk = tid + i * kThreads, row = chunk >> 3, part = chunk & 7;
      cp_async16(stage + kSmemK + swz128(row, part), ksrc + row * kAttnDim + part * 16);
    }
    const uint8_t *vsrc = sp.v + static_cast<std::size_t>(h) * kAttnDim * sp.n_pad + lb * kCtaK;
#pragma unroll
    for (int i = 0; i < kAttnDim * 4 / kThreads; ++i) {
      const int chunk = tid + i * kThreads, row = chunk >> 2, part = chunk & 3;
      cp_async16(stage + kSmemV + swz64(row, part), vsrc + static_cast<std::size_t>(row) * sp.n_pad + part * 16);
    }
  };

  // Prologue: Q, then block 0's K and V.
  {
    const int8_t *src = p.q + (static_cast<std::size_t>(h) * p.nq_pad + q0) * kAttnDim;
#pragma unroll
    for (int i = 0; i < kCtaQ * 8 / kThreads; ++i) {
      const int chunk = tid + i * kThreads, row = chunk >> 3, part = chunk & 7;
      cp_async16(smem + kSmemQ + swz128(row, part), src + row * kAttnDim + part * 16);
    }
    cp_async_commit();
  }
  load_kv(0);
  cp_async_commit();

  const float qscale = p.qscale[static_cast<std::size_t>(h) * (p.nq_pad / kAttnQScaleRows) + q0 / kAttnQScaleRows + warp];

  float ro[kTilesQ][kTilesV][2][4];
#pragma unroll
  for (int fq = 0; fq < kTilesQ; ++fq)
#pragma unroll
    for (int fv = 0; fv < kTilesV; ++fv)
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e) ro[fq][fv][nt][e] = 0.0f;
  float m[kTilesQ][2], dsum[kTilesQ][4];
#pragma unroll
  for (int fq = 0; fq < kTilesQ; ++fq) {
    m[fq][0] = m[fq][1] = neg_inf();
    dsum[fq][0] = dsum[fq][1] = dsum[fq][2] = dsum[fq][3] = 0.0f;
  }

  // ldmatrix lanes: A (Q) rows lane & 15, chunk half lane >> 4; B (K, V^T) rows (lane & 7) plus 8
  // for lanes 16..31, chunk half (lane >> 3) & 1 (image_mma.cuh's fragment table).
  const int a_row = warp * 32 + (lane & 15), a_half = lane >> 4;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_half = (lane >> 3) & 1;

  for (int it = 0; it < blocks; ++it) {
    const int s = block_span(it), lb = s == 0 ? it : it - nb[0];
    const Span &sp = p.span[s];
    // One barrier per block: block it's K and V (and Q) have landed and are visible, and every warp
    // is done with block it - 1, whose stage the next block's loads now overwrite.
    cp_async_wait<0>();
    __syncthreads();
    if (it + 1 < blocks) load_kv(it + 1);
    cp_async_commit();
    const uint8_t *stage = smem + kSmemStage0 + (it & 1) * kStageBytes;

    // S = Q K^T, exact in int32.
    int32_t rs[kTilesQ][kTilesK][2][4];
#pragma unroll
    for (int fq = 0; fq < kTilesQ; ++fq)
#pragma unroll
      for (int fk = 0; fk < kTilesK; ++fk)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 4; ++e) rs[fq][fk][nt][e] = 0;
#pragma unroll
    for (int kk = 0; kk < kInnerQK; ++kk) {
      uint32_t qa[kTilesQ][4];
#pragma unroll
      for (int fq = 0; fq < kTilesQ; ++fq) ldmatrix_x4(qa[fq], smem + kSmemQ + swz128(a_row + fq * 16, 2 * kk + a_half));
#pragma unroll
      for (int fk = 0; fk < kTilesK; ++fk) {
        uint32_t kb[4];
        ldmatrix_x4(kb, stage + kSmemK + swz128(fk * 16 + b_row, 2 * kk + b_half));
        const uint32_t b0[2] = {kb[0], kb[1]}, b1[2] = {kb[2], kb[3]};
#pragma unroll
        for (int fq = 0; fq < kTilesQ; ++fq) {
          mma_s8_s32(rs[fq][fk][0], qa[fq], b0);
          mma_s8_s32(rs[fq][fk][1], qa[fq], b1);
        }
      }
    }

    // The online softmax over this block (update_mdo, with exact -inf handling).
    const float c = __fmul_rn(__fmul_rn(qscale, sp.kscale[static_cast<std::size_t>(h) * (sp.n_pad / kCtaK) + lb]), p.c_sm);
    const int key0 = lb * kCtaK;  // within the run
    const bool partial = key0 + kCtaK > sp.keys;
    const bool diag = causal && sp.first + key0 + kCtaK - 1 > q0 + warp * 32 + p.causal_offset;
    float sf[kTilesQ][kTilesK][2][4];
    // int32 -> fp32 exactly, at full rate: |S| <= 128 * 128^2 < 2^22, so 1.5 * 2^23 + S has S as
    // its low bits (I2F is quarter rate). One warp-uniform branch picks the masked or the plain
    // conversion, so the plain path carries no per-element branch.
    auto convert = [&](auto masked_path) {
#pragma unroll
      for (int fq = 0; fq < kTilesQ; ++fq)
#pragma unroll
        for (int fk = 0; fk < kTilesK; ++fk)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              float v = __fsub_rn(__int_as_float(0x4B400000 + rs[fq][fk][nt][e]), 12582912.0f);
              if constexpr (decltype(masked_path)::value) {
                const int key = key0 + fk * 16 + nt * 8 + 2 * t + (e & 1);
                const int row = q0 + warp * 32 + fq * 16 + g + 8 * (e >> 1);
                const bool masked = key >= sp.keys || (causal && sp.first + key > row + p.causal_offset);
                v = masked ? neg_inf() : v;
              }
              sf[fq][fk][nt][e] = v;
            }
    };
    if (partial || diag) {
      convert(std::true_type{});
    } else {
      convert(std::false_type{});
    }
    uint32_t pa[kTilesQ][2][4];  // P as PV's A fragments, per 32-key half
    float o[kTilesQ][2];
    bool rescale = false;
#pragma unroll
    for (int fq = 0; fq < kTilesQ; ++fq) {
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        float mx = neg_inf();
#pragma unroll
        for (int fk = 0; fk < kTilesK; ++fk)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt) mx = fmaxf(mx, fmaxf(sf[fq][fk][nt][2 * rr], sf[fq][fk][nt][2 * rr + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 2));
        const float m_new = fmaxf(m[fq][rr], fmaf(mx, c, -kExpOffset));
        o[fq][rr] = m_new == m[fq][rr] ? 1.0f : ex2(m[fq][rr] - m_new);
        rescale = rescale || m_new != m[fq][rr];
        m[fq][rr] = m_new;
      }
    }
    // Once the running max has settled most blocks raise no row's max: skip the multiplications
    // by 1 when no row of the warp moved.
    if (__any_sync(0xFFFFFFFFu, rescale)) {
#pragma unroll
      for (int fq = 0; fq < kTilesQ; ++fq) {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          dsum[fq][2 * rr] *= o[fq][rr];
          dsum[fq][2 * rr + 1] *= o[fq][rr];
#pragma unroll
          for (int fv = 0; fv < kTilesV; ++fv)
#pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
              ro[fq][fv][nt][2 * rr] *= o[fq][rr];
              ro[fq][fv][nt][2 * rr + 1] *= o[fq][rr];
            }
        }
      }
    }
#pragma unroll
    for (int fq = 0; fq < kTilesQ; ++fq) {
#pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
#pragma unroll
        for (int fk = 0; fk < kTilesK; ++fk)
#pragma unroll
          for (int nt = 0; nt < 2; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e)
              sf[fq][fk][nt][2 * rr + e] = ex2(fmaf(sf[fq][fk][nt][2 * rr + e], c, -m[fq][rr]));
      }
      // RS_32_to_8: half hh's a[0] = row g, keys {2t, 2t+1, 8+2t, 9+2t} of tile 2hh; a[1] row g+8;
      // a[2], a[3] the same of tile 2hh+1 -- the V layout stores those keys at k 4t..4t+3.
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
#pragma unroll
          for (int ft = 0; ft < 2; ++ft) {
            const int fk = 2 * hh + ft;
            pa[fq][hh][2 * ft + rr] = e4m3x4(sf[fq][fk][0][2 * rr], sf[fq][fk][0][2 * rr + 1], sf[fq][fk][1][2 * rr],
                                             sf[fq][fk][1][2 * rr + 1]);
          }
        }
        // The row sum of the same E4M3 P, on the tensor cores.
        const uint32_t ones[2] = {kOnesE4m3, kOnesE4m3};
        mma_e4m3_mx_unit_f32(dsum[fq], pa[fq][hh], ones);
      }
    }

    // Crossing from run A to run B: the accumulator moves from A's V scales to B's.
    if (s == 1 && lb == 0) {
#pragma unroll
      for (int fv = 0; fv < kTilesV; ++fv)
#pragma unroll
        for (int nt = 0; nt < 2; ++nt)
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            const int ch = h * kAttnDim + fv * 16 + nt * 8 + 2 * t + e;
            const float ratio = __fdiv_rn(p.span[0].vscale[ch], p.span[1].vscale[ch]);
#pragma unroll
            for (int fq = 0; fq < kTilesQ; ++fq) {
              ro[fq][fv][nt][e] = __fmul_rn(ro[fq][fv][nt][e], ratio);
              ro[fq][fv][nt][2 + e] = __fmul_rn(ro[fq][fv][nt][2 + e], ratio);
            }
          }
    }

    // O += P V on block-scaled FP8 tensor cores, fp32 accumulate.
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
#pragma unroll
      for (int fv = 0; fv < kTilesV; ++fv) {
        uint32_t vb[4];
        ldmatrix_x4(vb, stage + kSmemV + swz64(fv * 16 + b_row, 2 * hh + b_half));
        const uint32_t b0[2] = {vb[0], vb[1]}, b1[2] = {vb[2], vb[3]};
#pragma unroll
        for (int fq = 0; fq < kTilesQ; ++fq) {
          mma_e4m3_mx_unit_f32(ro[fq][fv][0], pa[fq][hh], b0);
          mma_e4m3_mx_unit_f32(ro[fq][fv][1], pa[fq][hh], b1);
        }
      }
    }
  }
  cp_async_wait<0>();

  // O = (acc * vscale) * (1 / sum P), in the scale of the last run the loop read.
  const float *vscale = p.span[nb[1] > 0 ? 1 : 0].vscale + h * kAttnDim;
  float inv[kTilesQ][2];
#pragma unroll
  for (int fq = 0; fq < kTilesQ; ++fq) {
    inv[fq][0] = __frcp_rn(dsum[fq][0]);
    inv[fq][1] = __frcp_rn(dsum[fq][2]);
  }
#pragma unroll
  for (int fv = 0; fv < kTilesV; ++fv) {
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      const int ch = fv * 16 + nt * 8 + 2 * t;
      const float s0 = vscale[ch], s1 = vscale[ch + 1];
#pragma unroll
      for (int fq = 0; fq < kTilesQ; ++fq) {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int row = q0 + warp * 32 + fq * 16 + g + 8 * rr;
          if (row >= p.nq) continue;
          const float y0 = __fmul_rn(__fmul_rn(ro[fq][fv][nt][2 * rr], s0), inv[fq][rr]);
          const float y1 = __fmul_rn(__fmul_rn(ro[fq][fv][nt][2 * rr + 1], s1), inv[fq][rr]);
          const std::size_t at = static_cast<std::size_t>(row) * p.o_stride + h * kAttnDim + ch;
          if (p.o_f32) {
            *reinterpret_cast<float2 *>(static_cast<float *>(p.o) + at) = make_float2(y0, y1);
          } else {
            *reinterpret_cast<__nv_bfloat162 *>(static_cast<__nv_bfloat16 *>(p.o) + at) = __floats2bfloat162_rn(y0, y1);
          }
        }
      }
    }
  }
}

Span make_span(const void *run, uint32_t tokens, uint32_t keys, int first) {
  const Kv8Layout l = kv8_layout(tokens);
  const uint8_t *b = static_cast<const uint8_t *>(run);
  Span s;
  s.k = reinterpret_cast<const int8_t *>(b);
  s.v = b + l.v;
  s.kscale = reinterpret_cast<const float *>(b + l.kscale);
  s.vscale = reinterpret_cast<const float *>(b + l.vscale);
  s.keys = static_cast<int>(keys);
  s.n_pad = static_cast<int>(l.n_pad);
  s.first = first;
  return s;
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_attn(const void *q8, uint32_t q_tokens, const void *kv_a, uint32_t a_tokens,
                                    uint32_t a_keys, const void *kv_b, uint32_t b_tokens, uint32_t causal, float scale,
                                    void *o, uint32_t o_row_stride, uint32_t o_f32, void *stream) {
  const std::string op = "ignis_image_attn";
  if (q8 == nullptr || kv_a == nullptr || o == nullptr) return fail(op + ": NULL pointer");
  if (q_tokens == 0 || q_tokens > (1u << 30)) return fail(op + ": q_tokens must be in [1, 2^30]");
  if (a_tokens > (1u << 30) || b_tokens > (1u << 30)) return fail(op + ": runs are at most 2^30 tokens");
  if (a_keys == 0 || a_keys > a_tokens) return fail(op + ": a_keys must be in [1, a_tokens]");
  if ((kv_b == nullptr) != (b_tokens == 0)) return fail(op + ": run B is a pointer and a token count, or neither");
  const uint64_t keys = static_cast<uint64_t>(a_keys) + b_tokens;
  if (causal != 0 && keys < q_tokens) return fail(op + ": causal needs at least as many keys as queries");
  if (o_row_stride < static_cast<uint32_t>(kAttnHD) || o_row_stride % 8 != 0) {
    return fail(op + ": o_row_stride must be >= 4096 and a multiple of 8");
  }
  if (!aligned(q8, 16) || !aligned(kv_a, 16) || (kv_b != nullptr && !aligned(kv_b, 16)) || !aligned(o, 16)) {
    return fail(op + ": q8, the runs and o must be 16-byte aligned");
  }
  if (!std::isfinite(scale) || !(scale > 0.0f)) return fail(op + ": scale must be finite and > 0");
  const Q8Layout ql = q8_layout(q_tokens);
  AttnArgs p{};
  p.q = static_cast<const int8_t *>(q8);
  p.qscale = reinterpret_cast<const float *>(static_cast<const uint8_t *>(q8) + ql.scales);
  p.nq = static_cast<int>(q_tokens);
  p.nq_pad = static_cast<int>(ql.n_pad);
  p.span[0] = make_span(kv_a, a_tokens, a_keys, 0);
  p.spans = 1;
  if (kv_b != nullptr) {
    p.span[1] = make_span(kv_b, b_tokens, b_tokens, static_cast<int>(a_keys));
    p.spans = 2;
  }
  p.causal_offset = causal != 0 ? static_cast<int>(keys - q_tokens) : -1;
  p.c_sm = static_cast<float>(static_cast<double>(scale) * 1.4426950408889634);
  p.o = o;
  p.o_stride = static_cast<int>(o_row_stride);
  p.o_f32 = o_f32 != 0 ? 1 : 0;
  attn8_kernel<<<dim3(ql.n_pad / kCtaQ, kAttnHeads), kThreads, kSmemBytes, static_cast<cudaStream_t>(stream)>>>(p);
  return check_launch(op.c_str());
}
