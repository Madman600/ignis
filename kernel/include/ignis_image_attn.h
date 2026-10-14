/* ignis kernel leaf: the image model's 8-bit DiT attention (spec image-model/02 acceptance 4,
 * GitHub #319) -- an ADAPTED op (ADR 0051 tier 2): the attention kernel adapts SageAttention2++'s
 * sm89/sm120 path (thu-ml/SageAttention d1a57a5, csrc/qattn, Apache-2.0; its mma, shared-memory
 * and cp.async helpers are themselves adapted from FlashInfer v0.1.5). Provenance per file in
 * kernel/src/image/attn8*.cu and kernel/NOTICE. The quantizers are our own.
 *
 * Geometry: 32 heads of 128 (the DiT's), non-causal or causal aligned bottom-right.
 *
 * The scheme (SageAttention2): Q and K in INT8, K smoothed by subtracting a mean over keys; V in
 * E4M3 with one scale per channel; QK^T on INT8 tensor cores (exact int32), the softmax in fp32
 * with an online max, P rounded to E4M3 and PV on block-scaled FP8 tensor cores with fp32
 * accumulation (mma kind::mxf8f6f4, unit scales: spec image-model/02 "As built" for why).
 *
 * The condition prefix and the target. A K/V *run* is one call's keys and values, quantized and
 * stored in the byte layout below. The image leaf (spec image-model/04) stores the condition
 * prefix's run once (step 1) and reads it in every denoising step; each step stores the target's
 * run with the PREFIX's K mean (`mean_from`), so one constant is subtracted from every key a query
 * sees (exact: the softmax ignores a constant added to all of a query's scores). An attention call
 * reads up to two runs, A then B; when it crosses from A to B it rescales its fp32 accumulator
 * per channel by vscale_A / vscale_B, so the two runs' V scales may differ (exact up to rounding).
 *
 * The calls the DiT makes (diffusers' QwenImage21AttnProcessor):
 *   prefix image block [start, end)  queries = the block, run A = the prefix truncated to `end`
 *                                    keys, non-causal;
 *   prefix text segment [start, end) the same with `causal`: query i of the segment sees keys
 *                                    0 .. start + i (bottom-right aligned, ones(len, start) ||
 *                                    tril(len));
 *   target                           queries = the target, run A = the whole prefix, run B = the
 *                                    target's own run, non-causal.
 *
 * Byte layouts (H = 32 heads, D = 128; every region starts 16-byte aligned):
 *   q8 buffer for n queries, n_pad = round_up(n, 128):
 *     codes   INT8  [H][n_pad][D]       q rows; rows >= n are zero
 *     scales  fp32  [H][n_pad / 32]     one per 32-row block: amax / 127
 *   K/V run for n tokens, n_pad = round_up(n, 64):
 *     k       INT8  [H][n_pad][D]       K - mean; rows >= n are zero
 *     v       E4M3  [H][D][n_pad]       V transposed: channel c's keys contiguous, the keys of
 *                                       each group of 16 stored in the order
 *                                       0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15 (SageAttention's,
 *                                       so P's accumulator fragment is PV's A fragment as it
 *                                       is); keys >= n are zero
 *     kscale  fp32  [H][n_pad / 64]     one per 64-key block: amax |K - mean| / 127
 *     vscale  fp32  [H][D]              one per channel over the run's n keys: amax |V| / 448
 *     mean    fp32  [H][D]              the K mean the run was smoothed with
 *   at byte offsets 0, H*n_pad*D, 2*H*n_pad*D, 2*H*n_pad*D + H*(n_pad/64)*4, then + H*D*4.
 *
 * Quantization (each step one fp32 operation, rounded to nearest even):
 *   Q    per head and 32-row block: amax = max |q|; scale = amax / 127; code = sat_s8(q * (127 /
 *        amax)).
 *   K    k' = k - mean[h][c]; per head and 64-key block of the run: amax = max |k'| over its
 *        valid keys; scale = amax / 127; code = sat_s8(k' * (127 / amax)). The run's own mean is
 *        the fp32 mean over its n keys (a fixed-order sum, deterministic).
 *   V    per head and channel: amax = max |v| over the run's n keys; scale = amax / 448;
 *        code = E4M3(v / scale), saturated to +-448.
 *   An amax below 127 * 2^-126 (INT8) or 448 * 2^-126 (E4M3) -- the smallest whose scale is a
 *   normal fp32, an all-zero block among them -- gets scale 1, and so zero codes (Q and K: the
 *   multiplier is 0; V: E4M3(v / 1) of so small a v is +-0).
 *
 * The attention: for each query, s_j = int32(q8 . k8_j) * ((qscale * kscale_j) * c) with
 * c = fl32(scale * log2(e)); an online max m over 64-key blocks (offset by -8.807 so P <= 448),
 * P_j = E4M3(2^(s_j - m)), the denominator the fp32 sum of the same E4M3 P_j (a tensor-core row
 * sum), O = (sum_j P_j v8_j * vscale) / sum_j P_j.
 *
 * Every call enqueues on `stream` (NULL: the legacy stream), allocates nothing, is
 * graph-capturable and deterministic (no atomics; a fixed order per output). Return 0, or -1 with
 * the reason in ignis_image_attn_last_error().
 */
#ifndef IGNIS_IMAGE_ATTN_H
#define IGNIS_IMAGE_ATTN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes of a q8 buffer for `tokens` queries. */
uint64_t ignis_image_attn_q8_bytes(uint32_t tokens);

/* Bytes of a K/V run of `tokens` keys. */
uint64_t ignis_image_attn_kv8_bytes(uint32_t tokens);

/* Quantizes BF16 queries into a q8 buffer. Token t's head h is at q + t * row_stride + h * 128
 * (elements); row_stride >= 4096 and a multiple of 8, `q` and `q8` 16-byte aligned. */
int32_t ignis_image_attn_quantize_q(const void *q, uint32_t tokens, uint32_t row_stride, void *q8, void *stream);

/* Quantizes BF16 keys and values (same addressing as the queries, one row_stride for both) into
 * the run `kv8` of `tokens` keys. `mean_from` NULL: the run is smoothed with its own K mean;
 * else with the mean stored in the run `mean_from` of `mean_from_tokens` keys (the prefix's, for
 * the target), which is copied into this run. mean_from_tokens must be that run's exact token
 * count: it locates the mean in the run's layout, and a wrong count silently reads another region
 * (spec image-model/04's prefix handle carries the count). */
int32_t ignis_image_attn_store_kv(const void *k, const void *v, uint32_t tokens, uint32_t row_stride,
                                  const void *mean_from, uint32_t mean_from_tokens, void *kv8, void *stream);

/* O = attention of the q8 buffer's `q_tokens` queries over the first `a_keys` keys of run A
 * (`a_tokens` keys, 1 <= a_keys <= a_tokens) followed by all `b_tokens` keys of run B (NULL and
 * 0 for none). Run B must have been stored with mean_from = run A: the kernel never reads the
 * means, so a run B with another mean gives silently wrong scores. `causal`: query i sees keys
 * 0 .. i + (keys - q_tokens), keys = a_keys + b_tokens >= q_tokens. `scale` (finite, > 0)
 * multiplies q . k (1 / sqrt(128) for the DiT).
 * O: token t's head h at o + t * o_row_stride + h * 128, BF16, or fp32 when `o_f32`;
 * o_row_stride >= 4096 and a multiple of 8, `o` 16-byte aligned. */
int32_t ignis_image_attn(const void *q8, uint32_t q_tokens, const void *kv_a, uint32_t a_tokens, uint32_t a_keys,
                         const void *kv_b, uint32_t b_tokens, uint32_t causal, float scale, void *o,
                         uint32_t o_row_stride, uint32_t o_f32, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_attn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_ATTN_H */
