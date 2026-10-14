/* ignis kernel leaf: the image model's encoder attention (spec image-model/02 acceptance 5, GitHub
 * #319) -- an ADAPTED op (ADR 0051 tier 2): FlashAttention-2's forward (Dao-AILab/flash-attention
 * v2.8.3, BSD-3-Clause), its structure re-expressed without CuTe. Provenance in
 * kernel/src/image/encoder_attn.cu and kernel/NOTICE.
 *
 * Geometry: the text encoder's (spec image-model/03, Qwen3-VL-8B's text model): BF16, 32 query
 * heads over 8 KV heads (GQA: query head h reads KV head h / 4), head size 128, ONE sequence of
 * `tokens` (the encoder runs one prompt per image job: no batch, no CFG, the vision tokens inside
 * it). Causal (query i sees keys 0 .. i), or non-causal (every key).
 *
 * Addressing: token t's query head h at q + t * q_row_stride + h * 128 (elements), its KV head g
 * at k + t * kv_row_stride + g * 128 and v + t * kv_row_stride + g * 128. The fused QKV row
 * [tokens][6144], after the QK norm and mrope applied to it in place, is read as it is: q = qkv,
 * k = qkv + 4096, v = qkv + 5120, both strides 6144. O: token t's head h at o + t * o_row_stride +
 * h * 128, BF16, or fp32 when `o_f32`.
 *
 * The numerics (what the CTest replays): keys are visited in blocks of 64, from the last block a
 * query sees down to block 0. Per block, S = q . k in fp32 on BF16 tensor cores; the running max
 * m of the raw scores; when it rises, the accumulator and the row sum are rescaled by
 * 2^((m_old - m_new) * c), and by exactly 1 when it does not; P = BF16(2^(S * c - m * c)) with
 * c = fl32(scale * log2 e) (one FMA, ex2.approx); O accumulates P V in fp32 on tensor cores and
 * the row sum is the fp32 sum of the same BF16 P; at the end O * (1 / sum).
 *
 * The call enqueues on `stream` (NULL: the legacy stream), allocates nothing, is
 * graph-capturable and deterministic (no atomics; a fixed order per output). `o` must not
 * overlap q, k or v. Return 0, or -1 with the reason in ignis_image_encoder_attn_last_error().
 */
#ifndef IGNIS_IMAGE_ENCODER_ATTN_H
#define IGNIS_IMAGE_ENCODER_ATTN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* O = softmax(scale * Q K^T) V over one sequence of `tokens` (1 .. 2^20), causal when `causal`.
 * q_row_stride >= 4096, kv_row_stride >= 1024 and o_row_stride >= 4096, each a multiple of 8;
 * q, k, v and o 16-byte aligned; scale finite and > 0 (1 / sqrt(128) for the encoder). */
int32_t ignis_image_encoder_attn(const void *q, const void *k, const void *v, uint32_t tokens, uint32_t q_row_stride,
                                 uint32_t kv_row_stride, uint32_t causal, float scale, void *o, uint32_t o_row_stride,
                                 uint32_t o_f32, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_encoder_attn_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_ENCODER_ATTN_H */
