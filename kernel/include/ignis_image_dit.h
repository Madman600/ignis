/* ignis kernel leaf: the image model's DiT elementwise ops -- the 3-axis RoPE positions and
 * table, the modulated LayerNorm, the zero-centred RMSNorm, the gated residual, the QK RMSNorm with
 * RoPE and the activations (spec image-model/02 acceptance 7, GitHub #319) -- OURS (ADR 0051 tier
 * 1: nothing read or copied; FlashInfer's fused_dit_layernorm.cuh and fused_qk_rmsnorm_rope.cuh
 * were allowed and not used). The definitions are diffusers 0.41.0's
 * QwenImage21Transformer2DModel (models/transformers/transformer_qwenimage21.py).
 *
 * Arithmetic. Activations come in BF16, modulation rows in fp32. Every step is one fp32 operation
 * rounded to nearest (expf, tanhf and rsqrtf are CUDA's, within 2 ulp), and the result is rounded
 * once to the output type. diffusers' BF16 path rounds in between (the norm's output, 1 + scale,
 * tanh(gate), the product before the residual add, the normed q and k before RoPE, SiLU's output
 * before the SwiGLU product); these ops skip those roundings, as the VAE convolution's epilogue
 * does.
 *
 * Output kinds, for the ops whose output is an FP8 GEMM's input (ignis_image_gemm.h):
 *   IGNIS_IMAGE_DIT_OUT_BF16  y BF16 [tokens][cols];
 *   IGNIS_IMAGE_DIT_OUT_F32   y fp32 [tokens][cols] (the tests' form: the BF16 output is it rounded);
 *   IGNIS_IMAGE_DIT_OUT_E4M3  y = E4M3 codes [tokens][cols] and y_scales [tokens] fp32: exactly
 *                             what ignis_image_quantize_tokens makes of the BF16 output, bit for
 *                             bit, in the same pass (the producer-fused quantization of spec 02
 *                             acceptance 3).
 *
 * The per-token modulation row. The DiT modulates the condition prefix (text and condition-image
 * tokens) from t = 0 and the target's tokens from the sampled timestep (diffusers'
 * _select_modulation_rows with causal_condition), and the target block is the sequence's last. So
 * a call takes both rows and `first_target`: tokens [0, first_target) read the t = 0 row,
 * tokens [first_target, tokens) the t row. The leaf's prefix pass is first_target = tokens, a
 * denoising step's target-only pass first_target = 0. A row a call does not read may be NULL.
 *
 * The RoPE (QwenImage21Rope, theta 10000, axes 16 / 56 / 56 channels). A token's position is
 * (frame, height, width). Text tokens advance one shared position on all three axes. An image
 * block of h x w latent tokens (row-major) takes frame = the position the text before it reached,
 * height in [-(h - h / 2), h / 2) and width in [-(w - w / 2), w / 2), then advances the position by
 * max(h, w); text after the last block continues from there. The table holds per token 64 (cos,
 * sin) pairs, fp32 [tokens][64][2]: 8 frame, 28 height and 28 width frequencies, diffusers'
 * pos_embed in the same layout. Pair i of an axis of d channels turns by fl32(position * f_i),
 * where f_i = 1 / 10000^(2i / d) is the fp32 value PyTorch computes: fl32(1 / fl32(10000^fl32(2i /
 * d))) for every i but (d = 56, i = 11), where torch's powf rounds one ulp up and we take its value.
 * cos and sin of that fp32 angle are evaluated in fp64 and rounded to fp32. The rotation multiplies
 * channel pair (x[2i], x[2i + 1]) as a complex number by (cos, sin): adjacent pairs.
 *
 * Every device call enqueues on `stream` (NULL: the legacy stream), allocates nothing, is
 * graph-capturable and deterministic (no atomics; each reduction in a fixed order). No call needs a
 * prepare. Return 0, or -1 with the reason in ignis_image_dit_last_error().
 */
#ifndef IGNIS_IMAGE_DIT_H
#define IGNIS_IMAGE_DIT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IGNIS_IMAGE_DIT_OUT_BF16 0u
#define IGNIS_IMAGE_DIT_OUT_F32 1u
#define IGNIS_IMAGE_DIT_OUT_E4M3 2u

#define IGNIS_IMAGE_DIT_SILU 0u
#define IGNIS_IMAGE_DIT_GELU_TANH 1u
#define IGNIS_IMAGE_DIT_SWIGLU 2u

/* Host only. positions [tokens][3] int32 (frame, height, width) for a joint sequence of `tokens`
 * whose image block b covers tokens [block_start[b], block_start[b] + block_height[b] *
 * block_width[b]), blocks in sequence order (condition images first, the target last), not
 * overlapping. Refuses a position outside diffusers' table, [-1024, 8192). */
int32_t ignis_image_dit_rope_positions(uint32_t tokens, uint32_t blocks, const uint32_t *block_start,
                                       const uint32_t *block_height, const uint32_t *block_width,
                                       int32_t *positions);

/* table [tokens][64][2] fp32 (cos, sin) as above, from device positions [tokens][3]. positions
 * 4-byte aligned, table 16-byte aligned. */
int32_t ignis_image_dit_rope_table(const int32_t *positions, uint32_t tokens, float *table, void *stream);

/* y = LayerNorm(x) * (1 + s), no affine, with s = scale_0 [cols] for tokens < first_target, else
 * scale_t [cols] (img_norm1 / img_norm2 with _modulate, and norm_out; Qwen-Image 2.1 has no shift).
 * Mean and variance in two passes: mu = sum(x) / cols, var = sum((x - mu)^2) / cols, then y = ((x -
 * mu) * rsqrt(var + eps)) * (1 + s). x BF16 [tokens][cols]. */
int32_t ignis_image_dit_modulated_norm(const void *x, uint32_t tokens, uint32_t cols, const float *scale_t,
                                       const float *scale_0, uint32_t first_target, float eps, void *y,
                                       float *y_scales, uint32_t y_kind, void *stream);

/* y = (x * rsqrt(sum(x^2) / cols + eps)) * (w + 1), w BF16 [cols] (QwenImage21ZeroCenterRMSNorm:
 * txt_in's text_norm). x BF16 [tokens][cols]. */
int32_t ignis_image_dit_rms_norm(const void *x, uint32_t tokens, uint32_t cols, const void *weight, float eps,
                                 void *y, float *y_scales, uint32_t y_kind, void *stream);

/* out = x + tanh(g) * y (one fma), g = gate_0 [cols] for tokens < first_target, else gate_t [cols]:
 * the block's two gated residual adds (diffusers: hidden_states + gate.tanh() * output). x and y
 * BF16 [tokens][cols]; out BF16, or fp32 when `out_f32`; out may be x
 * only with BF16 output (an fp32 row is twice as wide and would overwrite x rows not yet read). */
int32_t ignis_image_dit_gated_residual(const void *x, const void *y, uint32_t tokens, uint32_t cols,
                                       const float *gate_t, const float *gate_0, uint32_t first_target, void *out,
                                       uint32_t out_f32, void *stream);

/* The QK RMSNorm with RoPE, for q and k in one call. Head h < 32 of token t is at q + t *
 * row_stride + h * 128 (and k alike); v is not touched, so the fused QKV row [tokens][12288] is
 * rewritten in place. Per head: n = (x * rsqrt(sum(x^2) / 128 + eps)) * w, w = q_weight /
 * k_weight BF16 [128] (diffusers' RMSNorm norm_q / norm_k), then pair i of n turns by table row t's
 * pair i (rope [tokens][64][2], the table above for these tokens). Output at q_out / k_out with
 * out_row_stride, BF16, or fp32 when `out_f32`; q_out = q and k_out = k with out_row_stride =
 * row_stride is the in-place form. Strides >= 4096 and multiples of 8; pointers 16-byte aligned. */
int32_t ignis_image_dit_qk_norm_rope(const void *q, const void *k, uint32_t tokens, uint32_t row_stride,
                                     const void *q_weight, const void *k_weight, float eps, const float *rope,
                                     void *q_out, void *k_out, uint32_t out_row_stride, uint32_t out_f32,
                                     void *stream);

/* The activations, x (and up) BF16 rows at row_stride elements, y [tokens][cols]:
 *   IGNIS_IMAGE_DIT_SILU       y = x / (1 + exp(-x))                  (the timestep path's SiLUs);
 *   IGNIS_IMAGE_DIT_GELU_TANH  y = x / (1 + exp(-2z)), z = sqrt(2 / pi) (x + 0.044715 x^3), which
 *                              is 0.5 x (1 + tanh z) without its cancellation (txt_in's GELU);
 *   IGNIS_IMAGE_DIT_SWIGLU     y = silu(x) * up, x the gate projection (the MLP's SwiGLU); separate
 *                              gate and up outputs, or one fused [tokens][2 cols] output with up at
 *                              x + cols and row_stride 2 cols. `up` is NULL for the other kinds. */
int32_t ignis_image_dit_act(uint32_t kind, const void *x, const void *up, uint32_t tokens, uint32_t cols,
                            uint32_t row_stride, void *y, float *y_scales, uint32_t y_kind, void *stream);

/* For the row ops above (norms, activations): cols a positive multiple of 16, at most 24560 (the row sits in
 * 48 KiB of shared memory); x, up,
 * y and codes 16-byte aligned, y_scales 4-byte aligned (NULL unless y_kind is E4M3); row_stride >=
 * cols and a multiple of 8; y overlaps no input. The gated residual takes cols a multiple of 8 and
 * 16-byte aligned x, y and out. */

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_dit_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_DIT_H */
