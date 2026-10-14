/* ignis kernel leaf: the image model's text encoder -- Qwen3-VL-8B's text model (spec
 * image-model/03, GitHub #320) -- OURS (ADR 0051 tier 1: the program and its three ops are written
 * here; the linears and the attention are spec 02's ops, ignis_image_gemm.h and
 * ignis_image_encoder_attn.h). The definitions are transformers 5.17.0's Qwen3VLTextModel
 * (models/qwen3_vl/modeling_qwen3_vl.py).
 *
 * The forward runs decoder layers [first_layer, end_layer) of the 36 on one prompt, in place on the
 * residual stream x, fp32 [tokens][4096]. Per layer:
 *   h = RMSNorm(x) * input_layernorm                      -> E4M3 (ignis_image_encoder_residual_norm)
 *   q = h Wq^T, k = h Wk^T (fp32), v = h Wv^T (BF16)        ignis_image_gemm_w8a8, three calls
 *   q, k = mrope(RMSNorm_head(q) * q_norm), ... k_norm     -> BF16 (ignis_image_encoder_qk_norm_mrope)
 *   o = causal GQA attention 32 / 8 of 128, scale 128^-1/2  ignis_image_encoder_attn, BF16
 *   x += o Wo^T                                             quantize_tokens, then the GEMM in fp32
 *   h = RMSNorm(x) * post_attention_layernorm             -> E4M3 (fused with the add above)
 *   x += (silu(h Wg^T) * h Wu^T) Wd^T                       gate / up BF16, ignis_image_dit_act
 *                                                           (SWIGLU, E4M3), down in fp32
 * The add after layer l's MLP is fused with layer l + 1's input norm. Layer l < 3 adds the
 * DeepStack features at the vision rows after its MLP (transformers' _deepstack_process). No final
 * norm, no lm_head: x after the last layer is the encoder's output (diffusers takes the final norm's
 * input); the leaf takes its rows from drop_idx on.
 *
 * Arithmetic: the residual stream and the two out-projections' outputs are fp32, as in the fp32
 * pipeline the oracle recorded; the norms quantize their fp32 output to E4M3 directly; q and k are
 * fp32 until the QK norm and mrope round them once to BF16 for the attention; v, gate and up are
 * BF16 (the attention reads BF16, and SwiGLU's E4M3 output is the quantization of its BF16 output,
 * ignis_image_dit.h).
 *
 * The mrope (Qwen3VLTextRotaryEmbedding with mrope_interleaved). positions are int32 [3][tokens],
 * rows t, h, w (transformers' position_ids after the text row is split off; the oracle's
 * mrope_position_ids). Slot i < 64 of the head's 64 frequency slots turns by
 * fl32(positions[axis(i)][t] * f_i), f_i = fl32(1 / fl32(theta^fl32(2i / 128))), each step
 * correctly rounded (torch's own powf is one ulp up at i = 19 for theta 5e6; that moves an angle by
 * under 4e-5 rad at 41,000 positions). axis(i) is 1 (h) when i % 3 == 1 and i < 3 section[1], 2 (w)
 * when i % 3 == 2 and i < 3 section[2], else 0 (t): transformers' interleaved layout
 * [THWTHW...TT] for mrope_section (section[0], section[1], section[2]) = (24, 20, 20). The rotation
 * is rotate_half's: channel d < 64 and channel d + 64 are slot d's pair, out[d] = x[d] cos - x[d +
 * 64] sin and out[d + 64] = x[d + 64] cos + x[d] sin, with cos and sin of the fp32 angle (sincosf).
 *
 * Every device call enqueues on `stream` (NULL: the legacy stream), allocates nothing and is
 * deterministic (no atomics; each reduction in a fixed order). The GEMM needs
 * ignis_image_gemm_prepare() on the device first. Return 0, or -1 with the reason in
 * ignis_image_encoder_last_error().
 */
#ifndef IGNIS_IMAGE_ENCODER_H
#define IGNIS_IMAGE_ENCODER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The geometry the leaf runs: Qwen3-VL-8B's text model. */
#define IGNIS_IMAGE_ENCODER_HIDDEN 4096u
#define IGNIS_IMAGE_ENCODER_HEADS 32u
#define IGNIS_IMAGE_ENCODER_KV_HEADS 8u
#define IGNIS_IMAGE_ENCODER_HEAD_DIM 128u
#define IGNIS_IMAGE_ENCODER_INTERMEDIATE 12288u
#define IGNIS_IMAGE_ENCODER_DEEPSTACK 3u

/* One decoder layer's weights, device pointers: the norms BF16 ([4096], q_norm and k_norm [128]),
 * the linears row-scale-v1 FP8 payloads as the artifact stores them (ignis_image_gemm.h): q [4096 x
 * 4096], k and v [1024 x 4096], o [4096 x 4096], gate and up [12288 x 4096], down [4096 x 12288]. */
struct ignis_image_encoder_layer {
  const void *input_norm;
  const void *q_proj;
  const void *k_proj;
  const void *v_proj;
  const void *q_norm;
  const void *k_norm;
  const void *o_proj;
  const void *post_norm;
  const void *gate_proj;
  const void *up_proj;
  const void *down_proj;
};

/* The text model: `layers` decoder layers (a host array of `layer_count`), the RMSNorms' eps and
 * the mrope's theta and section, all from the checkpoint's text config. */
struct ignis_image_encoder_weights {
  const struct ignis_image_encoder_layer *layers;
  uint32_t layer_count;
  float rms_norm_eps;
  float rope_theta;
  uint32_t mrope_section[3];
};

/* The vision tokens' hooks (spec 03: the tower's merged features and DeepStack). `rows` sequence
 * positions (the image-pad tokens), `row_index`, device int32 [rows], distinct and in [0, tokens). When the
 * forward starts at layer 0, `embeds` (BF16 [rows][4096], or NULL) is written over those rows of x
 * first. After layer l < 3, `deepstack[l]` (BF16 [rows][4096], or NULL) is added to them; feature
 * row j goes to sequence row row_index[j] (transformers' masked rows, in sequence order). */
struct ignis_image_encoder_vision {
  uint32_t rows;
  const int32_t *row_index;
  const void *embeds;
  const void *deepstack[IGNIS_IMAGE_ENCODER_DEEPSTACK];
};

/* The workspace the forward needs for `tokens` (1 .. 2^20), in bytes: about 60 KiB a token. */
uint64_t ignis_image_encoder_workspace_bytes(uint32_t tokens);

/* Runs decoder layers [first_layer, end_layer) (end_layer <= weights->layer_count) in place on x,
 * fp32 [tokens][4096], with mrope positions [3][tokens] (device int32). `vision` is NULL for a
 * prompt without vision tokens. `workspace` holds workspace_bytes(tokens) or more, 256-byte
 * aligned. x is 16-byte aligned; x, positions and workspace do not overlap. A range of layers
 * gives the same x, bit for bit, as the same layers run in two consecutive calls. */
int32_t ignis_image_encoder_forward(const struct ignis_image_encoder_weights *weights, uint32_t first_layer,
                                    uint32_t end_layer, float *x, uint32_t tokens, const int32_t *positions,
                                    const struct ignis_image_encoder_vision *vision, void *workspace,
                                    uint64_t workspace_bytes, void *stream);

/* The forward's three ops of its own, exposed for the tests: */

/* x += delta, when delta is not NULL (stored back to x); then, when weight is not NULL, y =
 * (x * rsqrt(sum(x^2) / cols + eps)) * weight, each step one fp32 operation, quantized per token to
 * E4M3 from fp32 with ignis_image_gemm.h's steps (amax / 448, round to nearest even, saturate):
 * codes y [tokens][cols] and y_scales [tokens]. With y_f32 (the tests' form) y is the fp32
 * [tokens][cols] value instead and y_scales NULL. x and delta fp32 [tokens][cols], weight BF16
 * [cols]; cols a positive multiple of 8, at most 8192; x, delta, weight and y 16-byte aligned,
 * y_scales 4-byte aligned; y overlaps neither x nor delta. */
int32_t ignis_image_encoder_residual_norm(float *x, const float *delta, uint32_t tokens, uint32_t cols,
                                          const void *weight, float eps, void *y, float *y_scales, uint32_t y_f32,
                                          void *stream);

/* The QK RMSNorm with the mrope above, q's 32 heads and k's 8: head h of token t is at q + t *
 * q_stride + h * 128 (k alike), fp32; per head n = (x * rsqrt(sum(x^2) / 128 + eps)) * w, w =
 * q_weight / k_weight BF16 [128], then n turns by the mrope. Out BF16 at q_out (stride 4096) and
 * k_out (stride 1024). positions int32 [3][tokens]; q_stride >= 4096 and k_stride >= 1024, each a
 * multiple of 4; the section sums to 64 with 3 section[1] and 3 section[2] at most 64; theta finite
 * and > 1; pointers 16-byte aligned (positions 4-byte). */
int32_t ignis_image_encoder_qk_norm_mrope(const float *q, uint32_t q_stride, const float *k, uint32_t k_stride,
                                          uint32_t tokens, const void *q_weight, const void *k_weight, float eps,
                                          const int32_t *positions, float rope_theta, const uint32_t *mrope_section,
                                          void *q_out, void *k_out, void *stream);

/* For j < rows: x[row_index[j]] = src[j] (add == 0) or x[row_index[j]] += src[j] (add != 0), x fp32
 * [tokens][cols], src BF16 [rows][cols], cols a positive multiple of 8; x and src 16-byte aligned.
 * The indices are device data the call cannot check before it runs: one outside [0, tokens) is
 * skipped, so nothing outside x is written (the caller builds them from the image-pad mask). */
int32_t ignis_image_encoder_rows(float *x, uint32_t tokens, uint32_t cols, const int32_t *row_index, uint32_t rows,
                                 const void *src, uint32_t add, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_encoder_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_ENCODER_H */
