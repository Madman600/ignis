/* ignis kernel leaf: the image model's diffusion transformer -- Qwen-Image-2.1's DiT (spec
 * image-model/04, GitHub #321) -- OURS (ADR 0051 tier 1: the program and its ops of its own are
 * written here; the linears, the attention and the elementwise ops are spec 02's,
 * ignis_image_gemm.h, ignis_image_attn.h and ignis_image_dit.h). The definitions are diffusers
 * 0.41.0's QwenImage21Transformer2DModel (models/transformers/transformer_qwenimage21.py).
 *
 * The joint sequence is [condition prefix ; target]. The prefix's tokens (the encoder's text rows
 * through txt_in; a condition image's latents through img_in) are modulated at t = 0 and attend
 * only to the tokens before them and to their own image block, so their keys and values are the
 * same at every denoising step: the PREFIX PASS computes them once and stores each block's K/V
 * run in 8 bits (ignis_image_attn.h); a TARGET PASS, one per denoising step, runs the target's
 * tokens alone, attending to [the stored prefix ; the target].
 *
 * Per block, on the residual stream x (fp32 [tokens][4096]), with the shared modulation row
 * (scale1 | gate1 | scale2 | gate2, 4096 each):
 *   h = LayerNorm(x) * (1 + scale1)                      -> E4M3 (ignis_image_transformer_residual_norm)
 *   q, k, v = h Wq^T, h Wk^T, h Wv^T                        BF16, three GEMMs
 *   q, k = RoPE(RMSNorm_head(.) * norm_q / norm_k)          in place, BF16 (ignis_image_dit_qk_norm_rope)
 *   the K/V run: the prefix pass stores block b's run (its own K mean); the target pass stores
 *   the target's run smoothed with the prefix run's K mean
 *   o = attention, BF16: the prefix pass one call per segment, over the run's first `end` keys,
 *       causal for a text segment; the target pass over the whole prefix run, then the target's
 *   x += tanh(gate1) * (o Wo^T)                             quantize_tokens, the GEMM in fp32, then
 *   h = LayerNorm(x) * (1 + scale2)                      -> E4M3 (fused with the add)
 *   x += tanh(gate2) * ((silu(h Wg^T) * h Wu^T) Wd^T)       gate and up BF16, SWIGLU -> E4M3, down fp32
 * The add after block b's MLP is fused with block b + 1's first norm, so a range of blocks gives
 * the same x, bit for bit, as the same blocks run in consecutive calls. Wg is diffusers'
 * img_mlp.gate_layer (the SiLU'd one), Wu its img_mlp.proj.
 *
 * The rest of the model: txt_in = out_layer(GELU_tanh(in_layer(RMSNorm(x) * (w + 1)))) on the
 * encoder's rows; img_in = a linear on the latents (fp32 [tokens][64]); the head = proj_out(
 * LayerNorm(x) * (1 + out_scale)) (norm_out); the modulation from the timestep embedding (below).
 *
 * Arithmetic: the residual stream and every GEMM that adds into it (img_in, txt_in's out_layer,
 * to_out, the MLP's down) are fp32, as in the fp32 pipeline the oracle recorded; the norms
 * quantize their fp32 output to E4M3 directly; q, k, v, the attention output, the MLP's gate and up
 * and txt_in's hidden are BF16 (the ops reading them take BF16). The latents are quantized to E4M3
 * from fp32; the encoder's rows are rounded to BF16 once for txt_in's norm. Each step of an op of
 * its own is one fp32 operation rounded to nearest; tanhf, expf, rsqrtf, cosf and sinf are CUDA's
 * (no fast math).
 *
 * The timestep embedding at sigma (QwenImage21TimestepProjEmbeddings, then the shared modulation
 * and norm_out.linear): features [256] = cos(a_i) for i < 128, then sin(a_i), a_i = fl32(fl32(1000
 * sigma) * f_i), f_i = PyTorch's fp32 exp(fl32(fl32(-ln 10000 * i) / 128)); temb = linear_2(silu(
 * linear_1(features))); mod [16384] = modulation(silu(temb)); out_scale [4096] = norm_out(silu(
 * temb)). These linears are BF16 weights on fp32 inputs, fp32 accumulation in a fixed order.
 *
 * Every device call enqueues on `stream` (NULL: the legacy stream), allocates nothing and is
 * deterministic (no atomics; each reduction in a fixed order). The GEMM needs
 * ignis_image_gemm_prepare() on the device first. Return 0, or -1 with the reason in
 * ignis_image_transformer_last_error().
 */
#ifndef IGNIS_IMAGE_TRANSFORMER_H
#define IGNIS_IMAGE_TRANSFORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The geometry the leaf runs: Qwen-Image-2.1's DiT. */
#define IGNIS_IMAGE_TRANSFORMER_HIDDEN 4096u
#define IGNIS_IMAGE_TRANSFORMER_HEADS 32u
#define IGNIS_IMAGE_TRANSFORMER_HEAD_DIM 128u
#define IGNIS_IMAGE_TRANSFORMER_MLP 12288u
#define IGNIS_IMAGE_TRANSFORMER_LATENT 64u
#define IGNIS_IMAGE_TRANSFORMER_TIMESTEP 256u
/* The shared modulation row: scale1 | gate1 | scale2 | gate2. */
#define IGNIS_IMAGE_TRANSFORMER_MODULATION (4u * IGNIS_IMAGE_TRANSFORMER_HIDDEN)

/* One block's weights, device pointers: the linears row-scale-v1 FP8 payloads as the artifact
 * stores them (ignis_image_gemm.h), to_q / to_k / to_v / to_out [4096 x 4096], mlp_gate
 * (img_mlp.gate_layer) and mlp_up (img_mlp.proj) [12288 x 4096], mlp_down (img_mlp.out) [4096 x
 * 12288]; norm_q and norm_k BF16 [128]. */
struct ignis_image_transformer_block {
  const void *to_q;
  const void *to_k;
  const void *to_v;
  const void *to_out;
  const void *norm_q;
  const void *norm_k;
  const void *mlp_gate;
  const void *mlp_up;
  const void *mlp_down;
};

/* The model: FP8 img_in [4096 x 64], txt_in (in_layer) and txt_out (out_layer) [4096 x 4096],
 * proj_out [64 x 4096]; BF16 txt_norm [4096] (stored as scale - 1), time_linear_1 [4096 x 256],
 * time_linear_2 [4096 x 4096], modulation [16384 x 4096], norm_out (norm_out.linear) [4096 x 4096];
 * `blocks` a host array of `block_count`; `eps` every norm's (the transformer config's). */
struct ignis_image_transformer_weights {
  const void *img_in;
  const void *txt_norm;
  const void *txt_in;
  const void *txt_out;
  const void *time_linear_1;
  const void *time_linear_2;
  const void *modulation;
  const void *norm_out;
  const void *proj_out;
  const struct ignis_image_transformer_block *blocks;
  uint32_t block_count;
  float eps;
};

/* A run of the prefix: tokens [start, end), text (causal) or one image block (bidirectional);
 * diffusers' _qwenimage21_prefix_segments. */
struct ignis_image_transformer_segment {
  uint32_t start;
  uint32_t end;
  uint32_t text;
};

/* The condition prefix: `tokens` tokens, `runs` a host array of block_count device K/V runs of
 * `tokens` keys (ignis_image_attn_kv8_bytes(tokens) bytes each, 16-byte aligned): written by the
 * prefix pass, read by the target pass. `segments` (host, `segment_count`, in order, covering
 * [0, tokens)) is read by the prefix pass only. */
struct ignis_image_transformer_prefix {
  uint32_t tokens;
  void *const *runs;
  const struct ignis_image_transformer_segment *segments;
  uint32_t segment_count;
};

/* The workspace a call over `tokens` (1 .. 2^20) needs, in bytes: about 60 KiB a token. */
uint64_t ignis_image_transformer_workspace_bytes(uint32_t tokens);

/* The timestep embedding at `sigma` (finite, >= 0): mod fp32 [16384] and out_scale fp32 [4096].
 * The workspace holds at least 64 KiB. */
int32_t ignis_image_transformer_modulation(const struct ignis_image_transformer_weights *weights, float sigma,
                                           float *mod, float *out_scale, void *workspace, uint64_t workspace_bytes,
                                           void *stream);

/* txt_in on the encoder's rows: x fp32 [rows][4096] -> y fp32 [rows][4096]. */
int32_t ignis_image_transformer_txt_in(const struct ignis_image_transformer_weights *weights, const float *x,
                                       uint32_t rows, float *y, void *workspace, uint64_t workspace_bytes,
                                       void *stream);

/* img_in: latents fp32 [tokens][64] -> y fp32 [tokens][4096]. */
int32_t ignis_image_transformer_img_in(const struct ignis_image_transformer_weights *weights, const float *latents,
                                       uint32_t tokens, float *y, void *workspace, uint64_t workspace_bytes,
                                       void *stream);

/* Blocks [first_block, end_block) in place on x, fp32 [tokens][4096]. `target` 0: x is the prefix
 * (tokens == prefix->tokens) and block b's run is written to prefix->runs[b]; `target` 1: x is
 * the target, attending to prefix->runs[b]. rope: fp32 [tokens][64][2], the table
 * (ignis_image_dit_rope_table) of these tokens' joint-sequence positions. mod: the modulation row
 * of the pass (sigma 0 for the prefix, the step's for the target), fp32 [16384]. x and rope
 * 16-byte aligned, workspace 256-byte. */
int32_t ignis_image_transformer_blocks(const struct ignis_image_transformer_weights *weights, uint32_t first_block,
                                       uint32_t end_block, float *x, uint32_t tokens, uint32_t target,
                                       const float *rope, const float *mod,
                                       const struct ignis_image_transformer_prefix *prefix, void *workspace,
                                       uint64_t workspace_bytes, void *stream);

/* The prefix pass in chunks, for a prefix whose whole pass does not fit a workspace: blocks
 * [first_block, end_block) on the prefix's x (prefix->tokens rows) as ignis_image_transformer_blocks
 * with target 0 computes them, bit for bit, the runs included, through a workspace for `chunk`
 * tokens (ignis_image_transformer_workspace_bytes(chunk); chunk a positive multiple of 128) and
 * `kv`, the BF16 K and V of every prefix token (ignis_image_transformer_prefix_kv_bytes(tokens),
 * 256-byte aligned). Per block:
 *   phase A, chunk by chunk of `chunk` tokens: img_norm1, K (its norm and RoPE) and V into kv;
 *            then the block's run from all of them, one store (a run's K mean and V scales are
 *            over all its tokens);
 *   phase B, chunk by chunk of whole segments (a segment longer than `chunk` cut every `chunk`
 *            queries, so its pieces' query blocks fall as in one call): img_norm1 again, Q, the
 *            attention of each segment or piece over the run (a causal piece's keys end at its own
 *            end), to_out and the MLP.
 * The MLP's add is not fused with the next block's img_norm1, which phase A recomputes from x:
 * the same values. rope: the prefix's table; mod: the t = 0 modulation row. */
uint64_t ignis_image_transformer_prefix_kv_bytes(uint32_t tokens);
int32_t ignis_image_transformer_prefix_blocks(const struct ignis_image_transformer_weights *weights,
                                              uint32_t first_block, uint32_t end_block, float *x, const float *rope,
                                              const float *mod, const struct ignis_image_transformer_prefix *prefix,
                                              void *kv, uint32_t chunk, void *workspace, uint64_t workspace_bytes,
                                              void *stream);

/* The head: velocity fp32 [tokens][64] = proj_out(LayerNorm(x) * (1 + out_scale)). Then, when
 * `latents` (fp32 [tokens][64]) is not NULL, the Euler update latents += (sigma_next - sigma) *
 * velocity: dt = fl32(sigma_next - sigma), then fl32(latents + fl32(dt * velocity)). x is read,
 * not written. */
int32_t ignis_image_transformer_head(const struct ignis_image_transformer_weights *weights, const float *x,
                                     uint32_t tokens, const float *out_scale, float *velocity, float *latents,
                                     float sigma, float sigma_next, void *workspace, uint64_t workspace_bytes,
                                     void *stream);

/* The program's ops of its own, exposed for the tests: */

/* When delta is not NULL: x = fma(tanh(gate), delta, x), stored back to x (gate fp32 [cols]).
 * Then, when scale is not NULL: y = ((x - mu) * rsqrt(var + eps)) * (1 + scale), mu and var in
 * two passes as ignis_image_dit_modulated_norm takes them, scale fp32 [cols], quantized per token to
 * E4M3 from fp32 (ignis_image_gemm.h's steps): codes y [tokens][cols] and y_scales [tokens]; with
 * y_f32 (the tests' form) y is fp32 [tokens][cols] and y_scales NULL. x, delta fp32 [tokens][cols];
 * cols a positive multiple of 8, at most 8192; pointers 16-byte aligned (y_scales 4-byte); y
 * overlaps neither x nor delta. */
int32_t ignis_image_transformer_residual_norm(float *x, const float *delta, const float *gate, uint32_t tokens,
                                              uint32_t cols, const float *scale, float eps, void *y, float *y_scales,
                                              uint32_t y_f32, void *stream);

/* y [n][rows] = x [n][cols] W^T, W BF16 [rows][cols], fp32 accumulation in a fixed order (per
 * output: each lane's 8-element chunks in order, then a warp butterfly). rows and cols positive
 * multiples of 8; W, x and y 16-byte aligned. */
int32_t ignis_image_transformer_linear_bf16(const void *weight, uint32_t rows, uint32_t cols, const float *x,
                                            uint32_t n, float *y, void *stream);

/* The per-token E4M3 quantization of fp32 x [tokens][cols] (ignis_image_gemm.h's steps, from
 * fp32): codes [tokens][cols], scales [tokens]. cols a positive multiple of 8; x and codes 16-byte
 * aligned, scales 4-byte. */
int32_t ignis_image_transformer_quantize_f32(const float *x, uint32_t tokens, uint32_t cols, void *codes,
                                             float *scales, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_transformer_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_TRANSFORMER_H */
