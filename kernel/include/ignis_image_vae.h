/* ignis kernel leaf: the image model's VAE ops -- the convolution, the mid-block attention and the
 * residual shortcuts (spec image-model/02 acceptance 6, GitHub #319), and the channel RMS norm
 * (acceptance 7) -- OURS (ADR 0051 tier 1:
 * written from our FP8 GEMM, kernel/src/image/fp8_gemm_kernel.cuh; no convolution source was
 * read or copied). The convolution's shared-memory plumbing is attn_smem.cuh's (FlashInfer through
 * SageAttention, Apache-2.0; kernel/NOTICE).
 *
 * Layout. Activations are one image, BF16, NHWC: pixel (y, x)'s channels at
 * (y * width + x) * channels. A band of whole rows is one contiguous range, which is how spec 04
 * decodes 2048² (horizontal bands with halos after the global mid block). The VAE's tensors are
 * diffusers' AutoencoderKLQwenImage21 for one frame, where every causal 3D convolution is a 2D one.
 *
 * The convolution: y = conv(x) + bias (+ residual), an implicit GEMM on BF16 tensor cores with fp32
 * accumulation. Rows are output pixels, columns output channels, and the reduction runs over the
 * taps (ky, kx) in order and, within a tap, the input channels in order, in k16 MMA steps: that
 * fixed order per output makes every call deterministic, and an output does not depend on which
 * call or band computed it. The epilogue adds the bias and the residual in fp32, one rounding
 * each, and rounds once to the output type. (diffusers' BF16 path rounds between the convolution,
 * the bias and the residual add; the fused epilogue skips those roundings.) The kinds are the VAE's:
 *   IGNIS_IMAGE_VAE_CONV_3X3     3 x 3, stride 1, zero padding 1 (resnets, conv_in, conv_out,
 *                                and with `upsample` the decoder's resample: nearest-exact 2x of
 *                                the input first, read in place, never materialized);
 *   IGNIS_IMAGE_VAE_CONV_3X3_S2  3 x 3, stride 2 after zero padding (0, 1, 0, 1): one column on
 *                                the right and one row at the bottom (the encoder's downsample);
 *   IGNIS_IMAGE_VAE_CONV_1X1     1 x 1 (the shortcuts, quant_conv, post_quant_conv, to_qkv, proj).
 * Weights are packed from PyTorch's [out][in][ky][kx] to [out][ky][kx][in]
 * (ignis_image_vae_pack_conv_weight) by the caller before the calls that use them (the image leaf
 * packs the 3x3 weights per call, into its workspace); the bias is BF16 [out].
 *
 * Row windows (for bands). `x` holds input rows [x_row0, x_row0 + x_rows) of an input image of
 * height x width (before the upsample), and the call computes output rows [y_row0, y_row0 + y_rows)
 * into `y` (and reads `residual` laid out the same way). Padding is decided by the global position:
 * only rows outside [0, height) are zero, so a band whose x holds its halo rows computes exactly
 * the rows the whole-image call computes, bit for bit. The call refuses a window whose x misses
 * a row the output rows read. The whole image is x_row0 = 0, x_rows = height, y_row0 = 0, y_rows =
 * the output height.
 *
 * The shortcuts (QwenImage21AvgDown3D / QwenImage21DupUp3D for one frame, factor_t and factor_s
 * in {1, 2}; with factor_t = 2 the frame is the last of the factor_t slots, as diffusers' front
 * padding and its first_chunk slice make it):
 *   IGNIS_IMAGE_VAE_AVG_DOWN  y[o] at (Y, X) = the mean over g < group of the value at j = o *
 *                             group + g, where j = ((c * factor_t + t) * factor_s + sh) *
 *                             factor_s + sw names channel c at (Y * factor_s + sh, X * factor_s
 *                             + sw) in slot t, which is x for t = factor_t - 1 and 0 otherwise;
 *                             group = in * factor_t * factor_s^2 / out. The mean is fp32: a sum
 *                             from 0 in g order, then one division by group (PyTorch's on the
 *                             CPU, bit for bit in fp32);
 *   IGNIS_IMAGE_VAE_DUP_UP    y[o] at (h * factor_s + sh, w * factor_s + sw) = x[c] at (h, w),
 *                             c = (o * factor + (factor_t - 1) * factor_s^2 + sh * factor_s +
 *                             sw) / repeats, factor = factor_t * factor_s^2, repeats = out *
 *                             factor / in (a copy).
 * With `accumulate` the call adds the shortcut into y (the block's x + shortcut(x_copy)): the
 * shortcut is rounded to y's type first, then one add, as diffusers' BF16 add. Tensors are BF16,
 * or fp32 when `f32`.
 *
 * The mid-block attention: O = softmax(scale Q K^T) V with one head of `dim` channels over the
 * latent grid's `tokens` (QwenImage21AttentionBlock; scale = 1 / sqrt(dim)). `qkv` is to_qkv's
 * output [tokens][3 * dim] (q at 0, k at dim, v at 2 * dim). Queries go in tiles of `row_tile`
 * rows, each through a fixed workspace: S = Q K^T in fp32 (the convolution's GEMM), then per row
 * P = BF16(2^(S * c - fl(max S * c))) with c = fl32(scale * log2 e) (one FMA, ex2.approx) and its
 * sum D, the fp32 sum of the BF16 P (each thread adds its columns j = t, t + 256, ... in order, then
 * a fixed tree), then O = (P V) * (1 / D) on the GEMM again, BF16 or fp32. An output row does not
 * depend on the tile it fell in.
 *
 * Every call enqueues on `stream` (NULL: the legacy stream), allocates nothing, is
 * graph-capturable and deterministic (no atomics). No call needs a prepare. Return 0, or -1 with
 * the reason in ignis_image_vae_last_error().
 */
#ifndef IGNIS_IMAGE_VAE_H
#define IGNIS_IMAGE_VAE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IGNIS_IMAGE_VAE_CONV_3X3 0u
#define IGNIS_IMAGE_VAE_CONV_3X3_S2 1u
#define IGNIS_IMAGE_VAE_CONV_1X1 2u

#define IGNIS_IMAGE_VAE_AVG_DOWN 0u
#define IGNIS_IMAGE_VAE_DUP_UP 1u

/* A BF16 weight [out_channels][in_channels][kernel][kernel] repacked to [out_channels][kernel]
 * [kernel][in_channels] (kernel 1 or 3). `w_packed` must not overlap `w`. */
int32_t ignis_image_vae_pack_conv_weight(const void *w, uint32_t out_channels, uint32_t in_channels, uint32_t kernel,
                                         void *w_packed, void *stream);

/* y = conv(x) + bias (+ residual) as above. in_channels is a multiple of 8, or 4 (RGBA);
 * out_channels is even. The stride-2 kind takes an even height and width; `upsample` takes the 3 x
 * 3 stride-1 kind only. x and weight are 16-byte aligned (8-byte when in_channels is 4), y 8-byte,
 * bias and residual 4-byte. `bias` and `residual` may be NULL; `y_f32` selects fp32 output.
 * Aliasing: x and weight must not overlap y. With BF16 output `residual == y` is allowed, an
 * in-place add (the thread that writes an output element reads its residual element first, and no
 * other thread touches it); any other overlap of residual and y is not, nor is residual == y with
 * `y_f32` (the elements are then of different widths). */
int32_t ignis_image_vae_conv(uint32_t kind, uint32_t upsample, const void *x, uint32_t height, uint32_t width,
                             uint32_t in_channels, uint32_t x_row0, uint32_t x_rows, const void *weight,
                             const void *bias, uint32_t out_channels, const void *residual, void *y, uint32_t y_row0,
                             uint32_t y_rows, uint32_t y_f32, void *stream);

/* y (+)= the shortcut of x [height][width][in_channels] as above: AVG_DOWN writes [height /
 * factor_s][width / factor_s][out_channels] (height and width multiples of factor_s), DUP_UP
 * [height * factor_s][width * factor_s][out_channels]. The channel counts must divide as the
 * definitions need. x and y aligned to their element; x must not overlap y. */
int32_t ignis_image_vae_shortcut(uint32_t kind, const void *x, uint32_t height, uint32_t width, uint32_t in_channels,
                                 uint32_t out_channels, uint32_t factor_t, uint32_t factor_s, void *y,
                                 uint32_t accumulate, uint32_t f32, void *stream);

/* The attention's workspace for `tokens` (1 .. 2^20), `dim` (a multiple of 8) and `row_tile`
 * (>= 1): V^T [dim][n8] BF16, S [row_tile][n8] fp32, P [row_tile][n8] BF16 and 1 / D
 * [row_tile] fp32, n8 = tokens rounded up to 8, each region 256-byte aligned. 0 if the arguments
 * are out of range. */
uint64_t ignis_image_vae_attn_workspace_bytes(uint32_t tokens, uint32_t dim, uint32_t row_tile);

/* O [tokens][dim] = the attention above, BF16, or fp32 when `o_f32`. qkv and o 16-byte aligned,
 * the workspace 256-byte aligned and at least ignis_image_vae_attn_workspace_bytes; scale finite
 * and > 0. o must not overlap qkv or the workspace. */
int32_t ignis_image_vae_attn(const void *qkv, uint32_t tokens, uint32_t dim, float scale, uint32_t row_tile,
                             void *workspace, uint64_t workspace_bytes, void *o, uint32_t o_f32, void *stream);

/* The channel RMS norm (QwenImage21RMS_norm: channel_first, no bias), then SiLU when `silu`. Per
 * pixel, y = (x * (fl32(sqrt(channels)) / max(||x||_2, 1e-12))) * gamma, ||x||_2 = sqrt of the
 * fp32 sum of squares in a fixed order, and with `silu` y = y / (1 + exp(-y)) on the fp32 value.
 * The resnets' norm1 / norm2 and the encoder's and decoder's norm_out are followed by SiLU; the
 * mid-block attention's norm is not. One rounding to the output type; diffusers' BF16 path rounds
 * the normalized x, the product with sqrt(channels) and the one with gamma, and then SiLU's output.
 * x BF16 NHWC [pixels][channels] (a band of rows is a range of pixels), gamma BF16 [channels]; y
 * BF16, or fp32 when `y_f32`; y may be x only for BF16 output (fp32 rows are twice as
 * wide and would overwrite x rows not yet read). channels a positive multiple of 8, pixels 1 .. 2^24; x,
 * gamma and y 16-byte aligned. */
int32_t ignis_image_vae_rms_norm(const void *x, uint32_t pixels, uint32_t channels, const void *gamma, uint32_t silu,
                                 void *y, uint32_t y_f32, void *stream);

/* Thread-local message from the most recent failed call. Never NULL. */
const char *ignis_image_vae_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* IGNIS_IMAGE_VAE_H */
