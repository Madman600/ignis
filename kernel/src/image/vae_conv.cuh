// ignis kernel leaf: the VAE convolution's implicit GEMM, shared by the convolution and the
// mid-block attention (kernel/include/ignis_image_vae.h; spec image-model/02 acceptance 6, GitHub
// #319) -- OURS. Leaf-internal: vae_conv.cu defines the launch, vae_attn.cu runs its Q K^T and P V
// through it as 1 x 1 convolutions over strided rows.
#ifndef IGNIS_IMAGE_VAE_CONV_CUH
#define IGNIS_IMAGE_VAE_CONV_CUH

#include <cuda_bf16.h>

#include <cstdint>

namespace ignis_image {

// One implicit GEMM: y[m][n] = ((acc * row_scale[m]) + bias[n]) + residual[m][n], each step one
// fp32 operation, acc = the sum over the taps, then the input channels, of x * w.
//   m  the window's output pixels: m = (oy - y_row0) * out_w + ox, oy in [y_row0, y_row0 + y_rows)
//   x  input pixel (iy, ix) (of height x width, before the upsample) at
//      x + ((iy - x_row0) * width + ix) * in_stride, its in_ch channels contiguous
//   w  output channel n's reduction row at w + n * w_stride: [ksize][ksize][in_ch]
//   y  y + m * out_stride + n, fp32 or BF16; residual (BF16) the same way
// row_scale, bias and residual may be NULL.
struct VaeConvArgs {
  const __nv_bfloat16 *x;
  int height, width, in_ch, in_stride, x_row0;
  int ksize, stride, pad, upsample;  // 3 / 1 / 1 (upsample 0 or 1), 3 / 2 / 0 or 1 / 1 / 0
  const __nv_bfloat16 *w;
  int w_stride;
  const float *row_scale;
  const __nv_bfloat16 *bias, *residual;
  void *y;
  int out_ch, out_stride, out_w, y_row0, y_rows, y_f32;
};

// Enqueues the GEMM. The callers check its preconditions: in_ch a multiple of 8 (16-byte loads),
// or 4 (8-byte loads), with in_stride, w_stride and the x and w alignments to match; out_stride
// even; every input row the window reads present in x; y_rows * out_w * ceil(out_ch / 96) tiles
// below 2^31.
int32_t vae_conv_launch(const VaeConvArgs &a, void *stream, const char *op);

// The input rows [lo, hi) (in the input's own resolution, before the upsample, clamped to
// [0, height)) that output rows [y_row0, y_row0 + y_rows) read: what x must hold. hi <= lo when
// they read none. Public in this header for the band planner (spec image-model/04, which sizes a
// band's input from it); the launch itself refuses a short window with the same arithmetic.
void vae_conv_input_rows(int ksize, int stride, int pad, int upsample, int height, int y_row0, int y_rows, int &lo,
                         int &hi);

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_VAE_CONV_CUH
