// ignis kernel leaf: the VAE's residual shortcuts for one frame, AvgDown and DupUp
// (kernel/include/ignis_image_vae.h; spec image-model/02 acceptance 6, GitHub #319) -- OURS:
// diffusers' QwenImage21AvgDown3D and QwenImage21DupUp3D (first_chunk) restated per output
// element, on NHWC tensors. One thread per output element, a grid-stride loop: memory-bound, read
// once and written once.

#include "ignis_image_vae.h"
#include "image_common.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <string>

namespace ignis_image {
namespace {

struct ShortcutArgs {
  int avg, height, width, in_ch, out_ch, ft, fs, accumulate;
  int out_h, out_w;
  int factor, group, repeats;  // factor_t * factor_s^2; AvgDown's group; DupUp's repeats
};

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T> __device__ __forceinline__ T from_f32(float v);
template <> __device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <class T> __global__ void shortcut_kernel(const T *x, T *y, ShortcutArgs a) {
  const std::size_t total = static_cast<std::size_t>(a.out_h) * a.out_w * a.out_ch;
  const int fs2 = a.fs * a.fs;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < total;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const int o = static_cast<int>(e % a.out_ch);
    const std::size_t pix = e / a.out_ch;
    const int X = static_cast<int>(pix % a.out_w), Y = static_cast<int>(pix / a.out_w);
    float s;
    if (a.avg) {
      // The mean over the group: a sum from 0 in g order, then one division (PyTorch's, bit for
      // bit in fp32). Slots before the frame's (t < factor_t - 1) are the zero padding.
      float sum = 0.0f;
      for (int g = 0; g < a.group; ++g) {
        const int j = o * a.group + g, c = j / a.factor, rem = j % a.factor;
        const int t = rem / fs2, sh = (rem % fs2) / a.fs, sw = rem % a.fs;
        const float v = t == a.ft - 1
                            ? to_f32(x[(static_cast<std::size_t>(Y * a.fs + sh) * a.width + X * a.fs + sw) * a.in_ch + c])
                            : 0.0f;
        sum = __fadd_rn(sum, v);
      }
      s = __fdiv_rn(sum, static_cast<float>(a.group));
    } else {
      // The frame's slot (factor_t - 1) of the replicated channels, at sub-pixel (sh, sw).
      const int sh = Y % a.fs, sw = X % a.fs;
      const int c = (o * a.factor + (a.ft - 1) * fs2 + sh * a.fs + sw) / a.repeats;
      s = to_f32(x[(static_cast<std::size_t>(Y / a.fs) * a.width + X / a.fs) * a.in_ch + c]);
    }
    // The shortcut in y's type, then (accumulating) one add in fp32 rounded to that type: the
    // block's x + shortcut(x_copy) as diffusers' BF16 add computes it.
    const T st = from_f32<T>(s);
    y[e] = a.accumulate ? from_f32<T>(__fadd_rn(to_f32(y[e]), to_f32(st))) : st;
  }
}

}  // namespace
}  // namespace ignis_image

using namespace ignis_image;

extern "C" int32_t ignis_image_vae_shortcut(uint32_t kind, const void *x, uint32_t height, uint32_t width,
                                            uint32_t in_channels, uint32_t out_channels, uint32_t factor_t,
                                            uint32_t factor_s, void *y, uint32_t accumulate, uint32_t f32,
                                            void *stream) {
  const std::string op = "ignis_image_vae_shortcut";
  if (x == nullptr || y == nullptr) return fail(op + ": NULL pointer");
  if (kind > IGNIS_IMAGE_VAE_DUP_UP) return fail(op + ": unknown kind");
  if ((factor_t != 1 && factor_t != 2) || (factor_s != 1 && factor_s != 2)) {
    return fail(op + ": factor_t and factor_s must be 1 or 2");
  }
  if (height == 0 || width == 0 || height > 32768 || width > 32768) return fail(op + ": height and width must be in [1, 32768]");
  if (in_channels == 0 || out_channels == 0 || in_channels > 65536 || out_channels > 65536) {
    return fail(op + ": channel counts must be in [1, 65536]");
  }
  const uint32_t factor = factor_t * factor_s * factor_s;
  ShortcutArgs a{};
  a.avg = kind == IGNIS_IMAGE_VAE_AVG_DOWN ? 1 : 0;
  if (a.avg) {
    if (in_channels * factor % out_channels != 0) {
      return fail(op + ": AVG_DOWN needs in_channels * factor_t * factor_s^2 divisible by out_channels");
    }
    if (height % factor_s != 0 || width % factor_s != 0) return fail(op + ": AVG_DOWN needs height and width multiples of factor_s");
    a.group = static_cast<int>(in_channels * factor / out_channels);
    a.out_h = static_cast<int>(height / factor_s);
    a.out_w = static_cast<int>(width / factor_s);
  } else {
    if (out_channels * factor % in_channels != 0) {
      return fail(op + ": DUP_UP needs out_channels * factor_t * factor_s^2 divisible by in_channels");
    }
    a.repeats = static_cast<int>(out_channels * factor / in_channels);
    a.out_h = static_cast<int>(height * factor_s);
    a.out_w = static_cast<int>(width * factor_s);
  }
  const uint32_t elem = f32 != 0 ? 4 : 2;
  if (!aligned(x, elem) || !aligned(y, elem)) return fail(op + ": x and y must be aligned to their element");
  a.height = static_cast<int>(height);
  a.width = static_cast<int>(width);
  a.in_ch = static_cast<int>(in_channels);
  a.out_ch = static_cast<int>(out_channels);
  a.ft = static_cast<int>(factor_t);
  a.fs = static_cast<int>(factor_s);
  a.factor = static_cast<int>(factor);
  a.accumulate = accumulate != 0 ? 1 : 0;
  const std::size_t total = static_cast<std::size_t>(a.out_h) * a.out_w * a.out_ch;
  const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((total + 255) / 256, 170 * 64));
  if (f32 != 0) {
    shortcut_kernel<float><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const float *>(x),
                                                                                  static_cast<float *>(y), a);
  } else {
    shortcut_kernel<__nv_bfloat16><<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        static_cast<const __nv_bfloat16 *>(x), static_cast<__nv_bfloat16 *>(y), a);
  }
  return check_launch(op.c_str());
}
