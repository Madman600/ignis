// ignis kernel leaf: the 8-bit DiT attention's buffer layouts (kernel/include/ignis_image_attn.h)
// -- OURS (spec image-model/02 acceptance 4, GitHub #319). Shared by the quantizers
// (attn8_quant.cu) and the attention (attn8.cu).
#ifndef IGNIS_IMAGE_ATTN8_LAYOUT_CUH
#define IGNIS_IMAGE_ATTN8_LAYOUT_CUH

#include "fp8_quant.cuh"

#include <cstddef>
#include <cstdint>

namespace ignis_image {

constexpr int kAttnHeads = 32;
constexpr int kAttnDim = 128;
constexpr int kAttnHD = kAttnHeads * kAttnDim;
constexpr int kAttnQBlock = 128;  // q8 rows are padded to the attention's query tile
constexpr int kAttnQScaleRows = 32;  // one Q scale per warp's 32 rows
constexpr int kAttnKBlock = 64;  // one K scale per key tile

// The INT8 floor: the smallest amax whose scale amax / 127 is a normal fp32. V's E4M3 floor and
// scale are the GEMM activations' (fp8_quant.cuh: kE4m3MinAmax, e4m3_token_scale).
constexpr float kAttnInt8MinAmax = 0x1.fcp-120f;  // 127 * 2^-126

struct Q8Layout {
  uint32_t n, n_pad;
  std::size_t scales;  // byte offset of the scales
  std::size_t bytes;
};

__host__ __device__ inline Q8Layout q8_layout(uint32_t n) {
  Q8Layout l;
  l.n = n;
  l.n_pad = (n + kAttnQBlock - 1) / kAttnQBlock * kAttnQBlock;
  l.scales = static_cast<std::size_t>(kAttnHeads) * l.n_pad * kAttnDim;
  l.bytes = l.scales + static_cast<std::size_t>(kAttnHeads) * (l.n_pad / kAttnQScaleRows) * 4;
  return l;
}

struct Kv8Layout {
  uint32_t n, n_pad;
  std::size_t v, kscale, vscale, mean;  // byte offsets (K codes at 0)
  std::size_t bytes;
};

__host__ __device__ inline Kv8Layout kv8_layout(uint32_t n) {
  Kv8Layout l;
  l.n = n;
  l.n_pad = (n + kAttnKBlock - 1) / kAttnKBlock * kAttnKBlock;
  const std::size_t codes = static_cast<std::size_t>(kAttnHeads) * l.n_pad * kAttnDim;
  l.v = codes;
  l.kscale = 2 * codes;
  l.vscale = l.kscale + static_cast<std::size_t>(kAttnHeads) * (l.n_pad / kAttnKBlock) * 4;
  l.mean = l.vscale + static_cast<std::size_t>(kAttnHD) * 4;
  l.bytes = l.mean + static_cast<std::size_t>(kAttnHD) * 4;
  return l;
}

// The V layout's key order within a group of 16: key j sits at position kv8_v_position(j).
// Position p holds key 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15 [p] (SageAttention's), the order in
// which the attention's P accumulator fragment is PV's A fragment.
__host__ __device__ inline int kv8_v_position(int j) {
  const int r = j & 15;
  const int pos = r < 8 ? (r >> 1) * 4 + (r & 1) : ((r - 8) >> 1) * 4 + 2 + (r & 1);
  return (j & ~15) + pos;
}

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_ATTN8_LAYOUT_CUH
