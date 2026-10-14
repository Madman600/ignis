// ignis kernel leaf: the shared-memory plumbing of the image model's attention kernels (attn8.cu,
// encoder_attn.cu, vae_conv.cu, vae_attn.cu; spec image-model/02, GitHub #319) -- ADAPTED (ADR 0051 tier 2).
//
// Upstream: thu-ml/SageAttention, commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5, Apache-2.0:
// csrc/permuted_smem.cuh, csrc/cp_async.cuh and csrc/mma.cuh, which say "Adapted from FlashInfer"
// (flashinfer-ai/flashinfer v0.1.5 include/flashinfer/{permuted_smem,cp_async,mma}.cuh, Copyright
// (c) 2023 by FlashInfer team, Apache-2.0): the 128-byte / 64-byte shared-memory swizzles, cp.async
// (with its zero-filling form) and ldmatrix (with its transposing form). ex2 and -inf are ours.
#ifndef IGNIS_IMAGE_ATTN_SMEM_CUH
#define IGNIS_IMAGE_ATTN_SMEM_CUH

#include <cstdint>

namespace ignis_image {

// Byte offset of 16-byte chunk `chunk` of `row` in a tile of 128-byte rows (chunk ^ row % 8) or
// 64-byte rows (chunk ^ (row / 2) % 4) -- FlashInfer's k128B / k64B swizzles. Eight consecutive
// rows read at one logical chunk hit eight different bank groups.
__device__ __forceinline__ int swz128(int row, int chunk) { return row * 128 + ((chunk ^ (row & 7)) << 4); }
__device__ __forceinline__ int swz64(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
// The same, or 16 zero bytes when !valid (gmem is not read then).
__device__ __forceinline__ void cp_async16_zfill(void *smem, const void *gmem, bool valid) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void *smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}
// Each 8x8 b16 matrix transposed: thread (g, t) gets rows 2t, 2t+1 of column g.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void *smem) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ float neg_inf() { return __int_as_float(0xFF800000u); }

// 2^x, ex2.approx.ftz.f32: within 2 ulp of the correctly rounded result (PTX ISA); a subnormal
// result is flushed to +0, 2^-inf is +0.
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_ATTN_SMEM_CUH
