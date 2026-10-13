// ignis kernel leaf: the tensor-core instructions of the image model's GEMMs (spec image-model/02,
// GitHub #319) -- OURS (ADR 0051 tier 1; the instruction forms are the PTX ISA's).
//
// One warp-wide mma.sync per wrapper, operands in registers in the PTX ISA's fragment layouts
// (groupID g = lane >> 2, t = lane & 3):
//
//   m16n8k16 BF16, fp32 accumulate:
//     a[0] row g, k 2t..2t+1   a[1] row g+8, same k   a[2] row g, k 2t+8..   a[3] row g+8, k 2t+8..
//     b[0] k 2t..2t+1, col g   b[1] k 2t+8..2t+9, col g
//   m16n8k32 E4M3 (both FP8 forms) and INT8:
//     a[0] row g, k 4t..4t+3   a[1] row g+8, same k   a[2] row g, k 16+4t..  a[3] row g+8, k 16+4t..
//     b[0] k 4t..4t+3, col g   b[1] k 16+4t..16+4t+3, col g
//   accumulator (fp32 or int32): d[0..1] row g, cols 2t..2t+1; d[2..3] row g+8, same cols.
//
// The k-contiguous bytes of an 8-bit operand row are exactly what ldmatrix.b16 delivers when its
// byte pairs are taken as b16 elements, so the FP8 GEMM loads both operands with ldmatrix as the
// BF16 path does.
//
// The bench (kernel/tests/bench_mma_peak.cu) times every form here; the GEMM and the 8-bit
// attention use the ones that bench chose (spec image-model/02 "As built").
#ifndef IGNIS_IMAGE_MMA_CUH
#define IGNIS_IMAGE_MMA_CUH

#include <cstdint>

namespace ignis_image {

// D = A B + D, BF16 inputs, fp32 accumulate.
__device__ __forceinline__ void mma_bf16_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D = A B + D, E4M3 inputs, fp32 accumulate: the plain FP8 form (kind::f8f6f4, sm_120a).
__device__ __forceinline__ void mma_e4m3_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The UE8M0 encoding of 2^0: a block scale of one, in every byte of a scale register.
constexpr uint32_t kUnitScaleUe8m0 = 0x7F7F7F7Fu;

// D = (A * 1) (B * 1) + D, E4M3 inputs, fp32 accumulate: the block-scaled FP8 form
// (kind::mxf8f6f4, scale_vec::1X, sm_120a) with every block scale 2^0, so the product is the
// plain form's. Byte 0 of thread pair / thread 0 of each quad supplies the scales.
__device__ __forceinline__ void mma_e4m3_mx_unit_f32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0, 0}, %11, {0, 0};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(kUnitScaleUe8m0),
        "r"(kUnitScaleUe8m0));
}

// D = A B + D, INT8 inputs, int32 accumulate (exact): the 8-bit attention's QK^T.
__device__ __forceinline__ void mma_s8_s32(int32_t (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// D = A B + D, E4M3 inputs, fp16 accumulate (two f16x2 registers): the 838-class reference rate.
__device__ __forceinline__ void mma_e4m3_f16(uint32_t (&d)[2], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f16.e4m3.e4m3.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(d[0]), "+r"(d[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

}  // namespace ignis_image

#endif  // IGNIS_IMAGE_MMA_CUH
