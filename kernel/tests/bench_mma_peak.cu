// The tensor-core instruction microbenchmark (spec image-model/02 acceptance 1, GitHub #319) --
// OURS. It decides the FP8 GEMM's inner instruction.
//
// The loops run on register-resident operands, no memory traffic, eight independent accumulator
// chains per warp so the issue rate is measured rather than the latency:
//
//   bf16      mma.m16n8k16 BF16, fp32 accumulate                         (4096 FLOP / instruction)
//   fp8       mma.m16n8k32 kind::f8f6f4 E4M3, fp32 accumulate            (8192)
//   fp8-mx    mma.m16n8k32 kind::mxf8f6f4 E4M3, unit UE8M0 block scales,
//             fp32 accumulate                                            (8192)
//   fp8-f16   mma.m16n8k32 kind::f8f6f4 E4M3, fp16 accumulate (reference: the 838-class rate)
//   int8      mma.m16n8k32 s8.s8, int32 accumulate: the 8-bit attention's QK^T (8192 OP)
//   qk+pv     int8 and fp8-mx alternating, one of each per chain step: the 8-bit attention's mix
//
// Each loop warms the card for --warmup-ms first (the boost clock ramps), then times --reps
// launches with events. Block 0 reads clock64 and %globaltimer around its loop, so every rate is
// printed with the SM clock it ran at; the ratio between two FP8 forms does not depend on it.
//
// Before timing, every form runs once on known operands against an fp64 product (and the two FP8
// fp32 forms against each other, bit for bit), and a set of probes measures how many bits the
// accumulation keeps -- what the GEMM test's error bound may assume. A failed check exits 1.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself.
//
//   ignis_mma_peak_bench [--iters N] [--reps N] [--warmup-ms N]

#include "image/image_mma.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ignis_image;

#define BENCH_CUDA(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

enum Form { kBf16 = 0, kFp8 = 1, kFp8Mx = 2, kFp8F16 = 3, kInt8 = 4, kQkPv = 5 };
constexpr int kForms = 6;
const char *const kFormName[kForms] = {"bf16", "fp8", "fp8-mx", "fp8-f16", "int8", "qk+pv"};
// FLOP (or integer OP) per chain step: one instruction, two for qk+pv.
constexpr double kFlopPerStep[kForms] = {2.0 * 16 * 8 * 16, 2.0 * 16 * 8 * 32, 2.0 * 16 * 8 * 32,
                                         2.0 * 16 * 8 * 32, 2.0 * 16 * 8 * 32, 2.0 * 2.0 * 16 * 8 * 32};
constexpr int kChains = 8;
constexpr int kThreads = 256;

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// Four E4M3 codes of magnitude in [0.5, 2) with random signs and mantissas (exponent field 6..8).
__device__ __forceinline__ uint32_t random_e4m3x4(uint32_t h) {
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t e = 6 + (h >> (8 * i)) % 3;
    const uint32_t byte = ((h >> (8 * i + 2)) & 0x80u) | (e << 3) | ((h >> (8 * i + 4)) & 7u);
    r |= byte << (8 * i);
  }
  return r;
}

// Two BF16 of magnitude in [0.5, 2) with random signs and mantissas.
__device__ __forceinline__ uint32_t random_bf16x2(uint32_t h) {
  uint32_t r = 0;
  for (int i = 0; i < 2; ++i) {
    const uint32_t e = 126 + (h >> (16 * i)) % 2;
    const uint32_t v = ((h >> (16 * i + 1)) & 0x8000u) | (e << 7) | ((h >> (16 * i + 3)) & 0x7Fu);
    r |= v << (16 * i);
  }
  return r;
}

// Four INT8 in [-127, 127], so each one's negation is an INT8 too.
__device__ __forceinline__ uint32_t random_int8x4(uint32_t h) {
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) r |= static_cast<uint32_t>(static_cast<uint8_t>(static_cast<int>((h >> (8 * i)) & 0xFFu) % 255 - 127)) << (8 * i);
  return r;
}

__device__ __forceinline__ uint32_t negate_int8x4(uint32_t v) {
  uint32_t r = 0;
  for (int i = 0; i < 4; ++i) r |= static_cast<uint32_t>(static_cast<uint8_t>(-static_cast<int8_t>((v >> (8 * i)) & 0xFFu))) << (8 * i);
  return r;
}

// Each chain accumulates +A B then -A B, so the accumulators stay bounded however long it runs.
// qk+pv reads the E4M3 operands as INT8 too (its INT8 chains wrap, which does not change the rate).
template <int FORM>
__global__ void __launch_bounds__(kThreads) peak_kernel(int iters, uint32_t seed, float *sink,
                                                        unsigned long long *clocks) {
  const uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t a[4], b[2], bn[2];
  const uint32_t sign = FORM == kBf16 ? 0x80008000u : 0x80808080u;
  for (int i = 0; i < 4; ++i) {
    const uint32_t h = mix(id * 8u + i + seed);
    a[i] = FORM == kBf16 ? random_bf16x2(h) : FORM == kInt8 ? random_int8x4(h) : random_e4m3x4(h);
  }
  for (int i = 0; i < 2; ++i) {
    const uint32_t h = mix(id * 8u + 4 + i + seed);
    b[i] = FORM == kBf16 ? random_bf16x2(h) : FORM == kInt8 ? random_int8x4(h) : random_e4m3x4(h);
    bn[i] = FORM == kInt8 ? negate_int8x4(b[i]) : b[i] ^ sign;
  }
  float acc[kChains][4] = {};
  uint32_t hacc[kChains][2] = {};
  int32_t iacc[kChains][4] = {};
  unsigned long long c0 = 0, g0 = 0;
  if (id == 0) {
    c0 = clock64();
    g0 = globaltimer();
  }
  for (int i = 0; i < iters; ++i) {
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      if (FORM == kBf16) mma_bf16_f32(acc[c], a, b);
      if (FORM == kFp8) mma_e4m3_f32(acc[c], a, b);
      if (FORM == kFp8Mx || FORM == kQkPv) mma_e4m3_mx_unit_f32(acc[c], a, b);
      if (FORM == kFp8F16) mma_e4m3_f16(hacc[c], a, b);
      if (FORM == kInt8 || FORM == kQkPv) mma_s8_s32(iacc[c], a, b);
    }
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      if (FORM == kBf16) mma_bf16_f32(acc[c], a, bn);
      if (FORM == kFp8) mma_e4m3_f32(acc[c], a, bn);
      if (FORM == kFp8Mx || FORM == kQkPv) mma_e4m3_mx_unit_f32(acc[c], a, bn);
      if (FORM == kFp8F16) mma_e4m3_f16(hacc[c], a, bn);
      if (FORM == kInt8 || FORM == kQkPv) mma_s8_s32(iacc[c], a, bn);
    }
  }
  if (id == 0) {
    clocks[0] = clock64() - c0;
    clocks[1] = globaltimer() - g0;
  }
  float s = 0.0f;
  for (int c = 0; c < kChains; ++c) {
    s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    s += __uint_as_float(hacc[c][0]) + __uint_as_float(hacc[c][1]);
    s += static_cast<float>(iacc[c][0] + iacc[c][1] + iacc[c][2] + iacc[c][3]);
  }
  sink[id] = s;
}

// One INT8 instruction on known operands: A [16][32], B^T [8][32] (k contiguous), C and D [16][8]
// int32, fragments read by the PTX ISA's formulas.
__global__ void one_mma_int8_kernel(const int8_t *A, const int8_t *Bt, const int32_t *C, int32_t *D) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  auto word = [](const int8_t *p) { return *reinterpret_cast<const uint32_t *>(p); };
  const uint32_t a[4] = {word(A + g * 32 + 4 * t), word(A + (g + 8) * 32 + 4 * t), word(A + g * 32 + 16 + 4 * t),
                         word(A + (g + 8) * 32 + 16 + 4 * t)};
  const uint32_t b[2] = {word(Bt + g * 32 + 4 * t), word(Bt + g * 32 + 16 + 4 * t)};
  const int rows[4] = {g, g, g + 8, g + 8};
  const int cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
  int32_t d[4];
  for (int i = 0; i < 4; ++i) d[i] = C[rows[i] * 8 + cols[i]];
  mma_s8_s32(d, a, b);
  for (int i = 0; i < 4; ++i) D[rows[i] * 8 + cols[i]] = d[i];
}

// ---- one instruction on known operands ------------------------------------------------------

// One warp: A [16][kb] and B^T [8][kb] (kb = 16 BF16 or 32 E4M3, as raw bytes, k contiguous), C
// [16][8] fp32; D [16][8] fp32. Fragments are read straight from global memory by the PTX ISA's
// formulas, so this checks the layouts the header documents.
template <int FORM>
__global__ void one_mma_kernel(const uint8_t *A, const uint8_t *Bt, const float *C, float *D) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2, t = lane & 3;
  const int row_bytes = 32;  // 16 BF16 or 32 E4M3
  uint32_t a[4], b[2];
  auto word = [](const uint8_t *p) { return *reinterpret_cast<const uint32_t *>(p); };
  const int rs[4] = {g, g + 8, g, g + 8};
  if (FORM == kBf16) {
    const int ks[4] = {2 * t, 2 * t, 2 * t + 8, 2 * t + 8};
    for (int i = 0; i < 4; ++i) a[i] = word(A + rs[i] * row_bytes + 2 * ks[i]);
    b[0] = word(Bt + g * row_bytes + 2 * (2 * t));
    b[1] = word(Bt + g * row_bytes + 2 * (2 * t + 8));
  } else {
    const int ks[4] = {4 * t, 4 * t, 16 + 4 * t, 16 + 4 * t};
    for (int i = 0; i < 4; ++i) a[i] = word(A + rs[i] * row_bytes + ks[i]);
    b[0] = word(Bt + g * row_bytes + 4 * t);
    b[1] = word(Bt + g * row_bytes + 16 + 4 * t);
  }
  const int rows[4] = {g, g, g + 8, g + 8};
  const int cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
  float d[4];
  for (int i = 0; i < 4; ++i) d[i] = C[rows[i] * 8 + cols[i]];
  if (FORM == kBf16) mma_bf16_f32(d, a, b);
  if (FORM == kFp8) mma_e4m3_f32(d, a, b);
  if (FORM == kFp8Mx) mma_e4m3_mx_unit_f32(d, a, b);
  if (FORM == kFp8F16) {
    union {
      __half2 h2[2];
      uint32_t u[2];
    } h;
    h.h2[0] = __floats2half2_rn(d[0], d[1]);
    h.h2[1] = __floats2half2_rn(d[2], d[3]);
    mma_e4m3_f16(h.u, a, b);
    const __half2 rlo = h.h2[0], rhi = h.h2[1];
    d[0] = __low2float(rlo);
    d[1] = __high2float(rlo);
    d[2] = __low2float(rhi);
    d[3] = __high2float(rhi);
  }
  for (int i = 0; i < 4; ++i) D[rows[i] * 8 + cols[i]] = d[i];
}

// Host operands of one instruction: values, and their encodings.
struct Operands {
  int k = 32;                 // 16 (BF16) or 32 (E4M3)
  std::vector<double> a, b;   // a [16][k], b [8][k] (B transposed: column n's k values)
  std::vector<float> c;       // [16][8]
};

uint16_t bf16_bits(double v) {  // exact for the values used here
  const float f = static_cast<float>(v);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>(u >> 16);
}

// E4M3FN encoding of v (v must be representable).
uint8_t e4m3_bits(double v) {
  if (v == 0.0) return 0;
  const uint8_t sign = v < 0 ? 0x80 : 0;
  const double m = std::fabs(v);
  for (int code = 1; code < 0x7F; ++code) {
    const int e = code >> 3, man = code & 7;
    const double val = e == 0 ? std::ldexp(man, -9) : std::ldexp(1.0 + man / 8.0, e - 7);
    if (val == m) return static_cast<uint8_t>(sign | code);
  }
  std::fprintf(stderr, "FATAL: %g is not an E4M3 value\n", v);
  std::exit(EXIT_FAILURE);
}

template <int FORM> std::vector<float> run_one(const Operands &op) {
  std::vector<uint8_t> ha(16 * 32, 0), hb(8 * 32, 0);
  for (int r = 0; r < 16; ++r) {
    for (int k = 0; k < op.k; ++k) {
      if (FORM == kBf16) {
        const uint16_t v = bf16_bits(op.a[r * op.k + k]);
        std::memcpy(&ha[r * 32 + 2 * k], &v, 2);
      } else {
        ha[r * 32 + k] = e4m3_bits(op.a[r * op.k + k]);
      }
    }
  }
  for (int n = 0; n < 8; ++n) {
    for (int k = 0; k < op.k; ++k) {
      if (FORM == kBf16) {
        const uint16_t v = bf16_bits(op.b[n * op.k + k]);
        std::memcpy(&hb[n * 32 + 2 * k], &v, 2);
      } else {
        hb[n * 32 + k] = e4m3_bits(op.b[n * op.k + k]);
      }
    }
  }
  uint8_t *da, *db;
  float *dc, *dd;
  BENCH_CUDA(cudaMalloc(&da, ha.size()));
  BENCH_CUDA(cudaMalloc(&db, hb.size()));
  BENCH_CUDA(cudaMalloc(&dc, 128 * 4));
  BENCH_CUDA(cudaMalloc(&dd, 128 * 4));
  BENCH_CUDA(cudaMemcpy(da, ha.data(), ha.size(), cudaMemcpyHostToDevice));
  BENCH_CUDA(cudaMemcpy(db, hb.data(), hb.size(), cudaMemcpyHostToDevice));
  BENCH_CUDA(cudaMemcpy(dc, op.c.data(), 128 * 4, cudaMemcpyHostToDevice));
  one_mma_kernel<FORM><<<1, 32>>>(da, db, dc, dd);
  BENCH_CUDA(cudaGetLastError());
  std::vector<float> d(128);
  BENCH_CUDA(cudaMemcpy(d.data(), dd, 128 * 4, cudaMemcpyDeviceToHost));
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
  return d;
}

std::vector<float> run_form(int form, const Operands &op) {
  switch (form) {
  case kBf16: return run_one<kBf16>(op);
  case kFp8: return run_one<kFp8>(op);
  case kFp8Mx: return run_one<kFp8Mx>(op);
  default: return run_one<kFp8F16>(op);
  }
}

int g_failed = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
    ++g_failed;
  }
}

uint32_t lcg(uint32_t &s) {
  s = s * 1664525u + 1013904223u;
  return s >> 8;
}

// Random operands: values exact in both encodings' ranges (E4M3 [0.5, 2) mantissa 3 bits; for
// BF16 the same values), C random fp32 small.
Operands random_operands(int k, uint32_t seed) {
  Operands op;
  op.k = k;
  op.a.resize(16 * k);
  op.b.resize(8 * k);
  op.c.resize(128);
  auto value = [&]() {
    const double man = 1.0 + (lcg(seed) % 8) / 8.0;
    const int e = static_cast<int>(lcg(seed) % 3) - 1;
    return (lcg(seed) & 1 ? -1.0 : 1.0) * std::ldexp(man, e);
  };
  for (auto &v : op.a) v = value();
  for (auto &v : op.b) v = value();
  for (auto &v : op.c) v = static_cast<float>(value());
  return op;
}

// D = A B + C against fp64: the largest |error| / (|C| + sum |terms|) over the tile.
double worst_relative(const Operands &op, const std::vector<float> &d) {
  double worst = 0.0;
  for (int r = 0; r < 16; ++r) {
    for (int n = 0; n < 8; ++n) {
      double s = op.c[r * 8 + n], mag = std::fabs(op.c[r * 8 + n]);
      for (int k = 0; k < op.k; ++k) {
        const double p = op.a[r * op.k + k] * op.b[n * op.k + k];
        s += p;
        mag += std::fabs(p);
      }
      worst = std::max(worst, std::fabs(d[r * 8 + n] - s) / mag);
    }
  }
  return worst;
}

// Accumulation probes on element (0, 0), with 2^8 as the large magnitude and 2^(8-e) the small:
//   P1  C = 2^8, one product 2^(8-e)            the accumulator input plus a product
//   P2  C = 0, products 2^8 and 2^(8-e)         two products inside one instruction
//   P3  C = 0, 2^8 and k-1 products of 2^(8-e)  one large and many small products
//   P4  C = 2^8, k products of 2^(8-e)          the accumulator input and many small products
// The result is the largest e such that every e' <= e came out exact (fp32 keeps 23).
Operands probe(int which, int k, int e) {
  Operands op;
  op.k = k;
  op.a.assign(16 * k, 0.0);
  op.b.assign(8 * k, 0.0);
  op.c.assign(128, 0.0f);
  const double small_a = std::ldexp(1.0, (8 - e) / 2), small_b = std::ldexp(1.0, (8 - e) - (8 - e) / 2);
  auto put = [&](int kk, double av, double bv) {
    op.a[kk] = av;
    op.b[kk] = bv;
  };
  switch (which) {
  case 1:
    op.c[0] = 256.0f;
    put(0, small_a, small_b);
    break;
  case 2:
    put(0, 16.0, 16.0);
    put(1, small_a, small_b);
    break;
  case 3:
    put(0, 16.0, 16.0);
    for (int kk = 1; kk < k; ++kk) put(kk, small_a, small_b);
    break;
  default:
    op.c[0] = 256.0f;
    for (int kk = 0; kk < k; ++kk) put(kk, small_a, small_b);
    break;
  }
  return op;
}

int exact_bits(int form, int which) {
  const int k = form == kBf16 ? 16 : 32;
  // E4M3's smallest product is 2^-18 (two subnormals of 2^-9): e reaches 26.
  const int max_e = form == kBf16 ? 30 : 26;
  int best = 0;
  for (int e = 1; e <= max_e; ++e) {
    const Operands op = probe(which, k, e);
    const std::vector<float> d = run_form(form, op);
    double expect = op.c[0];
    for (int kk = 0; kk < k; ++kk) expect += op.a[kk] * op.b[kk];
    if (static_cast<double>(d[0]) != expect) break;
    best = e;
  }
  return best;
}

void semantics() {
  std::printf("instruction semantics (one mma on known operands, fp64 reference)\n");
  for (int form = 0; form < 4; ++form) {
    const int k = form == kBf16 ? 16 : 32;
    double worst = 0.0;
    for (uint32_t seed = 1; seed <= 8; ++seed) {
      const Operands op = random_operands(k, seed * 7919u);
      worst = std::max(worst, worst_relative(op, run_form(form, op)));
    }
    // The check is the layouts': a fragment read from the wrong place is an O(1) error. How close
    // to fp64 each form gets is a finding (printed), bounded below by the accumulation probes.
    std::printf("  %-8s worst |D - fp64| / (|C| + sum|AB|) = %.3g (2^%.1f)\n", kFormName[form], worst,
                worst > 0 ? std::log2(worst) : -99.0);
    check(worst <= std::ldexp(1.0, -8), std::string(kFormName[form]) + ": one instruction matches fp64 (layouts)");
  }
  bool same = true;
  for (uint32_t seed = 1; seed <= 8; ++seed) {
    const Operands op = random_operands(32, seed * 104729u);
    same = same && run_form(kFp8, op) == run_form(kFp8Mx, op);
  }
  std::printf("  fp8 and fp8-mx (unit scales) agree bit for bit: %s\n", same ? "yes" : "NO");
  check(same, "fp8-mx with unit scales equals fp8, bit for bit");
  std::printf("accumulation probes: largest e with every e' <= e exact (fp32 keeps 23)\n");
  std::printf("  %-8s %4s %4s %4s %4s\n", "form", "P1", "P2", "P3", "P4");
  for (int form = 0; form < 3; ++form) {
    std::printf("  %-8s %4d %4d %4d %4d\n", kFormName[form], exact_bits(form, 1), exact_bits(form, 2),
                exact_bits(form, 3), exact_bits(form, 4));
  }
}

// INT8 against the exact integer product: random operands over the whole INT8 range, and the
// extremes (-128 x -128 in every product, on a large C).
void int8_semantics() {
  std::vector<int8_t> a(16 * 32), b(8 * 32);
  std::vector<int32_t> c(128), d(128);
  int8_t *da, *db;
  int32_t *dc, *dd;
  BENCH_CUDA(cudaMalloc(&da, a.size()));
  BENCH_CUDA(cudaMalloc(&db, b.size()));
  BENCH_CUDA(cudaMalloc(&dc, 128 * 4));
  BENCH_CUDA(cudaMalloc(&dd, 128 * 4));
  bool exact = true;
  uint32_t seed = 31337u;
  for (int trial = 0; trial < 9; ++trial) {
    const bool extreme = trial == 8;
    for (auto &v : a) v = extreme ? -128 : static_cast<int8_t>(lcg(seed) & 0xFF);
    for (auto &v : b) v = extreme ? -128 : static_cast<int8_t>(lcg(seed) & 0xFF);
    for (auto &v : c) v = extreme ? 0x7FFFFFFF - 32 * 16384 : static_cast<int32_t>(lcg(seed)) - (1 << 23);
    BENCH_CUDA(cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice));
    BENCH_CUDA(cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice));
    BENCH_CUDA(cudaMemcpy(dc, c.data(), 128 * 4, cudaMemcpyHostToDevice));
    one_mma_int8_kernel<<<1, 32>>>(da, db, dc, dd);
    BENCH_CUDA(cudaGetLastError());
    BENCH_CUDA(cudaMemcpy(d.data(), dd, 128 * 4, cudaMemcpyDeviceToHost));
    for (int r = 0; r < 16; ++r) {
      for (int n = 0; n < 8; ++n) {
        int64_t s = c[r * 8 + n];
        for (int k = 0; k < 32; ++k) s += static_cast<int64_t>(a[r * 32 + k]) * b[n * 32 + k];
        exact = exact && s == d[r * 8 + n];
      }
    }
  }
  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  cudaFree(dd);
  std::printf("  %-8s one instruction equals the exact integer product (random, and -128^2 x 32 on 2^31-1 - 2^19): %s\n",
              "int8", exact ? "yes" : "NO");
  check(exact, "int8: one instruction is exact");
}

// ---- timing ---------------------------------------------------------------------------------

template <int FORM> void launch_peak(int blocks, int iters, float *sink, unsigned long long *clocks) {
  peak_kernel<FORM><<<blocks, kThreads>>>(iters, 12345u, sink, clocks);
}

void launch_form(int form, int blocks, int iters, float *sink, unsigned long long *clocks) {
  switch (form) {
  case kBf16: launch_peak<kBf16>(blocks, iters, sink, clocks); break;
  case kFp8: launch_peak<kFp8>(blocks, iters, sink, clocks); break;
  case kFp8Mx: launch_peak<kFp8Mx>(blocks, iters, sink, clocks); break;
  case kFp8F16: launch_peak<kFp8F16>(blocks, iters, sink, clocks); break;
  case kInt8: launch_peak<kInt8>(blocks, iters, sink, clocks); break;
  default: launch_peak<kQkPv>(blocks, iters, sink, clocks); break;
  }
}

struct Rate {
  double tflops = 0.0, mhz = 0.0;
};

Rate time_form(int form, int blocks, int iters, int reps, int warmup_ms, float *sink, unsigned long long *clocks) {
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  // Warm-up: launches until warmup_ms of this loop have run, so the clock has ramped.
  float spent = 0.0f;
  while (spent < static_cast<float>(warmup_ms)) {
    BENCH_CUDA(cudaEventRecord(e0));
    launch_form(form, blocks, iters, sink, clocks);
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    spent += ms;
  }
  BENCH_CUDA(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) launch_form(form, blocks, iters, sink, clocks);
  BENCH_CUDA(cudaEventRecord(e1));
  BENCH_CUDA(cudaEventSynchronize(e1));
  BENCH_CUDA(cudaGetLastError());
  float ms = 0.0f;
  BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
  unsigned long long hc[2];
  BENCH_CUDA(cudaMemcpy(hc, clocks, sizeof(hc), cudaMemcpyDeviceToHost));
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  const double mmas = static_cast<double>(blocks) * (kThreads / 32) * iters * 2.0 * kChains * reps;
  Rate rate;
  rate.tflops = mmas * kFlopPerStep[form] / (ms * 1e-3) / 1e12;
  rate.mhz = hc[1] > 0 ? static_cast<double>(hc[0]) / static_cast<double>(hc[1]) * 1e3 : 0.0;
  return rate;
}

int arg_int(int argc, char **argv, const char *name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = arg_int(argc, argv, "--iters", 8192);
  const int reps = arg_int(argc, argv, "--reps", 10);
  const int warmup_ms = arg_int(argc, argv, "--warmup-ms", 300);
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("%s, %d SMs, sm_%d%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
  semantics();
  int8_semantics();
  const int blocks = prop.multiProcessorCount * 4;
  float *sink;
  unsigned long long *clocks;
  BENCH_CUDA(cudaMalloc(&sink, static_cast<std::size_t>(blocks) * kThreads * sizeof(float)));
  BENCH_CUDA(cudaMalloc(&clocks, 2 * sizeof(unsigned long long)));
  std::printf("instruction rate (%d blocks x %d threads, %d chains, %d iterations, %d timed launches)\n", blocks,
              kThreads, kChains, iters, reps);
  Rate rates[kForms];
  for (int form = 0; form < kForms; ++form) {
    rates[form] = time_form(form, blocks, iters, reps, warmup_ms, sink, clocks);
    std::printf("  %-8s %8.1f TFLOPS at %6.0f MHz  (%.1f TFLOPS/GHz)\n", kFormName[form], rates[form].tflops,
                rates[form].mhz, rates[form].tflops / (rates[form].mhz * 1e-3));
  }
  auto per_clock = [&](int form) { return rates[form].tflops / rates[form].mhz; };
  std::printf("  fp8-mx / fp8 = %.2f (clock-adjusted %.2f)\n", rates[kFp8Mx].tflops / rates[kFp8].tflops,
              per_clock(kFp8Mx) / per_clock(kFp8));
  std::printf("  int8 / fp8-mx = %.2f, qk+pv / fp8-mx = %.2f (clock-adjusted)\n", per_clock(kInt8) / per_clock(kFp8Mx),
              per_clock(kQkPv) / per_clock(kFp8Mx));
  cudaFree(sink);
  cudaFree(clocks);
  if (g_failed != 0) {
    std::fprintf(stderr, "ignis_mma_peak_bench: %d check(s) failed\n", g_failed);
    return 1;
  }
  return 0;
}
