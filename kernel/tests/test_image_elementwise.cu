// The image model's elementwise ops (spec image-model/02 acceptance 7, GitHub #319) -- OURS. The
// ops are kernel/include/ignis_image_dit.h's and ignis_image_vae.h's ignis_image_vae_rms_norm;
// every reference is our own fp64 CPU code, except the RoPE oracle's, which is diffusers' own
// pos_embed (spec 01's recordings).
//
//   rope        the frequencies: PyTorch's fp32 formula restated, and the one pair where torch's
//               value is not it; the table op against the host's table at every position of
//               diffusers' range [-1024, 8192) on every axis, within one ulp; the positions op
//               against diffusers' QwenImage21Rope.forward restated in its mask form (text before,
//               between and after blocks, adjacent blocks, ten condition images and the preset's
//               target), a layout worked by hand, and its refusals.
//   modnorm     the modulated LayerNorm over 4096 columns, 1 .. 16,512 tokens, with the t2i and
//               edit layouts' prefix / target split.
//   rmsnorm     the zero-centred RMSNorm (txt_in's) over 4096 columns.
//   act         SiLU and GELU-tanh over 4096 columns, SwiGLU over 12288 from separate gate and up
//               rows and from the fused [tokens][24576] output.
//   residual    the gated residual over 4096 columns, in place and not.
//   qk          the QK RMSNorm with RoPE in place on a fused [tokens][12288] QKV row, with the
//               table the ops build for the t2i and edit layouts and a 128 x 128 image.
//   vae         the VAE channel norm, with and without SiLU, at every channel count of the VAE.
//   refusals    the ops' argument checks.
//   rope-oracle (not in the default list) the positions and table ops on the oracle's t2i and edit
//               layouts against its rope.npy, within one ulp.
//
// Each op is checked these ways, as the other image ops' tests do:
//   mechanics  the fp32 output against the fp64 reference at sampled rows, every column, within
//              the bound below;
//   BF16       the BF16 output is the fp32 output rounded, bit for bit, over the whole output (and
//              the in-place forms equal the out-of-place ones);
//   E4M3       for the ops that feed an FP8 GEMM, the codes and scales are what
//              ignis_image_quantize_tokens makes of the BF16 output, bit for bit;
//   NaN        the outputs start as NaN, so an unwritten output shows.
// The modulated norm and the gated residual also check the per-token row choice: in a [prefix ;
// target] call the target's rows equal the all-t call's and the prefix's the all-t0 call's, bit for
// bit, and the reference with the two rows swapped fails (a negative control).
//
// The bound. The reference follows the kernel's operations in its order, tracking error (Err
// below): each value v (exact, in fp64) carries e, a bound on how far the kernel's fp32 value can
// be from it. Every fp32 operation rounded to nearest adds 2^-24 of its result's magnitude (and
// 2^-149 for underflow); expf, tanhf and rsqrtf add 2 ulp (CUDA Math API, no fast math). An input's
// error propagates through the operation's own sensitivity: exp, v (e^e - 1); tanh, e times sech^2
// at the end of the interval nearest 0; rsqrt, e / 2 (v - e)^-3/2; a quotient, (e_a + |q| e_b) /
// (|b| - e_b). A sum in a fixed tree whose terms each go through at most d additions adds gamma_d
// = d 2^-24 / (1 - d 2^-24) of sum |terms|: d is each thread's element count, plus the butterfly's
// 5 levels, plus the 8 warps of a CTA. The bound is multiplied by --bound-factor F (default 1).
//
//   ignis_kernel_image_elementwise_test [--bound-factor F]
//                                       [--cases rope,modnorm,rmsnorm,act,residual,qk,vae,refusals]
//   ignis_kernel_image_elementwise_test --cases rope-oracle [--oracle DIR]
//
// rope-oracle reads DIR, else $IGNIS_IMAGE_ORACLE, else F:/ai/models/ignis-image-2.1-turbo/oracle
// (spec 01's recordings live outside the repo): with no oracle there it prints SKIP and passes, as
// the 8-bit attention's qkv-probe case does (the coordinator's call for #319).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_dit.h"
#include "ignis_image_gemm.h"
#include "ignis_image_vae.h"
#include "moe_fixture.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace moe_test;

namespace {

struct Options {
  double bound_factor = 1.0;
};

#define IMG_RC(expr, last_error)                                                                   \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, last_error());               \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)
#define DIT_RC(expr) IMG_RC(expr, ignis_image_dit_last_error)
#define VAE_RC(expr) IMG_RC(expr, ignis_image_vae_last_error)

constexpr uint32_t kBf16 = IGNIS_IMAGE_DIT_OUT_BF16, kF32 = IGNIS_IMAGE_DIT_OUT_F32, kE4m3 = IGNIS_IMAGE_DIT_OUT_E4M3;
constexpr float kEps = 1e-6f;  // every norm of the DiT (txt_in passes 1e-6 to its ZeroCenterRMSNorm too)

// ---- the error-tracking reference -------------------------------------------------------------

constexpr double kU = 0x1p-24;       // fp32 rounding to nearest: |fl(x) - x| <= 2^-24 |x|
constexpr double kTiny = 0x1p-149;   // ... plus half of fp32's subnormal spacing, twice over
constexpr double kUlp2 = 0x1p-22;    // 2 ulp: ulp(x) <= 2^-23 |x|
constexpr double kRefSlack = 0x1p-40;  // the fp64 reference's own roundings

struct Err {
  double v, e;  // the exact value, and how far the kernel's fp32 value can be from it
};

Err exact(double v) { return {v, 0.0}; }
// An fp32 constant: its rounding from the definition's value is its whole error.
Err constant(double v) { return {v, std::fabs(static_cast<double>(static_cast<float>(v)) - v)}; }
// One rounding to nearest of an operation whose exact result on the kernel's inputs is within e of v.
Err rounded(double v, double e) { return {v, e + kU * (std::fabs(v) + e) + kTiny}; }
Err within2ulp(double v, double e) { return {v, e + kUlp2 * (std::fabs(v) + e) + kTiny}; }
Err add(Err a, Err b) { return rounded(a.v + b.v, a.e + b.e); }
Err sub(Err a, Err b) { return rounded(a.v - b.v, a.e + b.e); }
Err neg(Err a) { return {-a.v, a.e}; }
Err mul(Err a, Err b) { return rounded(a.v * b.v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e); }
Err fma(Err a, Err b, Err c) {
  return rounded(a.v * b.v + c.v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e + c.e);
}
Err div(Err a, Err b) {  // |b.v| > b.e
  const double q = a.v / b.v;
  return rounded(q, (a.e + std::fabs(q) * b.e) / (std::fabs(b.v) - b.e));
}
Err exp_f(Err a) {
  const double v = std::exp(a.v);
  return within2ulp(v, v * std::expm1(a.e));
}
Err tanh_f(Err a) {
  const double s = 1.0 / std::cosh(std::max(0.0, std::fabs(a.v) - a.e));
  return within2ulp(std::tanh(a.v), a.e * s * s);
}
Err rsqrt_f(Err a) {  // a.v > a.e
  const double v = 1.0 / std::sqrt(a.v);
  return within2ulp(v, 0.5 * a.e * std::pow(a.v - a.e, -1.5));
}
Err sqrt_f(Err a) {  // correctly rounded (sqrtf, __fsqrt_rn)
  const double v = std::sqrt(a.v);
  return rounded(v, a.v > a.e ? a.e / (std::sqrt(a.v - a.e) + v) : std::sqrt(a.e));
}
// A sum whose terms each go through at most `depth` fp32 additions (or fmas), in any tree.
Err sum(const std::vector<Err> &t, int depth) {
  double v = 0.0, e = 0.0, mag = 0.0;
  for (const Err &x : t) {
    v += x.v;
    e += x.e;
    mag += std::fabs(x.v) + x.e;
  }
  return {v, e + depth * kU / (1.0 - depth * kU) * mag};
}
// The square of a computed value, exact as an fma's product is.
Err square(Err a) { return {a.v * a.v, 2.0 * std::fabs(a.v) * a.e + a.e * a.e}; }

Err silu_ref(Err x) { return div(x, add(exact(1.0), exp_f(neg(x)))); }

Err gelu_ref(Err x) {
  const Err x3 = mul(mul(x, x), x);
  const Err two_z = mul(constant(1.5957691216057308), fma(constant(0.044715), x3, x));
  return div(x, add(exact(1.0), exp_f(neg(two_z))));
}

// The row kernels' summation depth: each thread's elements (8 per 16-byte chunk, chunks tid, tid +
// 256, ...), the warp butterfly, the 8 warps.
int row_depth(uint32_t cols) { return 8 * static_cast<int>((cols / 8 + 255) / 256) + 5 + 8; }

struct Verdict {
  long long checked = 0, over = 0;
  double worst = 0.0;
  void take(double got, const Err &r, double factor) {
    const double bound = r.e * factor + kRefSlack * (std::fabs(r.v) + r.e);
    const double d = std::fabs(got - r.v);
    ++checked;
    if (!(d <= bound)) ++over;
    if (bound > 0.0) worst = std::max(worst, d / bound);
  }
};

void report(const std::string &what, const Verdict &v) {
  std::printf("  %-58s %9lld checked, worst %.3f of the bound\n", what.c_str(), v.checked, v.worst);
  check(v.checked > 0 && v.over == 0, what + ": " + std::to_string(v.over) + " outputs over the bound");
}

// ---- inputs, the same on the host and the device ------------------------------------------------

__host__ __device__ inline uint32_t mix32(uint32_t x) {  // moe_fixture.h's lowbias32
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

__host__ __device__ inline float hash_unit(uint32_t stream, uint64_t i) {  // [-1, 1)
  const uint32_t h = mix32(static_cast<uint32_t>(i) * 0x9E3779B9u + stream * 0x85EBCA6Bu);
  return static_cast<float>(h >> 8) * 0x1p-23f - 1.0f;
}

__host__ __device__ inline uint16_t bf16_bits(float v) {  // round to nearest even (finite)
#ifdef __CUDA_ARCH__
  uint32_t b = __float_as_uint(v);
#else
  uint32_t b;
  std::memcpy(&b, &v, 4);
#endif
  b += 0x7FFFu + ((b >> 16) & 1u);
  return static_cast<uint16_t>(b >> 16);
}

// Input families. Row r's element c (of `cols` per row), BF16:
//   kNormIn   rows a norm meets (r % 8): plain, tiny, large, an offset, a large offset with a small
//             spread (the mean cancels), outlier channels, all zeros, a constant;
//   kActIn    activation inputs (r % 4): amplitude 1, 3, 8, 0.01;
//   kPlainIn  uniform in [-1, 1).
// Each times `amplitude`.
enum Family : int { kNormIn, kActIn, kPlainIn };

__host__ __device__ inline uint16_t input_bf16(int family, uint32_t stream, uint64_t r, uint64_t c, uint64_t cols,
                                               float amplitude) {
  const float u = hash_unit(stream, r * cols + c);
  float v = u;
  if (family == kNormIn) {
    switch (r % 8) {
      case 1: v = u * 0x1p-6f; break;
      case 2: v = u * 40.0f; break;
      case 3: v = 3.0f + u; break;
      case 4: v = -96.0f + 0.75f * u; break;
      case 5: v = c % 97 == 5 ? 64.0f * u : u; break;
      case 6: v = 0.0f; break;
      case 7: v = 2.5f; break;
      default: break;
    }
  } else if (family == kActIn) {
    const float amp[4] = {1.0f, 3.0f, 8.0f, 0.01f};
    v = amp[r % 4] * u;
  }
  return bf16_bits(v * amplitude);
}

__global__ void fill_kernel(uint16_t *p, std::size_t rows, std::size_t cols, int family, uint32_t stream,
                            float amplitude) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < rows * cols;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    p[e] = input_bf16(family, stream, e / cols, e % cols, cols, amplitude);
  }
}

void fill(const DeviceBytes &d, std::size_t rows, std::size_t cols, int family, uint32_t stream, float amplitude) {
  fill_kernel<<<2048, 256>>>(d.as<uint16_t>(), rows, cols, family, stream, amplitude);
  MOE_CUDA(cudaGetLastError());
}

double host_in(int family, uint32_t stream, uint64_t r, uint64_t c, uint64_t cols, float amplitude) {
  return bf16_to_f32(input_bf16(family, stream, r, c, cols, amplitude));
}

std::vector<float> host_row(uint32_t stream, std::size_t n, float lo, float hi) {  // fp32 rows (modulation)
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = lo + (hi - lo) * 0.5f * (hash_unit(stream, i) + 1.0f);
  return v;
}

std::vector<uint16_t> host_bf16_row(uint32_t stream, std::size_t n, float centre, float spread) {  // weights
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = bf16_bits(centre + spread * hash_unit(stream, i));
  return v;
}

// A device buffer of all-0xFF bytes (NaN in fp32 and BF16): an unwritten output shows.
struct Poisoned : DeviceBytes {
  explicit Poisoned(std::size_t bytes) : DeviceBytes(bytes) { MOE_CUDA(cudaMemset(p, 0xFF, bytes)); }
};

// ---- device comparisons -----------------------------------------------------------------------

__device__ unsigned long long g_count;

// Counts outputs whose BF16 is not the fp32 rounded (or either is NaN), rows x cols at strides.
__global__ void bf16_vs_f32_kernel(const float *y32, std::size_t ld32, const uint16_t *y16, std::size_t ld16,
                                   std::size_t rows, std::size_t cols) {
  unsigned long long diff = 0;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < rows * cols;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t r = e / cols, c = e % cols;
    const float v = y32[r * ld32 + c];
    diff += v != v || __bfloat16_as_ushort(__float2bfloat16_rn(v)) != y16[r * ld16 + c];
  }
  if (diff != 0) atomicAdd(&g_count, diff);
}

// Counts differing 16-bit words, rows x cols at strides.
__global__ void diff_u16_kernel(const uint16_t *a, std::size_t lda, const uint16_t *b, std::size_t ldb,
                                std::size_t rows, std::size_t cols) {
  unsigned long long diff = 0;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < rows * cols;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const std::size_t r = e / cols, c = e % cols;
    diff += a[r * lda + c] != b[r * ldb + c];
  }
  if (diff != 0) atomicAdd(&g_count, diff);
}

template <class Launch> unsigned long long count_on_device(Launch &&launch) {
  const unsigned long long zero = 0;
  MOE_CUDA(cudaMemcpyToSymbol(g_count, &zero, 8));
  launch();
  MOE_CUDA(cudaGetLastError());
  MOE_CUDA(cudaDeviceSynchronize());
  unsigned long long r = 0;
  MOE_CUDA(cudaMemcpyFromSymbol(&r, g_count, 8));
  return r;
}

unsigned long long bf16_vs_f32(const void *y32, std::size_t ld32, const void *y16, std::size_t ld16, std::size_t rows,
                               std::size_t cols) {
  return count_on_device([&] {
    bf16_vs_f32_kernel<<<2048, 256>>>(static_cast<const float *>(y32), ld32, static_cast<const uint16_t *>(y16), ld16,
                                      rows, cols);
  });
}

unsigned long long diff_u16(const void *a, std::size_t lda, const void *b, std::size_t ldb, std::size_t rows,
                            std::size_t cols) {
  return count_on_device([&] {
    diff_u16_kernel<<<2048, 256>>>(static_cast<const uint16_t *>(a), lda, static_cast<const uint16_t *>(b), ldb, rows,
                                   cols);
  });
}

// The E4M3 output against ignis_image_quantize_tokens of the BF16 output, bit for bit.
void check_e4m3(const std::string &what, const DeviceBytes &y16, const DeviceBytes &codes, const DeviceBytes &scales,
                uint32_t tokens, uint32_t cols) {
  DeviceBytes ref_codes(static_cast<std::size_t>(tokens) * cols), ref_scales(static_cast<std::size_t>(tokens) * 4);
  IMG_RC(ignis_image_quantize_tokens(y16.p, tokens, cols, ref_codes.p, ref_scales.as<float>(), nullptr),
         ignis_image_gemm_last_error);
  MOE_CUDA(cudaDeviceSynchronize());
  const auto dc = diff_u16(codes.p, cols / 2, ref_codes.p, cols / 2, tokens, cols / 2);
  const auto ds = diff_u16(scales.p, 2, ref_scales.p, 2, tokens, 2);
  check(dc == 0 && ds == 0, what + ": E4M3 differs from ignis_image_quantize_tokens of the BF16 output (" +
                                std::to_string(dc) + " code pairs, " + std::to_string(ds) + " scale halves)");
}

void check_bf16(const std::string &what, const DeviceBytes &y32, const DeviceBytes &y16, std::size_t rows,
                std::size_t cols) {
  const auto n = bf16_vs_f32(y32.p, cols, y16.p, cols, rows, cols);
  check(n == 0, what + ": " + std::to_string(n) + " BF16 outputs are not the fp32 output rounded (or NaN)");
}

// The smallest odd stride >= (count - 1) / (want - 1), rounded up, coprime with count: i * stride
// mod count for i < want are distinct and reach the end of the range (test_image_vae.cu's).
long long span_stride(long long count, int want) {
  long long stride = std::max(1LL, (count - 1 + want - 2) / (want - 1));
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, count) != 1) stride += 2;
  return stride;
}

// Up to `want` rows spread over [0, count), plus `extra` (boundary rows) inside it.
std::vector<uint32_t> sample_rows(uint32_t count, int want, std::vector<uint32_t> extra = {}) {
  std::vector<uint32_t> rows;
  if (count <= static_cast<uint32_t>(want)) {
    for (uint32_t r = 0; r < count; ++r) rows.push_back(r);
  } else {
    const long long stride = span_stride(count, want);
    for (int i = 0; i < want; ++i) rows.push_back(static_cast<uint32_t>(i * stride % count));
  }
  for (uint32_t r : extra) {
    if (r < count) rows.push_back(r);
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

std::vector<float> download_row_f32(const DeviceBytes &d, std::size_t row, std::size_t ld, std::size_t cols) {
  std::vector<float> v(cols);
  MOE_CUDA(cudaMemcpy(v.data(), d.as<float>() + row * ld, cols * 4, cudaMemcpyDeviceToHost));
  return v;
}

double arg_double(int argc, char **argv, const char *name, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
  }
  return fallback;
}

// ---- RoPE ----------------------------------------------------------------------------------------

constexpr int kRopePairs = 64;

// PyTorch's fp32 frequency for pair i of an axis of d channels (diffusers' rope_params: 1.0 /
// torch.pow(10000, arange(0, d, 2).float() / d)): fl32(1 / fl32(10000^fl32(2i / d))), each step
// correctly rounded, except (d = 56, i = 11), where torch's powf rounds the other way and its value
// is one ulp up. Measured against torch 2.13 on the CPU, where diffusers computes the table.
float formula_freq(int d, int i) {
  const float e = static_cast<float>(2 * i) / static_cast<float>(d);
  const float p = static_cast<float>(std::pow(10000.0, static_cast<double>(e)));
  return static_cast<float>(1.0 / static_cast<double>(p));
}
constexpr float kTorchFreq56At11 = 0x1.b7886ep-6f;

float torch_freq(int d, int i) { return d == 56 && i == 11 ? kTorchFreq56At11 : formula_freq(d, i); }

// Pair p (0..63) of the table: its axis (0 frame, 1 height, 2 width) and frequency.
int pair_axis(int p) { return p < 8 ? 0 : (p < 36 ? 1 : 2); }
float pair_freq(int p) { return p < 8 ? torch_freq(16, p) : torch_freq(56, (p - 8) % 28); }

// The table on the host: angle fl32(position * f), then cos and sin in fp64 rounded to fp32.
void host_table_row(const int32_t *pos, float *out) {
  for (int p = 0; p < kRopePairs; ++p) {
    const float angle = static_cast<float>(static_cast<double>(pos[pair_axis(p)]) * static_cast<double>(pair_freq(p)));
    out[2 * p] = static_cast<float>(std::cos(static_cast<double>(angle)));
    out[2 * p + 1] = static_cast<float>(std::sin(static_cast<double>(angle)));
  }
}

// The larger of the two values' ulps: a one-ulp difference is within it.
double ulp_of(float a, float b) {
  const float m = std::max(std::fabs(a), std::fabs(b));
  return std::max(static_cast<double>(std::nextafter(m, std::numeric_limits<float>::infinity()) - m), 0x1p-149);
}

struct TableDiff {
  long long equal = 0, one_ulp = 0, worse = 0;
  double worst_abs = 0.0;
};

TableDiff compare_tables(const std::vector<float> &got, const std::vector<float> &want) {
  TableDiff d;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(got[i]) - want[i]);
    d.worst_abs = std::max(d.worst_abs, diff);
    if (got[i] == want[i]) {
      ++d.equal;
    } else if (diff <= ulp_of(got[i], want[i])) {
      ++d.one_ulp;
    } else {
      ++d.worse;
    }
  }
  return d;
}

// diffusers' QwenImage21Rope.forward restated from the image-token mask and the blocks' (h, w):
// each block starts at the first image token after the previous one ends.
std::vector<int32_t> positions_from_mask(const std::vector<uint8_t> &mask, const std::vector<std::pair<int, int>> &hw) {
  std::vector<int64_t> frame, hi, wi;
  std::size_t cursor = 0;
  int64_t position = 0;
  for (const auto &[h, w] : hw) {
    std::size_t start = cursor;
    while (!mask[start]) ++start;
    for (std::size_t i = cursor; i < start; ++i) frame.push_back(position++);
    cursor = start + static_cast<std::size_t>(h) * w;
    for (int i = 0; i < h * w; ++i) frame.push_back(position);
    position += std::max(h, w);
    for (int y = -(h - h / 2); y < h / 2; ++y) {
      for (int x = -(w - w / 2); x < w / 2; ++x) {
        hi.push_back(y);
        wi.push_back(x);
      }
    }
  }
  while (frame.size() < mask.size()) frame.push_back(position++);
  std::vector<int32_t> pos(3 * mask.size());
  std::size_t k = 0;
  for (std::size_t t = 0; t < mask.size(); ++t) {
    pos[3 * t] = static_cast<int32_t>(frame[t]);
    pos[3 * t + 1] = static_cast<int32_t>(mask[t] ? hi[k] : frame[t]);
    pos[3 * t + 2] = static_cast<int32_t>(mask[t] ? wi[k] : frame[t]);
    k += mask[t];
  }
  return pos;
}

// A layout: text runs and image blocks in order. `text_before[b]` text tokens precede block b;
// `text_after` follow the last block.
struct Layout {
  std::string name;
  std::vector<uint32_t> text_before;
  std::vector<std::pair<int, int>> hw;
  uint32_t text_after = 0;
  uint32_t tokens() const {
    uint32_t n = text_after;
    for (std::size_t b = 0; b < hw.size(); ++b) n += text_before[b] + hw[b].first * hw[b].second;
    return n;
  }
  std::vector<uint32_t> starts() const {
    std::vector<uint32_t> s;
    uint32_t at = 0;
    for (std::size_t b = 0; b < hw.size(); ++b) {
      at += text_before[b];
      s.push_back(at);
      at += hw[b].first * hw[b].second;
    }
    return s;
  }
  std::vector<uint8_t> mask() const {
    std::vector<uint8_t> m;
    for (std::size_t b = 0; b < hw.size(); ++b) {
      m.insert(m.end(), text_before[b], 0);
      m.insert(m.end(), static_cast<std::size_t>(hw[b].first) * hw[b].second, 1);
    }
    m.insert(m.end(), text_after, 0);
    return m;
  }
};

std::vector<int32_t> op_positions(const Layout &l) {
  std::vector<uint32_t> s = l.starts(), h, w;
  for (const auto &[bh, bw] : l.hw) {
    h.push_back(bh);
    w.push_back(bw);
  }
  std::vector<int32_t> pos(3 * static_cast<std::size_t>(l.tokens()));
  DIT_RC(ignis_image_dit_rope_positions(l.tokens(), static_cast<uint32_t>(s.size()), s.data(), h.data(), w.data(),
                                        pos.data()));
  return pos;
}

// The table the ops build on the device for host positions ([tokens][64][2] fp32).
struct DeviceTable : Poisoned {
  explicit DeviceTable(const std::vector<int32_t> &pos) : Poisoned(pos.size() / 3 * kRopePairs * 8) {
    DeviceBytes d_pos(pos.size() * 4);
    upload(d_pos, pos);
    DIT_RC(ignis_image_dit_rope_table(d_pos.as<int32_t>(), static_cast<uint32_t>(pos.size() / 3), as<float>(), nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  }
};

const Layout kT2i{"t2i (text 31, 64 x 64)", {31}, {{64, 64}}, 0};
const Layout kEdit{"edit (text 8, 64 x 64, text 18, 64 x 64)", {8, 18}, {{64, 64}, {64, 64}}, 0};

void case_rope() {
  std::printf("rope:\n");
  // The frequencies: the formula is torch's at every pair but one, one ulp below it.
  int differ = 0;
  for (int d : {16, 56}) {
    for (int i = 0; i < d / 2; ++i) {
      // Near the exact value: the exponent's fp32 rounding (2^-24 of 2i / d) moves 10000^-e by
      // ln(10000) times that, relatively; pow and the reciprocal round once each.
      const double f = formula_freq(d, i), exact = std::pow(10000.0, -2.0 * i / d);
      differ += formula_freq(d, i) != torch_freq(d, i);
      check(std::fabs(f - exact) <= (std::log(10000.0) * 2.0 * i / d + 2.0) * kU * exact,
            "frequency d " + std::to_string(d) + " i " + std::to_string(i) + " is not near 10000^(-2i/d)");
    }
  }
  check(differ == 1 && std::nextafter(formula_freq(56, 11), std::numeric_limits<float>::infinity()) == kTorchFreq56At11,
        "torch's frequencies differ from the formula exactly at (56, 11), by one ulp up");

  // The table op against the host's at every position of diffusers' range, on every axis.
  std::vector<int32_t> pos;
  for (int p = -1024; p < 8192; ++p) pos.insert(pos.end(), {p, p, p});
  {
    const uint32_t tokens = static_cast<uint32_t>(pos.size() / 3);
    const DeviceTable table(pos);
    const std::vector<float> got = download<float>(table.p, static_cast<std::size_t>(tokens) * 2 * kRopePairs);
    std::vector<float> want(got.size());
    for (uint32_t t = 0; t < tokens; ++t) host_table_row(&pos[3 * t], &want[static_cast<std::size_t>(t) * 2 * kRopePairs]);
    const TableDiff d = compare_tables(got, want);
    std::printf("  table op vs host, positions [-1024, 8192) x 64 pairs: %lld equal, %lld at one ulp, %lld worse\n",
                d.equal, d.one_ulp, d.worse);
    check(d.worse == 0, "the table op differs from the host's table by more than one ulp");
  }

  // The positions op against the mask-form restatement.
  const std::vector<Layout> layouts = {
      kT2i,
      kEdit,
      {"text 3, 2 x 3, text 2, 4 x 5, text 2", {3, 2}, {{2, 3}, {4, 5}}, 2},
      {"text 5, 3 x 7 and 7 x 3 adjacent, text 4, 5 x 5, text 3", {5, 0, 4}, {{3, 7}, {7, 3}, {5, 5}}, 3},
      {"an image first, no text", {0}, {{6, 4}}, 0},
      {"text only", {}, {}, 40},
      {"ten condition images (text 3 each) and the 172 x 96 preset",
       {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
       {{64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {64, 64}, {172, 96}},
       0},
  };
  for (const Layout &l : layouts) {
    const std::vector<int32_t> got = op_positions(l), want = positions_from_mask(l.mask(), l.hw);
    check(got == want, "positions of " + l.name + " differ from diffusers' restated");
    std::printf("  positions, %s: %u tokens, %s\n", l.name.c_str(), l.tokens(), got == want ? "equal" : "DIFFER");
  }
  // Worked by hand: text 3, 2 x 3, text 2, 4 x 5, text 2.
  {
    const std::vector<int32_t> p = op_positions(layouts[2]);
    auto at = [&](int t) { return std::array<int32_t, 3>{p[3 * t], p[3 * t + 1], p[3 * t + 2]}; };
    check(at(2) == std::array<int32_t, 3>{2, 2, 2} && at(3) == std::array<int32_t, 3>{3, -1, -2} &&
              at(8) == std::array<int32_t, 3>{3, 0, 0} && at(9) == std::array<int32_t, 3>{6, 6, 6} &&
              at(11) == std::array<int32_t, 3>{8, -2, -3} && at(30) == std::array<int32_t, 3>{8, 1, 1} &&
              at(31) == std::array<int32_t, 3>{13, 13, 13},
          "positions of the hand-worked layout");
  }
  // Refusals.
  auto refused = [](uint32_t tokens, std::vector<uint32_t> s, std::vector<uint32_t> h, std::vector<uint32_t> w,
                    const char *what) {
    std::vector<int32_t> pos(3 * static_cast<std::size_t>(tokens));
    const int32_t rc = ignis_image_dit_rope_positions(tokens, static_cast<uint32_t>(s.size()), s.data(), h.data(),
                                                      w.data(), pos.data());
    check(rc != 0, std::string("rope_positions refuses ") + what);
  };
  refused(20, {2, 5}, {2, 2}, {2, 2}, "overlapping blocks");
  refused(20, {16}, {2, 3}, {3, 2}, "a block past the end");
  refused(20, {4}, {0}, {3}, "an empty block");
  refused(20, {0}, {0xFFFFFFFFu}, {0xFFFFFFFFu}, "a block whose h * w overflows 64 bits");
  refused(20, {0xFFFFFFFFu}, {2}, {2}, "a block that starts past the sequence");
  refused(20, {0}, {21}, {1}, "a block taller than the sequence");
  refused(8193, {}, {}, {}, "a text position at 8192");
  refused(2049 * 2, {0}, {2049}, {2}, "a block 2049 rows tall (height position -1025)");
}

// ---- the row ops ----------------------------------------------------------------------------------

// Reference of row r of the modulated LayerNorm, scale row s.
std::vector<Err> ref_layer_norm(const std::vector<double> &x, const float *s, int depth) {
  const std::size_t n = x.size();
  std::vector<Err> t(n);
  for (std::size_t c = 0; c < n; ++c) t[c] = exact(x[c]);
  const Err mean = div(sum(t, depth), exact(static_cast<double>(n)));
  std::vector<Err> centred(n), sq(n);
  for (std::size_t c = 0; c < n; ++c) {
    centred[c] = sub(exact(x[c]), mean);
    sq[c] = square(centred[c]);
  }
  const Err var = div(sum(sq, depth), exact(static_cast<double>(n)));
  const Err r = rsqrt_f(add(var, exact(kEps)));
  std::vector<Err> y(n);
  for (std::size_t c = 0; c < n; ++c) y[c] = mul(mul(centred[c], r), add(exact(1.0), exact(s[c])));
  return y;
}

// Reference of an RMS norm row: (x * rsqrt(sum x^2 / n + eps)) * m(c), m from `scale_of`.
template <class Scale> std::vector<Err> ref_rms_norm(const std::vector<double> &x, Scale &&scale_of, int depth) {
  const std::size_t n = x.size();
  std::vector<Err> sq(n);
  for (std::size_t c = 0; c < n; ++c) sq[c] = square(exact(x[c]));
  const Err r = rsqrt_f(add(div(sum(sq, depth), exact(static_cast<double>(n))), exact(kEps)));
  std::vector<Err> y(n);
  for (std::size_t c = 0; c < n; ++c) y[c] = mul(mul(exact(x[c]), r), scale_of(c));
  return y;
}

// The modulated norm at `tokens` with the prefix / target split at `first_target`.
void case_modnorm(const Options &o, uint32_t tokens, uint32_t first_target, const std::string &name,
                  uint32_t cols = 4096) {
  const std::size_t n = static_cast<std::size_t>(tokens) * cols;
  const std::string what = "modnorm " + name;
  DeviceBytes x(n * 2), st(cols * 4), s0(cols * 4);
  const std::vector<float> scale_t = host_row(21, cols, -1.5f, 1.5f), scale_0 = host_row(22, cols, -0.25f, 0.75f);
  upload(st, scale_t);
  upload(s0, scale_0);
  fill(x, tokens, cols, kNormIn, 11, 1.0f);
  Poisoned y32(n * 4), y16(n * 2), codes(n), scales(tokens * 4ull);
  auto run = [&](void *y, float *ys, uint32_t kind, uint32_t ft) {
    DIT_RC(ignis_image_dit_modulated_norm(x.p, tokens, cols, st.as<float>(), s0.as<float>(), ft, kEps, y, ys, kind,
                                          nullptr));
  };
  run(y32.p, nullptr, kF32, first_target);
  run(y16.p, nullptr, kBf16, first_target);
  run(codes.p, scales.as<float>(), kE4m3, first_target);
  MOE_CUDA(cudaDeviceSynchronize());

  Verdict v, swapped;
  for (uint32_t r : sample_rows(tokens, 48, {first_target == 0 ? 0 : first_target - 1, first_target})) {
    std::vector<double> xr(cols);
    for (uint32_t c = 0; c < cols; ++c) xr[c] = host_in(kNormIn, 11, r, c, cols, 1.0f);
    const bool prefix = r < first_target;
    const std::vector<Err> ref = ref_layer_norm(xr, (prefix ? scale_0 : scale_t).data(), row_depth(cols));
    const std::vector<Err> wrong = ref_layer_norm(xr, (prefix ? scale_t : scale_0).data(), row_depth(cols));
    const std::vector<float> got = download_row_f32(y32, r, cols, cols);
    for (uint32_t c = 0; c < cols; ++c) {
      v.take(got[c], ref[c], o.bound_factor);
      // The swapped rows only differ where the row is not zero (profiles 6 and 7 are).
      if (r % 8 < 6) swapped.take(got[c], wrong[c], o.bound_factor);
    }
  }
  report(what, v);
  check(swapped.checked == 0 || swapped.over * 2 > swapped.checked,
        what + ": the reference with the t and t=0 rows swapped passes (negative control)");
  check_bf16(what, y32, y16, tokens, cols);
  check_e4m3(what, y16, codes, scales, tokens, cols);
  if (first_target > 0 && first_target < tokens) {
    // The joint call's rows against the all-t and all-t0 calls, bit for bit.
    Poisoned all(n * 4);
    run(all.p, nullptr, kF32, 0);
    const auto dt = diff_u16(static_cast<uint16_t *>(y32.p) + 2ull * first_target * cols, 2 * cols,
                             static_cast<uint16_t *>(all.p) + 2ull * first_target * cols, 2 * cols,
                             tokens - first_target, 2 * cols);
    run(all.p, nullptr, kF32, tokens);
    const auto d0 = diff_u16(y32.p, 2 * cols, all.p, 2 * cols, first_target, 2 * cols);
    check(dt == 0 && d0 == 0, what + ": the per-token row choice differs from the all-t / all-t0 calls");
    std::printf("    row choice: target rows = the all-t call's, prefix rows = the all-t0 call's (%s); swapped "
                "rows: %lld of %lld over the bound\n",
                dt == 0 && d0 == 0 ? "bit for bit" : "DIFFER", swapped.over, swapped.checked);
  }
}

void case_rmsnorm(const Options &o, uint32_t tokens) {
  const uint32_t cols = 4096;
  const std::size_t n = static_cast<std::size_t>(tokens) * cols;
  const std::string what = "rmsnorm " + std::to_string(tokens) + " x 4096";
  DeviceBytes x(n * 2), w(cols * 2);
  const std::vector<uint16_t> weight = host_bf16_row(31, cols, 0.0f, 0.4f);  // stored zero-centred: w + 1 ~ 1
  upload(w, weight);
  fill(x, tokens, cols, kNormIn, 12, 1.0f);
  Poisoned y32(n * 4), y16(n * 2), codes(n), scales(tokens * 4ull);
  DIT_RC(ignis_image_dit_rms_norm(x.p, tokens, cols, w.p, kEps, y32.p, nullptr, kF32, nullptr));
  DIT_RC(ignis_image_dit_rms_norm(x.p, tokens, cols, w.p, kEps, y16.p, nullptr, kBf16, nullptr));
  DIT_RC(ignis_image_dit_rms_norm(x.p, tokens, cols, w.p, kEps, codes.p, scales.as<float>(), kE4m3, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  Verdict v;
  for (uint32_t r : sample_rows(tokens, 48)) {
    std::vector<double> xr(cols);
    for (uint32_t c = 0; c < cols; ++c) xr[c] = host_in(kNormIn, 12, r, c, cols, 1.0f);
    const std::vector<Err> ref =
        ref_rms_norm(xr, [&](std::size_t c) { return add(exact(bf16_to_f32(weight[c])), exact(1.0)); }, row_depth(cols));
    const std::vector<float> got = download_row_f32(y32, r, cols, cols);
    for (uint32_t c = 0; c < cols; ++c) v.take(got[c], ref[c], o.bound_factor);
  }
  report(what, v);
  check_bf16(what, y32, y16, tokens, cols);
  check_e4m3(what, y16, codes, scales, tokens, cols);
}

// An activation: `fused` puts gate and up in one [tokens][2 cols] buffer (SwiGLU only).
void case_act(const Options &o, uint32_t kind, uint32_t tokens, uint32_t cols, bool fused) {
  static const char *names[] = {"silu", "gelu-tanh", "swiglu"};
  const bool swiglu = kind == IGNIS_IMAGE_DIT_SWIGLU;
  const std::string what = std::string(names[kind]) + " " + std::to_string(tokens) + " x " + std::to_string(cols) +
                           (swiglu ? (fused ? " (fused gate | up)" : " (separate gate, up)") : "");
  const uint32_t stride = fused ? 2 * cols : cols;
  const std::size_t n = static_cast<std::size_t>(tokens) * cols;
  DeviceBytes in(static_cast<std::size_t>(tokens) * stride * 2), up_sep(swiglu && !fused ? n * 2 : 0);
  // Gate and up of the fused buffer are one fill (up in its second half); the separate up is its own.
  fill(in, tokens, stride, kActIn, 13, 1.0f);
  if (swiglu && !fused) fill(up_sep, tokens, cols, kPlainIn, 14, 2.0f);
  const void *x = in.p;
  const void *up = !swiglu ? nullptr : (fused ? static_cast<const void *>(static_cast<uint16_t *>(in.p) + cols) : up_sep.p);
  Poisoned y32(n * 4), y16(n * 2), codes(n), scales(tokens * 4ull);
  DIT_RC(ignis_image_dit_act(kind, x, up, tokens, cols, stride, y32.p, nullptr, kF32, nullptr));
  DIT_RC(ignis_image_dit_act(kind, x, up, tokens, cols, stride, y16.p, nullptr, kBf16, nullptr));
  DIT_RC(ignis_image_dit_act(kind, x, up, tokens, cols, stride, codes.p, scales.as<float>(), kE4m3, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  Verdict v;
  for (uint32_t r : sample_rows(tokens, 48)) {
    const std::vector<float> got = download_row_f32(y32, r, cols, cols);
    for (uint32_t c = 0; c < cols; ++c) {
      const Err g = exact(host_in(kActIn, 13, r, c, stride, 1.0f));
      Err ref = kind == IGNIS_IMAGE_DIT_GELU_TANH ? gelu_ref(g) : silu_ref(g);
      if (swiglu) {
        const double u = fused ? host_in(kActIn, 13, r, cols + c, stride, 1.0f) : host_in(kPlainIn, 14, r, c, cols, 2.0f);
        ref = mul(ref, exact(u));
      }
      v.take(got[c], ref, o.bound_factor);
    }
  }
  report(what, v);
  check_bf16(what, y32, y16, tokens, cols);
  check_e4m3(what, y16, codes, scales, tokens, cols);
}

void case_residual(const Options &o, uint32_t tokens, uint32_t first_target, const std::string &name) {
  const uint32_t cols = 4096;
  const std::size_t n = static_cast<std::size_t>(tokens) * cols;
  const std::string what = "residual " + name;
  DeviceBytes x(n * 2), y(n * 2), gt(cols * 4), g0(cols * 4);
  const std::vector<float> gate_t = host_row(41, cols, -3.0f, 3.0f), gate_0 = host_row(42, cols, -1.0f, 2.0f);
  upload(gt, gate_t);
  upload(g0, gate_0);
  fill(x, tokens, cols, kNormIn, 15, 1.0f);
  fill(y, tokens, cols, kActIn, 16, 1.0f);
  Poisoned out32(n * 4), out16(n * 2);
  DeviceBytes inplace(n * 2);
  MOE_CUDA(cudaMemcpy(inplace.p, x.p, n * 2, cudaMemcpyDeviceToDevice));
  auto run = [&](const void *xi, void *out, uint32_t f32, uint32_t ft) {
    DIT_RC(ignis_image_dit_gated_residual(xi, y.p, tokens, cols, gt.as<float>(), g0.as<float>(), ft, out, f32, nullptr));
  };
  run(x.p, out32.p, 1, first_target);
  run(x.p, out16.p, 0, first_target);
  run(inplace.p, inplace.p, 0, first_target);
  MOE_CUDA(cudaDeviceSynchronize());
  Verdict v, swapped;
  for (uint32_t r : sample_rows(tokens, 48, {first_target == 0 ? 0 : first_target - 1, first_target})) {
    const std::vector<float> got = download_row_f32(out32, r, cols, cols);
    const bool prefix = r < first_target;
    for (uint32_t c = 0; c < cols; ++c) {
      const Err xv = exact(host_in(kNormIn, 15, r, c, cols, 1.0f)), yv = exact(host_in(kActIn, 16, r, c, cols, 1.0f));
      v.take(got[c], fma(tanh_f(exact((prefix ? gate_0 : gate_t)[c])), yv, xv), o.bound_factor);
      if (yv.v != 0.0) swapped.take(got[c], fma(tanh_f(exact((prefix ? gate_t : gate_0)[c])), yv, xv), o.bound_factor);
    }
  }
  report(what, v);
  check_bf16(what, out32, out16, tokens, cols);
  const auto d = diff_u16(inplace.p, cols, out16.p, cols, tokens, cols);
  check(d == 0, what + ": the in-place output differs from the out-of-place one");
  if (first_target > 0 && first_target < tokens) {
    check(swapped.over * 2 > swapped.checked, what + ": the reference with the gate rows swapped passes");
    Poisoned all(n * 4);
    run(x.p, all.p, 1, 0);
    const auto dt = diff_u16(static_cast<uint16_t *>(out32.p) + 2ull * first_target * cols, 2 * cols,
                             static_cast<uint16_t *>(all.p) + 2ull * first_target * cols, 2 * cols,
                             tokens - first_target, 2 * cols);
    run(x.p, all.p, 1, tokens);
    const auto d0 = diff_u16(out32.p, 2 * cols, all.p, 2 * cols, first_target, 2 * cols);
    check(dt == 0 && d0 == 0, what + ": the per-token gate choice differs from the all-t / all-t0 calls");
    std::printf("    row choice: %s; swapped gate rows: %lld of %lld over the bound\n",
                dt == 0 && d0 == 0 ? "bit for bit" : "DIFFER", swapped.over, swapped.checked);
  }
}

// The QK norm with RoPE in place on a fused QKV row, the table from the ops for layout `l`.
void case_qk(const Options &o, const Layout &l) {
  const uint32_t tokens = l.tokens(), stride = 12288, width = 8192;  // q | k | v; the fp32 output holds q | k
  const std::string what = "qk norm+rope " + l.name;
  const std::size_t n = static_cast<std::size_t>(tokens) * stride;
  DeviceBytes qkv(n * 2), pristine(n * 2), qw(256), kw(256);
  const std::vector<uint16_t> q_weight = host_bf16_row(51, 128, 1.0f, 0.3f), k_weight = host_bf16_row(52, 128, 0.8f, 0.5f);
  upload(qw, q_weight);
  upload(kw, k_weight);
  // Rows by r % 8 (kNormIn): includes zero and constant heads.
  fill(qkv, tokens, stride, kNormIn, 17, 1.0f);
  MOE_CUDA(cudaMemcpy(pristine.p, qkv.p, n * 2, cudaMemcpyDeviceToDevice));
  const DeviceTable table(op_positions(l));
  Poisoned out32(static_cast<std::size_t>(tokens) * width * 4);
  uint16_t *q = qkv.as<uint16_t>();
  DIT_RC(ignis_image_dit_qk_norm_rope(q, q + 4096, tokens, stride, qw.p, kw.p, kEps, table.as<float>(), out32.p,
                                      out32.as<float>() + 4096, width, 1, nullptr));
  DIT_RC(ignis_image_dit_qk_norm_rope(q, q + 4096, tokens, stride, qw.p, kw.p, kEps, table.as<float>(), q, q + 4096,
                                      stride, 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  Verdict v;
  for (uint32_t t : sample_rows(tokens, 24, {tokens - 1})) {
    const std::vector<float> got = download_row_f32(out32, t, width, width);
    const std::vector<float> cs = download<float>(table.as<float>() + static_cast<std::size_t>(t) * 2 * kRopePairs,
                                                  2 * kRopePairs);
    for (int head = 0; head < 64; ++head) {
      const std::vector<uint16_t> &w = head < 32 ? q_weight : k_weight;
      std::vector<double> x(128);
      for (int c = 0; c < 128; ++c) x[c] = host_in(kNormIn, 17, t, head * 128 + c, stride, 1.0f);
      const std::vector<Err> nrm = ref_rms_norm(x, [&](std::size_t c) { return exact(bf16_to_f32(w[c])); }, 4 + 5);
      for (int p = 0; p < 64; ++p) {
        const Err c = exact(cs[2 * p]), s = exact(cs[2 * p + 1]);
        v.take(got[head * 128 + 2 * p], fma(nrm[2 * p], c, neg(mul(nrm[2 * p + 1], s))), o.bound_factor);
        v.take(got[head * 128 + 2 * p + 1], fma(nrm[2 * p], s, mul(nrm[2 * p + 1], c)), o.bound_factor);
      }
    }
  }
  report(what, v);
  const auto b = bf16_vs_f32(out32.p, width, qkv.p, stride, tokens, width);
  check(b == 0, what + ": " + std::to_string(b) + " in-place BF16 outputs are not the fp32 output rounded");
  const auto dv = diff_u16(static_cast<uint16_t *>(qkv.p) + width, stride, static_cast<uint16_t *>(pristine.p) + width,
                           stride, tokens, stride - width);
  check(dv == 0, what + ": v changed");
}

// The VAE channel norm over `pixels` of `channels`, with and without SiLU.
void case_vae(const Options &o, uint32_t pixels, uint32_t channels) {
  const std::size_t n = static_cast<std::size_t>(pixels) * channels;
  DeviceBytes x(n * 2), g(channels * 2ull), inplace(n * 2);
  Poisoned y32(n * 4), y16(n * 2);
  const std::vector<uint16_t> gamma = host_bf16_row(61, channels, 1.0f, 0.5f);
  upload(g, gamma);
  fill(x, pixels, channels, kNormIn, 18, 1.0f);
  const double scale = std::sqrt(static_cast<double>(channels));
  const int depth = 8 * static_cast<int>((channels / 8 + 31) / 32) + 5;
  for (uint32_t silu : {0u, 1u}) {
    const std::string what = "vae norm " + std::to_string(pixels) + " px x " + std::to_string(channels) +
                             (silu ? " + silu" : "");
    MOE_CUDA(cudaMemcpy(inplace.p, x.p, n * 2, cudaMemcpyDeviceToDevice));
    VAE_RC(ignis_image_vae_rms_norm(x.p, pixels, channels, g.p, silu, y32.p, 1, nullptr));
    VAE_RC(ignis_image_vae_rms_norm(x.p, pixels, channels, g.p, silu, y16.p, 0, nullptr));
    VAE_RC(ignis_image_vae_rms_norm(inplace.p, pixels, channels, g.p, silu, inplace.p, 0, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    Verdict v;
    for (uint32_t p : sample_rows(pixels, 64)) {
      std::vector<Err> sq(channels);
      std::vector<double> xr(channels);
      for (uint32_t c = 0; c < channels; ++c) {
        xr[c] = host_in(kNormIn, 18, p, c, channels, 1.0f);
        sq[c] = square(exact(xr[c]));
      }
      const Err nrm = sqrt_f(sum(sq, depth));
      // max(norm, 1e-12) is 1-Lipschitz; the clamp's fp32 rounding adds its own error.
      const Err clamped{std::max(nrm.v, 1e-12), nrm.e + 1e-12 * kU};
      const Err inv = div(Err{scale, std::fabs(static_cast<float>(scale) - scale)}, clamped);
      const std::vector<float> got = download_row_f32(y32, p, channels, channels);
      for (uint32_t c = 0; c < channels; ++c) {
        Err y = mul(mul(exact(xr[c]), inv), exact(bf16_to_f32(gamma[c])));
        if (silu) y = silu_ref(y);
        v.take(got[c], y, o.bound_factor);
      }
    }
    report(what, v);
    check_bf16(what, y32, y16, pixels, channels);
    const auto d = diff_u16(inplace.p, channels, y16.p, channels, pixels, channels);
    check(d == 0, what + ": the in-place output differs from the out-of-place one");
  }
}

void refusals() {
  std::printf("refusals:\n");
  DeviceBytes a(1 << 20), b(1 << 20), c(1 << 20), w(1 << 16);
  auto refused = [](int32_t rc, const char *what) {
    check(rc != 0, std::string("refuses ") + what);
    if (rc != 0) std::printf("  %s: %s\n", what, ignis_image_dit_last_error());
  };
  float *s = c.as<float>();
  refused(ignis_image_dit_modulated_norm(a.p, 4, 4104, s, s, 0, kEps, b.p, nullptr, kF32, nullptr), "cols % 16");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 24592, s, s, 0, kEps, b.p, nullptr, kF32, nullptr), "cols > 24560");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 24576, s, s, 0, kEps, b.p, nullptr, kF32, nullptr), "cols 24576 (row + reduction > 48 KiB)");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 64, s, s, 5, kEps, b.p, nullptr, kF32, nullptr), "first_target > tokens");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 64, nullptr, s, 2, kEps, b.p, nullptr, kF32, nullptr), "a NULL t row read");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 64, s, s, 0, kEps, b.p, nullptr, kE4m3, nullptr), "E4M3 without scales");
  refused(ignis_image_dit_modulated_norm(a.p, 4, 64, s, s, 0, kEps, b.p, s, kBf16, nullptr), "scales without E4M3");
  refused(ignis_image_dit_modulated_norm(static_cast<char *>(a.p) + 2, 4, 64, s, s, 0, kEps, b.p, nullptr, kF32, nullptr),
          "x misaligned");
  refused(ignis_image_dit_rms_norm(a.p, 0, 64, w.p, kEps, b.p, nullptr, kF32, nullptr), "zero tokens");
  refused(ignis_image_dit_rms_norm(a.p, 4, 64, nullptr, kEps, b.p, nullptr, kF32, nullptr), "a NULL weight");
  refused(ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, a.p, nullptr, 4, 64, 64, b.p, nullptr, kF32, nullptr),
          "SwiGLU without up");
  refused(ignis_image_dit_act(IGNIS_IMAGE_DIT_SILU, a.p, a.p, 4, 64, 64, b.p, nullptr, kF32, nullptr), "SiLU with up");
  refused(ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, a.p, a.p, 4, 64, 60, b.p, nullptr, kF32, nullptr),
          "row_stride < cols");
  refused(ignis_image_dit_act(7, a.p, nullptr, 4, 64, 64, b.p, nullptr, kF32, nullptr), "an unknown kind");
  refused(ignis_image_dit_gated_residual(a.p, b.p, 4, 12, s, s, 0, a.p, 0, nullptr), "residual cols % 8");
  refused(ignis_image_dit_gated_residual(a.p, b.p, 4, 64, s, nullptr, 1, a.p, 0, nullptr), "a NULL t=0 gate read");
  refused(ignis_image_dit_qk_norm_rope(a.p, a.p, 4, 4000, w.p, w.p, kEps, s, a.p, a.p, 4096, 0, nullptr),
          "row_stride < 4096");
  refused(ignis_image_dit_qk_norm_rope(a.p, a.p, 4, 12288, w.p, w.p, kEps, nullptr, a.p, a.p, 12288, 0, nullptr),
          "a NULL table");
  refused(ignis_image_dit_rope_table(nullptr, 4, s, nullptr), "a NULL positions");
  check(ignis_image_vae_rms_norm(a.p, 4, 100, w.p, 1, b.p, 0, nullptr) != 0, "vae norm refuses channels % 8");
  check(ignis_image_vae_rms_norm(a.p, 0, 96, w.p, 1, b.p, 0, nullptr) != 0, "vae norm refuses zero pixels");
  check(ignis_image_vae_rms_norm(static_cast<char *>(a.p) + 8, 4, 96, w.p, 1, b.p, 0, nullptr) != 0,
        "vae norm refuses x misaligned");
}

// ---- the oracle's RoPE tables (--cases rope-oracle) ----------------------------------------------

struct Npy {
  std::string descr;
  std::vector<std::size_t> shape;
  std::vector<char> data;
};

// A little-endian C-order .npy; false when the file is missing.
bool load_npy(const std::string &path, Npy &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[8];
  f.read(magic, 8);
  uint32_t header_len = 0;
  if (magic[6] == 1) {
    uint16_t h = 0;
    f.read(reinterpret_cast<char *>(&h), 2);
    header_len = h;
  } else {
    f.read(reinterpret_cast<char *>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(&header[0], header_len);
  const std::size_t d = header.find("'descr': '"), s = header.find("'shape': (");
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0 || d == std::string::npos || s == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos) {
    std::fprintf(stderr, "FATAL: %s is not a C-order .npy\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  out.descr = header.substr(d + 10, header.find('\'', d + 10) - (d + 10));
  out.shape.clear();
  std::size_t count = 1;
  for (std::size_t i = s + 10; header[i] != ')'; ++i) {
    if (std::isdigit(static_cast<unsigned char>(header[i]))) {
      std::size_t v = 0;
      while (std::isdigit(static_cast<unsigned char>(header[i]))) v = v * 10 + (header[i++] - '0');
      out.shape.push_back(v);
      count *= v;
      --i;
    }
  }
  const std::size_t item = out.descr == "<f4" ? 4 : (out.descr == "|u1" ? 1 : 0);
  if (item == 0) {
    std::fprintf(stderr, "FATAL: %s: dtype %s not read here\n", path.c_str(), out.descr.c_str());
    std::exit(EXIT_FAILURE);
  }
  out.data.resize(count * item);
  f.read(out.data.data(), static_cast<std::streamsize>(out.data.size()));
  if (!f) {
    std::fprintf(stderr, "FATAL: %s is short\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  return true;
}

// case.json's "img_shapes": [[f, h, w], ...] as (h, w) pairs.
std::vector<std::pair<int, int>> img_shapes(const std::string &path) {
  std::ifstream f(path);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string json = ss.str();
  std::size_t at = json.find("\"img_shapes\"");
  if (at == std::string::npos || (at = json.find('[', at)) == std::string::npos) {
    std::fprintf(stderr, "FATAL: %s has no img_shapes\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::vector<int> nums;
  int depth = 0;
  for (std::size_t i = at; i < json.size(); ++i) {
    if (json[i] == '[') {
      ++depth;
    } else if (json[i] == ']') {
      if (--depth == 0) break;
    } else if (std::isdigit(static_cast<unsigned char>(json[i]))) {
      int v = 0;
      while (std::isdigit(static_cast<unsigned char>(json[i]))) v = v * 10 + (json[i++] - '0');
      nums.push_back(v);
      --i;
    }
  }
  std::vector<std::pair<int, int>> hw;
  for (std::size_t i = 0; i + 2 < nums.size(); i += 3) hw.push_back({nums[i + 1], nums[i + 2]});
  return hw;
}

void case_rope_oracle(const std::string &dir) {
  std::printf("rope-oracle (%s):\n", dir.c_str());
  for (const char *name : {"t2i", "edit"}) {
    const std::string at = dir + "/" + name + "/";
    Npy rope, mask;
    if (!load_npy(at + "rope.npy", rope) || !load_npy(at + "joint_image_mask.npy", mask)) {
      check(false, std::string("rope-oracle: ") + name + ": rope.npy or joint_image_mask.npy missing");
      continue;
    }
    const std::vector<std::pair<int, int>> hw = img_shapes(at + "case.json");
    const uint32_t tokens = static_cast<uint32_t>(mask.data.size());
    check(rope.descr == "<f4" && rope.shape == std::vector<std::size_t>{tokens, 64, 2},
          std::string("rope-oracle: ") + name + ": rope.npy is not float32 (tokens, 64, 2)");
    // Block starts as diffusers' build_token_metadata finds them: the image tokens in order, cut by
    // each block's h * w; each block must be contiguous.
    std::vector<uint32_t> starts, h, w;
    std::size_t t = 0;
    for (const auto &[bh, bw] : hw) {
      while (t < tokens && !mask.data[t]) ++t;
      starts.push_back(static_cast<uint32_t>(t));
      h.push_back(bh);
      w.push_back(bw);
      for (int i = 0; i < bh * bw; ++i, ++t) check(t < tokens && mask.data[t], "rope-oracle: a block is not contiguous");
    }
    std::vector<int32_t> pos(3 * static_cast<std::size_t>(tokens));
    DIT_RC(ignis_image_dit_rope_positions(tokens, static_cast<uint32_t>(starts.size()), starts.data(), h.data(), w.data(),
                                          pos.data()));
    const DeviceTable table(pos);
    const std::vector<float> got = download<float>(table.p, static_cast<std::size_t>(tokens) * 2 * kRopePairs);
    std::vector<float> want(got.size());
    std::memcpy(want.data(), rope.data.data(), want.size() * 4);
    const TableDiff d = compare_tables(got, want);
    std::printf("  %s: %u tokens, %zu blocks: %lld equal, %lld at one ulp (%.4f%%), %lld worse; max |diff| %.3g\n", name,
                tokens, hw.size(), d.equal, d.one_ulp, 100.0 * d.one_ulp / got.size(), d.worse, d.worst_abs);
    check(d.worse == 0, std::string("rope-oracle: ") + name + ": the table differs from diffusers' by more than one ulp");
  }
}

}  // namespace

int main(int argc, char **argv) {
  Options o;
  o.bound_factor = arg_double(argc, argv, "--bound-factor", o.bound_factor);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "rope,modnorm,rmsnorm,act,residual,qk,vae,refusals", oracle;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
    if (std::strcmp(argv[i], "--oracle") == 0) oracle = argv[i + 1];
  }
  if (oracle.empty()) {
    const char *env = std::getenv("IGNIS_IMAGE_ORACLE");
    oracle = env != nullptr && *env != '\0' ? env : "F:/ai/models/ignis-image-2.1-turbo/oracle";
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("image elementwise ops (bound factor %.3f; cases %s)\n", o.bound_factor, cases.c_str());
  if (want("rope")) case_rope();
  if (want("modnorm")) {
    std::printf("modulated LayerNorm:\n");
    case_modnorm(o, 1, 0, "1 x 4096, target");
    case_modnorm(o, 1, 1, "1 x 4096, prefix");
    case_modnorm(o, 7, 3, "7 x 4096, prefix 3");
    case_modnorm(o, 4127, 31, "4127 x 4096 (t2i: prefix 31)");
    case_modnorm(o, 8218, 4122, "8218 x 4096 (edit: prefix 4122)");
    case_modnorm(o, 16512, 0, "16512 x 4096 (the preset's target)");
    case_modnorm(o, 3, 1, "3 x 24560 (the cols cap: the row fills 48 KiB of shared memory)", 24560);
  }
  if (want("rmsnorm")) {
    std::printf("zero-centred RMSNorm:\n");
    for (uint32_t tokens : {1u, 31u, 2048u}) case_rmsnorm(o, tokens);
  }
  if (want("act")) {
    std::printf("activations:\n");
    case_act(o, IGNIS_IMAGE_DIT_SILU, 2, 4096, false);
    case_act(o, IGNIS_IMAGE_DIT_SILU, 33, 4096, false);
    case_act(o, IGNIS_IMAGE_DIT_GELU_TANH, 31, 4096, false);
    case_act(o, IGNIS_IMAGE_DIT_GELU_TANH, 2048, 4096, false);
    case_act(o, IGNIS_IMAGE_DIT_SWIGLU, 1, 12288, false);
    case_act(o, IGNIS_IMAGE_DIT_SWIGLU, 4127, 12288, false);
    case_act(o, IGNIS_IMAGE_DIT_SWIGLU, 4096, 12288, true);
    case_act(o, IGNIS_IMAGE_DIT_SWIGLU, 16512, 12288, true);
  }
  if (want("residual")) {
    std::printf("gated residual:\n");
    case_residual(o, 1, 0, "1 x 4096");
    case_residual(o, 7, 3, "7 x 4096, prefix 3");
    case_residual(o, 4127, 31, "4127 x 4096 (t2i: prefix 31)");
    case_residual(o, 16512, 0, "16512 x 4096");
  }
  if (want("qk")) {
    std::printf("QK RMSNorm + RoPE:\n");
    case_qk(o, {"1 token", {}, {}, 1});
    case_qk(o, kT2i);
    case_qk(o, kEdit);
    case_qk(o, {"text 200, 128 x 128", {200}, {{128, 128}}, 0});
  }
  if (want("vae")) {
    std::printf("VAE channel norm:\n");
    for (uint32_t channels : {96u, 144u, 192u, 288u, 384u, 576u, 768u, 1152u}) case_vae(o, 4096, channels);
    case_vae(o, 1, 96);
    case_vae(o, 1024 * 1024, 144);
    case_vae(o, 128 * 128, 1152);
  }
  if (want("rope-oracle")) {
    if (std::ifstream(oracle + "/t2i/rope.npy").good()) {
      case_rope_oracle(oracle);
    } else {
      // A skip is never green in the GPU profile (docs/agents/testing.md).
      const char *profile = std::getenv("IGNIS_GPU_PROFILE");
      if (profile != nullptr && std::string(profile) == "1") {
        check(false, "rope-oracle: no oracle at " + oracle + " (--oracle DIR or IGNIS_IMAGE_ORACLE)");
      } else {
        std::printf("SKIP: rope-oracle: no oracle at %s (--oracle DIR or IGNIS_IMAGE_ORACLE)\n", oracle.c_str());
      }
    }
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_elementwise: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_elementwise: OK\n");
  return 0;
}
