// The image model's text encoder (spec image-model/03, GitHub #320) -- OURS. The ops are
// kernel/include/ignis_image_encoder.h's; every reference is our own fp64 CPU code, and the
// forward's checks are equalities between calls of its own.
//
//   norm      the residual add with the RMSNorm, 1 .. 2049 tokens x 4096: the add against the
//             host's fp32 add, the norm's fp32 output against fp64, the E4M3 output against the
//             host's quantization of that fp32 output (ignis_image_gemm.h's steps), bit for bit.
//   qk        the QK RMSNorm with the interleaved mrope, q's 32 heads and k's 8 from strided fp32
//             rows, with three different position rows (t, h, w) up to 40,000: against fp64, and
//             two negative controls that a wrong layout fails (the axes in chunks [24 | 20 | 20],
//             and the pairs adjacent (2i, 2i + 1) instead of rotate_half's (d, d + 64)).
//   rows      the vision rows written over and added to x, against the host, and an index past
//             the sequence skipped (the rows after x unchanged).
//   forward   the decoder layers on synthetic FP8 weights (two sets, layers A B A), 37 tokens with
//             9 vision rows: the vision embeddings and the three DeepStack adds through the hook
//             equal, bit for bit, the same additions made on the host between one-layer calls
//             (which also checks that a range of layers equals its layers run in consecutive
//             calls); a second run is identical; without DeepStack the output differs.
//   refusals  the ops' argument checks.
//
// The bound (test_image_elementwise.cu's error-tracking reference): each value v (exact, in fp64)
// carries e, a bound on how far the kernel's fp32 value can be from it. Every fp32 operation
// rounded to nearest adds 2^-24 of its result; rsqrtf, sincosf add 2 ulp (CUDA Math API, no fast
// math); a sum whose terms go through at most d additions adds gamma_d of sum |terms|. A BF16
// output adds its own rounding, 2^-8 of the value. The bound is multiplied by --bound-factor F
// (default 1).
//
//   ignis_kernel_image_encoder_test [--bound-factor F] [--cases norm,qk,rows,forward,refusals]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_encoder.h"
#include "ignis_image_gemm.h"
#include "moe_fixture.h"

#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

#define ENC_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_encoder_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

constexpr float kEps = 1e-6f;          // the text config's rms_norm_eps
constexpr float kTheta = 5000000.0f;   // its rope_theta
constexpr uint32_t kSection[3] = {24, 20, 20};
constexpr uint32_t kHidden = 4096, kKv = 1024, kInter = 12288;

// ---- the error-tracking reference (test_image_elementwise.cu's) ---------------------------------

constexpr double kU = 0x1p-24;
constexpr double kTiny = 0x1p-149;
constexpr double kUlp2 = 0x1p-22;
constexpr double kRefSlack = 0x1p-40;
constexpr double kBf16Round = 0x1p-8;  // BF16 rounding to nearest: |bf16(x) - x| <= 2^-8 |x| (8-bit significand)

struct Err {
  double v, e;
};

Err exact(double v) { return {v, 0.0}; }
Err rounded(double v, double e) { return {v, e + kU * (std::fabs(v) + e) + kTiny}; }
Err within2ulp(double v, double e) { return {v, e + kUlp2 * (std::fabs(v) + e) + kTiny}; }
Err add(Err a, Err b) { return rounded(a.v + b.v, a.e + b.e); }
Err sub(Err a, Err b) { return rounded(a.v - b.v, a.e + b.e); }
Err mul(Err a, Err b) { return rounded(a.v * b.v, std::fabs(a.v) * b.e + std::fabs(b.v) * a.e + a.e * b.e); }
Err div(Err a, Err b) {
  const double q = a.v / b.v;
  return rounded(q, (a.e + std::fabs(q) * b.e) / (std::fabs(b.v) - b.e));
}
Err rsqrt_f(Err a) {
  const double v = 1.0 / std::sqrt(a.v);
  return within2ulp(v, 0.5 * a.e * std::pow(a.v - a.e, -1.5));
}
Err sum(const std::vector<Err> &t, int depth) {
  double v = 0.0, e = 0.0, mag = 0.0;
  for (const Err &x : t) {
    v += x.v;
    e += x.e;
    mag += std::fabs(x.v) + x.e;
  }
  return {v, e + depth * kU / (1.0 - depth * kU) * mag};
}
Err square(Err a) { return {a.v * a.v, 2.0 * std::fabs(a.v) * a.e + a.e * a.e}; }

// y = (x * rsqrt(sum(x^2) / n + eps)) * w, the sum through `depth` additions.
std::vector<Err> ref_rms_norm(const std::vector<double> &x, const std::vector<uint16_t> &w, int depth) {
  const std::size_t n = x.size();
  std::vector<Err> sq(n);
  for (std::size_t c = 0; c < n; ++c) sq[c] = square(exact(x[c]));
  const Err r = rsqrt_f(add(div(sum(sq, depth), exact(static_cast<double>(n))), exact(kEps)));
  std::vector<Err> y(n);
  for (std::size_t c = 0; c < n; ++c) y[c] = mul(mul(exact(x[c]), r), exact(bf16_to_f32(w[c])));
  return y;
}

struct Verdict {
  long long checked = 0, over = 0;
  double worst = 0.0;
  // `extra` is a rounding of the output itself (BF16), as a fraction of the value.
  void take(double got, const Err &r, double factor, double extra = 0.0) {
    const double bound = r.e * factor + (kRefSlack + extra) * (std::fabs(r.v) + r.e);
    const double d = std::fabs(got - r.v);
    ++checked;
    if (!(d <= bound)) ++over;
    if (bound > 0.0) worst = std::max(worst, d / bound);
  }
};

void report(const std::string &what, const Verdict &v) {
  std::printf("  %-62s %9lld checked, worst %.3f of the bound\n", what.c_str(), v.checked, v.worst);
  check(v.checked > 0 && v.over == 0, what + ": " + std::to_string(v.over) + " outputs over the bound");
}

// ---- inputs -------------------------------------------------------------------------------------

// Row r's profile (r % 8): plain, tiny, large, an offset, outlier channels, all zeros, a
// constant, plain again; each element from the counter hash.
float profiled(uint32_t stream, uint64_t r, uint64_t c, uint64_t cols) {
  const float u = hash_uniform(stream, r * cols + c, 1.0f);
  switch (r % 8) {
    case 1: return u * 0x1p-6f;
    case 2: return u * 40.0f;
    case 3: return 3.0f + u;
    case 4: return c % 97 == 5 ? 64.0f * u : u;
    case 5: return 0.0f;
    case 6: return 2.5f;
    default: return u;
  }
}

std::vector<float> profiled_rows(uint32_t stream, std::size_t rows, std::size_t cols) {
  std::vector<float> v(rows * cols);
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < cols; ++c) v[r * cols + c] = profiled(stream, r, c, cols);
  }
  return v;
}

std::vector<uint16_t> bf16_row(uint32_t stream, std::size_t n, float centre, float spread) {
  std::vector<uint16_t> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = f32_to_bf16(centre + hash_uniform(stream, i, spread));
  return v;
}

// A device buffer of all-0xFF bytes (NaN in fp32 and BF16): an unwritten output shows.
struct Poisoned : DeviceBytes {
  explicit Poisoned(std::size_t bytes) : DeviceBytes(bytes) { MOE_CUDA(cudaMemset(p, 0xFF, bytes)); }
};

long long span_stride(long long count, int want) {
  long long stride = std::max(1LL, (count - 1 + want - 2) / (want - 1));
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, count) != 1) stride += 2;
  return stride;
}

std::vector<uint32_t> sample_rows(uint32_t count, int want) {
  std::vector<uint32_t> rows;
  if (count <= static_cast<uint32_t>(want)) {
    for (uint32_t r = 0; r < count; ++r) rows.push_back(r);
  } else {
    const long long stride = span_stride(count, want);
    for (int i = 0; i < want; ++i) rows.push_back(static_cast<uint32_t>(i * stride % count));
    rows.push_back(count - 1);
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  return rows;
}

uint32_t bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}

// ignis_image_gemm.h's quantization of one fp32 row, on the host.
void host_quantize(const float *y, std::size_t cols, uint8_t *codes, float *scale) {
  float amax = 0.0f;
  for (std::size_t c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(y[c]));
  const float floor = 0x1.cp-118f;
  *scale = amax >= floor ? amax / 448.0f : 1.0f;
  const float inverse = amax >= floor ? 448.0f / amax : 0.0f;
  for (std::size_t c = 0; c < cols; ++c) codes[c] = __nv_cvt_float_to_fp8(y[c] * inverse, __NV_SATFINITE, __NV_E4M3);
}

double arg_double(int argc, char **argv, const char *name, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
  }
  return fallback;
}

// ---- norm ---------------------------------------------------------------------------------------

void case_norm(double factor, uint32_t tokens) {
  const uint32_t cols = kHidden;
  const std::size_t n = static_cast<std::size_t>(tokens) * cols;
  const std::string what = "residual norm " + std::to_string(tokens) + " x 4096";
  const std::vector<float> x0 = profiled_rows(61, tokens, cols);
  std::vector<float> delta(n);
  for (std::size_t i = 0; i < n; ++i) delta[i] = (i / cols) % 8 == 5 ? 0.0f : hash_uniform(62, i, 0.75f);
  const std::vector<uint16_t> w = bf16_row(63, cols, 1.0f, 0.3f);
  DeviceBytes x(n * 4), d(n * 4), dw(cols * 2);
  upload(x, x0);
  upload(d, delta);
  upload(dw, w);
  Poisoned y32(n * 4), codes(n), scales(tokens * 4ull);

  // add + norm (fp32 out), then the norm alone on the sum (E4M3 out), then the add alone.
  ENC_RC(ignis_image_encoder_residual_norm(x.as<float>(), d.as<float>(), tokens, cols, dw.p, kEps, y32.p, nullptr, 1,
                                           nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> sum1 = download<float>(x.p, n);
  ENC_RC(ignis_image_encoder_residual_norm(x.as<float>(), nullptr, tokens, cols, dw.p, kEps, codes.p,
                                           scales.as<float>(), 0, nullptr));
  ENC_RC(ignis_image_encoder_residual_norm(x.as<float>(), d.as<float>(), tokens, cols, nullptr, kEps, nullptr, nullptr,
                                           0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> sum2 = download<float>(x.p, n);
  long long add_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float once = x0[i] + delta[i];
    add_diff += bits(sum1[i]) != bits(once) || bits(sum2[i]) != bits(once + delta[i]);
  }
  check(add_diff == 0, what + ": " + std::to_string(add_diff) + " sums differ from the host's fp32 add");

  const std::vector<float> y = download<float>(y32.p, n);
  Verdict v;
  for (uint32_t r : sample_rows(tokens, 48)) {
    std::vector<double> xr(cols);
    for (uint32_t c = 0; c < cols; ++c) xr[c] = sum1[static_cast<std::size_t>(r) * cols + c];
    const std::vector<Err> ref = ref_rms_norm(xr, w, 8 * (cols / 8 / 256) + 5 + 8);
    for (uint32_t c = 0; c < cols; ++c) v.take(y[static_cast<std::size_t>(r) * cols + c], ref[c], factor);
  }
  report(what, v);

  // The E4M3 output is the host's quantization of the fp32 output, bit for bit, every row.
  const std::vector<uint8_t> got_codes = download<uint8_t>(codes.p, n);
  const std::vector<float> got_scales = download<float>(scales.p, tokens);
  std::vector<uint8_t> want_codes(cols);
  long long code_diff = 0, scale_diff = 0;
  for (uint32_t r = 0; r < tokens; ++r) {
    float want_scale = 0.0f;
    host_quantize(y.data() + static_cast<std::size_t>(r) * cols, cols, want_codes.data(), &want_scale);
    scale_diff += bits(got_scales[r]) != bits(want_scale);
    for (uint32_t c = 0; c < cols; ++c) code_diff += got_codes[static_cast<std::size_t>(r) * cols + c] != want_codes[c];
  }
  check(code_diff == 0 && scale_diff == 0, what + ": E4M3 differs from the host's quantization of the fp32 output (" +
                                               std::to_string(code_diff) + " codes, " + std::to_string(scale_diff) +
                                               " scales)");
}

// ---- qk -----------------------------------------------------------------------------------------

// The mrope's frequency of slot i (ignis_image_encoder.h): fl32(1 / fl32(theta^fl32(2i / 128))).
float slot_freq(int i) {
  const float e = static_cast<float>(2 * i) / 128.0f;
  const float p = static_cast<float>(std::pow(static_cast<double>(kTheta), static_cast<double>(e)));
  return static_cast<float>(1.0 / static_cast<double>(p));
}

// transformers' interleaved rule, and the chunked layout a negative control uses.
int slot_axis(int i) { return (i % 3 == 1 && i < 60) ? 1 : ((i % 3 == 2 && i < 60) ? 2 : 0); }
int chunked_axis(int i) { return i < 24 ? 0 : (i < 44 ? 1 : 2); }

enum Layout { kRotateHalf, kChunkedAxes, kAdjacentPairs };

// One head's 128 outputs: the norm, then the rotation of each pair by its slot's angle.
std::vector<Err> ref_qk_head(const std::vector<double> &x, const std::vector<uint16_t> &w, const int32_t *pos,
                             uint32_t tokens, uint32_t t, Layout layout) {
  const std::vector<Err> n = ref_rms_norm(x, w, 4 + 5);
  std::vector<Err> out(128);
  for (int i = 0; i < 64; ++i) {
    const int axis = layout == kChunkedAxes ? chunked_axis(i) : slot_axis(i);
    const float angle = static_cast<float>(pos[axis * tokens + t]) * slot_freq(i);
    const Err c = within2ulp(std::cos(static_cast<double>(angle)), 0.0);
    const Err s = within2ulp(std::sin(static_cast<double>(angle)), 0.0);
    const int a = layout == kAdjacentPairs ? 2 * i : i, b = layout == kAdjacentPairs ? 2 * i + 1 : i + 64;
    out[a] = sub(mul(n[a], c), mul(n[b], s));
    out[b] = add(mul(n[b], c), mul(n[a], s));
  }
  return out;
}

void case_qk(double factor, uint32_t tokens) {
  const uint32_t q_stride = kHidden + 64, k_stride = kKv + 32;  // strided rows: the op reads only its heads
  const std::string what = "qk norm + mrope " + std::to_string(tokens) + " tokens";
  const std::vector<float> q = profiled_rows(71, tokens, q_stride), k = profiled_rows(72, tokens, k_stride);
  const std::vector<uint16_t> qw = bf16_row(73, 128, 1.0f, 0.3f), kw = bf16_row(74, 128, 0.8f, 0.5f);
  // Three different rows: t a text-like ramp, h and w far from it (w up to 40,000).
  std::vector<int32_t> pos(3 * static_cast<std::size_t>(tokens));
  for (uint32_t t = 0; t < tokens; ++t) {
    pos[t] = static_cast<int32_t>(t + 7);
    pos[tokens + t] = static_cast<int32_t>((t * 37) % 211) - 50;
    pos[2 * tokens + t] = 40000 - 3 * static_cast<int32_t>(t);
  }
  DeviceBytes dq(q.size() * 4), dk(k.size() * 4), dqw(256), dkw(256), dpos(pos.size() * 4);
  upload(dq, q);
  upload(dk, k);
  upload(dqw, qw);
  upload(dkw, kw);
  upload(dpos, pos);
  Poisoned qo(static_cast<std::size_t>(tokens) * kHidden * 2), ko(static_cast<std::size_t>(tokens) * kKv * 2);
  ENC_RC(ignis_image_encoder_qk_norm_mrope(dq.as<float>(), q_stride, dk.as<float>(), k_stride, tokens, dqw.p, dkw.p,
                                           kEps, dpos.as<int32_t>(), kTheta, kSection, qo.p, ko.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint16_t> gq = download<uint16_t>(qo.p, static_cast<std::size_t>(tokens) * kHidden);
  const std::vector<uint16_t> gk = download<uint16_t>(ko.p, static_cast<std::size_t>(tokens) * kKv);
  Verdict v, chunked, adjacent;
  for (uint32_t t : sample_rows(tokens, 24)) {
    for (int head = 0; head < 40; ++head) {
      const bool is_k = head >= 32;
      const int h = is_k ? head - 32 : head;
      const float *src = is_k ? k.data() + static_cast<std::size_t>(t) * k_stride : q.data() + static_cast<std::size_t>(t) * q_stride;
      std::vector<double> x(128);
      for (int c = 0; c < 128; ++c) x[c] = src[h * 128 + c];
      const uint16_t *got = is_k ? gk.data() + static_cast<std::size_t>(t) * kKv : gq.data() + static_cast<std::size_t>(t) * kHidden;
      const std::vector<Err> ref = ref_qk_head(x, is_k ? kw : qw, pos.data(), tokens, t, kRotateHalf);
      const std::vector<Err> wrong_axes = ref_qk_head(x, is_k ? kw : qw, pos.data(), tokens, t, kChunkedAxes);
      const std::vector<Err> wrong_pairs = ref_qk_head(x, is_k ? kw : qw, pos.data(), tokens, t, kAdjacentPairs);
      const bool flat = t % 8 == 5 || t % 8 == 6;  // zero and constant rows: a rotation of nothing, or of a constant
      for (int c = 0; c < 128; ++c) {
        const double g = bf16_to_f32(got[h * 128 + c]);
        v.take(g, ref[c], factor, kBf16Round);
        if (!flat) {
          chunked.take(g, wrong_axes[c], factor, kBf16Round);
          adjacent.take(g, wrong_pairs[c], factor, kBf16Round);
        }
      }
    }
  }
  report(what, v);
  std::printf("    negative controls: chunked axes %lld of %lld over, adjacent pairs %lld of %lld over\n", chunked.over,
              chunked.checked, adjacent.over, adjacent.checked);
  check(chunked.over * 4 > chunked.checked, what + ": the reference with the axes in chunks passes (negative control)");
  check(adjacent.over * 2 > adjacent.checked, what + ": the reference with adjacent pairs passes (negative control)");
}

// ---- rows ---------------------------------------------------------------------------------------

void case_rows() {
  const uint32_t tokens = 37, cols = kHidden, guard = 2;
  const std::vector<int32_t> index = {3, 4, 5, 36, 20, 37};  // 37 is past the sequence: skipped
  const uint32_t rows = static_cast<uint32_t>(index.size());
  const std::size_t n = static_cast<std::size_t>(tokens + guard) * cols;
  const std::vector<float> x0 = profiled_rows(81, tokens + guard, cols);
  std::vector<uint16_t> src(static_cast<std::size_t>(rows) * cols);
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = f32_to_bf16(hash_uniform(82, i, 3.0f));
  DeviceBytes x(n * 4), di(index.size() * 4), ds(src.size() * 2);
  upload(di, index);
  upload(ds, src);
  for (uint32_t add = 0; add < 2; ++add) {
    upload(x, x0);
    ENC_RC(ignis_image_encoder_rows(x.as<float>(), tokens, cols, di.as<int32_t>(), rows, ds.p, add, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    std::vector<float> want = x0;
    for (uint32_t j = 0; j < rows; ++j) {
      if (index[j] >= static_cast<int32_t>(tokens)) continue;
      for (uint32_t c = 0; c < cols; ++c) {
        float &e = want[static_cast<std::size_t>(index[j]) * cols + c];
        const float f = bf16_to_f32(src[static_cast<std::size_t>(j) * cols + c]);
        e = add ? e + f : f;
      }
    }
    const std::vector<float> got = download<float>(x.p, n);
    long long diff = 0;
    for (std::size_t i = 0; i < n; ++i) diff += bits(got[i]) != bits(want[i]);
    const std::string what = add ? "rows, added" : "rows, written over";
    std::printf("  %-62s %zu values, %lld differ (row 37 skipped, the guard rows kept)\n", what.c_str(), n, diff);
    check(diff == 0, what + ": " + std::to_string(diff) + " values differ from the host");
  }
}

// ---- forward ------------------------------------------------------------------------------------

// A row-scale-v1 FP8 payload [rows][cols]: codes E4M3(u * 448), u uniform in [-1, 1), then the BF16
// row scales amplitude / 448, so a weight is about uniform in [-amplitude, amplitude).
__device__ inline uint32_t mix32(uint32_t x) {  // moe_fixture.h's lowbias32, on the device
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

__global__ void fill_codes_kernel(uint8_t *codes, std::size_t n, uint32_t stream) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const uint32_t h = mix32(static_cast<uint32_t>(e) * 0x9E3779B9u + stream * 0x85EBCA6Bu);
    const float u = static_cast<float>(h >> 8) * 0x1p-23f - 1.0f;
    codes[e] = __nv_cvt_float_to_fp8(u * 448.0f, __NV_SATFINITE, __NV_E4M3);
  }
}

struct Payload : DeviceBytes {
  Payload(uint32_t rows, uint32_t cols, uint32_t stream)
      : DeviceBytes(static_cast<std::size_t>(rows) * cols + static_cast<std::size_t>(rows) * 2) {
    const std::size_t n = static_cast<std::size_t>(rows) * cols;  // a multiple of 256 at every encoder shape
    fill_codes_kernel<<<2048, 256>>>(as<uint8_t>(), n, stream);
    MOE_CUDA(cudaGetLastError());
    // amplitude sqrt(3 / cols): the output's RMS about the input's
    const float amplitude = std::sqrt(3.0f / static_cast<float>(cols));
    const std::vector<uint16_t> scales(rows, f32_to_bf16(amplitude / 448.0f));
    MOE_CUDA(cudaMemcpy(as<uint8_t>() + n, scales.data(), rows * 2ull, cudaMemcpyHostToDevice));
  }
};

struct Norm : DeviceBytes {
  Norm(uint32_t n, uint32_t stream, float centre, float spread) : DeviceBytes(n * 2ull) {
    upload(*this, bf16_row(stream, n, centre, spread));
  }
};

struct LayerSet {
  Norm input, post, qn, kn;
  Payload q, k, v, o, gate, up, down;
  explicit LayerSet(uint32_t s)
      : input(kHidden, s, 1.0f, 0.2f), post(kHidden, s + 1, 1.0f, 0.2f), qn(128, s + 2, 1.0f, 0.3f),
        kn(128, s + 3, 1.0f, 0.3f), q(kHidden, kHidden, s + 4), k(kKv, kHidden, s + 5), v(kKv, kHidden, s + 6),
        o(kHidden, kHidden, s + 7), gate(kInter, kHidden, s + 8), up(kInter, kHidden, s + 9),
        down(kHidden, kInter, s + 10) {}
  ignis_image_encoder_layer layer() const {
    return {input.p, q.p, k.p, v.p, qn.p, kn.p, o.p, post.p, gate.p, up.p, down.p};
  }
};

void case_forward() {
  ENC_RC(ignis_image_gemm_prepare());
  const uint32_t tokens = 37;
  const LayerSet a(1000), b(2000);
  const ignis_image_encoder_layer layers[3] = {a.layer(), b.layer(), a.layer()};
  const ignis_image_encoder_weights weights{layers, 3, kEps, kTheta, {kSection[0], kSection[1], kSection[2]}};

  // The prompt: embeddings about unit RMS; 9 vision rows (8 together, one apart); mrope positions
  // with an image block's (t, h, w) at the vision rows, text continuing after it.
  const std::size_t n = static_cast<std::size_t>(tokens) * kHidden;
  std::vector<float> x0(n);
  for (std::size_t i = 0; i < n; ++i) x0[i] = hash_uniform(91, i, 1.7f);
  const std::vector<int32_t> index = {5, 6, 7, 8, 9, 10, 11, 12, 30};
  const uint32_t rows = static_cast<uint32_t>(index.size());
  std::vector<int32_t> pos(3 * static_cast<std::size_t>(tokens));
  for (uint32_t t = 0; t < tokens; ++t) {
    const bool image = t >= 5 && t <= 12;
    const int32_t text = t < 5 ? static_cast<int32_t>(t) : static_cast<int32_t>(t) - 8 + 5 + 2;
    pos[t] = image ? 5 : text;
    pos[tokens + t] = image ? 5 + static_cast<int32_t>(t - 5) / 4 : text;
    pos[2 * tokens + t] = image ? 5 + static_cast<int32_t>(t - 5) % 4 : text;
  }
  // The features: embeddings, then DeepStack 0-2, BF16 [rows][4096] each, one after the other.
  const std::size_t per = static_cast<std::size_t>(rows) * kHidden;
  std::vector<std::vector<uint16_t>> feat(4, std::vector<uint16_t>(per));
  std::vector<uint16_t> all_feat;
  for (uint32_t f = 0; f < 4; ++f) {
    for (std::size_t i = 0; i < per; ++i) feat[f][i] = f32_to_bf16(hash_uniform(92 + f, i, f == 0 ? 2.0f : 0.5f));
    all_feat.insert(all_feat.end(), feat[f].begin(), feat[f].end());
  }
  DeviceBytes x(n * 4), dpos(pos.size() * 4), di(index.size() * 4), df(all_feat.size() * 2),
      ws(ignis_image_encoder_workspace_bytes(tokens));
  upload(dpos, pos);
  upload(di, index);
  upload(df, all_feat);
  const uint16_t *f0 = df.as<uint16_t>();
  auto run = [&](uint32_t first, uint32_t end, const ignis_image_encoder_vision *vision) {
    ENC_RC(ignis_image_encoder_forward(&weights, first, end, x.as<float>(), tokens, dpos.as<int32_t>(), vision, ws.p,
                                       ws.bytes, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  };
  const ignis_image_encoder_vision hooks{rows, di.as<int32_t>(), f0, {f0 + per, f0 + 2 * per, f0 + 3 * per}};
  const ignis_image_encoder_vision no_deepstack{rows, di.as<int32_t>(), f0, {nullptr, nullptr, nullptr}};

  upload(x, x0);
  run(0, 3, &hooks);
  const std::vector<float> full = download<float>(x.p, n);
  upload(x, x0);
  run(0, 3, &hooks);
  const std::vector<float> again = download<float>(x.p, n);
  upload(x, x0);
  run(0, 3, &no_deepstack);
  const std::vector<float> plain = download<float>(x.p, n);

  // The same additions on the host, between one-layer calls.
  std::vector<float> h = x0;
  auto host_rows = [&](const std::vector<uint16_t> &f, bool add) {
    for (uint32_t j = 0; j < rows; ++j) {
      for (uint32_t c = 0; c < kHidden; ++c) {
        float &e = h[static_cast<std::size_t>(index[j]) * kHidden + c];
        const float v = bf16_to_f32(f[static_cast<std::size_t>(j) * kHidden + c]);
        e = add ? e + v : v;
      }
    }
  };
  host_rows(feat[0], false);
  for (uint32_t l = 0; l < 3; ++l) {
    upload(x, h);
    run(l, l + 1, nullptr);
    h = download<float>(x.p, n);
    host_rows(feat[1 + l], true);
  }

  long long chain_diff = 0, again_diff = 0, plain_same = 0, nonfinite = 0;
  double ss = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    chain_diff += bits(full[i]) != bits(h[i]);
    again_diff += bits(full[i]) != bits(again[i]);
    nonfinite += !std::isfinite(full[i]);
    ss += static_cast<double>(full[i]) * full[i];
  }
  long long vision_values = 0;
  for (int32_t r : index) {
    for (uint32_t c = 0; c < kHidden; ++c) {
      ++vision_values;
      plain_same += bits(full[static_cast<std::size_t>(r) * kHidden + c]) == bits(plain[static_cast<std::size_t>(r) * kHidden + c]);
    }
  }
  std::printf("  forward, 3 layers (A B A), 37 tokens, 9 vision rows: output RMS %.3f, %lld non-finite\n",
              std::sqrt(ss / static_cast<double>(n)), nonfinite);
  std::printf("    hooks vs the host's additions between one-layer calls: %lld of %zu values differ\n", chain_diff, n);
  std::printf("    a second run: %lld differ; without DeepStack: %lld of %lld vision values unchanged\n", again_diff,
              plain_same, vision_values);
  check(nonfinite == 0, "forward: non-finite outputs");
  check(chain_diff == 0, "forward: the hooks differ from the host's additions between one-layer calls");
  check(again_diff == 0, "forward: a second run differs");
  check(plain_same * 100 < vision_values, "forward: the output barely changes without DeepStack (negative control)");
}

// ---- refusals -----------------------------------------------------------------------------------

void refusals() {
  std::printf("refusals:\n");
  DeviceBytes a(1 << 22), b(1 << 22), w(1 << 16), p(1 << 16);
  auto refused = [](int32_t rc, const char *what) {
    check(rc != 0, std::string("refuses ") + what);
    if (rc != 0) std::printf("  %s: %s\n", what, ignis_image_encoder_last_error());
  };
  float *x = a.as<float>(), *s = b.as<float>();
  const int32_t *pos = p.as<int32_t>();
  refused(ignis_image_encoder_residual_norm(x, nullptr, 4, 64, nullptr, kEps, nullptr, nullptr, 0, nullptr),
          "neither delta nor weight");
  refused(ignis_image_encoder_residual_norm(x, nullptr, 4, 8200, w.p, kEps, b.p, s, 0, nullptr), "cols > 8192");
  refused(ignis_image_encoder_residual_norm(x, nullptr, 4, 60, w.p, kEps, b.p, s, 0, nullptr), "cols % 8");
  refused(ignis_image_encoder_residual_norm(x, nullptr, 4, 64, w.p, kEps, b.p, nullptr, 0, nullptr),
          "E4M3 without scales");
  refused(ignis_image_encoder_residual_norm(x + 1, nullptr, 4, 64, w.p, kEps, b.p, s, 0, nullptr), "x misaligned");
  const uint32_t bad_sum[3] = {24, 20, 21}, bad_h[3] = {0, 22, 42};
  refused(ignis_image_encoder_qk_norm_mrope(x, 4096, x, 1024, 4, w.p, w.p, kEps, pos, kTheta, bad_sum, b.p, b.p, nullptr),
          "a section not summing to 64");
  refused(ignis_image_encoder_qk_norm_mrope(x, 4096, x, 1024, 4, w.p, w.p, kEps, pos, kTheta, bad_h, b.p, b.p, nullptr),
          "3 section[1] > 64");
  refused(ignis_image_encoder_qk_norm_mrope(x, 4000, x, 1024, 4, w.p, w.p, kEps, pos, kTheta, kSection, b.p, b.p, nullptr),
          "q_stride < 4096");
  refused(ignis_image_encoder_qk_norm_mrope(x, 4096, x, 1024, 4, w.p, w.p, kEps, pos, 1.0f, kSection, b.p, b.p, nullptr),
          "theta 1");
  refused(ignis_image_encoder_rows(x, 4, 64, pos, 5, w.p, 0, nullptr), "more rows than tokens");
  const ignis_image_encoder_layer layer{w.p, w.p, w.p, w.p, w.p, w.p, w.p, w.p, w.p, nullptr, w.p};
  const ignis_image_encoder_weights weights{&layer, 1, kEps, kTheta, {24, 20, 20}};
  refused(ignis_image_encoder_forward(&weights, 0, 2, x, 4, pos, nullptr, b.p, b.bytes, nullptr), "end_layer > layers");
  refused(ignis_image_encoder_forward(&weights, 0, 1, x, 4, pos, nullptr, b.p, b.bytes, nullptr), "a NULL weight");
  refused(ignis_image_encoder_forward(&weights, 0, 0, x, 4, pos, nullptr, b.p, 1024, nullptr), "a short workspace");
}

}  // namespace

int main(int argc, char **argv) {
  const double factor = arg_double(argc, argv, "--bound-factor", 1.0);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "norm,qk,rows,forward,refusals";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("image text encoder (bound factor %.3f; cases %s)\n", factor, cases.c_str());
  if (want("norm")) {
    std::printf("residual add + RMSNorm:\n");
    for (uint32_t tokens : {1u, 45u, 2049u}) case_norm(factor, tokens);
  }
  if (want("qk")) {
    std::printf("QK RMSNorm + interleaved mrope:\n");
    for (uint32_t tokens : {1u, 45u, 1100u}) case_qk(factor, tokens);
  }
  if (want("rows")) {
    std::printf("vision rows:\n");
    case_rows();
  }
  if (want("forward")) {
    std::printf("forward:\n");
    case_forward();
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_encoder: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_encoder: OK\n");
  return 0;
}
