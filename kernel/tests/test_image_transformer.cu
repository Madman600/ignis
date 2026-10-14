// The image model's diffusion transformer (spec image-model/04, GitHub #321) -- OURS. The program
// and its ops are kernel/include/ignis_image_transformer.h's; every reference is our own fp64 CPU
// code, and the program's checks are equalities between calls of its own.
//
//   norm        the gated residual add with the modulated LayerNorm, 1 .. 2049 tokens x 4096: the
//               add against the host's fp32 fma, the norm's fp32 output against fp64, the E4M3
//               output against the host's quantization of that fp32 output, bit for bit.
//   linear      the BF16 linear at the timestep embedding's shapes against fp64.
//   quantize    the fp32 per-token E4M3 quantization against the host's, bit for bit.
//   modulation  the timestep embedding on synthetic BF16 weights at three sigmas against an fp64
//               restatement of diffusers' chain (QwenImage21TemporalTimesteps, the embedder, the
//               shared modulation, norm_out.linear), with cos and sin swapped as a negative control.
//   program     txt_in and img_in per row (a call on all rows equals calls on parts of them, bit for
//               bit), img_in against fp64; the blocks on synthetic FP8 weights (two blocks, A B):
//               the prefix pass (13 tokens: text, a 2 x 3 image, text) and the target pass (70
//               tokens) run whole equal the same blocks run one call each, bit for bit, runs
//               included; a second run is identical; the target pass reads the prefix (another
//               prefix changes it); the head's Euler update against the host's on its own velocity,
//               bit for bit; the prefix pass in chunks (834 tokens: text | image | text | image |
//               a long text, chunks of 128 and 384) equals the whole pass, bit for bit, runs
//               included.
//   refusals    the calls' argument checks.
//
// The bound (test_image_elementwise.cu's error-tracking reference): each value v (exact, in fp64)
// carries e, a bound on how far the kernel's fp32 value can be from it. Every fp32 operation
// rounded to nearest adds 2^-24 of its result; rsqrtf adds 2 ulp (CUDA Math API, no fast math); a
// sum whose terms go through at most d additions adds gamma_d of sum |terms|. The bound is
// multiplied by --bound-factor F (default 1).
//
//   ignis_kernel_image_transformer_test [--bound-factor F]
//                                       [--cases norm,linear,quantize,modulation,program,refusals]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_transformer.h"
#include "ignis_image_attn.h"
#include "ignis_image_dit.h"
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

#define TF_RC(expr)                                                                                \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_transformer_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

constexpr float kEps = 1e-6f;  // the transformer config's eps
constexpr uint32_t kHidden = 4096, kMlp = 12288, kMod = 4 * kHidden;

// ---- the error-tracking reference (test_image_elementwise.cu's) ---------------------------------

constexpr double kU = 0x1p-24;
constexpr double kTiny = 0x1p-149;
constexpr double kUlp2 = 0x1p-22;
constexpr double kRefSlack = 0x1p-40;

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

// y = ((x - mu) * rsqrt(var + eps)) * (1 + s), mu and var each a sum through `depth` additions.
std::vector<Err> ref_layer_norm(const std::vector<double> &x, const float *s, int depth) {
  const std::size_t n = x.size();
  std::vector<Err> terms(n);
  for (std::size_t c = 0; c < n; ++c) terms[c] = exact(x[c]);
  const Err mean = div(sum(terms, depth), exact(static_cast<double>(n)));
  std::vector<Err> centred(n), sq(n);
  for (std::size_t c = 0; c < n; ++c) {
    centred[c] = sub(exact(x[c]), mean);
    sq[c] = mul(centred[c], centred[c]);
  }
  const Err r = rsqrt_f(add(div(sum(sq, depth), exact(static_cast<double>(n))), exact(kEps)));
  std::vector<Err> y(n);
  for (std::size_t c = 0; c < n; ++c) y[c] = mul(mul(centred[c], r), add(exact(1.0), exact(s[c])));
  return y;
}

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

std::vector<float> uniform(uint32_t stream, std::size_t n, float centre, float spread) {
  std::vector<float> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = centre + hash_uniform(stream, i, spread);
  return v;
}

std::vector<uint16_t> bf16_values(uint32_t stream, std::size_t n, float centre, float spread) {
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
  const std::string what = "residual + modulated LayerNorm " + std::to_string(tokens) + " x 4096";
  const std::vector<float> x0 = profiled_rows(61, tokens, cols);
  std::vector<float> delta(n);
  for (std::size_t i = 0; i < n; ++i) delta[i] = (i / cols) % 8 == 5 ? 0.0f : hash_uniform(62, i, 3.0f);
  const std::vector<float> gate = uniform(63, cols, 0.0f, 0.9f), scale = uniform(64, cols, 0.0f, 0.6f);
  DeviceBytes x(n * 4), d(n * 4), dg(cols * 4), ds(cols * 4);
  upload(x, x0);
  upload(d, delta);
  upload(dg, gate);
  upload(ds, scale);
  Poisoned y32(n * 4), codes(n), scales(tokens * 4ull);

  // add + norm (fp32 out), then the norm alone on the sum (E4M3 out), then the add alone.
  TF_RC(ignis_image_transformer_residual_norm(x.as<float>(), d.as<float>(), dg.as<float>(), tokens, cols,
                                              ds.as<float>(), kEps, y32.p, nullptr, 1, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> sum1 = download<float>(x.p, n);
  TF_RC(ignis_image_transformer_residual_norm(x.as<float>(), nullptr, nullptr, tokens, cols, ds.as<float>(), kEps,
                                              codes.p, scales.as<float>(), 0, nullptr));
  TF_RC(ignis_image_transformer_residual_norm(x.as<float>(), d.as<float>(), dg.as<float>(), tokens, cols, nullptr, kEps,
                                              nullptr, nullptr, 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> sum2 = download<float>(x.p, n);
  long long add_diff = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float g = gate[i % cols];
    const float once = std::fmaf(g, delta[i], x0[i]);
    add_diff += bits(sum1[i]) != bits(once) || bits(sum2[i]) != bits(std::fmaf(g, delta[i], once));
  }
  check(add_diff == 0, what + ": " + std::to_string(add_diff) + " sums differ from the host's fp32 fma");

  const std::vector<float> y = download<float>(y32.p, n);
  Verdict v;
  for (uint32_t r : sample_rows(tokens, 48)) {
    std::vector<double> xr(cols);
    for (uint32_t c = 0; c < cols; ++c) xr[c] = sum1[static_cast<std::size_t>(r) * cols + c];
    const std::vector<Err> ref = ref_layer_norm(xr, scale.data(), 8 * (cols / 8 / 256) + 5 + 8);
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

// ---- linear -------------------------------------------------------------------------------------

void case_linear(double factor, uint32_t rows, uint32_t cols, uint32_t n) {
  const std::string what = "BF16 linear [" + std::to_string(rows) + " x " + std::to_string(cols) + "], " +
                           std::to_string(n) + " input(s)";
  const float amplitude = std::sqrt(3.0f / static_cast<float>(cols));
  const std::vector<uint16_t> w = bf16_values(31, static_cast<std::size_t>(rows) * cols, 0.0f, amplitude);
  const std::vector<float> x = profiled_rows(32, n, cols);
  DeviceBytes dw(w.size() * 2), dx(x.size() * 4);
  upload(dw, w);
  upload(dx, x);
  Poisoned y(static_cast<std::size_t>(rows) * n * 4);
  TF_RC(ignis_image_transformer_linear_bf16(dw.p, rows, cols, dx.as<float>(), n, y.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> got = download<float>(y.p, static_cast<std::size_t>(rows) * n);
  const int depth = static_cast<int>((cols / 8 + 31) / 32) * 8 + 5;
  Verdict v;
  for (uint32_t s = 0; s < n; ++s) {
    for (uint32_t r : sample_rows(rows, 512)) {
      std::vector<Err> terms(cols);
      for (uint32_t c = 0; c < cols; ++c) {
        terms[c] = mul(exact(bf16_to_f32(w[static_cast<std::size_t>(r) * cols + c])),
                       exact(x[static_cast<std::size_t>(s) * cols + c]));
      }
      v.take(got[static_cast<std::size_t>(s) * rows + r], sum(terms, depth), factor);
    }
  }
  report(what, v);
}

// ---- quantize -----------------------------------------------------------------------------------

void case_quantize(uint32_t tokens, uint32_t cols) {
  const std::string what = "fp32 E4M3 quantization " + std::to_string(tokens) + " x " + std::to_string(cols);
  const std::vector<float> x = profiled_rows(41, tokens, cols);
  DeviceBytes dx(x.size() * 4);
  upload(dx, x);
  Poisoned codes(x.size()), scales(tokens * 4ull);
  TF_RC(ignis_image_transformer_quantize_f32(dx.as<float>(), tokens, cols, codes.p, scales.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<uint8_t> got = download<uint8_t>(codes.p, x.size());
  const std::vector<float> got_scales = download<float>(scales.p, tokens);
  std::vector<uint8_t> want(cols);
  long long diff = 0;
  for (uint32_t t = 0; t < tokens; ++t) {
    float scale = 0.0f;
    host_quantize(x.data() + static_cast<std::size_t>(t) * cols, cols, want.data(), &scale);
    diff += bits(got_scales[t]) != bits(scale);
    for (uint32_t c = 0; c < cols; ++c) diff += got[static_cast<std::size_t>(t) * cols + c] != want[c];
  }
  std::printf("  %-62s %zu codes, %lld differ from the host\n", what.c_str(), x.size(), diff);
  check(diff == 0, what + ": " + std::to_string(diff) + " codes or scales differ from the host's");
}

// ---- modulation ---------------------------------------------------------------------------------

// y = W x in fp64, W BF16 [rows][cols].
std::vector<double> host_linear(const std::vector<uint16_t> &w, uint32_t rows, uint32_t cols,
                                const std::vector<double> &x) {
  std::vector<double> y(rows, 0.0);
  for (uint32_t r = 0; r < rows; ++r) {
    double acc = 0.0;
    for (uint32_t c = 0; c < cols; ++c) acc += static_cast<double>(bf16_to_f32(w[static_cast<std::size_t>(r) * cols + c])) * x[c];
    y[r] = acc;
  }
  return y;
}

std::vector<double> host_silu(const std::vector<double> &x) {
  std::vector<double> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = x[i] / (1.0 + std::exp(-x[i]));
  return y;
}

double rel_l2(const std::vector<float> &got, const std::vector<double> &want) {
  double e = 0.0, n = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    e += (got[i] - want[i]) * (got[i] - want[i]);
    n += want[i] * want[i];
  }
  return std::sqrt(e / n);
}

void case_modulation() {
  // Synthetic weights about the checkpoint's scale: each linear's output about its input's RMS.
  const std::vector<uint16_t> l1 = bf16_values(51, kHidden * 256ull, 0.0f, std::sqrt(3.0f / 256.0f));
  const std::vector<uint16_t> l2 = bf16_values(52, kHidden * 4096ull, 0.0f, std::sqrt(3.0f / 4096.0f));
  const std::vector<uint16_t> mw = bf16_values(53, kMod * 4096ull, 0.0f, 0.5f * std::sqrt(3.0f / 4096.0f));
  const std::vector<uint16_t> ow = bf16_values(54, kHidden * 4096ull, 0.0f, 0.5f * std::sqrt(3.0f / 4096.0f));
  DeviceBytes d1(l1.size() * 2), d2(l2.size() * 2), dm(mw.size() * 2), dout(ow.size() * 2), any(1 << 16);
  upload(d1, l1);
  upload(d2, l2);
  upload(dm, mw);
  upload(dout, ow);
  // The model's other weights are not read: any aligned pointer.
  ignis_image_transformer_weights weights{any.p, any.p, any.p, any.p, d1.p, d2.p, dm.p, dout.p, any.p, nullptr, 0, kEps};
  DeviceBytes mod(kMod * 4ull), out(kHidden * 4ull), ws(ignis_image_transformer_workspace_bytes(1));
  const float c = static_cast<float>(-std::log(10000.0));
  for (float sigma : {0.0f, 0.978453f, 0.414568f}) {
    TF_RC(ignis_image_transformer_modulation(&weights, sigma, mod.as<float>(), out.as<float>(), ws.p, ws.bytes, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    const std::vector<float> got_mod = download<float>(mod.p, kMod), got_out = download<float>(out.p, kHidden);
    // diffusers' chain in fp64 from the same fp32 angles; the negative control swaps cos and sin.
    std::vector<double> feat(256), swapped(256);
    const float t = 1000.0f * sigma;
    for (int i = 0; i < 128; ++i) {
      const float f = static_cast<float>(std::exp(static_cast<double>((c * static_cast<float>(i)) / 128.0f)));
      const double a = static_cast<double>(t * f);
      feat[i] = std::cos(a), feat[128 + i] = std::sin(a);
      swapped[i] = feat[128 + i], swapped[128 + i] = feat[i];
    }
    auto chain = [&](const std::vector<double> &features, std::vector<double> &want_mod, std::vector<double> &want_out) {
      const std::vector<double> temb = host_linear(l2, kHidden, 4096, host_silu(host_linear(l1, kHidden, 256, features)));
      const std::vector<double> st = host_silu(temb);
      want_mod = host_linear(mw, kMod, 4096, st);
      want_out = host_linear(ow, kHidden, 4096, st);
    };
    std::vector<double> want_mod, want_out, wrong_mod, wrong_out;
    chain(feat, want_mod, want_out);
    chain(swapped, wrong_mod, wrong_out);
    const double e_mod = rel_l2(got_mod, want_mod), e_out = rel_l2(got_out, want_out);
    const double w_mod = rel_l2(got_mod, wrong_mod);
    std::printf("  modulation at sigma %.6f: mod %.2e, out_scale %.2e rel L2 (cos and sin swapped: %.2e)\n", sigma, e_mod,
                e_out, w_mod);
    check(e_mod <= 1e-5 && e_out <= 1e-5, "modulation: over 1e-5 of the fp64 chain");
    if (sigma > 0.0f) check(w_mod > 1e-2, "modulation: the chain with cos and sin swapped passes (negative control)");
  }
}

// ---- program ------------------------------------------------------------------------------------

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
  uint32_t rows, cols;
  float amplitude;
  Payload(uint32_t r, uint32_t c, uint32_t stream, float gain = 1.0f)
      : DeviceBytes(((static_cast<std::size_t>(r) * c + 255) & ~std::size_t{255}) + static_cast<std::size_t>(r) * 2),
        rows(r), cols(c), amplitude(gain * std::sqrt(3.0f / static_cast<float>(c))) {
    const std::size_t n = static_cast<std::size_t>(r) * c;
    fill_codes_kernel<<<2048, 256>>>(as<uint8_t>(), n, stream);
    MOE_CUDA(cudaGetLastError());
    const std::vector<uint16_t> scales(r, f32_to_bf16(amplitude / 448.0f));
    MOE_CUDA(cudaMemcpy(as<uint8_t>() + ((n + 255) & ~std::size_t{255}), scales.data(), r * 2ull,
                        cudaMemcpyHostToDevice));
  }
};

struct Bf16 : DeviceBytes {
  Bf16(std::size_t n, uint32_t stream, float centre, float spread) : DeviceBytes(n * 2) {
    upload(*this, bf16_values(stream, n, centre, spread));
  }
};

struct BlockSet {
  Payload q, k, v, o, gate, up, down;
  Bf16 qn, kn;
  explicit BlockSet(uint32_t s)
      : q(kHidden, kHidden, s), k(kHidden, kHidden, s + 1), v(kHidden, kHidden, s + 2), o(kHidden, kHidden, s + 3),
        gate(kMlp, kHidden, s + 4), up(kMlp, kHidden, s + 5), down(kHidden, kMlp, s + 6), qn(128, s + 7, 1.0f, 0.3f),
        kn(128, s + 8, 1.0f, 0.3f) {}
  ignis_image_transformer_block block() const { return {q.p, k.p, v.p, o.p, qn.p, kn.p, gate.p, up.p, down.p}; }
};

long long differ(const std::vector<float> &a, const std::vector<float> &b) {
  long long d = 0;
  for (std::size_t i = 0; i < a.size(); ++i) d += bits(a[i]) != bits(b[i]);
  return d;
}

void case_program() {
  TF_RC(ignis_image_gemm_prepare());
  const BlockSet a(1000), b(2000);
  const Payload img_in(kHidden, 64, 3000), txt_in(kHidden, kHidden, 3001), txt_out(kHidden, kHidden, 3002),
      proj_out(64, kHidden, 3003);
  const Bf16 txt_norm(kHidden, 3004, 0.0f, 0.2f);
  DeviceBytes any(1 << 16);
  const ignis_image_transformer_block blocks[2] = {a.block(), b.block()};
  const ignis_image_transformer_weights weights{img_in.p, txt_norm.p, txt_in.p, txt_out.p, any.p, any.p,
                                                any.p,    any.p,      proj_out.p, blocks, 2,     kEps};

  // txt_in and img_in, per row: all rows in one call equal the rows in two calls.
  const uint32_t rows = 7, latents = 70;
  DeviceBytes ws(ignis_image_transformer_workspace_bytes(latents));
  const std::vector<float> enc = profiled_rows(71, rows, kHidden);
  DeviceBytes denc(enc.size() * 4), ty(enc.size() * 4);
  upload(denc, enc);
  TF_RC(ignis_image_transformer_txt_in(&weights, denc.as<float>(), rows, ty.as<float>(), ws.p, ws.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> txt_all = download<float>(ty.p, enc.size());
  TF_RC(ignis_image_transformer_txt_in(&weights, denc.as<float>(), 3, ty.as<float>(), ws.p, ws.bytes, nullptr));
  TF_RC(ignis_image_transformer_txt_in(&weights, denc.as<float>() + 3 * kHidden, rows - 3, ty.as<float>() + 3 * kHidden,
                                       ws.p, ws.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const long long txt_diff = differ(txt_all, download<float>(ty.p, enc.size()));
  long long txt_nonfinite = 0;
  for (float v : txt_all) txt_nonfinite += !std::isfinite(v);

  const std::vector<float> lat = uniform(72, latents * 64ull, 0.0f, 2.5f);
  DeviceBytes dlat(lat.size() * 4), iy(latents * kHidden * 4ull);
  upload(dlat, lat);
  TF_RC(ignis_image_transformer_img_in(&weights, dlat.as<float>(), latents, iy.as<float>(), ws.p, ws.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> img_all = download<float>(iy.p, latents * kHidden);
  TF_RC(ignis_image_transformer_img_in(&weights, dlat.as<float>(), 33, iy.as<float>(), ws.p, ws.bytes, nullptr));
  TF_RC(ignis_image_transformer_img_in(&weights, dlat.as<float>() + 33 * 64, latents - 33, iy.as<float>() + 33 * kHidden,
                                       ws.p, ws.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const long long img_diff = differ(img_all, download<float>(iy.p, latents * kHidden));
  // img_in against fp64 on the host's own quantization of the latents (bit for bit the device's):
  // y = (scale_t * sum_c code_tc * w_rc) * wscale_r, the 64-term sum in fp32.
  const std::size_t wbytes = (static_cast<std::size_t>(kHidden) * 64 + 255) & ~std::size_t{255};
  const std::vector<uint8_t> wcodes = download<uint8_t>(img_in.p, static_cast<std::size_t>(kHidden) * 64);
  const std::vector<uint16_t> wscale = download<uint16_t>(img_in.as<uint8_t>() + wbytes, kHidden);
  double img_worst = 0.0;
  for (uint32_t t = 0; t < latents; ++t) {
    uint8_t codes[64];
    float scale = 0.0f;
    host_quantize(lat.data() + static_cast<std::size_t>(t) * 64, 64, codes, &scale);
    for (uint32_t r = 0; r < kHidden; ++r) {
      double acc = 0.0, mag = 0.0;
      for (int c = 0; c < 64; ++c) {
        const double p = static_cast<double>(e4m3_to_f32(codes[c])) * e4m3_to_f32(wcodes[static_cast<std::size_t>(r) * 64 + c]);
        acc += p;
        mag += std::fabs(p);
      }
      const double ws_r = bf16_to_f32(wscale[r]);
      const double want = acc * scale * ws_r;
      // The 64 exact products (E4M3 x E4M3) summed in fp32 by a tensor core (taken as 2^-23 an
      // addition: the accumulator's rounding is not specified as to nearest), then two scalings.
      const double bound = (64 * 2 * kU * mag + 0x1p-20 * std::fabs(acc)) * scale * ws_r + 1e-30;
      img_worst = std::max(img_worst, std::fabs(img_all[static_cast<std::size_t>(t) * kHidden + r] - want) / bound);
    }
  }
  std::printf("  txt_in, 7 rows: whole vs 3 + 4 rows %lld differ, %lld non-finite; img_in, 70 tokens: whole vs 33 + 37 "
              "%lld differ, worst %.3f of its bound against fp64\n",
              txt_diff, txt_nonfinite, img_diff, img_worst);
  check(txt_diff == 0 && txt_nonfinite == 0, "txt_in: a row depends on the others, or is not finite");
  check(img_diff == 0, "img_in: a token depends on the others");
  check(img_worst <= 1.0, "img_in: over its bound against fp64");

  // The joint sequence: the prefix (text 5 | a 2 x 3 image | text 2), then a 7 x 10 target.
  const uint32_t P = 13, N = 70;
  const ignis_image_transformer_segment segments[3] = {{0, 5, 1}, {5, 11, 0}, {11, 13, 1}};
  const uint32_t starts[2] = {5, P}, heights[2] = {2, 7}, widths[2] = {3, 10};
  std::vector<int32_t> positions(3 * (P + N));
  TF_RC(ignis_image_dit_rope_positions(P + N, 2, starts, heights, widths, positions.data()));
  DeviceBytes dpos(positions.size() * 4), rope((P + N) * 512ull);
  upload(dpos, positions);
  TF_RC(ignis_image_dit_rope_table(dpos.as<int32_t>(), P + N, rope.as<float>(), nullptr));
  const float *rope_prefix = rope.as<float>(), *rope_target = rope.as<float>() + P * 128;
  // The modulation rows (scale ~ +-0.3, gate ~ +-1.5) and norm_out's scale.
  std::vector<float> mod0(kMod), modt(kMod);
  for (uint32_t i = 0; i < kMod; ++i) {
    const bool is_gate = (i / kHidden) % 2 == 1;
    mod0[i] = hash_uniform(81, i, is_gate ? 1.5f : 0.3f);
    modt[i] = hash_uniform(82, i, is_gate ? 1.5f : 0.3f);
  }
  const std::vector<float> out_scale = uniform(83, kHidden, 0.0f, 0.3f);
  DeviceBytes dmod0(kMod * 4ull), dmodt(kMod * 4ull), dout(kHidden * 4ull);
  upload(dmod0, mod0);
  upload(dmodt, modt);
  upload(dout, out_scale);

  const uint64_t run_bytes = ignis_image_attn_kv8_bytes(P);
  DeviceBytes run0(run_bytes), run1(run_bytes), other0(run_bytes), other1(run_bytes);
  void *runs[2] = {run0.p, run1.p}, *other_runs[2] = {other0.p, other1.p};
  const ignis_image_transformer_prefix prefix{P, runs, segments, 3};
  const ignis_image_transformer_prefix other{P, other_runs, segments, 3};
  const std::vector<float> xp0 = uniform(84, P * kHidden, 0.0f, 1.7f), xt0 = uniform(85, N * kHidden, 0.0f, 1.7f);
  DeviceBytes xp(xp0.size() * 4), xt(xt0.size() * 4);
  auto blocks_run = [&](DeviceBytes &x, uint32_t tokens, uint32_t target, const float *r, const DeviceBytes &m,
                        const ignis_image_transformer_prefix &pre, uint32_t first, uint32_t end) {
    TF_RC(ignis_image_transformer_blocks(&weights, first, end, x.as<float>(), tokens, target, r, m.as<float>(), &pre,
                                         ws.p, ws.bytes, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
  };

  // The prefix pass, whole and one block a call; then another prefix (other inputs) for the control.
  upload(xp, xp0);
  blocks_run(xp, P, 0, rope_prefix, dmod0, prefix, 0, 2);
  const std::vector<float> prefix_whole = download<float>(xp.p, xp0.size());
  const std::vector<uint8_t> runs_whole = download<uint8_t>(run1.p, run_bytes);
  upload(xp, xp0);
  blocks_run(xp, P, 0, rope_prefix, dmod0, prefix, 0, 1);
  blocks_run(xp, P, 0, rope_prefix, dmod0, prefix, 1, 2);
  const long long prefix_split = differ(prefix_whole, download<float>(xp.p, xp0.size()));
  const long long run_split = runs_whole != download<uint8_t>(run1.p, run_bytes);
  upload(xp, uniform(86, P * kHidden, 0.0f, 1.7f));
  blocks_run(xp, P, 0, rope_prefix, dmod0, other, 0, 2);

  // The target pass, whole, again, one block a call, and over the other prefix.
  upload(xt, xt0);
  blocks_run(xt, N, 1, rope_target, dmodt, prefix, 0, 2);
  const std::vector<float> target_whole = download<float>(xt.p, xt0.size());
  upload(xt, xt0);
  blocks_run(xt, N, 1, rope_target, dmodt, prefix, 0, 2);
  const long long target_again = differ(target_whole, download<float>(xt.p, xt0.size()));
  upload(xt, xt0);
  blocks_run(xt, N, 1, rope_target, dmodt, prefix, 0, 1);
  blocks_run(xt, N, 1, rope_target, dmodt, prefix, 1, 2);
  const long long target_split = differ(target_whole, download<float>(xt.p, xt0.size()));
  upload(xt, xt0);
  blocks_run(xt, N, 1, rope_target, dmodt, other, 0, 2);
  const long long target_other = differ(target_whole, download<float>(xt.p, xt0.size()));
  long long nonfinite = 0;
  double ss = 0.0;
  for (float v : target_whole) {
    nonfinite += !std::isfinite(v);
    ss += static_cast<double>(v) * v;
  }
  for (float v : prefix_whole) nonfinite += !std::isfinite(v);

  // The head on the target: the velocity, then the Euler update against the host's.
  upload(xt, target_whole);
  const std::vector<float> lat0 = uniform(87, N * 64ull, 0.0f, 1.0f);
  DeviceBytes vel(N * 64 * 4ull), vel2(N * 64 * 4ull), dl(N * 64 * 4ull);
  upload(dl, lat0);
  const float sigma = 0.9784529805183411f, sigma_next = 0.9541800022125244f;
  TF_RC(ignis_image_transformer_head(&weights, xt.as<float>(), N, dout.as<float>(), vel.as<float>(), dl.as<float>(),
                                     sigma, sigma_next, ws.p, ws.bytes, nullptr));
  TF_RC(ignis_image_transformer_head(&weights, xt.as<float>(), N, dout.as<float>(), vel2.as<float>(), nullptr, 0.0f, 0.0f,
                                     ws.p, ws.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> v1 = download<float>(vel.p, N * 64), v2 = download<float>(vel2.p, N * 64);
  const std::vector<float> stepped = download<float>(dl.p, N * 64);
  const float dt = sigma_next - sigma;
  long long euler_diff = 0, vel_nonfinite = 0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    const float prod = dt * v1[i];
    euler_diff += bits(stepped[i]) != bits(lat0[i] + prod);
    vel_nonfinite += !std::isfinite(v1[i]);
  }
  const long long vel_diff = differ(v1, v2);

  std::printf("  blocks (A B), prefix 13 tokens (text 5 | image 2 x 3 | text 2), target 70: target RMS %.3f, %lld "
              "non-finite\n",
              std::sqrt(ss / static_cast<double>(target_whole.size())), nonfinite);
  std::printf("    prefix: whole vs one block a call %lld differ (block 1's run %s); target: again %lld, one block a "
              "call %lld, over another prefix %lld of %zu differ\n",
              prefix_split, run_split ? "differs" : "equal", target_again, target_split, target_other,
              target_whole.size());
  std::printf("    head: %lld non-finite velocities, %lld differ without the update; Euler vs the host %lld differ\n",
              vel_nonfinite, vel_diff, euler_diff);
  check(nonfinite == 0 && vel_nonfinite == 0, "program: non-finite outputs");
  check(prefix_split == 0 && run_split == 0, "program: the prefix pass split by block differs");
  check(target_again == 0, "program: a second target pass differs");
  check(target_split == 0, "program: the target pass split by block differs");
  check(target_other * 100 > 99 * static_cast<long long>(target_whole.size()),
        "program: the target pass barely changes over another prefix (it must read it)");
  check(vel_diff == 0, "program: the velocity depends on the update");
  check(euler_diff == 0, "program: the Euler update differs from the host's two roundings");

  // The prefix pass in chunks equals the whole pass, bit for bit, x and runs: 834 tokens, text 37 |
  // image 12 x 24 | text 9 | image 10 x 20 | text 300, in chunks of 128 (both images and the long
  // text cut into pieces, the short segments packed) and of 384, in one call and one block a call.
  const uint32_t L = 834;
  const ignis_image_transformer_segment long_segments[5] = {{0, 37, 1}, {37, 325, 0}, {325, 334, 1}, {334, 534, 0}, {534, 834, 1}};
  const uint32_t long_starts[2] = {37, 334}, long_heights[2] = {12, 10}, long_widths[2] = {24, 20};
  std::vector<int32_t> long_positions(3 * L);
  TF_RC(ignis_image_dit_rope_positions(L, 2, long_starts, long_heights, long_widths, long_positions.data()));
  DeviceBytes dlong_pos(long_positions.size() * 4), long_rope(L * 512ull);
  upload(dlong_pos, long_positions);
  TF_RC(ignis_image_dit_rope_table(dlong_pos.as<int32_t>(), L, long_rope.as<float>(), nullptr));
  const uint64_t long_run = ignis_image_attn_kv8_bytes(L);
  DeviceBytes whole0(long_run), whole1(long_run), chunk0(long_run), chunk1(long_run);
  void *whole_runs[2] = {whole0.p, whole1.p}, *chunk_runs[2] = {chunk0.p, chunk1.p};
  const ignis_image_transformer_prefix whole{L, whole_runs, long_segments, 5}, chunked{L, chunk_runs, long_segments, 5};
  const std::vector<float> xl0 = uniform(88, L * kHidden, 0.0f, 1.7f);
  DeviceBytes xl(xl0.size() * 4), ws_whole(ignis_image_transformer_workspace_bytes(L)),
      kv(ignis_image_transformer_prefix_kv_bytes(L));
  // Every byte of a run starts as 0xFF on both sides, so the comparison covers the whole buffer.
  MOE_CUDA(cudaMemset(whole0.p, 0xFF, long_run));
  MOE_CUDA(cudaMemset(whole1.p, 0xFF, long_run));
  upload(xl, xl0);
  TF_RC(ignis_image_transformer_blocks(&weights, 0, 2, xl.as<float>(), L, 0, long_rope.as<float>(), dmod0.as<float>(),
                                       &whole, ws_whole.p, ws_whole.bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<float> long_whole = download<float>(xl.p, xl0.size());
  const std::vector<uint8_t> runs0 = download<uint8_t>(whole0.p, long_run), runs1 = download<uint8_t>(whole1.p, long_run);
  long long long_nonfinite = 0;
  for (float v : long_whole) long_nonfinite += !std::isfinite(v);
  check(long_nonfinite == 0, "program: the long prefix's whole pass is not finite");
  for (uint32_t chunk : {128u, 384u}) {
    DeviceBytes ws_chunk(ignis_image_transformer_workspace_bytes(chunk));
    for (int split = 0; split < 2; ++split) {
      MOE_CUDA(cudaMemset(chunk0.p, 0xFF, long_run));
      MOE_CUDA(cudaMemset(chunk1.p, 0xFF, long_run));
      upload(xl, xl0);
      for (uint32_t b = 0; b < 2; b += split ? 1 : 2) {
        TF_RC(ignis_image_transformer_prefix_blocks(&weights, b, split ? b + 1 : 2, xl.as<float>(), long_rope.as<float>(),
                                                    dmod0.as<float>(), &chunked, kv.p, chunk, ws_chunk.p, ws_chunk.bytes,
                                                    nullptr));
      }
      MOE_CUDA(cudaDeviceSynchronize());
      const long long x_diff = differ(long_whole, download<float>(xl.p, xl0.size()));
      const bool runs_differ = runs0 != download<uint8_t>(chunk0.p, long_run) || runs1 != download<uint8_t>(chunk1.p, long_run);
      std::printf("    prefix 834 tokens in chunks of %u%s: %lld of %zu x values differ from the whole pass, runs %s\n", chunk,
                  split ? ", one block a call" : "", x_diff, long_whole.size(), runs_differ ? "differ" : "equal");
      check(x_diff == 0 && !runs_differ, "program: the prefix pass in chunks differs from the whole pass");
    }
  }
}

// ---- refusals -----------------------------------------------------------------------------------

void refusals() {
  std::printf("refusals:\n");
  DeviceBytes a(1 << 22), b(1 << 22), w(1 << 16), ws(ignis_image_transformer_workspace_bytes(4));
  auto refused = [](int32_t rc, const char *what) {
    check(rc != 0, std::string("refuses ") + what);
    if (rc != 0) std::printf("  %s: %s\n", what, ignis_image_transformer_last_error());
  };
  float *x = a.as<float>(), *s = b.as<float>();
  refused(ignis_image_transformer_residual_norm(x, nullptr, nullptr, 4, 64, nullptr, kEps, nullptr, nullptr, 0, nullptr),
          "neither delta nor scale");
  refused(ignis_image_transformer_residual_norm(x, s, nullptr, 4, 64, nullptr, kEps, nullptr, nullptr, 0, nullptr),
          "a delta without its gate");
  refused(ignis_image_transformer_residual_norm(x, nullptr, nullptr, 4, 8200, s, kEps, b.p, s, 0, nullptr), "cols > 8192");
  refused(ignis_image_transformer_residual_norm(x, nullptr, nullptr, 4, 64, s, kEps, b.p, nullptr, 0, nullptr),
          "E4M3 without scales");
  refused(ignis_image_transformer_linear_bf16(w.p, 12, 64, x, 1, s, nullptr), "rows % 8");
  refused(ignis_image_transformer_quantize_f32(x + 1, 4, 64, b.p, s, nullptr), "x misaligned");
  const ignis_image_transformer_block block{w.p, w.p, w.p, w.p, w.p, w.p, w.p, nullptr, w.p};
  const ignis_image_transformer_weights weights{w.p, w.p, w.p, w.p, w.p, w.p, w.p, w.p, w.p, &block, 1, kEps};
  ignis_image_transformer_weights missing = weights;
  missing.proj_out = nullptr;
  void *runs[1] = {b.p};
  const ignis_image_transformer_segment gap[2] = {{0, 2, 1}, {3, 4, 0}};
  const ignis_image_transformer_prefix bad_segments{4, runs, gap, 2};
  const ignis_image_transformer_prefix short_prefix{3, runs, gap, 1};
  refused(ignis_image_transformer_blocks(&weights, 0, 2, x, 4, 1, s, s, &bad_segments, ws.p, ws.bytes, nullptr),
          "end_block > blocks");
  refused(ignis_image_transformer_blocks(&weights, 0, 1, x, 4, 1, s, s, &bad_segments, ws.p, ws.bytes, nullptr),
          "a NULL block weight");
  refused(ignis_image_transformer_blocks(&weights, 0, 0, x, 4, 0, s, s, &bad_segments, ws.p, ws.bytes, nullptr),
          "segments with a gap");
  refused(ignis_image_transformer_blocks(&weights, 0, 0, x, 4, 0, s, s, &short_prefix, ws.p, ws.bytes, nullptr),
          "a prefix pass over other than the prefix's tokens");
  refused(ignis_image_transformer_blocks(&weights, 0, 0, x, 4, 1, s, s, nullptr, ws.p, ws.bytes, nullptr), "no prefix");
  refused(ignis_image_transformer_blocks(&weights, 0, 0, x, 4, 1, s, s, &bad_segments, ws.p, 1024, nullptr),
          "a short workspace");
  refused(ignis_image_transformer_head(&missing, x, 4, s, s, nullptr, 0.0f, 0.0f, ws.p, ws.bytes, nullptr),
          "a NULL model weight");
  refused(ignis_image_transformer_modulation(&weights, -1.0f, x, s, ws.p, ws.bytes, nullptr), "a negative sigma");
  refused(ignis_image_transformer_head(&weights, x, 4, s, s, x, NAN, 0.0f, ws.p, ws.bytes, nullptr), "a NaN sigma");
  const ignis_image_transformer_segment one[1] = {{0, 4, 1}};
  const ignis_image_transformer_prefix whole_prefix{4, runs, one, 1};
  refused(ignis_image_transformer_prefix_blocks(&weights, 0, 0, x, s, s, &bad_segments, b.p, 128, ws.p, ws.bytes, nullptr),
          "chunked: segments with a gap");
  refused(ignis_image_transformer_prefix_blocks(&weights, 0, 0, x, s, s, &whole_prefix, b.p, 100, ws.p, ws.bytes, nullptr),
          "chunked: a chunk not a multiple of 128");
  refused(ignis_image_transformer_prefix_blocks(&weights, 0, 0, x, s, s, &whole_prefix, a.as<uint8_t>() + 16, 128, ws.p,
                                                ws.bytes, nullptr),
          "chunked: kv not 256-byte aligned");
  refused(ignis_image_transformer_prefix_blocks(&weights, 0, 0, x, s, s, &whole_prefix, b.p, 128, ws.p, ws.bytes, nullptr),
          "chunked: a workspace short of the chunk's");
  check(ignis_image_transformer_workspace_bytes(0) == 0, "workspace_bytes(0) is 0");
  check(ignis_image_transformer_prefix_kv_bytes(0) == 0 && ignis_image_transformer_prefix_kv_bytes(3) == 2 * 24576,
        "prefix_kv_bytes: 0 for 0 tokens, K and V of 4096 BF16 a token, each 256-byte aligned");
}

}  // namespace

int main(int argc, char **argv) {
  const double factor = arg_double(argc, argv, "--bound-factor", 1.0);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "norm,linear,quantize,modulation,program,refusals";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("image transformer (bound factor %.3f; cases %s)\n", factor, cases.c_str());
  if (want("norm")) {
    std::printf("gated residual + modulated LayerNorm:\n");
    for (uint32_t tokens : {1u, 45u, 2049u}) case_norm(factor, tokens);
  }
  if (want("linear")) {
    std::printf("BF16 linear:\n");
    case_linear(factor, kHidden, 256, 1);
    case_linear(factor, kHidden, 4096, 2);
    case_linear(factor, kMod, 4096, 1);
  }
  if (want("quantize")) {
    std::printf("fp32 quantization:\n");
    case_quantize(1, 64);
    case_quantize(4099, 64);
    case_quantize(33, 4096);
  }
  if (want("modulation")) {
    std::printf("timestep embedding and modulation:\n");
    case_modulation();
  }
  if (want("program")) {
    std::printf("program:\n");
    case_program();
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_transformer: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_transformer: OK\n");
  return 0;
}
