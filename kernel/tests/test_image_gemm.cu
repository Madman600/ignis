// The image model's FP8 W8A8 GEMM and per-token activation quantization (spec image-model/02
// acceptance 2 and 3, GitHub #319) -- OURS.
//
//   quantize   ignis_image_quantize_tokens against its definition restated on the host (codes and
//              scales bit for bit), and every dequantized value within E4M3's half ulp of x; an
//              all-zero row, a row with one outlier, magnitudes from 2^-10 to 2^9; the edge rows
//              (NaN, +-Inf, amax below / at / above 448 * 2^-126); the refusals.
//   mechanics  ignis_image_gemm_w8a8 against an fp64 GEMM of the *dequantized* operands (the
//              quantizer's own codes and scales, the payload's codes and BF16 row scales), at
//              every linear shape of the DiT and the text encoder (rows x cols below) and tokens
//              from 1 to a 10-image condition prefix (41,000). Outputs are poisoned first; each
//              call runs twice and must agree bit for bit. Every other tile configuration the
//              bench's sweep times (kernel/tests/image_gemm_sweep.h, built into this test and the
//              bench only) is held to the same check at the edge shapes: one token, partial token
//              and output-row tiles, K = 128.
//   quant      the W8A8 product's error against the unquantized one (fp64 of BF16 x and BF16 W,
//              W quantized per row as the converter does), reported with the weight-only error
//              beside it; gated only by a loose sanity flag.
//   refusals   what the GEMM does not take.
//
// The linear shapes (spec image-model/02 acceptance 2: K and N in {64, 4096, 6144, 12288,
// 24576} as the artifact's linears need them; spec.md's model table):
//   rows x cols  4096 x 64     DiT img_in (z 64 -> hidden)
//                64 x 4096     DiT proj_out
//                4096 x 4096   DiT to_q/to_k/to_v/to_out, txt_in; encoder q_proj/o_proj
//                1024 x 4096   encoder k_proj/v_proj (GQA 8 x 128)
//                6144 x 4096   encoder q/k/v fused
//                12288 x 4096  DiT q/k/v fused; SwiGLU gate or up (DiT x3 and encoder 12288)
//                24576 x 4096  SwiGLU gate+up fused
//                4096 x 12288  SwiGLU down
//
// Mechanics bound (from bench_mma_peak's probes: one mxf8f6f4 instruction keeps fp32's bits and
// drops addends more than 2^25 below its largest): per instruction, 32 products and the
// accumulator each lose at most 2^-25 of the largest, and its result one fp32 rounding. Walking
// each output's k32 steps in the kernel's order, with S_i the exact prefix sum after step i,
//   E_i = E_{i-1} + F * 35 * 2^-25 * (|S_{i-1}| + E_{i-1} + sum_{step i} |e4m3(a) e4m3(w)|)
//   |y - ref| <= E_last * |scales[t] * wscale[r]| + |ref| * (2^-23 + BF16's 2^-8)
// with F = --bound-factor (default 1). The rows sampled at large token counts reach the end of the
// range at an odd stride coprime with the count (span_stride), so up to 16,512 tokens every token
// tile is checked (at 41,000 the 161 rows hit 161 of 321 tiles), and the rows cover every residue
// mod 128, the columns every residue mod 256: every warp, fragment and lane position of a tile.
//
//   ignis_kernel_image_gemm_test [--bound-factor F] [--max-quant-rel-rms R]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_gemm.h"
#include "image_gemm_sweep.h"
#include "moe_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace moe_test;

#define IMG_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_gemm_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

// ---- E4M3FN on the host -----------------------------------------------------------------------

// float -> E4M3FN, round to nearest even, saturated to +-448; NaN stays NaN (0x7F).
uint8_t e4m3_rn_satfinite(float v) {
  if (std::isnan(v)) return 0x7F;
  const uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
  const double a = std::fabs(static_cast<double>(v));
  if (a >= 448.0) return sign | 0x7E;
  int lo = 0;  // the largest code whose value is <= a
  while (lo + 1 <= 0x7E && e4m3_to_f32(static_cast<uint8_t>(lo + 1)) <= a) ++lo;
  if (lo == 0x7E) return sign | 0x7E;
  const double vlo = e4m3_to_f32(static_cast<uint8_t>(lo)), vhi = e4m3_to_f32(static_cast<uint8_t>(lo + 1));
  int code;
  if (a - vlo < vhi - a) {
    code = lo;
  } else if (a - vlo > vhi - a) {
    code = lo + 1;
  } else {
    code = (lo & 1) == 0 ? lo : lo + 1;  // ties to the even mantissa
  }
  return static_cast<uint8_t>(sign | code);
}

// The header's definition of the quantization, one fp32 operation per step. std::max(a, NaN) is
// a, as the kernel's fmaxf: a NaN element does not raise amax.
void quantize_row_host(const uint16_t *x, int cols, uint8_t *codes, float *scale) {
  float amax = 0.0f;
  for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(bf16_to_f32(x[c])));
  const volatile float k448 = 448.0f;
  const float min_amax = std::ldexp(7.0f, -120);  // 448 * 2^-126: the scale is a normal fp32
  const float s = amax >= min_amax ? amax / k448 : 1.0f;
  const float inv = amax >= min_amax ? k448 / amax : 0.0f;
  for (int c = 0; c < cols; ++c) codes[c] = e4m3_rn_satfinite(bf16_to_f32(x[c]) * inv);
  *scale = s;
}

// Activations of `tokens` x `cols`: token t's magnitude 2^((t % 20) - 10); every 7th token gets one
// outlier channel 64x its row; tokens listed in `zero_rows` are all zero.
std::vector<uint16_t> make_activations(uint32_t stream, int tokens, int cols, const std::vector<int> &zero_rows) {
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (int t = 0; t < tokens; ++t) {
    const float mag = std::ldexp(1.0f, (t % 20) - 10);
    const bool zero = std::find(zero_rows.begin(), zero_rows.end(), t) != zero_rows.end();
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(t) * cols + c;
      float v = zero ? 0.0f : hash_uniform(stream, i, mag);
      if (!zero && t % 7 == 3 && c == (t * 37) % cols) v = 64.0f * mag;
      x[i] = f32_to_bf16(v);
    }
  }
  return x;
}

// ---- quantize -------------------------------------------------------------------------------

void quantize_arm(int tokens, int cols) {
  const std::vector<uint16_t> x =
      make_activations(500 + cols, tokens, cols, tokens >= 3 ? std::vector<int>{tokens / 2} : std::vector<int>{});
  DeviceBytes dx(x.size() * 2), dc(x.size()), ds(static_cast<std::size_t>(tokens) * 4);
  upload(dx, x);
  IMG_RC(ignis_image_quantize_tokens(dx.p, tokens, cols, dc.p, ds.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto codes = download<uint8_t>(dc.p, x.size());
  const auto scales = download<float>(ds.p, tokens);
  std::vector<uint8_t> want(cols);
  int code_diff = 0, scale_diff = 0, over = 0;
  double worst = 0.0;
  for (int t = 0; t < tokens; ++t) {
    float s = 0.0f;
    quantize_row_host(&x[static_cast<std::size_t>(t) * cols], cols, want.data(), &s);
    if (std::memcmp(&s, &scales[t], 4) != 0) ++scale_diff;
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(t) * cols + c;
      if (codes[i] != want[c]) ++code_diff;
      // Property, independent of the restatement: E4M3 keeps 3 fraction bits, so the round trip
      // is within 2^-4 relative, or 2^-10 * scale (half the subnormal step) absolute.
      const double xv = bf16_to_f32(x[i]);
      const double err = std::fabs(static_cast<double>(scales[t]) * e4m3_to_f32(codes[i]) - xv);
      const double tol = std::fabs(xv) * 0x1p-4 + static_cast<double>(scales[t]) * 0x1p-10;
      worst = std::isnan(err) ? 1e300 : std::max(worst, err / tol);
      if (!(err <= tol)) ++over;
    }
  }
  std::printf("  quantize %5d tokens x %5d: %d codes and %d scales off the definition, round trip worst %.2f of "
              "half an E4M3 ulp, %d over\n",
              tokens, cols, code_diff, scale_diff, worst, over);
  check(code_diff == 0 && scale_diff == 0, "quantize " + std::to_string(tokens) + " x " + std::to_string(cols) +
                                               ": codes and scales are the definition's, bit for bit");
  check(over == 0, "quantize " + std::to_string(tokens) + " x " + std::to_string(cols) + ": round trip within half an ulp");
  if (tokens >= 3) {
    const std::size_t zero_row = static_cast<std::size_t>(tokens / 2) * cols;
    bool zeros = scales[tokens / 2] == 1.0f;
    for (int c = 0; c < cols; ++c) zeros = zeros && (codes[zero_row + c] & 0x7F) == 0;
    check(zeros, "quantize: an all-zero row has scale 1 and zero codes");
  }
}

// The header's edge rows, each against the restatement (codes and scales bit for bit):
//   0  ordinary (a control in the same launch)
//   1  one NaN            -> a NaN code; amax, and so the row's scale, from the other elements
//   2  one +Inf, 3 -Inf   -> scale Inf: the row never dequantizes to finite values
//   4  amax ~ 1e-37       -> below 448 * 2^-126: scale 1 and zero codes (no 0 * Inf = NaN code)
//   5  amax ~ 1e-35       -> above it: a normal scale, codes over E4M3's whole range
//   6  amax = 7 * 2^-120  -> exactly the floor: scale 2^-126, the largest code 448
//   7  all zero           -> scale 1, zero codes
// Every non-finite x must dequantize to a non-finite value; the finite rows keep the round trip.
void quantize_edge_rows() {
  constexpr int kRows = 8, kCols = 256;
  const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
  std::vector<uint16_t> x(kRows * kCols);
  const float mags[kRows] = {1.0f, 1.0f, 1.0f, 1.0f, 1e-37f, 1e-35f, std::ldexp(7.0f, -120), 0.0f};
  for (int t = 0; t < kRows; ++t) {
    for (int c = 0; c < kCols; ++c) x[t * kCols + c] = f32_to_bf16(hash_uniform(800 + t, c, mags[t]));
  }
  auto put = [&](int t, int c, float v) {  // bf16 of v, NaN and Inf included
    uint32_t u;
    std::memcpy(&u, &v, 4);
    x[t * kCols + c] = static_cast<uint16_t>(u >> 16);
  };
  put(1, 17, nan);
  put(2, 100, inf);
  put(3, 3, -inf);
  put(6, 200, std::ldexp(7.0f, -120));
  for (int c = 0; c < kCols; ++c) x[7 * kCols + c] = 0;
  DeviceBytes dx(x.size() * 2), dc(x.size()), ds(kRows * 4);
  upload(dx, x);
  IMG_RC(ignis_image_quantize_tokens(dx.p, kRows, kCols, dc.p, ds.as<float>(), nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const auto codes = download<uint8_t>(dc.p, x.size());
  const auto scales = download<float>(ds.p, kRows);
  std::vector<uint8_t> want(kCols);
  for (int t = 0; t < kRows; ++t) {
    float s = 0.0f;
    quantize_row_host(&x[t * kCols], kCols, want.data(), &s);
    const bool same = std::memcmp(&s, &scales[t], 4) == 0 && std::memcmp(want.data(), &codes[t * kCols], kCols) == 0;
    int non_finite_kept = 0, non_finite = 0, over = 0;
    for (int c = 0; c < kCols; ++c) {
      const double xv = bf16_to_f32(x[t * kCols + c]);
      const double back = static_cast<double>(scales[t]) * e4m3_to_f32(codes[t * kCols + c]);
      if (!std::isfinite(xv)) {
        ++non_finite;
        if (!std::isfinite(back)) ++non_finite_kept;
      } else if (t != 2 && t != 3) {
        const double tol = std::fabs(xv) * 0x1p-4 + static_cast<double>(scales[t]) * 0x1p-10;
        if (!(std::fabs(back - xv) <= tol)) ++over;
      }
    }
    std::printf("  quantize edge row %d: scale %g, %s the definition, %d/%d non-finite x kept non-finite, %d over\n", t,
                scales[t], same ? "bit for bit" : "OFF", non_finite_kept, non_finite, over);
    const std::string what = "quantize edge row " + std::to_string(t);
    check(same, what + ": codes and scale are the definition's, bit for bit");
    check(non_finite_kept == non_finite, what + ": a non-finite x never dequantizes to a finite value");
    check(over == 0, what + ": the finite elements keep the round trip");
  }
  check(std::isinf(scales[2]) && std::isinf(scales[3]), "quantize: an infinite element makes the scale Inf");
  check(scales[4] == 1.0f && scales[7] == 1.0f, "quantize: a row below 448 * 2^-126 has scale 1");
  check(scales[6] == std::ldexp(1.0f, -126), "quantize: a row at 448 * 2^-126 has scale 2^-126");
  bool zero_codes = true;
  for (int t : {4, 7}) {
    for (int c = 0; c < kCols; ++c) zero_codes = zero_codes && (codes[t * kCols + c] & 0x7F) == 0;
  }
  check(zero_codes, "quantize: a row below 448 * 2^-126 has zero codes");
}

void quantize_refusals() {
  DeviceBytes dx(4096), dc(2048), ds(64);
  check(ignis_image_quantize_tokens(dx.p, 1, 24, dc.p, ds.as<float>(), nullptr) != 0, "quantize: cols off 16 refused");
  check(ignis_image_quantize_tokens(dx.p, 0, 64, dc.p, ds.as<float>(), nullptr) != 0, "quantize: zero tokens refused");
  check(ignis_image_quantize_tokens(static_cast<char *>(dx.p) + 8, 1, 64, dc.p, ds.as<float>(), nullptr) != 0,
        "quantize: x off 16 bytes refused");
  check(ignis_image_quantize_tokens(dx.p, 1, 64, static_cast<char *>(dc.p) + 4, ds.as<float>(), nullptr) != 0,
        "quantize: codes off 16 bytes refused");
  check(ignis_image_quantize_tokens(nullptr, 1, 64, dc.p, ds.as<float>(), nullptr) != 0, "quantize: NULL refused");
}

// ---- the GEMM ---------------------------------------------------------------------------------

struct Options {
  double bound_factor = 1.0;
  double max_quant_rel_rms = 0.1;
};

// A row-scale-v1 payload (layout.md §6.1) of random codes (every finite E4M3 value) and BF16 row
// scales around `scale_mag`; the host keeps only the payload, decoding codes through a table.
std::vector<uint8_t> make_weight(uint32_t stream, int rows, int cols, float scale_mag) {
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scale_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> payload(scale_at + static_cast<std::size_t>(rows) * 2, 0);
  for (std::size_t i = 0; i < codes; ++i) {
    uint8_t c = static_cast<uint8_t>(hash_u32(stream, i) & 0xFFu);
    if ((c & 0x7F) == 0x7F) c ^= 1u;  // E4M3FN's NaN codes are not weights
    payload[i] = c;
  }
  for (int r = 0; r < rows; ++r) {
    const uint16_t s = f32_to_bf16(scale_mag * (0.5f + std::fabs(hash_uniform(stream + 1, r, 1.0f))));
    std::memcpy(&payload[scale_at + 2 * static_cast<std::size_t>(r)], &s, 2);
  }
  return payload;
}

struct E4m3Table {
  double v[256];
  E4m3Table() {
    for (int c = 0; c < 256; ++c) v[c] = e4m3_to_f32(static_cast<uint8_t>(c));
  }
};
const E4m3Table g_e4m3;

double weight_scale(const std::vector<uint8_t> &payload, int rows, int cols, int r) {
  const std::size_t scale_at = (static_cast<std::size_t>(rows) * cols + 255) / 256 * 256;
  uint16_t s;
  std::memcpy(&s, &payload[scale_at + 2 * static_cast<std::size_t>(r)], 2);
  return bf16_to_f32(s);
}

// The sampling stride for `want` of `count` indices: the smallest odd number >= (count - 1) /
// (want - 1), rounded up, that is coprime with count. Then i * stride mod count for i < want are
// distinct, reach the end of the range, and -- an odd stride being invertible mod any power of
// two -- cover every residue mod a power of two up to `want` that divides count.
int span_stride(int count, int want) {
  int stride = (count - 1 + want - 2) / (want - 1);
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, count) != 1) stride += 2;
  return stride;
}

// `want` distinct indices in [0, count): all of them when few, else i * stride mod count for
// i < want, plus the last.
std::vector<int> sample(int count, int want) {
  std::vector<int> s;
  if (count <= want) {
    for (int i = 0; i < count; ++i) s.push_back(i);
    return s;
  }
  const long long stride = span_stride(count, want);
  for (int i = 0; i < want; ++i) s.push_back(static_cast<int>(i * stride % count));
  if (std::find(s.begin(), s.end(), count - 1) == s.end()) s.push_back(count - 1);
  return s;
}

// One sampled row of y, as doubles.
std::vector<double> y_row(const DeviceBytes &dy, int t, int rows, bool f32_out) {
  std::vector<double> v(rows);
  const std::size_t at = static_cast<std::size_t>(t) * rows;
  if (f32_out) {
    const auto f = download<float>(dy.as<float>() + at, rows);
    for (int r = 0; r < rows; ++r) v[r] = f[r];
  } else {
    const auto b = download<uint16_t>(dy.as<uint16_t>() + at, rows);
    for (int r = 0; r < rows; ++r) v[r] = bf16_to_f32(b[r]);
  }
  return v;
}

std::vector<uint8_t> y_row_bytes(const DeviceBytes &dy, int t, int rows, bool f32_out) {
  const std::size_t width = f32_out ? 4 : 2;
  return download<uint8_t>(static_cast<const uint8_t *>(dy.p) + static_cast<std::size_t>(t) * rows * width, rows * width);
}

// `config` < 0: the public entry point; else that tile configuration (image_gemm_sweep.h).
void mechanics_arm(const Options &o, int tokens, int rows, int cols, bool f32_out, int config = -1) {
  const std::vector<uint8_t> w = make_weight(static_cast<uint32_t>(7000 + rows + cols), rows, cols, 0.004f);
  const std::vector<uint16_t> x = make_activations(static_cast<uint32_t>(600 + cols), tokens, cols, {});
  const std::size_t width = f32_out ? 4 : 2;
  DeviceBytes dw(w.size()), dx(x.size() * 2), dc(x.size()), ds(static_cast<std::size_t>(tokens) * 4),
      dy(static_cast<std::size_t>(tokens) * rows * width);
  upload(dw, w);
  upload(dx, x);
  MOE_CUDA(cudaMemset(dy.p, 0xFF, dy.bytes));  // poison: an unwritten output is a NaN
  auto gemm = [&] {
    return config < 0 ? ignis_image_gemm_w8a8(dw.p, rows, cols, dc.p, ds.as<float>(), tokens, dy.p, f32_out ? 1 : 0, nullptr)
                      : image_gemm_sweep::gemm_w8a8(config, dw.p, rows, cols, dc.p, ds.as<float>(), tokens, dy.p,
                                                      f32_out ? 1 : 0, nullptr);
  };
  IMG_RC(ignis_image_quantize_tokens(dx.p, tokens, cols, dc.p, ds.as<float>(), nullptr));
  IMG_RC(gemm());
  MOE_CUDA(cudaDeviceSynchronize());
  const std::vector<int> ts = sample(tokens, 160), rs = sample(rows, 384);
  std::vector<std::vector<uint8_t>> first;
  for (int t : ts) first.push_back(y_row_bytes(dy, t, rows, f32_out));
  IMG_RC(gemm());
  MOE_CUDA(cudaDeviceSynchronize());
  bool same = true;
  for (std::size_t i = 0; i < ts.size(); ++i) same = same && y_row_bytes(dy, ts[i], rows, f32_out) == first[i];

  // The bound walks each output's k32 steps in the kernel's order (k ascending, one instruction
  // per step): step i's 32 products and its accumulator input lose at most 2^-25 of its largest
  // addend each and its result one rounding, so it adds at most 35 * 2^-25 * (|S_{i-1}| + E_{i-1}
  // + sum_i |p|), S the exact fp64 prefix sum and E the bound so far (the accumulator the
  // instruction actually reads is within E of S).
  const double per_step = 0x1p-25 * 35.0 * o.bound_factor;
  const double out_rel = 0x1p-23 + (f32_out ? 0.0 : 0x1p-8);
  int bad = 0;
  double worst = 0.0;
  std::vector<double> acc_rel;  // the accumulation part of the bound over |ref|
  for (int t : ts) {
    const auto codes = download<uint8_t>(dc.as<uint8_t>() + static_cast<std::size_t>(t) * cols, cols);
    const double sx = download<float>(ds.as<float>() + t, 1)[0];
    const std::vector<double> y = y_row(dy, t, rows, f32_out);
    std::vector<double> a(cols);
    for (int c = 0; c < cols; ++c) a[c] = g_e4m3.v[codes[c]];
    for (int r : rs) {
      const uint8_t *wr = &w[static_cast<std::size_t>(r) * cols];
      double s = 0.0, e = 0.0;
      for (int k0 = 0; k0 < cols; k0 += 32) {
        double step = 0.0, step_mag = 0.0;
        for (int c = k0; c < k0 + 32; ++c) {
          const double p = a[c] * g_e4m3.v[wr[c]];
          step += p;
          step_mag += std::fabs(p);
        }
        e += per_step * (std::fabs(s) + e + step_mag);
        s += step;
      }
      const double scale = sx * weight_scale(w, rows, cols, r);
      const double ref = s * scale;
      const double tol = e * std::fabs(scale) + std::fabs(ref) * out_rel + 1e-30;
      const double err = std::fabs(y[r] - ref);
      if (!(err <= tol)) ++bad;
      if (err / tol > worst || std::isnan(err)) worst = std::isnan(err) ? 1e300 : err / tol;
      if (ref != 0.0) acc_rel.push_back(e * std::fabs(scale) / std::fabs(ref));
    }
  }
  double median = 0.0;
  if (!acc_rel.empty()) {
    std::nth_element(acc_rel.begin(), acc_rel.begin() + acc_rel.size() / 2, acc_rel.end());
    median = acc_rel[acc_rel.size() / 2];
  }
  const std::string tile = config < 0 ? "" : ", configuration " + std::to_string(config);
  std::printf("  gemm %5d x %5d x %5d tokens, %s out%s: worst error %.3f of bound, %d over (%zu x %zu sampled), "
              "accumulation bound / |ref| median %.2e%s\n",
              rows, cols, tokens, f32_out ? "fp32" : "bf16", tile.c_str(), worst, bad, ts.size(), rs.size(), median,
              same ? "" : ", NOT deterministic");
  const std::string what =
      "gemm [" + std::to_string(rows) + "][" + std::to_string(cols) + "] x " + std::to_string(tokens) + tile;
  check(bad == 0, what + " within the bound of the dequantized fp64 GEMM");
  check(same, what + ": a second run agrees bit for bit");
}

// Approximately normal: the sum of three uniforms, unit variance.
float gaussian(uint32_t stream, uint64_t i) {
  return (hash_uniform(stream, 3 * i, 1.0f) + hash_uniform(stream, 3 * i + 1, 1.0f) + hash_uniform(stream, 3 * i + 2, 1.0f));
}

void quant_error_arm(const Options &o, int tokens, int rows, int cols, bool outliers) {
  // W: BF16, ~N(0, 0.02^2); quantized per row as the converter does (layout.md §6.1,
  // tools/flash-next-converter/fp8.py): scale = the smallest BF16 >= fp32(amax / 448), so no code
  // saturates, or 1 for an all-zero row; codes E4M3(w / scale).
  std::vector<float> wf(static_cast<std::size_t>(rows) * cols);
  for (std::size_t i = 0; i < wf.size(); ++i) wf[i] = bf16_to_f32(f32_to_bf16(0.02f * gaussian(901, i)));
  const std::size_t scale_at = (wf.size() + 255) / 256 * 256;
  std::vector<uint8_t> w(scale_at + static_cast<std::size_t>(rows) * 2, 0);
  std::vector<double> wq(wf.size());
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    for (int c = 0; c < cols; ++c) amax = std::max(amax, std::fabs(wf[static_cast<std::size_t>(r) * cols + c]));
    const float want = amax / 448.0f;
    uint16_t sb = f32_to_bf16(want);
    if (bf16_to_f32(sb) < want) ++sb;  // round up: the next BF16 (positive, so the next bit pattern)
    if (amax == 0.0f) sb = f32_to_bf16(1.0f);
    const float s = bf16_to_f32(sb);
    std::memcpy(&w[scale_at + 2 * static_cast<std::size_t>(r)], &sb, 2);
    for (int c = 0; c < cols; ++c) {
      const std::size_t i = static_cast<std::size_t>(r) * cols + c;
      w[i] = e4m3_rn_satfinite(wf[i] / s);
      wq[i] = g_e4m3.v[w[i]] * s;
    }
  }
  // x: BF16, ~N(0, 1) per token; with `outliers`, 8 channels carry 20x (a DiT-like activation).
  std::vector<uint16_t> x(static_cast<std::size_t>(tokens) * cols);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const int c = static_cast<int>(i % cols);
    const float boost = outliers && c % (cols / 8) == 5 ? 20.0f : 1.0f;
    x[i] = f32_to_bf16(boost * gaussian(902, i));
  }
  DeviceBytes dw(w.size()), dx(x.size() * 2), dc(x.size()), ds(static_cast<std::size_t>(tokens) * 4),
      dy(static_cast<std::size_t>(tokens) * rows * 2);
  upload(dw, w);
  upload(dx, x);
  IMG_RC(ignis_image_quantize_tokens(dx.p, tokens, cols, dc.p, ds.as<float>(), nullptr));
  IMG_RC(ignis_image_gemm_w8a8(dw.p, rows, cols, dc.p, ds.as<float>(), tokens, dy.p, 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  double e8 = 0.0, e16 = 0.0, ref2 = 0.0, dot = 0.0, y2 = 0.0;
  for (int t : sample(tokens, 64)) {
    const std::vector<double> y = y_row(dy, t, rows, false);
    std::vector<double> xr(cols);
    for (int c = 0; c < cols; ++c) xr[c] = bf16_to_f32(x[static_cast<std::size_t>(t) * cols + c]);
    for (int r : sample(rows, 256)) {
      double full = 0.0, wonly = 0.0;
      for (int c = 0; c < cols; ++c) {
        const std::size_t i = static_cast<std::size_t>(r) * cols + c;
        full += xr[c] * wf[i];
        wonly += xr[c] * wq[i];
      }
      e8 += (y[r] - full) * (y[r] - full);
      e16 += (wonly - full) * (wonly - full);
      ref2 += full * full;
      dot += y[r] * full;
      y2 += y[r] * y[r];
    }
  }
  const double rel8 = std::sqrt(e8 / ref2), rel16 = std::sqrt(e16 / ref2), cosine = dot / std::sqrt(ref2 * y2);
  std::printf("  quant error %5d x %5d x %4d tokens, %s x: W8A8 relative RMS %.4f (cosine %.6f); weights only %.4f\n",
              rows, cols, tokens, outliers ? "outlier-channel" : "gaussian", rel8, cosine, rel16);
  check(rel8 <= o.max_quant_rel_rms, "quant error [" + std::to_string(rows) + "][" + std::to_string(cols) +
                                         "]: W8A8 relative RMS within --max-quant-rel-rms");
}

void gemm_refusals() {
  DeviceBytes dw(64 * 64 + 256), dc(64 * 64), ds(256), dy(64 * 64 * 4);
  const float *s = ds.as<float>();
  check(ignis_image_gemm_w8a8(dw.p, 64, 96, dc.p, s, 1, dy.p, 0, nullptr) != 0, "gemm: cols off 64 refused");
  check(ignis_image_gemm_w8a8(dw.p, 40, 64, dc.p, s, 1, dy.p, 0, nullptr) != 0, "gemm: rows off 16 refused");
  check(ignis_image_gemm_w8a8(dw.p, 64, 64, dc.p, s, 0, dy.p, 0, nullptr) != 0, "gemm: zero tokens refused");
  check(ignis_image_gemm_w8a8(static_cast<char *>(dw.p) + 8, 64, 64, dc.p, s, 1, dy.p, 0, nullptr) != 0,
        "gemm: a weight off 16 bytes refused");
  check(ignis_image_gemm_w8a8(dw.p, 64, 64, static_cast<char *>(dc.p) + 8, s, 1, dy.p, 0, nullptr) != 0,
        "gemm: codes off 16 bytes refused");
  check(ignis_image_gemm_w8a8(dw.p, 64, 64, dc.p, s, 1, static_cast<char *>(dy.p) + 4, 0, nullptr) != 0,
        "gemm: y off 8 bytes refused");
  check(ignis_image_gemm_w8a8(dw.p, 64, 64, dc.p, nullptr, 1, dy.p, 0, nullptr) != 0, "gemm: NULL scales refused");
}

double arg_double(int argc, char **argv, const char *name, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  Options o;
  o.bound_factor = arg_double(argc, argv, "--bound-factor", o.bound_factor);
  o.max_quant_rel_rms = arg_double(argc, argv, "--max-quant-rel-rms", o.max_quant_rel_rms);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  IMG_RC(ignis_image_gemm_prepare());
  IMG_RC(image_gemm_sweep::prepare());
  std::printf("per-token E4M3 activation quantization\n");
  for (int cols : {16, 64, 4096, 12288}) quantize_arm(3, cols);
  for (int cols : {4096, 6144}) quantize_arm(300, cols);
  quantize_arm(1, 4096);
  quantize_edge_rows();
  quantize_refusals();

  std::printf("FP8 W8A8 GEMM against fp64 of the dequantized operands (bound factor %.2f)\n", o.bound_factor);
  struct Shape {
    int rows, cols;
  };
  // The encoder's shapes run to 41,000 tokens too: its sequence (text plus up to 10 x 1024 vision
  // slots) is never longer than the DiT prefix it becomes (each slot is 4 prefix tokens), and the
  // prompt text has no cap of its own (spec image-model/02 "As built").
  const Shape shapes[] = {{4096, 64},   {64, 4096},    {4096, 4096},  {1024, 4096},
                          {6144, 4096}, {12288, 4096}, {24576, 4096}, {4096, 12288}};
  for (const Shape &s : shapes) {
    int i = 0;
    for (int tokens : {1, 2, 17, 129, 4096, 16512}) mechanics_arm(o, tokens, s.rows, s.cols, (i++ + s.rows / 64) % 2 == 1);
    mechanics_arm(o, 41000, s.rows, s.cols, false);
  }
  std::printf("every tile configuration (the bench's sweep) at the edge shapes\n");
  for (int config = 0; config < image_gemm_sweep::config_count(); ++config) {
    mechanics_arm(o, 1, 64, 4096, true, config);
    mechanics_arm(o, 129, 4096, 128, false, config);
    mechanics_arm(o, 300, 1040, 4096, true, config);
    mechanics_arm(o, 4096, 4096, 12288, false, config);
  }
  gemm_refusals();

  std::printf("quantization error against the unquantized BF16 product (synthetic inputs)\n");
  for (bool outliers : {false, true}) {
    quant_error_arm(o, 4096, 4096, 4096, outliers);
    quant_error_arm(o, 4096, 4096, 12288, outliers);
  }
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_gemm: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_gemm: OK\n");
  return 0;
}
