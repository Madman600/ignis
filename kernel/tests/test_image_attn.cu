// The image model's 8-bit DiT attention (spec image-model/02 acceptance 4, GitHub #319) -- OURS.
// The op is kernel/include/ignis_image_attn.h; every reference here is our own fp64 CPU code.
//
//   quantizers  ignis_image_attn_quantize_q and ignis_image_attn_store_kv against their
//               definitions restated on the host, bit for bit: the Q and K codes and scales, the
//               V codes in their transposed, permuted layout, the V scales, the zero padding; the
//               K mean against fp64 (a fixed-order fp32 sum); the target run's mean copied from
//               the prefix's; edge blocks of zeros, of 1e-37 and at the floor.
//   calls       at the DiT's geometry (32 heads of 128, 1024² = 4096 latent tokens per image):
//               the target over [prefix ; target] for 0, 1 and 10 condition images (and a 2048²
//               target, 16,384 queries), image blocks over their truncated keys, and text segments
//               causal aligned bottom-right, each call checked two ways:
//     mechanics against fp64 attention on the same quantized, smoothed operands, the bound
//               derived below. The output is fp32 for this check; the same call with BF16 output
//               must then be the fp32 output rounded to BF16, bit for bit (and so deterministic).
//     quant     against fp64 attention on the unquantized BF16 inputs: cosine per head >= 0.99
//               (--min-cosine), with the centred cosine and the relative RMS error reported.
//   prefix      the target's call reads the prefix run (stored alone, its own mean and V scales)
//               and the target run (the prefix's mean, its own V scales), rescaling the
//               accumulator at the boundary. The mechanics bound holds; the same bound computed
//               WITHOUT the rescale must fail (a negative control: the check sees a lost
//               rescale); the prefix run is byte for byte as stored after every call that read
//               it; and the split's error against fp64 is reported beside that of quantizing
//               [prefix ; target] as one run, at 0, 1 and 10 images.
//   inputs      DiT-like (make_sequence), and three harder profiles at 0, 1 and 10 images: a heavy
//               tail (cosine gated); Q and K outlier channels at 20x, and prefix and target K means
//               far apart (both measured: the cosine reported, not gated).
//   refusals    what the op does not take.
//
// The mechanics reference replays the kernel's softmax: per 64-key block in the kernel's order,
// the same fp32 operations for the score scale c = (qscale * kscale) * fl(scale * log2 e), the
// running max m = max(m, fma(max_j S_j, c, -8.807)) and the exponent x_j = fma(S_j, c, -m), all
// exact replicas (S_j = q8 . k8_j is exact in int32). What it cannot replay it bounds:
//   P_j = E4M3(2^x_j): ex2.approx is within kExp2Rel of 2^x, so a code differs from the fp64
//         one only where 2^x_j (1 +- kExp2Rel) straddles an E4M3 rounding midpoint; such a key
//         adds that ulp * |v| to the numerator's and that ulp to the denominator's error.
//   the rescale by 2^fl32(m_old - m_new) when the max rises (the subtraction replayed): ex2
//         again plus one fp32 rounding.
//   PV and the row sum (fp32 accumulation on tensor cores): per k32 step 35 * 2^-25 * (|acc| +
//         error so far + the step's sum |P v|), as the GEMM's bound (bench_mma_peak's probes).
//   the run boundary: the ratio vscale_A / vscale_B and the product, two fp32 roundings.
//   the end: (acc * vscale) * (1 / sum P), three roundings; then the quotient's error,
//         |dO| <= (|dN| + |O| |dD|) / (D - |dD|).
// The whole bound is multiplied by --bound-factor F (default 1).
//
//   ignis_kernel_image_attn_test [--bound-factor F] [--min-cosine C]
//                                [--cases edges,small,0,1,10,16k,shift,tail,outliers,refusals]
//   ignis_kernel_image_attn_test --cases qkv-probe [--qkv-probe DIR]
//
// qkv-probe runs the same checks on real Q, K and V (case_qkv_probe below), read from DIR, else
// $IGNIS_IMAGE_QKV_PROBE, else F:/ai/models/ignis-image-2.1-turbo/qkv-probe. The data lives
// outside the repo: with no dump there it prints SKIP and passes (the coordinator's call for #319,
// against the no-skip rule below: a machine without the dump cannot run it at all).
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_attn.h"
#include "moe_fixture.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace moe_test;

#define ATT_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_attn_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

constexpr int kHeads = 32, kDim = 128, kHD = kHeads * kDim;
// ex2.approx.f32's relative error, with margin: the PTX ISA gives "2 ulp from the correctly rounded
// result across the full range", so within 2.5 * 2^-23 < 2^-21.6.
constexpr double kExp2Rel = 0x1p-20;
constexpr float kOffset = 8.807f;     // the kernel's exponent offset: 2^8.807 ~ 448
const float kScale = 1.0f / std::sqrt(128.0f);

struct Options {
  double bound_factor = 1.0;
  double min_cosine = 0.99;
};

template <class F> void parallel_for(int n, F &&f) {
  const int threads = std::max(1, std::min<int>(n, std::min(16u, std::max(1u, std::thread::hardware_concurrency()))));
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t) {
    pool.emplace_back([&] {
      for (int i = next++; i < n; i = next++) f(i);
    });
  }
  for (auto &th : pool) th.join();
}

// ---- E4M3 and the quantizers' definitions on the host -----------------------------------------

// E4M3FN's value nearest to p >= 0 (ties to even), saturated to 448.
double e4m3_rn(double p) {
  if (p >= 448.0) return 448.0;
  if (p < 0x1p-6) return std::nearbyint(p * 512.0) / 512.0;  // subnormals: steps of 2^-9
  int e = 0;
  std::frexp(p, &e);  // p in [2^(e-1), 2^e)
  const double step = std::ldexp(1.0, e - 4);
  return std::nearbyint(p / step) * step;
}

// float -> E4M3FN code, round to nearest even, saturated to +-448 (finite inputs).
uint8_t e4m3_code(float v) {
  const uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
  const double a = e4m3_rn(std::fabs(static_cast<double>(v)));
  if (a == 0.0) return sign;
  int e = 0;
  const double f = std::frexp(a, &e);  // a = f 2^e, f in [0.5, 1)
  if (a < 0x1p-6) return static_cast<uint8_t>(sign | static_cast<int>(a * 512.0));
  const int exp_field = e - 1 + 7;
  const int man = static_cast<int>((f * 2.0 - 1.0) * 8.0);
  return static_cast<uint8_t>(sign | (exp_field << 3) | man);
}

struct E4m3Table {
  float v[256];
  E4m3Table() {
    for (int c = 0; c < 256; ++c) v[c] = e4m3_to_f32(static_cast<uint8_t>(c));
  }
};
const E4m3Table g_e4m3;

int8_t sat_s8(float v) {  // cvt.rni.sat.s8.f32
  const float r = std::nearbyint(v);
  return static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, r)));
}

// One fp32 operation each (volatile keeps the host from fusing or reordering).
float f32_mul(float a, float b) {
  volatile float r = a * b;
  return r;
}
float f32_div(float a, float b) {
  volatile float r = a / b;
  return r;
}
float f32_sub(float a, float b) {
  volatile float r = a - b;
  return r;
}

const float kInt8MinAmax = std::ldexp(127.0f, -126), kE4m3MinAmax = std::ldexp(448.0f, -126);

// Where key j sits in the V layout: position p of a group of 16 holds key
// 0 1 8 9 2 3 10 11 4 5 12 13 6 7 14 15 [p] (the header's order, restated).
int key_position(int j) {
  static const int inv[16] = {0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15};
  return j / 16 * 16 + inv[j % 16];
}

// ---- the inputs -------------------------------------------------------------------------------

// Approximately N(0, 1): the Irwin-Hall sum of a hash's four bytes.
float gauss(uint32_t stream, uint64_t i) {
  const uint32_t h = hash_u32(stream, i);
  const int s = static_cast<int>((h & 255u) + ((h >> 8) & 255u) + ((h >> 16) & 255u) + (h >> 24));
  return (static_cast<float>(s) - 510.0f) * (1.0f / 147.8f);
}

// gauss() with a heavy tail: one value in 256 at 6x, so up to ~20 sigma.
float heavy(uint32_t stream, uint64_t i) {
  const float g = gauss(stream, i);
  return (hash_u32(stream + 1000, i) & 255u) == 0 ? 6.0f * g : g;
}

enum TokenKind { kText = 0, kImage = 1, kTarget = 2 };

// The inputs beyond the DiT-like default (the review of #319's AC 4):
//   kTail      a heavy tail on q, k and v (heavy(): up to ~20 sigma).
//   kOutliers  outlier channels as the GEMM test's "8 channels at 20x": q's channel 37 and k's
//              channel 91 in heads 0, 4, .., 28, Gaussian. They stretch a 32-row Q block's and a
//              64-key K block's amax for the other 127 channels.
//   kKShift    prefix and target K means far apart: in four channel pairs (d, d + 64) per head,
//              d in {10, 23, 36, 49} (RoPE's rotate-half pairs), an offset of 10 sigma at a
//              per-head angle for the prefix, rotated by 2 rad for the target (text and target
//              positions rotate a pair's mean differently). The prefix's mean is then ~17 off in
//              those channels of every target key.
enum class Profile { kPlain, kTail, kOutliers, kKShift };

const char *profile_name(Profile p) {
  return p == Profile::kTail       ? "heavy-tail"
         : p == Profile::kOutliers ? "outlier-channel"
         : p == Profile::kKShift   ? "K-means-apart"
                                   : "DiT-like";
}

// A joint sequence [prefix ; target] of BF16 q, k, v [tokens][4096] with DiT-like statistics:
// K with a per-channel mean (what smoothing removes), shifted for text and for the target; V
// with per-channel scales and offsets, the target's spread 2.5x the prefix's (so the run
// boundary's rescale matters); Q at a per-head temperature from flat to peaked. `profile` adds
// the above.
struct Sequence {
  std::vector<int> kinds;  // per token
  int prefix = 0, target = 0;
  std::vector<uint16_t> q, k, v;
  int tokens() const { return prefix + target; }
};

Sequence make_sequence(const std::vector<std::pair<TokenKind, int>> &prefix_runs, int target, uint32_t seed,
                       Profile profile = Profile::kPlain) {
  Sequence s;
  for (const auto &r : prefix_runs) s.kinds.insert(s.kinds.end(), r.second, r.first);
  s.prefix = static_cast<int>(s.kinds.size());
  s.target = target;
  s.kinds.insert(s.kinds.end(), target, kTarget);
  const std::size_t n = static_cast<std::size_t>(s.tokens()) * kHD;
  s.q.resize(n);
  s.k.resize(n);
  s.v.resize(n);
  std::vector<float> mu(kHD), shift_text(kHD), shift_target(kHD), vs(kHD), vo(kHD), qs(kHeads);
  for (int i = 0; i < kHD; ++i) {
    mu[i] = hash_uniform(seed + 11, i, 3.0f);
    shift_text[i] = hash_uniform(seed + 12, i, 0.75f);
    shift_target[i] = hash_uniform(seed + 13, i, 0.5f);
    vs[i] = 0.25f + 3.75f * std::fabs(hash_uniform(seed + 14, i, 1.0f));
    vo[i] = hash_uniform(seed + 15, i, 2.0f);
  }
  for (int h = 0; h < kHeads; ++h) qs[h] = 0.5f + 2.5f * std::fabs(hash_uniform(seed + 16, h, 1.0f));
  // kKShift: the offset of channel d's pair for the prefix (target: rotated by 2 rad).
  std::vector<float> kshift_prefix(kHD, 0.0f), kshift_target(kHD, 0.0f);
  if (profile == Profile::kKShift) {
    for (int h = 0; h < kHeads; ++h) {
      for (int d : {10, 23, 36, 49}) {
        const double phi = 3.141592653589793 * hash_uniform(seed + 17, h * kDim + d, 1.0f);
        kshift_prefix[h * kDim + d] = static_cast<float>(10.0 * std::cos(phi));
        kshift_prefix[h * kDim + d + 64] = static_cast<float>(10.0 * std::sin(phi));
        kshift_target[h * kDim + d] = static_cast<float>(10.0 * std::cos(phi + 2.0));
        kshift_target[h * kDim + d + 64] = static_cast<float>(10.0 * std::sin(phi + 2.0));
      }
    }
  }
  const bool tail = profile == Profile::kTail, outliers = profile == Profile::kOutliers;
  parallel_for(s.tokens(), [&](int t) {
    const int kind = s.kinds[t];
    for (int i = 0; i < kHD; ++i) {
      const std::size_t at = static_cast<std::size_t>(t) * kHD + i;
      const float shift = kind == kText ? shift_text[i] : kind == kTarget ? shift_target[i] : 0.0f;
      float qx = tail ? heavy(seed + 1, at) : gauss(seed + 1, at);
      float kx = tail ? heavy(seed + 2, at) : gauss(seed + 2, at);
      const float vx = tail ? heavy(seed + 3, at) : gauss(seed + 3, at);
      if (outliers && (i / kDim) % 4 == 0) {
        qx *= i % kDim == 37 ? 20.0f : 1.0f;
        kx *= i % kDim == 91 ? 20.0f : 1.0f;
      }
      const float kshift = kind == kTarget ? kshift_target[i] : kshift_prefix[i];
      s.q[at] = f32_to_bf16(qs[i / kDim] * qx);
      s.k[at] = f32_to_bf16(kx + mu[i] + shift + kshift);
      s.v[at] = f32_to_bf16(vs[i] * vx * (kind == kTarget ? 2.5f : 1.0f) + vo[i]);
    }
  });
  return s;
}

// ---- runs and q8 buffers on the host ----------------------------------------------------------

int round_up(int n, int m) { return (n + m - 1) / m * m; }

struct Run {
  int n = 0, n_pad = 0;
  std::vector<int8_t> k;      // [H][n_pad][D]
  std::vector<uint8_t> v;     // [H][D][n_pad], permuted
  std::vector<float> kscale;  // [H][n_pad / 64]
  std::vector<float> vscale;  // [H][D]
  std::vector<float> mean;    // [H][D]
  int8_t kc(int h, int j, int d) const { return k[(static_cast<std::size_t>(h) * n_pad + j) * kDim + d]; }
  uint8_t vc(int h, int c, int j) const { return v[(static_cast<std::size_t>(h) * kDim + c) * n_pad + key_position(j)]; }
};

Run download_run(const DeviceBytes &d, int n) {
  Run r;
  r.n = n;
  r.n_pad = round_up(n, 64);
  const std::size_t codes = static_cast<std::size_t>(kHeads) * r.n_pad * kDim;
  const std::size_t blocks = static_cast<std::size_t>(kHeads) * (r.n_pad / 64);
  const auto *base = static_cast<const uint8_t *>(d.p);
  r.k = download<int8_t>(base, codes);
  r.v = download<uint8_t>(base + codes, codes);
  r.kscale = download<float>(base + 2 * codes, blocks);
  r.vscale = download<float>(base + 2 * codes + blocks * 4, kHD);
  r.mean = download<float>(base + 2 * codes + blocks * 4 + kHD * 4, kHD);
  return r;
}

struct Q8 {
  int n = 0, n_pad = 0;
  std::vector<int8_t> codes;  // [H][n_pad][D]
  std::vector<float> scales;  // [H][n_pad / 32]
  const int8_t *row(int h, int i) const { return &codes[(static_cast<std::size_t>(h) * n_pad + i) * kDim]; }
  float scale(int h, int i) const { return scales[static_cast<std::size_t>(h) * (n_pad / 32) + i / 32]; }
};

Q8 download_q8(const DeviceBytes &d, int n) {
  Q8 q;
  q.n = n;
  q.n_pad = round_up(n, 128);
  const std::size_t codes = static_cast<std::size_t>(kHeads) * q.n_pad * kDim;
  q.codes = download<int8_t>(d.p, codes);
  q.scales = download<float>(static_cast<const uint8_t *>(d.p) + codes, static_cast<std::size_t>(kHeads) * (q.n_pad / 32));
  return q;
}

// ---- the quantizers against their definitions -------------------------------------------------

// Q rows [first, first + n) of the sequence against the q8 buffer.
void check_q8(const Sequence &s, int first, const Q8 &q, const std::string &what) {
  std::atomic<long long> code_diff{0}, scale_diff{0}, pad_diff{0};
  parallel_for(kHeads, [&](int h) {
    long long cd = 0, sd = 0, pd = 0;
    for (int b = 0; b < q.n_pad / 32; ++b) {
      float amax = 0.0f;
      for (int i = b * 32; i < std::min(q.n, b * 32 + 32); ++i) {
        for (int d = 0; d < kDim; ++d) {
          amax = std::max(amax, std::fabs(bf16_to_f32(s.q[static_cast<std::size_t>(first + i) * kHD + h * kDim + d])));
        }
      }
      const bool zero = !(amax >= kInt8MinAmax);
      const float scale = zero ? 1.0f : f32_div(amax, 127.0f);
      const float inv = zero ? 0.0f : f32_div(127.0f, amax);
      if (std::memcmp(&scale, &q.scales[static_cast<std::size_t>(h) * (q.n_pad / 32) + b], 4) != 0) ++sd;
      for (int i = b * 32; i < b * 32 + 32; ++i) {
        for (int d = 0; d < kDim; ++d) {
          const int8_t got = q.row(h, i)[d];
          if (i >= q.n) {
            pd += got != 0;
            continue;
          }
          const float x = bf16_to_f32(s.q[static_cast<std::size_t>(first + i) * kHD + h * kDim + d]);
          cd += got != sat_s8(f32_mul(x, inv));
        }
      }
    }
    code_diff += cd;
    scale_diff += sd;
    pad_diff += pd;
  });
  std::printf("  q8 %s (%d rows): %lld codes, %lld scales, %lld padding bytes off the definition\n", what.c_str(), q.n,
              code_diff.load(), scale_diff.load(), pad_diff.load());
  check(code_diff == 0 && scale_diff == 0 && pad_diff == 0, "q8 " + what + ": the definition's, bit for bit");
}

// Rows [first, first + n) of the sequence's K and V against the run; `mean` the expected mean
// source (nullptr: the run's own, checked against fp64).
void check_run(const Sequence &s, int first, const Run &r, const Run *mean_from, const std::string &what) {
  std::atomic<long long> kd{0}, ksd{0}, vd{0}, vsd{0}, pad{0};
  std::atomic<int> mean_bad{0};
  parallel_for(kHeads, [&](int h) {
    long long kdl = 0, ksdl = 0, vdl = 0, vsdl = 0, padl = 0;
    int mb = 0;
    auto kx = [&](int j, int c) { return bf16_to_f32(s.k[static_cast<std::size_t>(first + j) * kHD + h * kDim + c]); };
    auto vx = [&](int j, int c) { return bf16_to_f32(s.v[static_cast<std::size_t>(first + j) * kHD + h * kDim + c]); };
    for (int c = 0; c < kDim; ++c) {
      const float got = r.mean[h * kDim + c];
      if (mean_from != nullptr) {
        if (std::memcmp(&got, &mean_from->mean[h * kDim + c], 4) != 0) ++mb;
      } else {
        double sum = 0.0, mag = 0.0;
        for (int j = 0; j < r.n; ++j) {
          sum += kx(j, c);
          mag += std::fabs(kx(j, c));
        }
        // A fixed-order fp32 sum of n terms and one division: within n 2^-24 of the magnitudes.
        const double tol = (r.n * 0x1p-24) * mag / r.n + std::fabs(sum / r.n) * 0x1p-23 + 1e-30;
        if (!(std::fabs(got - sum / r.n) <= tol)) ++mb;
      }
    }
    for (int b = 0; b < r.n_pad / 64; ++b) {
      float amax = 0.0f;
      for (int j = b * 64; j < std::min(r.n, b * 64 + 64); ++j) {
        for (int c = 0; c < kDim; ++c) amax = std::max(amax, std::fabs(f32_sub(kx(j, c), r.mean[h * kDim + c])));
      }
      const bool zero = !(amax >= kInt8MinAmax);
      const float scale = zero ? 1.0f : f32_div(amax, 127.0f), inv = zero ? 0.0f : f32_div(127.0f, amax);
      if (std::memcmp(&scale, &r.kscale[static_cast<std::size_t>(h) * (r.n_pad / 64) + b], 4) != 0) ++ksdl;
      for (int j = b * 64; j < b * 64 + 64; ++j) {
        for (int c = 0; c < kDim; ++c) {
          if (j >= r.n) {
            padl += r.kc(h, j, c) != 0;
            continue;
          }
          kdl += r.kc(h, j, c) != sat_s8(f32_mul(f32_sub(kx(j, c), r.mean[h * kDim + c]), inv));
        }
      }
    }
    for (int c = 0; c < kDim; ++c) {
      float amax = 0.0f;
      for (int j = 0; j < r.n; ++j) amax = std::max(amax, std::fabs(vx(j, c)));
      const bool zero = !(amax >= kE4m3MinAmax);
      const float scale = zero ? 1.0f : f32_div(amax, 448.0f);
      if (std::memcmp(&scale, &r.vscale[h * kDim + c], 4) != 0) ++vsdl;
      for (int j = 0; j < r.n_pad; ++j) {
        if (j >= r.n) {
          padl += r.vc(h, c, j) != 0;
          continue;
        }
        vdl += r.vc(h, c, j) != e4m3_code(f32_div(vx(j, c), scale));
      }
    }
    kd += kdl;
    ksd += ksdl;
    vd += vdl;
    vsd += vsdl;
    pad += padl;
    mean_bad += mb;
  });
  std::printf("  run %s (%d keys): K %lld codes / %lld scales, V %lld codes / %lld scales, %lld padding bytes off; "
              "%d means %s\n",
              what.c_str(), r.n, kd.load(), ksd.load(), vd.load(), vsd.load(), pad.load(), mean_bad.load(),
              mean_from != nullptr ? "not copied from the prefix" : "outside fp32-sum error of fp64");
  check(kd == 0 && ksd == 0 && vd == 0 && vsd == 0 && pad == 0, "run " + what + ": the definition's, bit for bit");
  check(mean_bad == 0, "run " + what + (mean_from != nullptr ? ": the prefix's mean, copied" : ": the K mean"));
}

// ---- the references ---------------------------------------------------------------------------

// One attention call as the kernel sees it.
struct Call {
  std::string what;
  int q_first = 0, q_tokens = 0;  // the sequence rows the queries are
  const Run *a = nullptr;
  int a_keys = 0;
  int a_first = 0;  // the sequence row of run A's key 0
  const Run *b = nullptr;
  int b_first = 0;
  bool causal = false;
  bool cosine_gated = true;  // false: the cosine against fp64 is reported, not gated
  int keys() const { return a_keys + (b != nullptr ? b->n : 0); }
};

struct RowCheck {
  double out[kDim];    // the reference
  double bound[kDim];  // fp32-output bound
};

// One head's operands for a call, in the call's key order (run A's first a_keys keys, then run
// B's): the V code values de-permuted to [key][channel], and the BF16 inputs as floats.
struct HeadOperands {
  std::vector<float> v8;      // [keys][D] E4M3 values (no scale)
  std::vector<float> kx, vx;  // [keys][D] the unquantized K and V
};

HeadOperands head_operands(const Sequence &s, const Call &call, int h, bool exact) {
  HeadOperands op;
  const int keys = call.keys();
  op.v8.resize(static_cast<std::size_t>(keys) * kDim);
  for (int j = 0; j < keys; ++j) {
    const Run *run = j < call.a_keys ? call.a : call.b;
    const int jj = j < call.a_keys ? j : j - call.a_keys;
    for (int c = 0; c < kDim; ++c) op.v8[static_cast<std::size_t>(j) * kDim + c] = g_e4m3.v[run->vc(h, c, jj)];
  }
  if (exact) {
    op.kx.resize(op.v8.size());
    op.vx.resize(op.v8.size());
    for (int j = 0; j < keys; ++j) {
      const int t = j < call.a_keys ? call.a_first + j : call.b_first + (j - call.a_keys);
      for (int c = 0; c < kDim; ++c) {
        const std::size_t from = static_cast<std::size_t>(t) * kHD + h * kDim + c, to = static_cast<std::size_t>(j) * kDim + c;
        op.kx[to] = bf16_to_f32(s.k[from]);
        op.vx[to] = bf16_to_f32(s.v[from]);
      }
    }
  }
  return op;
}

// The mechanics reference for query i of head h (see the file header). `rescale` false computes
// the bound as if the kernel had not rescaled at the run boundary (the negative control).
void mechanics_row(const Call &call, const Q8 &q, const HeadOperands &op, int h, int i, bool rescale, double F,
                   RowCheck &rc) {
  const int8_t *qr = q.row(h, i);
  const float qs = q.scale(h, i);
  const float c_sm = static_cast<float>(static_cast<double>(kScale) * 1.4426950408889634);
  const int offset = call.keys() - call.q_tokens;
  double N[kDim] = {}, EN[kDim] = {}, D = 0.0, ED = 0.0;
  float m = -std::numeric_limits<float>::infinity();
  int global0 = 0;
  for (int span = 0; span < 2; ++span) {
    const Run *run = span == 0 ? call.a : call.b;
    if (run == nullptr) break;
    const int keys = span == 0 ? call.a_keys : run->n;
    if (span == 1) {
      for (int c = 0; c < kDim; ++c) {
        const double ratio = rescale ? static_cast<double>(call.a->vscale[h * kDim + c]) / run->vscale[h * kDim + c] : 1.0;
        EN[c] = EN[c] * ratio + std::fabs(N[c] * ratio) * 0x1p-23;
        N[c] *= ratio;
      }
    }
    const int8_t *kh = &run->k[static_cast<std::size_t>(h) * run->n_pad * kDim];
    for (int b0 = 0; b0 < keys; b0 += 64) {
      const int b1 = std::min(keys, b0 + 64);
      const float c = f32_mul(f32_mul(qs, run->kscale[static_cast<std::size_t>(h) * (run->n_pad / 64) + b0 / 64]), c_sm);
      int S[64];
      bool live[64];
      float smax = -std::numeric_limits<float>::infinity();
      for (int j = b0; j < b1; ++j) {
        const int8_t *kr = kh + static_cast<std::size_t>(j) * kDim;
        int s = 0;
        for (int d = 0; d < kDim; ++d) s += static_cast<int>(qr[d]) * kr[d];
        S[j - b0] = s;
        live[j - b0] = !call.causal || global0 + j <= i + offset;
        if (live[j - b0]) smax = std::max(smax, static_cast<float>(s));
      }
      const float m_new = std::max(m, std::fmaf(smax, c, -kOffset));
      if (m_new != m) {
        const double o = std::exp2(static_cast<double>(f32_sub(m, m_new)));  // fl32(m - m_new), as the kernel; 0 first
        for (int cc = 0; cc < kDim; ++cc) {
          EN[cc] = (EN[cc] + std::fabs(N[cc]) * (kExp2Rel + 0x1p-24)) * o * (1.0 + kExp2Rel);
          N[cc] *= o;
        }
        ED = (ED + D * (kExp2Rel + 0x1p-24)) * o * (1.0 + kExp2Rel);
        D *= o;
        m = m_new;
      }
      // P per key, then two k32 steps (keys b0..b0+31, b0+32..b0+63) of PV and the row sum.
      for (int half = 0; half < 2; ++half) {
        double step[kDim] = {}, step_abs[kDim] = {}, flip[kDim] = {};
        double step_d = 0.0, flip_d = 0.0;
        for (int j = b0 + 32 * half; j < std::min(b1, b0 + 32 * half + 32); ++j) {
          if (!live[j - b0]) continue;
          const float x = std::fmaf(static_cast<float>(S[j - b0]), c, -m);
          const double p_true = std::exp2(static_cast<double>(x));
          const double p = e4m3_rn(p_true);
          const double lo = e4m3_rn(p_true * (1.0 - kExp2Rel)), hi = e4m3_rn(p_true * (1.0 + kExp2Rel));
          const double ulp = hi - lo;  // > 0 only where the kernel's code may differ
          step_d += p;
          flip_d += ulp;
          const float *vr = &op.v8[static_cast<std::size_t>(global0 + j) * kDim];
          for (int cc = 0; cc < kDim; ++cc) {
            const double pv = p * vr[cc];
            step[cc] += pv;
            step_abs[cc] += std::fabs(pv);
          }
          if (ulp > 0.0) {
            for (int cc = 0; cc < kDim; ++cc) flip[cc] += ulp * std::fabs(vr[cc]);
          }
        }
        for (int cc = 0; cc < kDim; ++cc) {
          EN[cc] += 35.0 * 0x1p-25 * (std::fabs(N[cc]) + EN[cc] + step_abs[cc]) + flip[cc];
          N[cc] += step[cc];
        }
        ED += 35.0 * 0x1p-25 * (D + ED + step_d) + flip_d;
        D += step_d;
      }
    }
    global0 += keys;
  }
  const Run *last = call.b != nullptr ? call.b : call.a;
  for (int cc = 0; cc < kDim; ++cc) {
    const double s = last->vscale[h * kDim + cc];
    const double o = N[cc] * s / D;
    rc.out[cc] = o;
    rc.bound[cc] = F * (s * (EN[cc] + std::fabs(N[cc] / D) * ED) / (D - ED) + std::fabs(o) * 3.0 * 0x1p-24) + 1e-30;
  }
}

// fp64 attention of the call's query i (head h) over its keys, from the BF16 inputs.
void exact_row(const Sequence &s, const Call &call, const HeadOperands &op, int h, int i, double *out) {
  const int keys = call.keys(), offset = keys - call.q_tokens;
  const int visible = call.causal ? std::min(keys, i + offset + 1) : keys;
  std::vector<double> sc(visible);
  const uint16_t *qr = &s.q[static_cast<std::size_t>(call.q_first + i) * kHD + h * kDim];
  double qd[kDim];
  for (int d = 0; d < kDim; ++d) qd[d] = bf16_to_f32(qr[d]);
  double mx = -1e300;
  for (int j = 0; j < visible; ++j) {
    const float *kr = &op.kx[static_cast<std::size_t>(j) * kDim];
    double dot = 0.0;
    for (int d = 0; d < kDim; ++d) dot += qd[d] * kr[d];
    sc[j] = dot * kScale;
    mx = std::max(mx, sc[j]);
  }
  double sum = 0.0;
  for (int c = 0; c < kDim; ++c) out[c] = 0.0;
  for (int j = 0; j < visible; ++j) {
    const double p = std::exp(sc[j] - mx);
    sum += p;
    const float *vr = &op.vx[static_cast<std::size_t>(j) * kDim];
    for (int c = 0; c < kDim; ++c) out[c] += p * vr[c];
  }
  for (int c = 0; c < kDim; ++c) out[c] /= sum;
}

// ---- one call, end to end ---------------------------------------------------------------------

struct DeviceRun {
  DeviceBytes bytes;
  int n = 0;
  explicit DeviceRun(int tokens) : bytes(ignis_image_attn_kv8_bytes(tokens)), n(tokens) {}
};

// The rows checked: per head, i * stride + h mod q_tokens for i < want (all rows when few), plus
// the last row. The stride is the smallest odd number >= (q_tokens - 1) / (want - 1), rounded up,
// coprime with q_tokens: the rows are distinct and spread to the end of the call, and the head's
// shift makes the heads together cover every residue mod 128. The shift can stop a head's rows
// short of the last 128-query tile (at 16,384 queries they end at 16,182 + h), so the last row is
// added: every call compares its last CTA.
std::vector<int> rows_for(int q_tokens, int h, int want) {
  std::vector<int> rows;
  if (q_tokens <= want) {
    for (int i = 0; i < q_tokens; ++i) rows.push_back(i);
    return rows;
  }
  long long stride = (q_tokens - 1 + want - 2) / (want - 1);
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, static_cast<long long>(q_tokens)) != 1) stride += 2;
  for (int i = 0; i < want; ++i) rows.push_back(static_cast<int>((i * stride + h) % q_tokens));
  if (std::find(rows.begin(), rows.end(), q_tokens - 1) == rows.end()) rows.push_back(q_tokens - 1);
  return rows;
}

struct Verdict {
  int over = 0;
  double worst = 0.0;
  double min_cos = 1.0, min_centred = 1.0, max_rel_rms = 0.0;
  double median_bound = 0.0;  // the bound over |reference|, median over the checked outputs
  int worst_head = 0;         // the head with the lowest cosine
  // Per head, when the quantization error is checked; spread is the reference's norm with each
  // channel's mean over the rows removed, over its whole norm (the share the centred cosine sees).
  std::vector<double> head_cos, head_centred, head_rel, head_spread;
};

Verdict run_call(const Options &o, const Sequence &s, const Call &call, DeviceBytes &d_q8, const DeviceBytes &d_a,
                 const DeviceBytes *d_b, const DeviceBytes &d_seq_q, bool quant_check = true, int rows_per_head = 64) {
  ATT_RC(ignis_image_attn_quantize_q(static_cast<const uint16_t *>(d_seq_q.p) + static_cast<std::size_t>(call.q_first) * kHD,
                                     call.q_tokens, kHD, d_q8.p, nullptr));
  DeviceBytes d_o32(static_cast<std::size_t>(call.q_tokens) * kHD * 4), d_o16(static_cast<std::size_t>(call.q_tokens) * kHD * 2);
  MOE_CUDA(cudaMemset(d_o32.p, 0xFF, d_o32.bytes));  // poison: an unwritten output is a NaN
  MOE_CUDA(cudaMemset(d_o16.p, 0xFF, d_o16.bytes));
  auto attn = [&](void *out, uint32_t f32) {
    return ignis_image_attn(d_q8.p, call.q_tokens, d_a.p, call.a->n, call.a_keys, d_b != nullptr ? d_b->p : nullptr,
                            call.b != nullptr ? call.b->n : 0, call.causal ? 1 : 0, kScale, out, kHD, f32, nullptr);
  };
  ATT_RC(attn(d_o32.p, 1));
  ATT_RC(attn(d_o16.p, 0));
  MOE_CUDA(cudaDeviceSynchronize());
  const Q8 q = download_q8(d_q8, call.q_tokens);
  const auto o32 = download<float>(d_o32.p, static_cast<std::size_t>(call.q_tokens) * kHD);
  const auto o16 = download<uint16_t>(d_o16.p, static_cast<std::size_t>(call.q_tokens) * kHD);

  long long bf16_diff = 0;
  for (std::size_t i = 0; i < o32.size(); ++i) bf16_diff += f32_to_bf16(o32[i]) != o16[i] || std::isnan(o32[i]);

  std::vector<int> over(kHeads, 0);
  std::vector<std::vector<double>> tight(kHeads);
  std::vector<double> worst(kHeads, 0.0), cosv(kHeads, 1.0), centred(kHeads, 1.0), rel(kHeads, 0.0), spread(kHeads, 1.0);
  parallel_for(kHeads, [&](int h) {
    const std::vector<int> rows = rows_for(call.q_tokens, h, rows_per_head);
    const HeadOperands op = head_operands(s, call, h, quant_check);
    std::vector<double> got, want;
    RowCheck rc;
    for (int i : rows) {
      mechanics_row(call, q, op, h, i, true, o.bound_factor, rc);
      for (int c = 0; c < kDim; ++c) {
        const double y = o32[static_cast<std::size_t>(i) * kHD + h * kDim + c];
        const double err = std::fabs(y - rc.out[c]);
        if (!(err <= rc.bound[c])) ++over[h];
        if (rc.out[c] != 0.0) tight[h].push_back(rc.bound[c] / std::fabs(rc.out[c]));
        worst[h] = std::isnan(err) ? 1e300 : std::max(worst[h], err / rc.bound[c]);
      }
      if (quant_check) {
        double ex[kDim];
        exact_row(s, call, op, h, i, ex);
        for (int c = 0; c < kDim; ++c) {
          got.push_back(o32[static_cast<std::size_t>(i) * kHD + h * kDim + c]);
          want.push_back(ex[c]);
        }
      }
    }
    if (quant_check) {
      double dot = 0, gg = 0, ww = 0, ee = 0;
      for (std::size_t k = 0; k < got.size(); ++k) {
        dot += got[k] * want[k];
        gg += got[k] * got[k];
        ww += want[k] * want[k];
        ee += (got[k] - want[k]) * (got[k] - want[k]);
      }
      cosv[h] = dot / std::sqrt(gg * ww);
      rel[h] = std::sqrt(ee / ww);
      // Centred: each channel's mean over the checked rows removed from both (a V offset inflates
      // the raw cosine).
      double mean_g[kDim] = {}, mean_w[kDim] = {};
      const std::size_t nrows = got.size() / kDim;
      for (std::size_t k = 0; k < got.size(); ++k) {
        mean_g[k % kDim] += got[k] / nrows;
        mean_w[k % kDim] += want[k] / nrows;
      }
      const double ww_raw = ww;
      dot = gg = ww = 0;
      for (std::size_t k = 0; k < got.size(); ++k) {
        const double a = got[k] - mean_g[k % kDim], b = want[k] - mean_w[k % kDim];
        dot += a * b;
        gg += a * a;
        ww += b * b;
      }
      centred[h] = dot / std::sqrt(gg * ww);
      spread[h] = std::sqrt(ww / ww_raw);
    }
  });
  Verdict v;
  v.head_cos = cosv;
  v.head_centred = centred;
  v.head_rel = rel;
  v.head_spread = spread;
  for (int h = 0; h < kHeads; ++h) {
    v.over += over[h];
    v.worst = std::max(v.worst, worst[h]);
    if (cosv[h] < v.min_cos) v.worst_head = h;
    v.min_cos = std::min(v.min_cos, cosv[h]);
    v.min_centred = std::min(v.min_centred, centred[h]);
    v.max_rel_rms = std::max(v.max_rel_rms, rel[h]);
  }
  std::vector<double> all;
  for (const auto &t : tight) all.insert(all.end(), t.begin(), t.end());
  if (!all.empty()) {
    std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());
    v.median_bound = all[all.size() / 2];
  }
  std::printf("  %-44s %5d q x %6d keys%s: mechanics worst %.3f of bound (bound/|O| median %.1e), %d over; BF16 = RN(fp32) %s",
              call.what.c_str(), call.q_tokens, call.keys(), call.causal ? " causal" : "", v.worst, v.median_bound, v.over,
              bf16_diff == 0 ? "bit for bit" : "NO");
  if (quant_check) {
    std::printf("; per head cosine >= %.5f (head %d; centred %.5f), rel RMS <= %.4f", v.min_cos, v.worst_head,
                v.min_centred, v.max_rel_rms);
  }
  std::printf("\n");
  check(v.over == 0, call.what + ": within the mechanics bound of fp64 on the quantized operands");
  check(bf16_diff == 0, call.what + ": the BF16 output is the fp32 output rounded, bit for bit");
  if (quant_check && call.cosine_gated) check(v.min_cos >= o.min_cosine, call.what + ": cosine per head >= --min-cosine");
  return v;
}

// ---- the cases ---------------------------------------------------------------------------------

// A [prefix ; target] sequence with `images` condition images: text segments around 4096-token
// image blocks, then a 4096-token target.
std::vector<std::pair<TokenKind, int>> prefix_layout(int images) {
  if (images == 0) return {{kText, 200}};
  std::vector<std::pair<TokenKind, int>> runs{{kText, 20}};
  for (int i = 0; i < images; ++i) {
    runs.push_back({kImage, 4096});
    runs.push_back({kText, i + 1 < images ? 7 : 160});
  }
  return runs;
}

struct Segment {
  int start, end;
  bool text;
};

std::vector<Segment> segments_of(const std::vector<std::pair<TokenKind, int>> &runs) {
  std::vector<Segment> segs;
  int at = 0;
  for (const auto &r : runs) {
    segs.push_back({at, at + r.second, r.first == kText});
    at += r.second;
  }
  return segs;
}

enum class Segments { kNone, kSome, kAll };

// The calls for `images` condition images and a `target`-token target:
//   the target over [prefix ; target] as the leaf runs it (the split: the prefix run stored alone,
//   the target run with the prefix's mean), checked;
//   the same over [prefix ; target] stored as one run, checked, and the two errors against fp64
//   on the unquantized inputs reported side by side (the cost of the split, not gated);
//   the prefix's own segment calls (step 1).
// The prefix run must come out of all of them byte for byte as it was stored.
void case_images(const Options &o, int images, int target, Segments segments, Profile profile) {
  const auto layout = prefix_layout(images);
  const Sequence s = make_sequence(layout, target, 100 + images, profile);
  std::printf("%d condition image(s), %s inputs: prefix %d tokens, target %d\n", images, profile_name(profile), s.prefix,
              s.target);
  DeviceBytes dq(s.q.size() * 2), dk(s.k.size() * 2), dv(s.v.size() * 2);
  upload(dq, s.q);
  upload(dk, s.k);
  upload(dv, s.v);
  auto at = [](const DeviceBytes &d, int row) { return static_cast<const uint16_t *>(d.p) + static_cast<std::size_t>(row) * kHD; };
  DeviceRun prefix(s.prefix), tgt(s.target);
  ATT_RC(ignis_image_attn_store_kv(at(dk, 0), at(dv, 0), s.prefix, kHD, nullptr, 0, prefix.bytes.p, nullptr));
  ATT_RC(ignis_image_attn_store_kv(at(dk, s.prefix), at(dv, s.prefix), s.target, kHD, prefix.bytes.p, s.prefix,
                                   tgt.bytes.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const Run rp = download_run(prefix.bytes, s.prefix), rt = download_run(tgt.bytes, s.target);
  check_run(s, 0, rp, nullptr, "prefix");
  check_run(s, s.prefix, rt, &rp, "target (the prefix's mean)");
  const std::vector<uint8_t> stored = download<uint8_t>(prefix.bytes.p, prefix.bytes.bytes);

  DeviceBytes q8(ignis_image_attn_q8_bytes(std::max(s.target, s.prefix)));
  // The target over [prefix ; target]. Two profiles measure rather than gate the cosine: K means
  // apart (the split's cost, for spec 04's data to decide) and outlier channels, where the heads
  // with a 20x channel stay below 0.99 (per-block INT8 Q and K: spec image-model/02 "As built").
  const bool gated = profile != Profile::kKShift && profile != Profile::kOutliers;
  Call t;
  t.what = "target over [prefix ; target]";
  t.q_first = s.prefix;
  t.q_tokens = s.target;
  t.a = &rp;
  t.a_keys = s.prefix;
  t.b = &rt;
  t.b_first = s.prefix;
  t.cosine_gated = gated;
  const Verdict split = run_call(o, s, t, q8, prefix.bytes, &tgt.bytes, dq);
  check_q8(s, s.prefix, download_q8(q8, s.target), "target");

  if (images == 1 && profile == Profile::kPlain) {
    // Negative control: the same check without the boundary rescale must fail.
    const Q8 q = download_q8(q8, s.target);
    DeviceBytes d_o(static_cast<std::size_t>(s.target) * kHD * 4);
    ATT_RC(ignis_image_attn(q8.p, s.target, prefix.bytes.p, s.prefix, s.prefix, tgt.bytes.p, s.target, 0, kScale, d_o.p,
                            kHD, 1, nullptr));
    const auto y = download<float>(d_o.p, static_cast<std::size_t>(s.target) * kHD);
    std::atomic<long long> over{0}, total{0};
    parallel_for(kHeads, [&](int h) {
      RowCheck rc;
      long long ov = 0, tot = 0;
      const HeadOperands op = head_operands(s, t, h, false);
      for (int i : rows_for(s.target, h, 8)) {
        mechanics_row(t, q, op, h, i, false, o.bound_factor, rc);
        for (int c = 0; c < kDim; ++c) {
          ++tot;
          ov += !(std::fabs(y[static_cast<std::size_t>(i) * kHD + h * kDim + c] - rc.out[c]) <= rc.bound[c]);
        }
      }
      over += ov;
      total += tot;
    });
    std::printf("  negative control: without the boundary rescale, %lld of %lld outputs fall outside the bound\n",
                over.load(), total.load());
    check(over * 2 > total, "prefix: the mechanics check sees a lost boundary rescale");
  }

  // [prefix ; target] quantized as one run instead: its own mean and V scales.
  {
    DeviceRun joint(s.tokens());
    ATT_RC(ignis_image_attn_store_kv(at(dk, 0), at(dv, 0), s.tokens(), kHD, nullptr, 0, joint.bytes.p, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    const Run rj = download_run(joint.bytes, s.tokens());
    Call j = t;
    j.what = "target over [prefix ; target] stored as one run";
    j.a = &rj;
    j.a_keys = s.tokens();
    j.b = nullptr;
    const Verdict one = run_call(o, s, j, q8, joint.bytes, nullptr, dq);
    // Split against one run, directly (the q8 buffer holds the target's queries).
    DeviceBytes d_s(static_cast<std::size_t>(s.target) * kHD * 4), d_j(static_cast<std::size_t>(s.target) * kHD * 4);
    ATT_RC(ignis_image_attn(q8.p, s.target, prefix.bytes.p, s.prefix, s.prefix, tgt.bytes.p, s.target, 0, kScale, d_s.p,
                            kHD, 1, nullptr));
    ATT_RC(ignis_image_attn(q8.p, s.target, joint.bytes.p, s.tokens(), s.tokens(), nullptr, 0, 0, kScale, d_j.p, kHD, 1,
                            nullptr));
    const auto ys = download<float>(d_s.p, static_cast<std::size_t>(s.target) * kHD);
    const auto yj = download<float>(d_j.p, static_cast<std::size_t>(s.target) * kHD);
    double num = 0, den = 0;
    for (std::size_t i = 0; i < ys.size(); ++i) {
      num += (ys[i] - yj[i]) * (ys[i] - yj[i]);
      den += static_cast<double>(yj[i]) * yj[i];
    }
    std::printf("  the split's cost, against fp64 on the unquantized inputs: split cosine >= %.5f (centred %.5f), rel RMS "
                "<= %.4f; one run cosine >= %.5f (centred %.5f), rel RMS <= %.4f; split against one run: rel RMS %.4f\n",
                split.min_cos, split.min_centred, split.max_rel_rms, one.min_cos, one.min_centred, one.max_rel_rms,
                std::sqrt(num / den));
  }

  // The prefix's own calls (step 1): each segment over the stored prefix truncated at its end.
  const std::vector<Segment> segs = segments_of(layout);
  for (std::size_t si = 0; si < segs.size(); ++si) {
    const Segment &g = segs[si];
    const bool pick = segments == Segments::kAll ||
                      (segments == Segments::kSome && (si == 0 || si == 1 || si + 1 == segs.size() ||
                                                       si + 2 == segs.size() || (g.text && si == segs.size() / 2)));
    if (!pick) continue;
    Call c;
    c.what = std::string(g.text ? "text" : "image") + " segment [" + std::to_string(g.start) + ", " + std::to_string(g.end) +
             ")";
    c.q_first = g.start;
    c.q_tokens = g.end - g.start;
    c.a = &rp;
    c.a_keys = g.end;
    c.causal = g.text;
    c.cosine_gated = gated;
    run_call(o, s, c, q8, prefix.bytes, nullptr, dq);
  }

  // Storage: the calls only read the prefix run.
  MOE_CUDA(cudaDeviceSynchronize());
  const bool same = download<uint8_t>(prefix.bytes.p, prefix.bytes.bytes) == stored;
  std::printf("  the prefix run after every call that read it: %s\n", same ? "byte for byte as stored" : "CHANGED");
  check(same, "prefix: the stored run is unchanged by the calls that read it");
}

// Small, odd sizes: partial blocks everywhere, both runs, truncation, causal with an offset.
void case_small(const Options &o) {
  const Sequence s = make_sequence({{kText, 30}, {kImage, 100}}, 77, 7);
  std::printf("small: prefix %d, target %d\n", s.prefix, s.target);
  DeviceBytes dq(s.q.size() * 2), dk(s.k.size() * 2), dv(s.v.size() * 2);
  upload(dq, s.q);
  upload(dk, s.k);
  upload(dv, s.v);
  auto at = [](const DeviceBytes &d, int row) { return static_cast<const uint16_t *>(d.p) + static_cast<std::size_t>(row) * kHD; };
  DeviceRun prefix(s.prefix), tgt(s.target);
  ATT_RC(ignis_image_attn_store_kv(at(dk, 0), at(dv, 0), s.prefix, kHD, nullptr, 0, prefix.bytes.p, nullptr));
  ATT_RC(ignis_image_attn_store_kv(at(dk, s.prefix), at(dv, s.prefix), s.target, kHD, prefix.bytes.p, s.prefix,
                                   tgt.bytes.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const Run rp = download_run(prefix.bytes, s.prefix), rt = download_run(tgt.bytes, s.target);
  check_run(s, 0, rp, nullptr, "small prefix");
  check_run(s, s.prefix, rt, &rp, "small target");
  DeviceBytes q8(ignis_image_attn_q8_bytes(s.prefix + s.target));
  Call t;
  t.what = "small: target over [prefix ; target]";
  t.q_first = s.prefix;
  t.q_tokens = s.target;
  t.a = &rp;
  t.a_keys = s.prefix;
  t.b = &rt;
  t.b_first = s.prefix;
  run_call(o, s, t, q8, prefix.bytes, &tgt.bytes, dq);
  check_q8(s, s.prefix, download_q8(q8, s.target), "small target");
  Call t2 = t;
  t2.what = "small: target causal over [prefix ; target]";
  t2.causal = true;
  run_call(o, s, t2, q8, prefix.bytes, &tgt.bytes, dq);
  Call c;
  c.what = "small: image rows [30, 130) over keys [0, 130)";
  c.q_first = 30;
  c.q_tokens = 100;
  c.a = &rp;
  c.a_keys = 130;
  run_call(o, s, c, q8, prefix.bytes, nullptr, dq);
  Call c2 = c;
  c2.what = "small: rows [30, 95) causal over keys [0, 95)";
  c2.q_tokens = 65;
  c2.a_keys = 95;
  c2.causal = true;
  run_call(o, s, c2, q8, prefix.bytes, nullptr, dq);
  Call c3 = c;
  c3.what = "small: one row, one key";
  c3.q_first = 0;
  c3.q_tokens = 1;
  c3.a_keys = 1;
  run_call(o, s, c3, q8, prefix.bytes, nullptr, dq, false);
  Call c4 = t;
  c4.what = "small: target over [prefix truncated to 100 ; target]";
  c4.a_keys = 100;
  run_call(o, s, c4, q8, prefix.bytes, &tgt.bytes, dq);
  // Causal over two runs with more queries than run B has keys: the first 128-query tile sees no
  // key of run B, so it never crosses the boundary and ends in run A's V scales.
  Call c5 = t;
  c5.what = "small: 200 rows causal over [prefix ; 70 target keys]";
  c5.q_first = 0;
  c5.q_tokens = 200;
  c5.a_keys = s.prefix;
  c5.causal = true;
  DeviceRun tgt70(70);
  ATT_RC(ignis_image_attn_store_kv(at(dk, s.prefix), at(dv, s.prefix), 70, kHD, prefix.bytes.p, s.prefix, tgt70.bytes.p,
                                   nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const Run rt70 = download_run(tgt70.bytes, 70);
  c5.b = &rt70;
  run_call(o, s, c5, q8, prefix.bytes, &tgt70.bytes, dq);
}

// The quantizers' edge blocks, each restated bit for bit and its scale pinned:
//   zeros                    scale 1, zero codes;
//   every magnitude 1e-37    below the floor: scale 1, zero codes (a regression to `amax > 0`
//                            would give a subnormal scale and full codes);
//   every magnitude at the floor (127 * 2^-126 for Q and K, 448 * 2^-126 for V): scale 2^-126.
// Signs alternate by token, so a head whose keys are all +-a has a K mean of exactly 0 and
// |k - mean| = a.
void case_edges() {
  Sequence s = make_sequence({{kText, 100}}, 0, 9);
  const float tiny = 1e-37f;
  auto at = [](int t, int h, int d) { return static_cast<std::size_t>(t) * kHD + h * kDim + d; };
  auto pm = [](int t, float a) { return f32_to_bf16(t % 2 != 0 ? -a : a); };
  for (int t = 32; t < 64; ++t) {  // Q block 1 (rows 32..63) of heads 3, 4, 6
    for (int d = 0; d < kDim; ++d) {
      s.q[at(t, 3, d)] = 0;
      s.q[at(t, 4, d)] = pm(t, tiny);
      s.q[at(t, 6, d)] = pm(t, kInt8MinAmax);
    }
  }
  for (int t = 0; t < 100; ++t) {
    for (int d = 0; d < kDim; ++d) {  // every key of heads 9 and 10
      s.k[at(t, 9, d)] = pm(t, tiny);
      s.k[at(t, 10, d)] = pm(t, kInt8MinAmax);
    }
    s.v[at(t, 5, 17)] = 0;  // V channels
    s.v[at(t, 6, 17)] = pm(t, tiny);
    s.v[at(t, 6, 18)] = pm(t, kE4m3MinAmax);
  }
  DeviceBytes dq(s.q.size() * 2), dk(s.k.size() * 2), dv(s.v.size() * 2);
  upload(dq, s.q);
  upload(dk, s.k);
  upload(dv, s.v);
  DeviceRun run(100);
  DeviceBytes q8(ignis_image_attn_q8_bytes(100));
  ATT_RC(ignis_image_attn_store_kv(dk.p, dv.p, 100, kHD, nullptr, 0, run.bytes.p, nullptr));
  ATT_RC(ignis_image_attn_quantize_q(dq.p, 100, kHD, q8.p, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const Run r = download_run(run.bytes, 100);
  const Q8 q = download_q8(q8, 100);
  check_run(s, 0, r, nullptr, "edges");
  check_q8(s, 0, q, "edges");
  const float floor_scale = std::ldexp(1.0f, -126);
  auto kscale = [&](int h, int b) { return r.kscale[static_cast<std::size_t>(h) * (r.n_pad / 64) + b]; };
  check(q.scale(3, 40) == 1.0f && r.vscale[5 * kDim + 17] == 1.0f, "edges: a zero Q block and a zero V channel have scale 1");
  check(q.scale(4, 40) == 1.0f && kscale(9, 0) == 1.0f && kscale(9, 1) == 1.0f && r.vscale[6 * kDim + 17] == 1.0f,
        "edges: Q, K and V blocks of 1e-37 have scale 1");
  check(q.scale(6, 40) == floor_scale && kscale(10, 0) == floor_scale && kscale(10, 1) == floor_scale &&
            r.vscale[6 * kDim + 18] == floor_scale,
        "edges: Q, K and V blocks at the floor have scale 2^-126");
  std::printf("  edge blocks: zeros and 1e-37 at scale 1, the floor at scale 2^-126 (Q, K, V)\n");
}

void refusals() {
  DeviceBytes q8(ignis_image_attn_q8_bytes(256)), kv(ignis_image_attn_kv8_bytes(256)), o(256 * kHD * 4),
      x(256 * kHD * 2);
  auto attn = [&](uint32_t q, uint32_t a, uint32_t a_keys, const void *b, uint32_t bt, uint32_t causal, void *out,
                  uint32_t stride) {
    return ignis_image_attn(q8.p, q, kv.p, a, a_keys, b, bt, causal, kScale, out, stride, 0, nullptr);
  };
  check(attn(0, 256, 256, nullptr, 0, 0, o.p, kHD) != 0, "attn: zero queries refused");
  check(attn(16, 256, 0, nullptr, 0, 0, o.p, kHD) != 0, "attn: zero keys refused");
  check(attn(16, 128, 256, nullptr, 0, 0, o.p, kHD) != 0, "attn: a_keys > a_tokens refused");
  check(attn(16, 256, 256, nullptr, 7, 0, o.p, kHD) != 0, "attn: run B tokens without a run refused");
  check(attn(16, 256, 256, kv.p, 0, 0, o.p, kHD) != 0, "attn: a run B of zero tokens refused");
  check(attn(64, 256, 32, nullptr, 0, 1, o.p, kHD) != 0, "attn: causal with fewer keys than queries refused");
  check(attn(16, 256, 256, nullptr, 0, 0, static_cast<char *>(o.p) + 8, kHD) != 0, "attn: o off 16 bytes refused");
  check(attn(16, 256, 256, nullptr, 0, 0, o.p, kHD - 8) != 0, "attn: an output row stride below 4096 refused");
  check(attn(16, 256, 256, nullptr, 0, 0, o.p, kHD + 4) != 0, "attn: an output row stride off 8 refused");
  check(ignis_image_attn(nullptr, 16, kv.p, 256, 256, nullptr, 0, 0, kScale, o.p, kHD, 0, nullptr) != 0,
        "attn: NULL q8 refused");
  // A negative scale makes max(S) * c the row's minimum (P saturates); 0 makes NaN on masked keys.
  check(ignis_image_attn(q8.p, 16, kv.p, 256, 256, nullptr, 0, 0, 0.0f, o.p, kHD, 0, nullptr) != 0,
        "attn: a zero scale refused");
  check(ignis_image_attn(q8.p, 16, kv.p, 256, 256, nullptr, 0, 0, -kScale, o.p, kHD, 0, nullptr) != 0,
        "attn: a negative scale refused");
  check(ignis_image_attn_quantize_q(x.p, 0, kHD, q8.p, nullptr) != 0, "quantize_q: zero tokens refused");
  check(ignis_image_attn_quantize_q(static_cast<char *>(x.p) + 8, 16, kHD, q8.p, nullptr) != 0,
        "quantize_q: q off 16 bytes refused");
  check(ignis_image_attn_quantize_q(x.p, 16, kHD - 8, q8.p, nullptr) != 0, "quantize_q: a row stride below 4096 refused");
  check(ignis_image_attn_store_kv(x.p, x.p, 0, kHD, nullptr, 0, kv.p, nullptr) != 0, "store_kv: zero tokens refused");
  check(ignis_image_attn_store_kv(x.p, x.p, 16, kHD + 2, nullptr, 0, kv.p, nullptr) != 0,
        "store_kv: a row stride off 8 refused");
  check(ignis_image_attn_store_kv(x.p, x.p, 16, kHD, kv.p, 0, kv.p, nullptr) != 0,
        "store_kv: a mean source of zero tokens refused");
  check(ignis_image_attn_store_kv(x.p, nullptr, 16, kHD, nullptr, 0, kv.p, nullptr) != 0, "store_kv: NULL v refused");
}

// ---- real Q, K and V (--cases qkv-probe) ------------------------------------------------------
//
// Q, K and V the DiT computed, taken after QK-norm and RoPE in a bf16 diffusers run (dump_qkv.py,
// kept beside the dump): the t2i case (a 31-token text prefix) and the edit case (text 8, one
// condition image of 4096, text 18), each with a 4096-token target, at steps 0 and 4 and blocks 0,
// 15 and 31. Step 0's files hold [prefix ; target]; step 4's hold the target's rows only, read
// after step 0's prefix (the cached K and V, which the dump checked bit for bit). Each goes through
// the calls spec 04 makes: the prefix run stored alone, the target run with the prefix's mean, the
// target over both, and at step 0 the prefix's own segment calls. Beside them runs the target over
// [prefix ; target] stored as one run (the split's cost). The quantizers, the mechanics bound and
// BF16 = RN(fp32) are checked as above. The cosine against fp64 on the unquantized inputs is
// reported, not gated: per call, then min, p1 and median over every call's 32 heads, and every
// head below --min-cosine by name.

// A 2-D uint16 .npy (little-endian '<u2', C order): BF16 bits.
std::vector<uint16_t> load_npy_u16(const std::string &path, std::size_t &rows, std::size_t &cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "FATAL: cannot open %s\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  char magic[8];
  f.read(magic, 8);
  uint32_t header_len = 0;
  if (magic[6] == 1) {
    uint16_t n = 0;
    f.read(reinterpret_cast<char *>(&n), 2);
    header_len = n;
  } else {
    f.read(reinterpret_cast<char *>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(&header[0], header_len);
  const std::size_t at = header.find('(');
  if (!f || std::memcmp(magic, "\x93NUMPY", 6) != 0 || header.find("'<u2'") == std::string::npos ||
      header.find("'fortran_order': False") == std::string::npos || at == std::string::npos ||
      std::sscanf(header.c_str() + at, "(%zu, %zu)", &rows, &cols) != 2) {
    std::fprintf(stderr, "FATAL: %s is not a 2-D '<u2' C-order .npy\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::vector<uint16_t> v(rows * cols);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(v.size() * 2));
  if (!f) {
    std::fprintf(stderr, "FATAL: %s is short\n", path.c_str());
    std::exit(EXIT_FAILURE);
  }
  return v;
}

struct ProbeHead {
  std::string where;  // case, step, block, call
  int head;
  double cos, centred, rel, spread;
};

// Min, p1 and median of the cosines (raw and centred), the median and max relative RMS error, and
// the heads below `min_cosine`: by the cosine, then by the centred cosine with the share of the
// reference it sees (Verdict::head_spread).
void probe_summary(const std::string &what, const std::vector<ProbeHead> &heads, double min_cosine) {
  if (heads.empty()) return;
  auto sorted = [&](double ProbeHead::*field) {
    std::vector<double> x;
    for (const ProbeHead &p : heads) x.push_back(p.*field);
    std::sort(x.begin(), x.end());
    return x;
  };
  const auto c = sorted(&ProbeHead::cos), cc = sorted(&ProbeHead::centred), r = sorted(&ProbeHead::rel);
  const std::size_t n = heads.size(), p1 = (n - 1) / 100, mid = n / 2;
  std::size_t below = 0;
  for (const ProbeHead &p : heads) below += p.cos < min_cosine;
  check(below == 0, what + ": every head's cosine against fp64 is >= --min-cosine");
  std::printf("  %s, %zu heads: cosine min %.5f, p1 %.5f, median %.5f; centred min %.5f, p1 %.5f, median %.5f; "
              "rel RMS median %.4f, max %.4f; %zu below %.2f\n",
              what.c_str(), n, c[0], c[p1], c[mid], cc[0], cc[p1], cc[mid], r[mid], r[n - 1], below, min_cosine);
  for (const ProbeHead &p : heads) {
    if (p.cos < min_cosine) {
      std::printf("    below: %s, head %d: cosine %.5f (centred %.5f), rel RMS %.4f\n", p.where.c_str(), p.head, p.cos,
                  p.centred, p.rel);
    }
  }
  for (const ProbeHead &p : heads) {
    if (p.centred < min_cosine) {
      std::printf("    centred below: %s, head %d: centred %.5f (cosine %.5f), rel RMS %.4f, spread %.4f\n", p.where.c_str(),
                  p.head, p.centred, p.cos, p.rel, p.spread);
    }
  }
}

void case_qkv_probe(const Options &o, const std::string &dir) {
  std::vector<ProbeHead> split, one, segment;  // every head of every call, by the call's kind
  for (const char *name : {"t2i", "edit"}) {
    const std::string base = dir + "/" + name;
    std::vector<Segment> segs;
    {
      std::ifstream f(base + "/segments.txt");
      int start = 0, end = 0, text = 0;
      while (f >> start >> end >> text) segs.push_back({start, end, text != 0});
    }
    const int prefix = segs.empty() ? 0 : segs.back().end;
    check(prefix > 0, std::string(name) + ": a prefix in segments.txt");
    if (prefix == 0) continue;
    for (int step : {0, 4}) {
      for (int block : {0, 15, 31}) {
        Sequence s;
        auto load = [&](int st, const char *t) {
          std::size_t rows = 0, cols = 0;
          auto v = load_npy_u16(base + "/step" + std::to_string(st) + "/block" + std::to_string(block) + "_" + t + ".npy",
                                rows, cols);
          check(cols == static_cast<std::size_t>(kHD), std::string(name) + ": 4096 columns");
          return v;
        };
        s.q = load(0, "q");
        s.k = load(0, "k");
        s.v = load(0, "v");
        if (step != 0) {  // step 0's prefix rows, then this step's target rows
          const std::size_t keep = static_cast<std::size_t>(prefix) * kHD;
          for (auto *x : {&s.q, &s.k, &s.v}) {
            const auto t = load(step, x == &s.q ? "q" : x == &s.k ? "k" : "v");
            x->resize(keep);
            x->insert(x->end(), t.begin(), t.end());
          }
        }
        s.prefix = prefix;
        s.target = static_cast<int>(s.q.size() / kHD) - prefix;
        check(s.k.size() == s.q.size() && s.v.size() == s.q.size() && s.target > 0,
              std::string(name) + ": q, k and v of [prefix ; target]");
        const std::string where = std::string(name) + ", step " + std::to_string(step) + ", block " + std::to_string(block);
        std::printf("%s: prefix %d, target %d\n", where.c_str(), s.prefix, s.target);
        DeviceBytes dq(s.q.size() * 2), dk(s.k.size() * 2), dv(s.v.size() * 2);
        upload(dq, s.q);
        upload(dk, s.k);
        upload(dv, s.v);
        auto at = [](const DeviceBytes &d, int row) { return static_cast<const uint16_t *>(d.p) + static_cast<std::size_t>(row) * kHD; };
        DeviceRun pre(s.prefix), tgt(s.target), joint(s.tokens());
        ATT_RC(ignis_image_attn_store_kv(at(dk, 0), at(dv, 0), s.prefix, kHD, nullptr, 0, pre.bytes.p, nullptr));
        ATT_RC(ignis_image_attn_store_kv(at(dk, s.prefix), at(dv, s.prefix), s.target, kHD, pre.bytes.p, s.prefix,
                                         tgt.bytes.p, nullptr));
        ATT_RC(ignis_image_attn_store_kv(at(dk, 0), at(dv, 0), s.tokens(), kHD, nullptr, 0, joint.bytes.p, nullptr));
        MOE_CUDA(cudaDeviceSynchronize());
        const Run rp = download_run(pre.bytes, s.prefix), rt = download_run(tgt.bytes, s.target),
                  rj = download_run(joint.bytes, s.tokens());
        check_run(s, 0, rp, nullptr, "prefix");
        check_run(s, s.prefix, rt, &rp, "target (the prefix's mean)");
        DeviceBytes q8(ignis_image_attn_q8_bytes(std::max(s.target, s.prefix)));
        auto collect = [&](std::vector<ProbeHead> &into, const Call &c, const Verdict &v) {
          for (int h = 0; h < kHeads; ++h) {
            into.push_back({where + ", " + c.what, h, v.head_cos[h], v.head_centred[h], v.head_rel[h], v.head_spread[h]});
          }
        };
        Call t;
        t.what = "target over [prefix ; target]";
        t.q_first = s.prefix;
        t.q_tokens = s.target;
        t.a = &rp;
        t.a_keys = s.prefix;
        t.b = &rt;
        t.b_first = s.prefix;
        t.cosine_gated = false;
        collect(split, t, run_call(o, s, t, q8, pre.bytes, &tgt.bytes, dq));
        check_q8(s, s.prefix, download_q8(q8, s.target), "target");
        Call j = t;
        j.what = "target over [prefix ; target] stored as one run";
        j.a = &rj;
        j.a_keys = s.tokens();
        j.b = nullptr;
        collect(one, j, run_call(o, s, j, q8, joint.bytes, nullptr, dq));
        for (const Segment &g : step == 0 ? segs : std::vector<Segment>{}) {
          Call c;
          c.what = std::string(g.text ? "text" : "image") + " segment [" + std::to_string(g.start) + ", " +
                   std::to_string(g.end) + ")";
          c.q_first = g.start;
          c.q_tokens = g.end - g.start;
          c.a = &rp;
          c.a_keys = g.end;
          c.causal = g.text;
          c.cosine_gated = false;
          collect(segment, c, run_call(o, s, c, q8, pre.bytes, nullptr, dq));
        }
      }
    }
  }
  std::printf("real Q/K/V, per head against fp64 on the unquantized inputs (the cosine gated at --min-cosine, the rest reported):\n");
  probe_summary("target over [prefix ; target], the split (spec 04's layout)", split, o.min_cosine);
  probe_summary("target over [prefix ; target], one run", one, o.min_cosine);
  probe_summary("prefix segments (step 0)", segment, o.min_cosine);
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
  o.min_cosine = arg_double(argc, argv, "--min-cosine", o.min_cosine);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "edges,small,0,1,10,16k,shift,tail,outliers,refusals", probe;
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
    if (std::strcmp(argv[i], "--qkv-probe") == 0) probe = argv[i + 1];
  }
  if (probe.empty()) {
    const char *env = std::getenv("IGNIS_IMAGE_QKV_PROBE");
    probe = env != nullptr && *env != '\0' ? env : "F:/ai/models/ignis-image-2.1-turbo/qkv-probe";
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("8-bit DiT attention (bound factor %.2f, min cosine %.4f; cases %s)\n", o.bound_factor, o.min_cosine,
              cases.c_str());
  if (want("edges")) case_edges();
  if (want("small")) case_small(o);
  if (want("0")) case_images(o, 0, 4096, Segments::kAll, Profile::kPlain);
  if (want("1")) case_images(o, 1, 4096, Segments::kAll, Profile::kPlain);
  if (want("10")) case_images(o, 10, 4096, Segments::kSome, Profile::kPlain);
  // A 2048² target (16,384 queries), as the bench times it.
  if (want("16k")) case_images(o, 0, 16384, Segments::kNone, Profile::kPlain);
  for (int images : {0, 1, 10}) {
    if (want("shift")) case_images(o, images, 4096, Segments::kNone, Profile::kKShift);
  }
  for (int images : {0, 1, 10}) {
    if (want("tail")) case_images(o, images, 4096, Segments::kNone, Profile::kTail);
  }
  for (int images : {0, 1, 10}) {
    if (want("outliers")) case_images(o, images, 4096, Segments::kNone, Profile::kOutliers);
  }
  if (want("qkv-probe")) {
    if (std::ifstream(probe + "/t2i/segments.txt").good()) {
      case_qkv_probe(o, probe);
    } else {
      // A skip is never green in the GPU profile (docs/agents/testing.md).
      const char *profile = std::getenv("IGNIS_GPU_PROFILE");
      if (profile != nullptr && std::string(profile) == "1") {
        check(false, "qkv-probe: no Q/K/V dump at " + probe + " (--qkv-probe DIR or IGNIS_IMAGE_QKV_PROBE)");
      } else {
        std::printf("SKIP: qkv-probe: no Q/K/V dump at %s (--qkv-probe DIR or IGNIS_IMAGE_QKV_PROBE)\n", probe.c_str());
      }
    }
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_attn: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_attn: OK\n");
  return 0;
}
