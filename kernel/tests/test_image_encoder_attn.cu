// The image model's encoder attention (spec image-model/02 acceptance 5, GitHub #319) -- OURS.
// The op is kernel/include/ignis_image_encoder_attn.h; every reference here is our own fp64 CPU code.
//
// The calls are the text encoder's (spec image-model/03): one sequence, 32 query heads over 8 KV
// heads of 128, causal, at lengths from 1 to the longest 10-image edit prompt (41,000 tokens: the
// encoder shapes of the GEMM test), plus a few non-causal calls (the op takes both). Each call is
// checked three ways:
//   mechanics against fp64 attention that replays the kernel's softmax, the bound derived below.
//             The output is fp32 for this check.
//   BF16      the same call with BF16 output must be the fp32 output rounded to BF16, bit for bit
//             (so the call is deterministic too).
//   fp64      against plain fp64 attention on the BF16 inputs: the relative RMS error per head
//             <= --max-rel-rms (default 2^-8, BF16's unit roundoff); the worst is reported.
//
// The inputs are what the text model hands the attention, laid out as its fused QKV GEMM's output
// [tokens][6144] (q at 0, k at 4096, v at 5120): q and k RMSNorm'd per head with weights shared by
// the heads (Qwen3's q_norm / k_norm), four of them low-frequency outlier channels that make some
// heads attend sharply and the first token a sink there, then rotated by RoPE (rotate-half pairs,
// θ 5e6, position = token); V Gaussian with per-channel scales and offsets. QK-normed q and k keep
// |S| bounded, so the bound's band stays narrow. Every call reads a device copy of the first
// `tokens` rows followed by 64 rows of NaN: a kernel that multiplies V rows past the sequence turns
// them into NaN outputs. The guard covers V only: a K row past it is masked to -inf whatever it
// holds, and a Q row past it is never stored. Outputs start as NaN too, so an unwritten one fails.
//
// The mechanics reference replays the kernel per 64-key block (the header's), from the last block a
// row sees down to block 0. s_j = q . k_j is exact in fp64 (BF16 products are exact); the kernel's
// fp32 S'_j is not, and what follows from that is bounded:
//   S        per k16 step (d ascending) e_j += 19 * 2^-25 * (|partial| + e_j + sum_step |q_d k_d|):
//            bench_mma_peak's probes (bf16 P4 = 25): each of an instruction's 16 products and its
//            accumulator loses at most 2^-25 of the largest addend; the result is one fp32 rounding.
//   max      the kernel's running max m' is within em = max e_j (over the keys seen) of the
//            replay's m.
//   exponent the kernel's fl(S'_j c - fl(m' c)) (one FMA) is within dx_j = c (e_j + em)(1 + 2^-22)
//            + 2^-23 (|m c| + |x_j|) of x_j = (s_j - m) c, with c = fl32(scale * log2 e).
//   P        P'_j = BF16(ex2.approx(x')): ex2 is within kExp2Rel of 2^x and flushes below 2^-126,
//            so P'_j lies between the BF16 values of 2^(x_j -+ dx_j)(1 -+ kExp2Rel). The replay takes
//            P_j = BF16(2^x_j); a key whose interval holds more than one BF16 value adds the width
//            times |v| to the numerator's error and the width to the denominator's.
//   rescale  when the max rises, or comes within 2 em of rising (the kernel may rescale where the
//            replay does not): ex2, the argument's two roundings and the products' one, on the
//            accumulators so far.
//   PV       per k16 step 19 * 2^-25 * (|acc| + error so far + the step's sum |P v|).
//   sum      the fp32 sum of the BF16 P, per lane: a block adds 16 terms to a lane's partial
//            (64 keys over 4 lanes), each within 2^-24 of the row's sum; the quad's two adds at
//            the end.
//   end      O = N * (1 / D): two roundings, and the quotient |dO| <= (|dN| + |O| |dD|) / (D - |dD|).
// The whole bound is multiplied by --bound-factor F (default 1).
//
//   ignis_kernel_image_encoder_attn_test [--bound-factor F] [--max-rel-rms R]
//                                        [--cases small,noncausal,mid,max,refusals]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_encoder_attn.h"
#include "moe_fixture.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace moe_test;

namespace {

constexpr int kHeads = 32, kKvHeads = 8, kDim = 128, kHD = kHeads * kDim;
constexpr int kQkv = 6144, kKOff = 4096, kVOff = 5120;  // the fused QKV GEMM's row
constexpr int kBlockN = 64;                               // the header's key block
constexpr int kPoisonRows = 64;
constexpr int kMaxTokens = 41000;
// ex2.approx.f32's relative error, with margin: the PTX ISA gives "2 ulp from the correctly rounded
// result across the full range", so within 2.5 * 2^-23 < 2^-21.6.
constexpr double kExp2Rel = 0x1p-20;
constexpr double kMma = 19.0 * 0x1p-25;  // per k16 step: 16 products + the accumulator, + one rounding
const float kScale = 1.0f / std::sqrt(128.0f);
const double kInf = std::numeric_limits<double>::infinity();

struct Options {
  double bound_factor = 1.0;
  double max_rel_rms = 0x1p-8;
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

// The BF16 value nearest to p >= 0 (ties to even), as the kernel's P: 0 below 2^-126, where
// ex2.approx.ftz flushes.
double kernel_p(double p) {
  if (p < 0x1p-126) return 0.0;
  int e = 0;
  std::frexp(p, &e);  // p in [2^(e-1), 2^e)
  const double step = std::ldexp(1.0, e - 8);
  return std::nearbyint(p / step) * step;
}

// ---- the inputs -------------------------------------------------------------------------------

// Approximately N(0, 1): the Irwin-Hall sum of a hash's four bytes.
float gauss(uint32_t stream, uint64_t i) {
  const uint32_t h = hash_u32(stream, i);
  const int s = static_cast<int>((h & 255u) + ((h >> 8) & 255u) + ((h >> 16) & 255u) + (h >> 24));
  return (static_cast<float>(s) - 510.0f) * (1.0f / 147.8f);
}

bool outlier(int d) { return d == 58 || d == 61 || d == 122 || d == 125; }  // low-frequency RoPE pairs

// One head vector: RMSNorm over its 128 channels with weights w, then RoPE at position t.
void norm_rope(float *x, const float *w, int t) {
  double ss = 0.0;
  for (int d = 0; d < kDim; ++d) ss += static_cast<double>(x[d]) * x[d];
  const float inv = static_cast<float>(1.0 / std::sqrt(ss / kDim + 1e-6));
  for (int d = 0; d < kDim; ++d) x[d] *= inv * w[d];
  for (int i = 0; i < kDim / 2; ++i) {
    const double angle = t * std::pow(5.0e6, -static_cast<double>(i) / (kDim / 2));
    const float cs = static_cast<float>(std::cos(angle)), sn = static_cast<float>(std::sin(angle));
    const float a = x[i], b = x[i + kDim / 2];
    x[i] = a * cs - b * sn;
    x[i + kDim / 2] = b * cs + a * sn;
  }
}

// The sequence, [tokens][6144] BF16.
std::vector<uint16_t> make_qkv(int tokens, uint32_t seed) {
  std::vector<uint16_t> qkv(static_cast<std::size_t>(tokens) * kQkv);
  float wq[kDim], wk[kDim], sign[kDim], aq[kHeads], ak[kKvHeads], mu[kKvHeads * kDim], vs[kKvHeads * kDim],
      vo[kKvHeads * kDim];
  for (int d = 0; d < kDim; ++d) {
    wq[d] = outlier(d) ? 3.0f : 1.1f + hash_uniform(seed + 1, d, 0.5f);
    wk[d] = outlier(d) ? 2.5f : 1.1f + hash_uniform(seed + 2, d, 0.5f);
    sign[d] = hash_u32(seed + 3, d) & 1u ? 1.0f : -1.0f;
  }
  // Per head, how strongly q and k lean on the outlier channels: flat heads to sink heads.
  for (int h = 0; h < kHeads; ++h) aq[h] = 1.5f * std::fabs(hash_uniform(seed + 4, h, 1.0f));
  for (int g = 0; g < kKvHeads; ++g) ak[g] = std::fabs(hash_uniform(seed + 5, g, 1.0f));
  for (int i = 0; i < kKvHeads * kDim; ++i) {
    mu[i] = hash_uniform(seed + 6, i, 1.0f);
    vs[i] = 0.25f + 3.75f * std::fabs(hash_uniform(seed + 7, i, 1.0f));
    vo[i] = hash_uniform(seed + 8, i, 2.0f);
  }
  parallel_for(tokens, [&](int t) {
    uint16_t *row = &qkv[static_cast<std::size_t>(t) * kQkv];
    float x[kDim];
    for (int h = 0; h < kHeads; ++h) {
      // Head 7's query is zero at every 97th token: uniform attention over the keys it sees.
      if (h == 7 && t % 97 == 5) {
        for (int d = 0; d < kDim; ++d) row[h * kDim + d] = 0;
        continue;
      }
      for (int d = 0; d < kDim; ++d) {
        x[d] = gauss(seed + 9, (static_cast<uint64_t>(t) * kHeads + h) * kDim + d) + (outlier(d) ? aq[h] * sign[d] : 0.0f);
      }
      norm_rope(x, wq, t);
      for (int d = 0; d < kDim; ++d) row[h * kDim + d] = f32_to_bf16(x[d]);
    }
    for (int g = 0; g < kKvHeads; ++g) {
      for (int d = 0; d < kDim; ++d) {
        const float sink = t == 0 ? 3.0f : 1.0f;  // the first token: a sink in the outlier heads
        x[d] = gauss(seed + 10, (static_cast<uint64_t>(t) * kKvHeads + g) * kDim + d) + mu[g * kDim + d] +
               (outlier(d) ? ak[g] * sign[d] * sink : 0.0f);
      }
      norm_rope(x, wk, t);
      for (int d = 0; d < kDim; ++d) row[kKOff + g * kDim + d] = f32_to_bf16(x[d]);
      for (int d = 0; d < kDim; ++d) {
        const int i = g * kDim + d;
        row[kVOff + i] = f32_to_bf16(vs[i] * gauss(seed + 11, static_cast<uint64_t>(t) * 1024 + i) + vo[i]);
      }
    }
  });
  return qkv;
}

// One KV head's keys and values as floats, [tokens][128].
struct KvHead {
  std::vector<float> k, v;
};

std::vector<KvHead> kv_heads(const std::vector<uint16_t> &qkv, int tokens) {
  std::vector<KvHead> kv(kKvHeads);
  parallel_for(kKvHeads, [&](int g) {
    kv[g].k.resize(static_cast<std::size_t>(tokens) * kDim);
    kv[g].v.resize(static_cast<std::size_t>(tokens) * kDim);
    for (int t = 0; t < tokens; ++t) {
      for (int d = 0; d < kDim; ++d) {
        const std::size_t from = static_cast<std::size_t>(t) * kQkv + g * kDim + d, to = static_cast<std::size_t>(t) * kDim + d;
        kv[g].k[to] = bf16_to_f32(qkv[from + kKOff]);
        kv[g].v[to] = bf16_to_f32(qkv[from + kVOff]);
      }
    }
  });
  return kv;
}

// ---- the references ---------------------------------------------------------------------------

struct RowCheck {
  double out[kDim];    // the mechanics reference
  double bound[kDim];  // its fp32-output bound
  double exact[kDim];  // plain fp64 attention
};

// Query row i (q: its BF16 values as floats) over the keys it sees in a call of `tokens`.
void reference_row(const float *q, const KvHead &kv, int i, int tokens, bool causal, double F, RowCheck &rc) {
  const double c = static_cast<float>(static_cast<double>(kScale) * 1.4426950408889634);
  const int visible = causal ? i + 1 : tokens;
  std::vector<double> s(visible), e(visible);
  double smax = -kInf;
  for (int j = 0; j < visible; ++j) {
    const float *kr = &kv.k[static_cast<std::size_t>(j) * kDim];
    double acc = 0.0, err = 0.0;
    for (int st = 0; st < kDim / 16; ++st) {
      double step = 0.0, step_abs = 0.0;
      for (int d = st * 16; d < st * 16 + 16; ++d) {
        const double p = static_cast<double>(q[d]) * kr[d];
        step += p;
        step_abs += std::fabs(p);
      }
      err += kMma * (std::fabs(acc) + err + step_abs);
      acc += step;
    }
    s[j] = acc;
    e[j] = err + 0x1p-48 * std::fabs(acc);  // the fp64 sums' own rounding, generously
    smax = std::max(smax, acc);
  }

  // Plain fp64 attention.
  {
    double sum = 0.0;
    for (int cc = 0; cc < kDim; ++cc) rc.exact[cc] = 0.0;
    for (int j = 0; j < visible; ++j) {
      const double p = std::exp((s[j] - smax) * static_cast<double>(kScale));
      sum += p;
      const float *vr = &kv.v[static_cast<std::size_t>(j) * kDim];
      for (int cc = 0; cc < kDim; ++cc) rc.exact[cc] += p * vr[cc];
    }
    for (int cc = 0; cc < kDim; ++cc) rc.exact[cc] /= sum;
  }

  // The kernel's softmax, replayed block by block from the last.
  double N[kDim] = {}, EN[kDim] = {}, D = 0.0, ED = 0.0, m = -kInf, em = 0.0;
  for (int b = (visible - 1) / kBlockN; b >= 0; --b) {
    const int j0 = b * kBlockN, j1 = std::min(visible, j0 + kBlockN);
    double bmax = -kInf;
    for (int j = j0; j < j1; ++j) {
      bmax = std::max(bmax, s[j]);
      em = std::max(em, e[j]);
    }
    const double m_new = std::max(m, bmax);
    if (m > -kInf && bmax > m - 2.0 * em) {
      // The kernel may rescale here (the replay does when m_new > m).
      const double r = m_new > m ? std::exp2((m - m_new) * c) : 1.0;
      const double dlog = 2.0 * em * c * (1.0 + 0x1p-22) + 0x1p-23 * ((m_new - m) * c + 2.0 * em * c);
      const double rho = std::exp2(dlog) * (1.0 + kExp2Rel) - 1.0 + 0x1p-24;
      for (int cc = 0; cc < kDim; ++cc) {
        EN[cc] = (EN[cc] + std::fabs(N[cc]) * rho) * r * (1.0 + rho);
        N[cc] *= r;
      }
      ED = (ED + D * rho) * r * (1.0 + rho);
      D *= r;
    }
    m = m_new;
    // P per key, then the block's k16 steps of PV and its row-sum adds.
    double step_d = 0.0, flip_d = 0.0;
    for (int st = 0; st < kBlockN / 16; ++st) {
      double step[kDim] = {}, step_abs[kDim] = {}, flip[kDim] = {};
      bool any = false;
      for (int j = j0 + 16 * st; j < std::min(j1, j0 + 16 * st + 16); ++j) {
        any = true;
        const double x = (s[j] - m) * c;
        const double dx = c * (e[j] + em) * (1.0 + 0x1p-22) + 0x1p-23 * (std::fabs(m * c) + std::fabs(x)) + 0x1p-60;
        const double p = kernel_p(std::exp2(x));
        const double lo = kernel_p(std::exp2(x - dx) * (1.0 - kExp2Rel));
        const double hi = kernel_p(std::exp2(x + dx) * (1.0 + kExp2Rel));
        const double width = hi - lo;  // > 0 only where the kernel's P may differ from p
        step_d += p;
        flip_d += width;
        const float *vr = &kv.v[static_cast<std::size_t>(j) * kDim];
        for (int cc = 0; cc < kDim; ++cc) {
          const double pv = p * vr[cc];
          step[cc] += pv;
          step_abs[cc] += std::fabs(pv);
        }
        if (width > 0.0) {
          for (int cc = 0; cc < kDim; ++cc) flip[cc] += width * std::fabs(vr[cc]);
        }
      }
      if (!any) continue;
      for (int cc = 0; cc < kDim; ++cc) {
        EN[cc] += kMma * (std::fabs(N[cc]) + EN[cc] + step_abs[cc] + flip[cc]) + flip[cc];
        N[cc] += step[cc];
      }
    }
    ED += 16.0 * 0x1p-24 * (D + ED + step_d + flip_d) + flip_d;
    D += step_d;
  }
  ED += 2.0 * 0x1p-24 * (D + ED);
  for (int cc = 0; cc < kDim; ++cc) {
    const double o = N[cc] / D;
    rc.out[cc] = o;
    rc.bound[cc] = F * ((EN[cc] + std::fabs(o) * ED) / (D - ED) + std::fabs(o) * 2.0 * 0x1p-24) + 1e-30;
  }
}

// ---- one call, end to end ---------------------------------------------------------------------

// The rows checked: per head, i * stride + h mod tokens for i < want (all rows when few), the
// stride the smallest odd number >= (tokens - 1) / (want - 1), rounded up, coprime with tokens (as
// the 8-bit attention's test): distinct rows reaching the end of the call, the heads together
// covering every residue mod 128. Plus the first and last rows, the tile edges 63, 64, 127, 128
// and row 5 (head 7's zero query).
std::vector<int> rows_for(int tokens, int h, int want) {
  std::vector<int> rows;
  if (tokens <= want) {
    for (int i = 0; i < tokens; ++i) rows.push_back(i);
    return rows;
  }
  long long stride = (tokens - 1 + want - 2) / (want - 1);
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, static_cast<long long>(tokens)) != 1) stride += 2;
  for (int i = 0; i < want; ++i) rows.push_back(static_cast<int>((i * stride + h) % tokens));
  for (int r : {0, 5, 63, 64, 127, 128, tokens - 1}) {
    if (r < tokens && std::find(rows.begin(), rows.end(), r) == rows.end()) rows.push_back(r);
  }
  return rows;
}

struct Fixture {
  std::vector<uint16_t> qkv;  // [tokens][6144], the longest call's
  std::vector<KvHead> kv;
  DeviceBytes d_qkv;  // the call's rows, then kPoisonRows of NaN
  explicit Fixture(int tokens)
      : qkv(make_qkv(tokens, 1234)), kv(kv_heads(qkv, tokens)),
        d_qkv(static_cast<std::size_t>(tokens + kPoisonRows) * kQkv * 2) {}
};

void run_call(const Options &o, Fixture &fx, int tokens, bool causal) {
  const std::size_t row_bytes = static_cast<std::size_t>(kQkv) * 2;
  MOE_CUDA(cudaMemcpy(fx.d_qkv.p, fx.qkv.data(), tokens * row_bytes, cudaMemcpyHostToDevice));
  MOE_CUDA(cudaMemset(static_cast<char *>(fx.d_qkv.p) + tokens * row_bytes, 0xFF, kPoisonRows * row_bytes));
  const uint16_t *base = static_cast<const uint16_t *>(fx.d_qkv.p);
  DeviceBytes d_o32(static_cast<std::size_t>(tokens) * kHD * 4), d_o16(static_cast<std::size_t>(tokens) * kHD * 2);
  MOE_CUDA(cudaMemset(d_o32.p, 0xFF, d_o32.bytes));  // poison: an unwritten output is a NaN
  MOE_CUDA(cudaMemset(d_o16.p, 0xFF, d_o16.bytes));
  auto attn = [&](void *out, uint32_t f32) {
    return ignis_image_encoder_attn(base, base + kKOff, base + kVOff, tokens, kQkv, kQkv, causal ? 1 : 0, kScale, out,
                                    kHD, f32, nullptr);
  };
  const std::string what = std::to_string(tokens) + " tokens" + (causal ? ", causal" : ", non-causal");
  if (attn(d_o32.p, 1) != 0 || attn(d_o16.p, 0) != 0) {
    std::fprintf(stderr, "FATAL: ignis_image_encoder_attn (%s): %s\n", what.c_str(), ignis_image_encoder_attn_last_error());
    std::exit(EXIT_FAILURE);
  }
  MOE_CUDA(cudaDeviceSynchronize());
  const auto o32 = download<float>(d_o32.p, static_cast<std::size_t>(tokens) * kHD);
  const auto o16 = download<uint16_t>(d_o16.p, static_cast<std::size_t>(tokens) * kHD);
  long long bf16_diff = 0;
  for (std::size_t i = 0; i < o32.size(); ++i) bf16_diff += std::isnan(o32[i]) || f32_to_bf16(o32[i]) != o16[i];

  std::vector<int> over(kHeads, 0);
  std::vector<std::vector<double>> tight(kHeads);
  std::vector<double> worst(kHeads, 0.0), rel(kHeads, 0.0);
  parallel_for(kHeads, [&](int h) {
    const KvHead &kv = fx.kv[h / (kHeads / kKvHeads)];
    double ee = 0.0, ww = 0.0;
    RowCheck rc;
    float q[kDim];
    for (int i : rows_for(tokens, h, 64)) {
      for (int d = 0; d < kDim; ++d) q[d] = bf16_to_f32(fx.qkv[static_cast<std::size_t>(i) * kQkv + h * kDim + d]);
      reference_row(q, kv, i, tokens, causal, o.bound_factor, rc);
      for (int cc = 0; cc < kDim; ++cc) {
        const double y = o32[static_cast<std::size_t>(i) * kHD + h * kDim + cc];
        const double err = std::fabs(y - rc.out[cc]);
        if (!(err <= rc.bound[cc])) ++over[h];
        if (rc.out[cc] != 0.0) tight[h].push_back(rc.bound[cc] / std::fabs(rc.out[cc]));
        worst[h] = std::isnan(err) ? 1e300 : std::max(worst[h], err / rc.bound[cc]);
        ee += (y - rc.exact[cc]) * (y - rc.exact[cc]);
        ww += rc.exact[cc] * rc.exact[cc];
      }
    }
    rel[h] = std::isnan(ee) ? 1e300 : std::sqrt(ee / ww);
  });
  int total_over = 0;
  double worst_all = 0.0, rel_all = 0.0, median_bound = 0.0;
  for (int h = 0; h < kHeads; ++h) {
    total_over += over[h];
    worst_all = std::max(worst_all, worst[h]);
    rel_all = std::max(rel_all, rel[h]);
  }
  std::vector<double> all;
  for (const auto &t : tight) all.insert(all.end(), t.begin(), t.end());
  if (!all.empty()) {
    std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());
    median_bound = all[all.size() / 2];
  }
  std::printf("  %-28s mechanics worst %.3f of bound (bound/|O| median %.1e), %d over; BF16 = RN(fp32) %s; "
              "rel RMS vs fp64 <= %.2e\n",
              what.c_str(), worst_all, median_bound, total_over, bf16_diff == 0 ? "bit for bit" : "NO", rel_all);
  check(total_over == 0, what + ": within the mechanics bound of fp64");
  check(bf16_diff == 0, what + ": the BF16 output is the fp32 output rounded, bit for bit");
  check(rel_all <= o.max_rel_rms, what + ": relative RMS error against fp64 <= --max-rel-rms");
}

void refusals() {
  DeviceBytes x(static_cast<std::size_t>(256) * kQkv * 2), o(static_cast<std::size_t>(256) * kHD * 4);
  const uint16_t *q = x.as<uint16_t>();
  auto attn = [&](const void *qq, uint32_t tokens, uint32_t qs, uint32_t kvs, float scale, void *out, uint32_t os) {
    return ignis_image_encoder_attn(qq, q + kKOff, q + kVOff, tokens, qs, kvs, 1, scale, out, os, 0, nullptr);
  };
  check(attn(q, 0, kQkv, kQkv, kScale, o.p, kHD) != 0, "zero tokens refused");
  check(attn(q, (1u << 20) + 1, kQkv, kQkv, kScale, o.p, kHD) != 0, "more than 2^20 tokens refused");
  check(attn(nullptr, 16, kQkv, kQkv, kScale, o.p, kHD) != 0, "NULL q refused");
  check(ignis_image_encoder_attn(q, nullptr, q, 16, kQkv, kQkv, 1, kScale, o.p, kHD, 0, nullptr) != 0, "NULL k refused");
  check(attn(q, 16, kQkv, kQkv, kScale, nullptr, kHD) != 0, "NULL o refused");
  check(attn(q, 16, kHD - 8, kQkv, kScale, o.p, kHD) != 0, "a q row stride below 4096 refused");
  check(attn(q, 16, kQkv + 4, kQkv, kScale, o.p, kHD) != 0, "a q row stride off 8 refused");
  check(attn(q, 16, kQkv, 1016, kScale, o.p, kHD) != 0, "a kv row stride below 1024 refused");
  check(attn(q, 16, kQkv, kQkv + 2, kScale, o.p, kHD) != 0, "a kv row stride off 8 refused");
  check(attn(q, 16, kQkv, kQkv, kScale, o.p, kHD - 8) != 0, "an output row stride below 4096 refused");
  check(attn(q, 16, kQkv, kQkv, kScale, o.p, kHD + 4) != 0, "an output row stride off 8 refused");
  check(attn(q, 16, 0x80000000u, kQkv, kScale, o.p, kHD) != 0, "a row stride of 2^31 refused");
  check(attn(q + 4, 16, kQkv, kQkv, kScale, o.p, kHD) != 0, "q off 16 bytes refused");
  check(attn(q, 16, kQkv, kQkv, kScale, static_cast<char *>(o.p) + 8, kHD) != 0, "o off 16 bytes refused");
  check(attn(q, 16, kQkv, kQkv, 0.0f, o.p, kHD) != 0, "a zero scale refused");
  check(attn(q, 16, kQkv, kQkv, -kScale, o.p, kHD) != 0, "a negative scale refused");
  check(attn(q, 16, kQkv, kQkv, std::numeric_limits<float>::quiet_NaN(), o.p, kHD) != 0, "a NaN scale refused");
  const std::string message = ignis_image_encoder_attn_last_error();
  check(message.find("ignis_image_encoder_attn") != std::string::npos && message.find("scale") != std::string::npos,
        "a refusal names the op and the argument: " + message);
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
  o.max_rel_rms = arg_double(argc, argv, "--max-rel-rms", o.max_rel_rms);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "small,noncausal,mid,max,refusals";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("encoder attention (bound factor %.2f, max rel RMS %.2e; cases %s)\n", o.bound_factor, o.max_rel_rms,
              cases.c_str());
  const int longest = want("max") ? kMaxTokens : want("mid") ? 12400 : want("noncausal") ? 4097 : 300;
  if (want("small") || want("noncausal") || want("mid") || want("max")) {
    Fixture fx(longest);
    // Length 1, odd lengths, both sides of the 64-key blocks and of 64- and 128-query tiles, a t2i
    // prompt (~300 tokens).
    if (want("small")) {
      std::printf("small\n");
      for (int n : {1, 2, 7, 63, 64, 65, 127, 128, 129, 191, 192, 193, 300}) run_call(o, fx, n, true);
    }
    if (want("noncausal")) {
      std::printf("non-causal\n");
      for (int n : {1, 65, 300, 4097}) run_call(o, fx, n, false);
    }
    // A 2,000-token prompt, past 4096, and a 10-image edit with a 2,000-token prompt (~12,400).
    if (want("mid")) {
      std::printf("mid\n");
      for (int n : {2048, 4097, 12400}) run_call(o, fx, n, true);
    }
    if (want("max")) {
      std::printf("the longest 10-image edit prompt\n");
      run_call(o, fx, kMaxTokens, true);
    }
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_encoder_attn: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_encoder_attn: OK\n");
  return 0;
}
