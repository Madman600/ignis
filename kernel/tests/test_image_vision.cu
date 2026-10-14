// The image model's vision tower (spec image-model/03, GitHub #320) -- OURS: ignis_image_vision.h,
// the 27B's tower program (kernel/src/vision_encode.cu) run with the image model's parameters on
// synthetic W8G32 weights. The reference for the mergers is our own fp64 CPU code; the other checks
// are equalities between calls of the tower itself.
//
//   merger    every block neutral (all-zero codes and biases: each block adds exactly 0 to the
//             residual), so x after the 27 blocks is the patch embedding plus its bias plus the
//             bilinear position table. The merger and the three DeepStack mergers against fp64 that
//             rounds to BF16 wherever the device stores BF16, on one 4 x 4-patch image whose four
//             patches per merged token differ in scale; negative controls: each merger with the
//             other's norm placement (the DeepStack mergers normalize the merged 4608 after the 2x2
//             view, the merger each patch's 1152 before it).
//   taps      one live block among neutral ones: DeepStack merger i reads the residual after block
//             deepstack_layers[i] (8, 16, 24). With block 9 live, output 0 equals the all-neutral
//             run's bit for bit and outputs 1 and 2 do not; with block 8 live, output 0 does not. The
//             taps leave the merger untouched (DeepStack NULL: the same embeds), and one DeepStack
//             output asked alone equals the same output of the full run.
//   segments  two images (8 x 10 and 12 x 8 patches) in one call equal each run alone, bit for bit,
//             with one live block whose attention mixes patches; the same call as one segment (the
//             negative control) differs.
//   refusals  the argument checks, and the workspace size (the 27B's layout).
//
//   ignis_kernel_image_vision_test [--bound B] [--cases merger,taps,segments,refusals]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_vision.h"
#include "model_internal.h"
#include "moe_fixture.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace moe_test;

namespace {

#define VIS_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_vision_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

constexpr uint32_t kHidden = IGNIS_IMAGE_VISION_HIDDEN;
constexpr uint32_t kInter = IGNIS_IMAGE_VISION_INTERMEDIATE;
constexpr uint32_t kPatchDim = IGNIS_IMAGE_VISION_PATCH_DIM;
constexpr uint32_t kMerged = kHidden * IGNIS_IMAGE_VISION_MERGE_UNIT;
constexpr uint32_t kOut = IGNIS_IMAGE_VISION_OUT_HIDDEN;
constexpr uint32_t kDeep = IGNIS_IMAGE_VISION_DEEPSTACK;
constexpr int32_t kSide = 48;  // the position table's side
constexpr double kEps = 1e-6;  // the LayerNorms' eps

double bf(double v) { return bf16_to_f32(f32_to_bf16(static_cast<float>(v))); }

// One W8G32 matrix as the artifact stores it (row-split-k128-v1): codes int8 [rows][K padded to
// 128], zero in the padding, and FP16 scales [rows][padded / 32], on the host and the device.
// Random codes with scales of about 1 / sqrt(K var(code)) keep a unit-scale input at unit scale.
struct W8 {
  uint32_t rows, cols, padded;
  std::vector<int8_t> codes;
  std::vector<uint16_t> scales;
  DeviceBytes d_codes, d_scales;
  W8(uint32_t r, uint32_t c, uint32_t stream, bool zero)
      : rows(r), cols(c), padded((c + 127) / 128 * 128), codes(static_cast<std::size_t>(r) * padded, 0),
        scales(static_cast<std::size_t>(r) * (padded / 32)), d_codes(codes.size()),
        d_scales(scales.size() * 2) {
    const float base = 1.0f / std::sqrt(5376.0f * static_cast<float>(c));
    for (uint32_t row = 0; row < r && !zero; ++row) {
      for (uint32_t col = 0; col < c; ++col) {
        const std::size_t i = static_cast<std::size_t>(row) * padded + col;
        codes[i] = static_cast<int8_t>(static_cast<int>(hash_u32(stream, i) % 255u) - 127);
      }
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
      scales[i] = f32_to_f16(base * (0.5f + std::fabs(hash_uniform(stream ^ 0x5CA1Eu, i, 1.0f))));
    }
    upload(d_codes, codes);
    upload(d_scales, scales);
  }
  ignis_image_vision_w8 view() const { return {d_codes.p, d_scales.p}; }
  // Row `row` dotted with x[0, cols), in fp64, one scale per group of 32.
  double dot(uint32_t row, const double *x) const {
    const int8_t *code = &codes[static_cast<std::size_t>(row) * padded];
    const uint16_t *scale = &scales[static_cast<std::size_t>(row) * (padded / 32)];
    double d = 0.0;
    for (uint32_t g = 0; g * 32 < cols; ++g) {
      double s = 0.0;
      for (uint32_t k = g * 32; k < std::min(cols, g * 32 + 32); ++k) s += code[k] * x[k];
      d += s * f16_to_f32(scale[g]);
    }
    return d;
  }
};

// A BF16 vector: center + uniform(-amplitude, amplitude), or all `center` when amplitude is 0.
struct Vec {
  std::vector<uint16_t> h;
  DeviceBytes d;
  Vec(std::size_t n, uint32_t stream, float center, float amplitude) : h(n), d(n * 2) {
    for (std::size_t i = 0; i < n; ++i) h[i] = f32_to_bf16(center + hash_uniform(stream, i, amplitude));
    upload(d, h);
  }
  double operator[](std::size_t i) const { return bf16_to_f32(h[i]); }
};

struct Block {
  W8 qkv, output, fc1, fc2;
  Vec qkv_bias, output_bias, fc1_bias, fc2_bias, norm1_w, norm1_b, norm2_w, norm2_b;
  // live: random weights; else every code and bias 0, so the block adds exactly 0 to the residual.
  Block(uint32_t s, bool live)
      : qkv(3 * kHidden, kHidden, s + 1, !live), output(kHidden, kHidden, s + 2, !live),
        fc1(kInter, kHidden, s + 3, !live), fc2(kHidden, kInter, s + 4, !live),
        qkv_bias(3 * kHidden, s + 5, 0.0f, live ? 0.1f : 0.0f), output_bias(kHidden, s + 6, 0.0f, live ? 0.1f : 0.0f),
        fc1_bias(kInter, s + 7, 0.0f, live ? 0.1f : 0.0f), fc2_bias(kHidden, s + 8, 0.0f, live ? 0.1f : 0.0f),
        norm1_w(kHidden, s + 9, 1.0f, 0.25f), norm1_b(kHidden, s + 10, 0.0f, 0.1f),
        norm2_w(kHidden, s + 11, 1.0f, 0.25f), norm2_b(kHidden, s + 12, 0.0f, 0.1f) {}
  ignis_image_vision_block view() const {
    return {qkv.view(),     qkv_bias.d.p, output.view(),  output_bias.d.p, fc1.view(),     fc1_bias.d.p,
            fc2.view(),     fc2_bias.d.p, norm1_w.d.p,    norm1_b.d.p,     norm2_w.d.p,    norm2_b.d.p};
  }
};

// A merger; `norm` is its LayerNorm's width: 1152 (the merger) or 4608 (a DeepStack merger).
struct Merger {
  uint32_t norm;
  W8 fc1, fc2;
  Vec fc1_bias, fc2_bias, norm_w, norm_b;
  Merger(uint32_t s, uint32_t norm_cols)
      : norm(norm_cols), fc1(kMerged, kMerged, s + 1, false), fc2(kOut, kMerged, s + 2, false),
        fc1_bias(kMerged, s + 3, 0.0f, 0.1f), fc2_bias(kOut, s + 4, 0.0f, 0.1f),
        norm_w(norm_cols, s + 5, 1.0f, 0.25f), norm_b(norm_cols, s + 6, 0.0f, 0.1f) {}
  ignis_image_vision_merger view() const {
    return {fc1.view(), fc1_bias.d.p, fc2.view(), fc2_bias.d.p, norm_w.d.p, norm_b.d.p};
  }
};

struct Tower {
  W8 patch_embedding{kHidden, kPatchDim, 100, false};
  Vec patch_bias{kHidden, 101, 0.0f, 0.1f};
  Vec table{static_cast<std::size_t>(kSide) * kSide * kHidden, 102, 0.0f, 0.1f};
  Block neutral{200, false};
  Block live{300, true};
  Merger merger{400, kHidden};
  Merger deep0{500, kMerged}, deep1{600, kMerged}, deep2{700, kMerged};
  std::vector<ignis_image_vision_block> blocks;

  // The descriptor with block `live_at` live (-1: none) and the DeepStack taps after `layers`.
  ignis_image_vision_weights weights(int live_at, std::array<uint32_t, kDeep> layers = {8, 16, 24}) {
    blocks.assign(IGNIS_IMAGE_VISION_BLOCKS, neutral.view());
    if (live_at >= 0) blocks[live_at] = live.view();
    ignis_image_vision_weights w{};
    w.patch_embedding = patch_embedding.view();
    w.patch_embedding_bias = patch_bias.d.p;
    w.position_embedding = table.d.p;
    w.blocks = blocks.data();
    w.block_count = IGNIS_IMAGE_VISION_BLOCKS;
    w.merger = merger.view();
    w.deepstack[0] = deep0.view();
    w.deepstack[1] = deep1.view();
    w.deepstack[2] = deep2.view();
    std::copy(layers.begin(), layers.end(), w.deepstack_layers);
    return w;
  }
  const Merger &deep(uint32_t i) const { return i == 0 ? deep0 : i == 1 ? deep1 : deep2; }
};

// One call's host inputs, the images one after the other, each one frame: crates/core's
// vision_item_control per image (the patches in 2x2 merge-block order, the RoPE rows and columns,
// the bilinear corners at fl32(i * 47 / (side - 1))), the position ids as two planes over all the
// patches (every row, then every column).
struct Call {
  uint32_t patches = 0;
  std::vector<uint16_t> rows;  // BF16 [patches][1536]
  std::vector<int32_t> pos_rows, pos_cols, cu{0}, idx;
  std::vector<float> wts;
  std::vector<int32_t> position_ids() const {
    std::vector<int32_t> p(pos_rows);
    p.insert(p.end(), pos_cols.begin(), pos_cols.end());
    return p;
  }
};

float table_coordinate(int32_t i, int32_t size) {
  return size <= 1 ? 0.0f : static_cast<float>(i) * static_cast<float>(kSide - 1) / static_cast<float>(size - 1);
}

// `scaled`: patch p's values times 1 + p % 4, so a merged token's four patches differ in scale.
Call make_call(const std::vector<std::pair<int32_t, int32_t>> &grids, uint32_t stream, bool scaled) {
  Call c;
  for (const auto &[h, w] : grids) {
    for (int32_t by = 0; by < h / 2; ++by) {
      for (int32_t bx = 0; bx < w / 2; ++bx) {
        for (int32_t iy = 0; iy < 2; ++iy) {
          for (int32_t ix = 0; ix < 2; ++ix) {
            const int32_t y = by * 2 + iy, x = bx * 2 + ix;
            c.pos_rows.push_back(y);
            c.pos_cols.push_back(x);
            const float yf = table_coordinate(y, h), xf = table_coordinate(x, w);
            const int32_t y0 = static_cast<int32_t>(yf), x0 = static_cast<int32_t>(xf);
            const int32_t y1 = std::min(y0 + 1, kSide - 1), x1 = std::min(x0 + 1, kSide - 1);
            const float wy = yf - static_cast<float>(y0), wx = xf - static_cast<float>(x0);
            c.idx.insert(c.idx.end(), {y0 * kSide + x0, y0 * kSide + x1, y1 * kSide + x0, y1 * kSide + x1});
            c.wts.insert(c.wts.end(), {(1.0f - wy) * (1.0f - wx), (1.0f - wy) * wx, wy * (1.0f - wx), wy * wx});
          }
        }
      }
    }
    c.patches += static_cast<uint32_t>(h * w);
    c.cu.push_back(static_cast<int32_t>(c.patches));
  }
  c.rows.resize(static_cast<std::size_t>(c.patches) * kPatchDim);
  for (std::size_t i = 0; i < c.rows.size(); ++i) {
    const std::size_t p = i / kPatchDim;
    c.rows[i] = f32_to_bf16(hash_uniform(stream, i, 1.0f) * (scaled ? 1.0f + static_cast<float>(p % 4) : 1.0f));
  }
  return c;
}

struct Outputs {
  std::vector<uint16_t> embeds;
  std::vector<uint16_t> deep[kDeep];  // empty: not asked
};

// One forward: the outputs `want_deep` names, on the legacy stream, then synchronized.
Outputs run(const ignis_image_vision_weights &w, const Call &c, const bool (&want_deep)[kDeep],
            bool deepstack_null = false) {
  const uint32_t tokens = c.patches / IGNIS_IMAGE_VISION_MERGE_UNIT;
  const std::size_t out_n = static_cast<std::size_t>(tokens) * kOut;
  const std::vector<int32_t> pos = c.position_ids();
  ignis_image_vision_input in{c.patches, static_cast<uint32_t>(c.cu.size() - 1), c.rows.data(), pos.data(),
                              c.cu.data(), c.idx.data(), c.wts.data()};
  const uint64_t ws_bytes = ignis_image_vision_workspace_bytes(tokens, in.segments);
  DeviceBytes ws(ws_bytes), embeds(out_n * 2);
  std::unique_ptr<DeviceBytes> deep[kDeep];
  void *deep_ptrs[kDeep] = {nullptr, nullptr, nullptr};
  for (uint32_t i = 0; i < kDeep; ++i) {
    if (want_deep[i]) {
      deep[i] = std::make_unique<DeviceBytes>(out_n * 2);
      deep_ptrs[i] = deep[i]->p;
    }
  }
  VIS_RC(ignis_image_vision_forward(&w, &in, embeds.p, deepstack_null ? nullptr : deep_ptrs, ws.p, ws_bytes, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  Outputs o;
  o.embeds = download<uint16_t>(embeds.p, out_n);
  for (uint32_t i = 0; i < kDeep; ++i) {
    if (want_deep[i]) o.deep[i] = download<uint16_t>(deep[i]->p, out_n);
  }
  return o;
}

constexpr bool kAll[kDeep] = {true, true, true};

// ---- the fp64 references --------------------------------------------------------------------

// x after 27 neutral blocks: bf(bf(bf(patch . W) + bias) + sum_j w_j table[idx_j]), [patches][1152].
std::vector<double> ref_x0(const Tower &t, const Call &c) {
  std::vector<double> x(static_cast<std::size_t>(c.patches) * kHidden);
  for (uint32_t p = 0; p < c.patches; ++p) {
    std::vector<double> row(kPatchDim);
    for (uint32_t k = 0; k < kPatchDim; ++k) row[k] = bf16_to_f32(c.rows[static_cast<std::size_t>(p) * kPatchDim + k]);
    for (uint32_t n = 0; n < kHidden; ++n) {
      const double v = bf(bf(t.patch_embedding.dot(n, row.data())) + t.patch_bias[n]);
      double s = 0.0;
      for (int j = 0; j < 4; ++j) s += c.wts[p * 4 + j] * t.table[static_cast<std::size_t>(c.idx[p * 4 + j]) * kHidden + n];
      x[static_cast<std::size_t>(p) * kHidden + n] = bf(v + s);
    }
  }
  return x;
}

// LayerNorm of x[0, n) with weight/bias from w and b at `offset`, rounded to BF16.
void ref_layer_norm(const double *x, uint32_t n, const Vec &w, const Vec &b, uint32_t offset, double *y) {
  double mean = 0.0, var = 0.0;
  for (uint32_t i = 0; i < n; ++i) mean += x[i];
  mean /= n;
  for (uint32_t i = 0; i < n; ++i) var += (x[i] - mean) * (x[i] - mean);
  var /= n;
  const double r = 1.0 / std::sqrt(var + kEps);
  for (uint32_t i = 0; i < n; ++i) y[i] = bf((x[i] - mean) * r * w[(offset + i) % w.h.size()] + b[(offset + i) % b.h.size()]);
}

// A merger over x0: the norm before the 2x2 view (each patch's 1152) or after it (the merged 4608),
// then fc1 + bias, GELU-exact, fc2 + bias, each output rounded to BF16. [tokens][4096].
std::vector<double> ref_merger(const std::vector<double> &x0, uint32_t tokens, const Merger &m, bool post_shuffle) {
  std::vector<double> out(static_cast<std::size_t>(tokens) * kOut), y(kMerged), h(kMerged);
  for (uint32_t t = 0; t < tokens; ++t) {
    const double *x = &x0[static_cast<std::size_t>(t) * kMerged];
    if (post_shuffle) {
      ref_layer_norm(x, kMerged, m.norm_w, m.norm_b, 0, y.data());
    } else {
      for (uint32_t j = 0; j < IGNIS_IMAGE_VISION_MERGE_UNIT; ++j) ref_layer_norm(x + j * kHidden, kHidden, m.norm_w, m.norm_b, j * kHidden, &y[j * kHidden]);
    }
    for (uint32_t n = 0; n < kMerged; ++n) {
      const double v = bf(bf(m.fc1.dot(n, y.data())) + m.fc1_bias[n]);
      h[n] = bf(0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
    }
    for (uint32_t n = 0; n < kOut; ++n) {
      out[static_cast<std::size_t>(t) * kOut + n] = bf(bf(m.fc2.dot(n, h.data())) + m.fc2_bias[n]);
    }
  }
  return out;
}

double rel_l2(const std::vector<uint16_t> &got, const std::vector<double> &want) {
  double e = 0.0, n = 0.0;
  for (std::size_t i = 0; i < want.size(); ++i) {
    const double g = bf16_to_f32(got[i]);
    e += (g - want[i]) * (g - want[i]);
    n += want[i] * want[i];
  }
  return std::sqrt(e / n);
}

bool same(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b, std::size_t from = 0, std::size_t count = SIZE_MAX,
          std::size_t b_from = 0) {
  count = std::min(count, a.size() - from);
  return b.size() >= b_from + count && std::memcmp(&a[from], &b[b_from], count * 2) == 0;
}

// ---- the cases ------------------------------------------------------------------------------

void case_merger(Tower &t, double bound) {
  std::printf("merger: 27 neutral blocks, one 4 x 4-patch image (4 merged tokens), patches scaled 1..4 per token\n");
  const Call c = make_call({{4, 4}}, 11, true);
  const Outputs o = run(t.weights(-1), c, kAll);
  const std::vector<double> x0 = ref_x0(t, c);
  const uint32_t tokens = c.patches / 4;
  const double e = rel_l2(o.embeds, ref_merger(x0, tokens, t.merger, false));
  const double e_neg = rel_l2(o.embeds, ref_merger(x0, tokens, t.merger, true));
  std::printf("  merger:      rel L2 %.3e against fp64 (bound %.1e); with the norm after the view %.3e\n", e, bound, e_neg);
  check(e <= bound, "merger: the merger's output against fp64");
  check(e_neg > 10.0 * bound, "merger: the negative control (the norm after the view) fails");
  for (uint32_t i = 0; i < kDeep; ++i) {
    const double d = rel_l2(o.deep[i], ref_merger(x0, tokens, t.deep(i), true));
    const double d_neg = rel_l2(o.deep[i], ref_merger(x0, tokens, t.deep(i), false));
    std::printf("  DeepStack %u: rel L2 %.3e against fp64; with the norm before the view %.3e\n", i, d, d_neg);
    check(d <= bound, "merger: DeepStack merger " + std::to_string(i) + "'s output against fp64");
    check(d_neg > 10.0 * bound, "merger: the negative control (DeepStack " + std::to_string(i) + "'s norm before the view) fails");
  }
}

void case_taps(Tower &t) {
  std::printf("taps: one live block among neutral ones, DeepStack after blocks 8, 16, 24\n");
  const Call c = make_call({{4, 4}}, 12, false);
  const Outputs neutral = run(t.weights(-1), c, kAll);
  const Outputs at9 = run(t.weights(9), c, kAll);
  const Outputs at8 = run(t.weights(8), c, kAll);
  check(same(at9.deep[0], neutral.deep[0]), "taps: with block 9 live, DeepStack 0 (after block 8) is the neutral run's");
  check(!same(at9.deep[1], neutral.deep[1]) && !same(at9.deep[2], neutral.deep[2]),
        "taps: with block 9 live, DeepStack 1 and 2 (after blocks 16, 24) see it");
  check(!same(at9.embeds, neutral.embeds), "taps: with block 9 live, the merger sees it");
  check(!same(at8.deep[0], neutral.deep[0]), "taps: with block 8 live, DeepStack 0 (after block 8) sees it");
  // DeepStack 1 and 2 are pinned by the same pairs: a live block 16 is seen by the tap after block
  // 16 and one at 17 is not; a live block 24 is seen by the tap after block 24, a 25 is not.
  const Outputs at16 = run(t.weights(16), c, kAll), at17 = run(t.weights(17), c, kAll);
  check(!same(at16.deep[1], neutral.deep[1]) && same(at17.deep[1], neutral.deep[1]) && !same(at17.deep[2], neutral.deep[2]),
        "taps: DeepStack 1 sees block 16 and not block 17 (DeepStack 2 does)");
  const Outputs at24 = run(t.weights(24), c, kAll), at25 = run(t.weights(25), c, kAll);
  check(!same(at24.deep[2], neutral.deep[2]) && same(at25.deep[2], neutral.deep[2]) && !same(at25.embeds, neutral.embeds),
        "taps: DeepStack 2 sees block 24 and not block 25 (the merger does)");
  const Outputs bare = run(t.weights(9), c, kAll, true);
  check(same(bare.embeds, at9.embeds), "taps: the merger's output is the same without DeepStack (NULL)");
  const bool only1[kDeep] = {false, true, false};
  const Outputs one = run(t.weights(9), c, only1);
  check(same(one.deep[1], at9.deep[1]) && same(one.embeds, at9.embeds), "taps: DeepStack 1 asked alone is the full run's");
  const Outputs moved = run(t.weights(9, {9, 16, 24}), c, kAll);
  check(!same(moved.deep[0], at9.deep[0]) && same(moved.deep[1], at9.deep[1]),
        "taps: deepstack_layers moves the tap (9, 16, 24: DeepStack 0 sees block 9)");
}

void case_segments(Tower &t) {
  std::printf("segments: images of 8 x 10 and 12 x 8 patches (20 and 24 merged tokens), block 3 live\n");
  const Call a = make_call({{8, 10}}, 21, false), b = make_call({{12, 8}}, 22, false);
  Call ab = make_call({{8, 10}, {12, 8}}, 0, false);
  ab.rows = a.rows;
  ab.rows.insert(ab.rows.end(), b.rows.begin(), b.rows.end());
  const ignis_image_vision_weights w = t.weights(3);
  const Outputs oa = run(w, a, kAll), ob = run(w, b, kAll), oab = run(w, ab, kAll);
  const std::size_t na = static_cast<std::size_t>(a.patches / 4) * kOut, nb = static_cast<std::size_t>(b.patches / 4) * kOut;
  check(same(oab.embeds, oa.embeds, 0, na) && same(oab.embeds, ob.embeds, na, nb),
        "segments: the merger's rows of two images in one call equal each image alone, bit for bit");
  for (uint32_t i = 0; i < kDeep; ++i) {
    check(same(oab.deep[i], oa.deep[i], 0, na) && same(oab.deep[i], ob.deep[i], na, nb),
          "segments: DeepStack " + std::to_string(i) + "'s rows equal each image alone, bit for bit");
  }
  Call one = ab;
  one.cu = {0, static_cast<int32_t>(ab.patches)};
  const Outputs oone = run(w, one, kAll);
  check(!same(oone.embeds, oab.embeds), "segments: the negative control (one segment over both images) differs");
}

void expect_refusal(int32_t rc, const char *needle, const std::string &what) {
  const std::string message = ignis_image_vision_last_error();
  check(rc == -1 && message.find("ignis_image_vision_forward") != std::string::npos && message.find(needle) != std::string::npos,
        "refusals: " + what + " (got " + std::to_string(rc) + ": " + message + ")");
}

void case_refusals(Tower &t) {
  std::printf("refusals\n");
  check(ignis_image_vision_workspace_bytes(0, 1) == 0 && ignis_image_vision_workspace_bytes(4, 0) == 0,
        "refusals: no workspace for no tokens or no segments");
  check(ignis_image_vision_workspace_bytes(32769, 1) == 0 && ignis_image_vision_workspace_bytes(4, 11) == 0 &&
            ignis_image_vision_workspace_bytes(0xFFFFFFFFu, 1) == 0,
        "refusals: no workspace past the patch cap, the segment cap, or an overflowing token count");
  check(ignis_image_vision_workspace_bytes(12300, 10) > 0, "refusals: ten 1,230-token images have a workspace");
  for (const auto &[tokens, segments] : {std::pair<uint32_t, uint32_t>{4, 1}, {1024, 1}, {10240, 10}}) {
    const uint64_t got = ignis_image_vision_workspace_bytes(tokens, segments);
    std::printf("  workspace for %u merged tokens over %u segments: %llu bytes\n", tokens, segments,
                static_cast<unsigned long long>(got));
    check(got == ignis_vision_workspace_bytes(static_cast<int32_t>(tokens), static_cast<int32_t>(segments)),
          "refusals: the workspace is the 27B's layout's");
  }
  const Call c = make_call({{4, 4}}, 13, false);
  const std::vector<int32_t> pos = c.position_ids();
  const uint64_t ws_bytes = ignis_image_vision_workspace_bytes(4, 1);
  DeviceBytes ws(ws_bytes), out(4 * kOut * 2 * 4);
  void *outs = out.p;
  void *deep[kDeep] = {static_cast<char *>(outs) + 4 * kOut * 2, nullptr, nullptr};
  ignis_image_vision_weights w = t.weights(-1);
  ignis_image_vision_input in{c.patches, 1, c.rows.data(), pos.data(), c.cu.data(), c.idx.data(), c.wts.data()};
  const auto call = [&](const ignis_image_vision_weights *pw, const ignis_image_vision_input *pin, void *e, void *const *d,
                        uint64_t bytes) { return ignis_image_vision_forward(pw, pin, e, d, ws.p, bytes, nullptr); };
  expect_refusal(call(nullptr, &in, outs, deep, ws_bytes), "null argument", "null weights");
  ignis_image_vision_weights bad = w;
  bad.block_count = 26;
  expect_refusal(call(&bad, &in, outs, deep, ws_bytes), "26 blocks", "26 blocks");
  bad = w;
  bad.deepstack_layers[1] = 8;
  expect_refusal(call(&bad, &in, outs, deep, ws_bytes), "ascend", "DeepStack layers that do not ascend");
  bad = w;
  bad.deepstack_layers[2] = 27;
  expect_refusal(call(&bad, &in, outs, deep, ws_bytes), "ascend", "a DeepStack layer past the tower");
  bad = w;
  bad.deepstack[0].fc2.scales = nullptr;
  expect_refusal(call(&bad, &in, outs, deep, ws_bytes), "DeepStack merger 0", "a DeepStack merger without scales");
  ignis_image_vision_input bin = in;
  bin.segments = 0;
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "segments must be", "no segment");
  bin.segments = IGNIS_IMAGE_VISION_MAX_SEGMENTS + 1;
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "segments must be", "11 segments");
  std::vector<int32_t> cu = {0, 8};
  bin = in;
  bin.cu_seqlens = cu.data();
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "from 0 to patches", "segments that do not end at the patches");
  std::vector<int32_t> cu2 = {0, 6, 16};
  bin.segments = 2;
  bin.cu_seqlens = cu2.data();
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "multiple of 4", "a segment of 6 patches");
  bin = in;
  bin.patches = 18;
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "positive multiple", "18 patches");
  bin.patches = 131076;
  expect_refusal(call(&w, &bin, outs, deep, ws_bytes), "at most 131072", "131,076 patches");
  expect_refusal(call(&w, &in, outs, deep, ws_bytes - 256), "workspace must hold", "a workspace too small");
  expect_refusal(call(&w, &in, static_cast<char *>(outs) + 2, deep, ws_bytes), "16-byte aligned", "a misaligned output");
  void *overlap[kDeep] = {static_cast<char *>(outs) + 16, nullptr, nullptr};
  expect_refusal(call(&w, &in, outs, overlap, ws_bytes), "overlap", "DeepStack's output over the merger's");
  expect_refusal(ignis_image_vision_forward(&w, &in, ws.p, deep, ws.p, ws_bytes, nullptr), "overlap", "the merger's output over the workspace");
  // Not refused: no DeepStack at all.
  VIS_RC(call(&w, &in, outs, nullptr, ws_bytes));
  MOE_CUDA(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char **argv) {
  double bound = 1e-2;
  std::string cases = "merger,taps,segments,refusals";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--bound") bound = std::atof(argv[++i]);
    else if (std::string(argv[i]) == "--cases") cases = argv[++i];
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::fprintf(stderr, "FATAL: no CUDA device (ADR 0006: a GPU test fails without one)\n");
    return EXIT_FAILURE;
  }
  const auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  Tower tower;
  if (want("merger")) case_merger(tower, bound);
  if (want("taps")) case_taps(tower);
  if (want("segments")) case_segments(tower);
  if (want("refusals")) case_refusals(tower);
  if (g_failed != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_failed);
    return EXIT_FAILURE;
  }
  std::printf("all checks passed\n");
  return EXIT_SUCCESS;
}
