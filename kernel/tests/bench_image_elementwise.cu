// The elementwise ops' speed (spec image-model/02 acceptance 7, GitHub #319) -- OURS.
//
//   dit   at a 1024² (4096 target tokens) and a 2048² (16,384) denoising step: the modulated
//         norm (BF16 out, E4M3 out, and BF16 then ignis_image_quantize_tokens: the fused
//         quantization's saving), the gated residual in place, the QK norm with RoPE in place on
//         the fused QKV row, SwiGLU from the fused gate | up output (E4M3 out, and BF16 then the
//         quantization); then one step's elementwise time, 32 blocks of (two norms, two residual
//         adds, one QK norm, one SwiGLU) plus norm_out; and the RoPE table for 4127 and 57,505
//         tokens (ten condition images and the preset).
//   vae   the channel norm with SiLU at the decoder's stage sizes of a 2048² image.
// Each op is memory-bound: GB/s counts its reads plus writes once. Each case warms for 20
// launches, then times --reps launches with events; the best of three windows is kept.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself
// (GPU lock, timing mode).
//
//   ignis_image_elementwise_bench [--reps N] [--cases dit,vae]

#include "ignis_image_dit.h"
#include "ignis_image_gemm.h"
#include "ignis_image_vae.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define BENCH_CUDA(expr)                                                                           \
  do {                                                                                             \
    const cudaError_t err_ = (expr);                                                               \
    if (err_ != cudaSuccess) {                                                                     \
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, cudaGetErrorString(err_));                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

#define BENCH_RC(expr)                                                                             \
  do {                                                                                             \
    if ((expr) != 0) {                                                                             \
      std::fprintf(stderr, "FATAL: %s: %s / %s / %s\n", #expr, ignis_image_dit_last_error(),        \
                   ignis_image_vae_last_error(), ignis_image_gemm_last_error());                    \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

int g_reps = 50;

struct Buffer {
  void *p = nullptr;
  explicit Buffer(std::size_t bytes) {
    BENCH_CUDA(cudaMalloc(&p, bytes));
    BENCH_CUDA(cudaMemset(p, 0, bytes));  // zeros: finite inputs, the same work as any
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  template <typename T> T *as() const { return static_cast<T *>(p); }
};

// Milliseconds per launch of `f`: the best of three windows of g_reps launches.
template <class F> double time_ms(F &&f) {
  for (int i = 0; i < 20; ++i) f();
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  double best = 1e30;
  for (int w = 0; w < 3; ++w) {
    BENCH_CUDA(cudaEventRecord(e0));
    for (int i = 0; i < g_reps; ++i) f();
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    best = std::min(best, static_cast<double>(ms) / g_reps);
  }
  BENCH_CUDA(cudaGetLastError());
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return best;
}

void row(const std::string &what, double ms, double bytes) {
  std::printf("  %-56s %8.4f ms %8.0f GB/s\n", what.c_str(), ms, bytes / ms * 1e-6);
}

void bench_dit(uint32_t tokens, const char *size) {
  std::printf("dit, %s (%u target tokens):\n", size, tokens);
  const std::size_t t = tokens, d = 4096, ff = 12288;
  Buffer x(t * d * 2), y16(t * d * 2), codes(t * ff), scales(t * 4), rows(2 * d * 4), qkv(t * 3 * d * 2),
      table(t * 64 * 8), w(256), gu(t * 2 * ff * 2), a16(t * ff * 2), res(t * d * 2);
  const float *scale = rows.as<float>();
  auto norm = [&](uint32_t kind, void *out, float *s) {
    BENCH_RC(ignis_image_dit_modulated_norm(x.p, tokens, d, scale, scale + d, 0, 1e-6f, out, s, kind, nullptr));
  };
  const double n_bf16 = time_ms([&] { norm(IGNIS_IMAGE_DIT_OUT_BF16, y16.p, nullptr); });
  const double n_e4m3 = time_ms([&] { norm(IGNIS_IMAGE_DIT_OUT_E4M3, codes.p, scales.as<float>()); });
  const double q_d = time_ms([&] { BENCH_RC(ignis_image_quantize_tokens(y16.p, tokens, d, codes.p, scales.as<float>(), nullptr)); });
  row("modulated norm, BF16 out", n_bf16, t * d * 4.0);
  row("modulated norm, E4M3 out (fused)", n_e4m3, t * d * 3.0);
  row("modulated norm BF16 + quantize_tokens (unfused)", n_bf16 + q_d, t * d * 7.0);
  const double resid = time_ms([&] {
    BENCH_RC(ignis_image_dit_gated_residual(res.p, y16.p, tokens, d, scale, scale, 0, res.p, 0, nullptr));
  });
  row("gated residual, in place", resid, t * d * 6.0);
  uint16_t *q = qkv.as<uint16_t>();
  const double qk = time_ms([&] {
    BENCH_RC(ignis_image_dit_qk_norm_rope(q, q + d, tokens, 3 * d, w.p, w.p, 1e-6f, table.as<float>(), q, q + d, 3 * d, 0,
                                          nullptr));
  });
  row("QK norm + RoPE, in place on [tokens][12288]", qk, t * (2 * d * 4.0 + 512));
  const uint16_t *g = gu.as<uint16_t>();
  const double sw_e4m3 = time_ms([&] {
    BENCH_RC(ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, g, g + ff, tokens, ff, 2 * ff, codes.p, scales.as<float>(),
                                 IGNIS_IMAGE_DIT_OUT_E4M3, nullptr));
  });
  const double sw_bf16 = time_ms([&] {
    BENCH_RC(ignis_image_dit_act(IGNIS_IMAGE_DIT_SWIGLU, g, g + ff, tokens, ff, 2 * ff, a16.p, nullptr,
                                 IGNIS_IMAGE_DIT_OUT_BF16, nullptr));
  });
  const double q_ff = time_ms([&] { BENCH_RC(ignis_image_quantize_tokens(a16.p, tokens, ff, codes.p, scales.as<float>(), nullptr)); });
  row("SwiGLU from fused gate | up, E4M3 out (fused)", sw_e4m3, t * ff * 5.0);
  row("SwiGLU BF16 + quantize_tokens (unfused)", sw_bf16 + q_ff, t * ff * 9.0);
  const double step = 32 * (2 * n_e4m3 + 2 * resid + qk + sw_e4m3) + n_e4m3;
  const double unfused = 32 * (2 * (n_bf16 + q_d) + 2 * resid + qk + sw_bf16 + q_ff) + n_bf16 + q_d;
  std::printf("  one step's elementwise ops (32 blocks + norm_out): %.2f ms fused, %.2f ms with separate "
              "quantization\n",
              step, unfused);
}

void bench_rope() {
  std::printf("rope table:\n");
  for (uint32_t tokens : {4127u, 57505u}) {
    Buffer pos(tokens * 12ull), table(tokens * 512ull);
    const double ms = time_ms([&] { BENCH_RC(ignis_image_dit_rope_table(pos.as<int32_t>(), tokens, table.as<float>(), nullptr)); });
    row(std::to_string(tokens) + " tokens", ms, tokens * (12.0 + 512.0));
  }
}

void bench_vae() {
  std::printf("vae channel norm + SiLU, 2048² decoder stages:\n");
  struct Stage {
    uint32_t side, channels;
  };
  for (const Stage s : {Stage{128, 1152}, Stage{256, 1152}, Stage{512, 576}, Stage{1024, 576}, Stage{1024, 288},
                        Stage{2048, 288}, Stage{2048, 144}}) {
    const std::size_t pixels = static_cast<std::size_t>(s.side) * s.side;
    Buffer x(pixels * s.channels * 2), y(pixels * s.channels * 2), g(s.channels * 2ull);
    const double ms = time_ms([&] {
      BENCH_RC(ignis_image_vae_rms_norm(x.p, static_cast<uint32_t>(pixels), s.channels, g.p, 1, y.p, 0, nullptr));
    });
    row(std::to_string(s.side) + "² x " + std::to_string(s.channels), ms, pixels * s.channels * 4.0);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string cases = "dit,vae";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--reps") == 0) g_reps = std::max(1, std::atoi(argv[i + 1]));
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  if (want("dit")) {
    bench_dit(4096, "1024²");
    bench_dit(16384, "2048²");
    bench_rope();
  }
  if (want("vae")) bench_vae();
  return 0;
}
