// The FP8 W8A8 GEMM's speed at the DiT's shapes (spec image-model/02 acceptance 2 "Speed" and 3,
// GitHub #319) -- OURS.
//
// For a target of --tokens tokens (default 4096: a 1024² image; 16512 is the largest preset):
//   - each DiT linear shape alone, as TFLOPS against the literature bar (~690-740 for an FP8
//     GEMM on the 5090: spec image-model/02, the survey's aidxhxr and cuBLASLt numbers);
//   - the standalone per-token quantization's cost at the GEMM inputs' widths;
//   - one DiT block's linears as the leaf will run them (to_q, to_k, to_v, to_out, SwiGLU gate,
//     up and down, separate weights) with the four quantizations they need, and the 32 blocks'
//     total per denoising step.
// Each case warms for --warmup-ms, then times --reps launches with events; the SM clock and the
// board power are read through NVML halfway through the timed launches.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself
// (GPU lock, timing mode).
//
//   ignis_image_gemm_bench [--tokens N] [--reps N] [--warmup-ms N] [--sweep]
//
// --sweep times every tile configuration (image_gemm_sweep.h) instead of the
// default one alone.

#include "ignis_image_gemm.h"
#include "image_gemm_sweep.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
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
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, ignis_image_gemm_last_error());              \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

struct Timing {
  double ms = 0.0;  // per launch
  unsigned mhz = 0;
  double watts = 0.0;
};

nvmlDevice_t g_nvml = nullptr;

// Warm `run` for warmup_ms, then time `reps` calls; NVML is read halfway through.
Timing time_it(const std::function<void()> &run, int reps, int warmup_ms) {
  cudaEvent_t e0, e1;
  BENCH_CUDA(cudaEventCreate(&e0));
  BENCH_CUDA(cudaEventCreate(&e1));
  float spent = 0.0f;
  int warm = 0;
  while (spent < static_cast<float>(warmup_ms) || warm < 3) {
    BENCH_CUDA(cudaEventRecord(e0));
    run();
    BENCH_CUDA(cudaEventRecord(e1));
    BENCH_CUDA(cudaEventSynchronize(e1));
    float ms = 0.0f;
    BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
    spent += ms;
    ++warm;
  }
  const double estimate_ms = spent / warm;
  BENCH_CUDA(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) run();
  BENCH_CUDA(cudaEventRecord(e1));
  Timing t;
  std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(estimate_ms * reps * 500.0)));
  if (g_nvml != nullptr) {
    unsigned mw = 0;
    nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &t.mhz);
    if (nvmlDeviceGetPowerUsage(g_nvml, &mw) == NVML_SUCCESS) t.watts = mw / 1000.0;
  }
  BENCH_CUDA(cudaEventSynchronize(e1));
  BENCH_CUDA(cudaGetLastError());
  float ms = 0.0f;
  BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
  t.ms = ms / reps;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return t;
}

uint32_t mix(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

struct Weight {
  void *p = nullptr;
  int rows = 0, cols = 0;
};

// Random finite E4M3 codes and BF16 row scales near 0.004, in the row-scale-v1 layout.
Weight make_weight(int rows, int cols, uint32_t seed) {
  const std::size_t codes = static_cast<std::size_t>(rows) * cols;
  const std::size_t scale_at = (codes + 255) / 256 * 256;
  std::vector<uint8_t> host(scale_at + static_cast<std::size_t>(rows) * 2, 0);
  for (std::size_t i = 0; i < codes; ++i) {
    uint8_t c = static_cast<uint8_t>(mix(static_cast<uint32_t>(i) * 0x9E3779B9u + seed));
    if ((c & 0x7F) == 0x7F) c ^= 1u;
    host[i] = c;
  }
  for (int r = 0; r < rows; ++r) {
    const uint16_t s = 0x3B83;  // BF16 0.004
    std::memcpy(&host[scale_at + 2 * static_cast<std::size_t>(r)], &s, 2);
  }
  Weight w;
  w.rows = rows;
  w.cols = cols;
  BENCH_CUDA(cudaMalloc(&w.p, host.size()));
  BENCH_CUDA(cudaMemcpy(w.p, host.data(), host.size(), cudaMemcpyHostToDevice));
  return w;
}

int arg_int(int argc, char **argv, const char *name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  const int tokens = arg_int(argc, argv, "--tokens", 4096);
  const int reps = arg_int(argc, argv, "--reps", 20);
  const int warmup_ms = arg_int(argc, argv, "--warmup-ms", 300);
  bool sweep = false;
  for (int i = 1; i < argc; ++i) sweep = sweep || std::strcmp(argv[i], "--sweep") == 0;
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (nvmlInit_v2() == NVML_SUCCESS) nvmlDeviceGetHandleByIndex_v2(0, &g_nvml);
  BENCH_RC(ignis_image_gemm_prepare());
  BENCH_RC(image_gemm_sweep::prepare());
  std::printf("%s, %d SMs; %d target tokens, %d timed launches per case\n", prop.name, prop.multiProcessorCount, tokens,
              reps);

  const int hidden = 4096, inter = 12288, kmax = 12288, nmax = 24576;
  // Activations: random BF16, quantized once per width by the op itself.
  std::vector<uint16_t> xh(static_cast<std::size_t>(tokens) * kmax);
  for (std::size_t i = 0; i < xh.size(); ++i) {
    const float v = (static_cast<float>(mix(static_cast<uint32_t>(i) + 17u) >> 8) * 0x1p-23f - 1.0f);
    uint32_t b;
    std::memcpy(&b, &v, 4);
    xh[i] = static_cast<uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
  }
  void *x, *codes, *y;
  float *scales;
  BENCH_CUDA(cudaMalloc(&x, xh.size() * 2));
  BENCH_CUDA(cudaMalloc(&codes, xh.size()));
  BENCH_CUDA(cudaMalloc(&scales, static_cast<std::size_t>(tokens) * 4));
  BENCH_CUDA(cudaMalloc(&y, static_cast<std::size_t>(tokens) * nmax * 2));
  BENCH_CUDA(cudaMemcpy(x, xh.data(), xh.size() * 2, cudaMemcpyHostToDevice));
  BENCH_RC(ignis_image_quantize_tokens(x, tokens, kmax, codes, scales, nullptr));

  struct Case {
    const char *what;
    int rows, cols;
  };
  const Case cases[] = {
      {"to_q / to_k / to_v / to_out, txt_in", hidden, hidden},
      {"SwiGLU gate or up", inter, hidden},
      {"SwiGLU down", hidden, inter},
      {"q/k/v fused", 3 * hidden, hidden},
      {"SwiGLU gate+up fused", 2 * inter, hidden},
      {"img_in", hidden, 64},
      {"proj_out", 64, hidden},
  };
  std::vector<Weight> weights;
  for (const Case &c : cases) weights.push_back(make_weight(c.rows, c.cols, 99u + c.rows + c.cols));

  std::printf("standalone quantization                         cols      ms     GB/s\n");
  double quant_ms[2] = {};
  int qi = 0;
  for (int cols : {hidden, inter}) {
    const Timing t = time_it([&] { BENCH_RC(ignis_image_quantize_tokens(x, tokens, cols, codes, scales, nullptr)); }, reps,
                             warmup_ms);
    const double bytes = static_cast<double>(tokens) * cols * 3 + tokens * 4.0;
    std::printf("  %-44s %5d %7.4f %8.0f\n", "BF16 -> E4M3 + fp32 token scale", cols, t.ms, bytes / (t.ms * 1e-3) / 1e9);
    quant_ms[qi++] = t.ms;
  }

  const int first = sweep ? 0 : image_gemm_sweep::default_config();
  const int last = sweep ? image_gemm_sweep::config_count() - 1 : first;
  for (int config = first; config <= last; ++config) {
    std::printf("configuration %d: %s%s\n", config, image_gemm_sweep::config_name(config),
                config == image_gemm_sweep::default_config() ? " (the default)" : "");
    auto gemm = [&](const Weight &w) {
      BENCH_RC(image_gemm_sweep::gemm_w8a8(config, w.p, w.rows, w.cols, codes, scales, tokens, y, 0, nullptr));
    };
    std::printf("  GEMM alone (BF16 out)                        rows x  cols      ms   TFLOPS   SM MHz  board W\n");
    for (std::size_t i = 0; i < weights.size(); ++i) {
      const Weight &w = weights[i];
      if (image_gemm_sweep::gemm_w8a8(config, w.p, w.rows, w.cols, codes, scales, tokens, y, 0, nullptr) != 0) {
        std::printf("    %-42s %5d x %5d   (not taken: %s)\n", cases[i].what, w.rows, w.cols, ignis_image_gemm_last_error());
        continue;
      }
      const Timing t = time_it([&] { gemm(w); }, reps, warmup_ms);
      const double tflops = 2.0 * tokens * w.rows * w.cols / (t.ms * 1e-3) / 1e12;
      std::printf("    %-42s %5d x %5d %7.3f %8.1f %8u %8.0f\n", cases[i].what, w.rows, w.cols, t.ms, tflops, t.mhz, t.watts);
    }
    // One DiT block's linears with separate weights: q, k, v, out [4096][4096]; gate, up
    // [12288][4096]; down [4096][12288]; four quantizations (block input, attention output,
    // modulated MLP input, SwiGLU output).
    const Weight &sq = weights[0], &gu = weights[1], &dn = weights[2];
    const Timing block = time_it(
        [&] {
          BENCH_RC(ignis_image_quantize_tokens(x, tokens, hidden, codes, scales, nullptr));
          for (int i = 0; i < 3; ++i) gemm(sq);
          BENCH_RC(ignis_image_quantize_tokens(x, tokens, hidden, codes, scales, nullptr));
          gemm(sq);
          BENCH_RC(ignis_image_quantize_tokens(x, tokens, hidden, codes, scales, nullptr));
          for (int i = 0; i < 2; ++i) gemm(gu);
          BENCH_RC(ignis_image_quantize_tokens(x, tokens, inter, codes, scales, nullptr));
          gemm(dn);
        },
        reps, warmup_ms);
    const double block_flop = 2.0 * tokens * (4.0 * hidden * hidden + 3.0 * hidden * inter);
    const double quant_share = (3 * quant_ms[0] + quant_ms[1]) / block.ms;
    std::printf("  one DiT block's linears + 4 quantizations: %.3f ms, %.1f TFLOPS overall (quantization %.1f%%, %u MHz, "
                "%.0f W); x 32 blocks = %.1f ms per denoising step\n",
                block.ms, block_flop / (block.ms * 1e-3) / 1e12, 100.0 * quant_share, block.mhz, block.watts,
                32.0 * block.ms);
  }
  for (Weight &w : weights) cudaFree(w.p);
  cudaFree(x);
  cudaFree(codes);
  cudaFree(scales);
  cudaFree(y);
  if (g_nvml != nullptr) nvmlShutdown();
  return 0;
}
