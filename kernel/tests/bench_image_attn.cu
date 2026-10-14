// The 8-bit DiT attention's speed (spec image-model/02 acceptance 4 "Speed", GitHub #319) --
// OURS. TOPS = 4 * queries * keys * 128 * 32 / time (QK^T and PV, two operations per
// multiply-add; half that for a causal call), against SageAttention2++'s 5090 bar of ~618-643
// TOPS for head size 128, non-causal (its 5090_sageattn2++.png, 4K-32K, as the kernel survey reads it).
//
//   target   the target's queries over [condition prefix ; target], at 1024² (4096 queries) and
//            2048² (16384), for 0, 1 and 10 condition images: the call every denoising step
//            makes 32 times;
//   self     N queries over the same N keys, one run, N = 4096 .. 32768: the bar's geometry
//            (with 32 heads and no batch);
//   segments the prefix's calls (step 1): a 4096-token image block and a 160-token text
//            segment, both over a 10-image prefix;
//   quant    the standalone quantizers: Q of 4096 queries, the target's run, a 10-image prefix.
// Each case warms for --warmup-ms, then times --reps launches (at least a second of them) with
// events; the SM clock and the board power are NVML's means over the window's second half.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself
// (GPU lock, timing mode).
//
//   ignis_image_attn_bench [--reps N] [--warmup-ms N] [--self N]
//
// --self N times that one self-attention case only (for a profiler).

#include "ignis_image_attn.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>

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
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, ignis_image_attn_last_error());              \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

constexpr int kHD = 4096;
const float kScale = 1.0f / std::sqrt(128.0f);

struct Timing {
  double ms = 0.0;  // per launch
  unsigned mhz = 0;
  double watts = 0.0;
};

nvmlDevice_t g_nvml = nullptr;

// Warm `run` for warmup_ms, then time `reps` calls, repeated until they span at least
// kTimedMinMs (for calls of 0.1 ms or more). NVML's board power is a running average over about a second and its clock reading
// lags too, so a thread samples both every millisecond over the window's second half only and the
// mean is reported.
constexpr double kTimedMinMs = 1000.0;

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
  // Kernels under 0.1 ms keep --reps: a second of them back to back would time the launches.
  if (estimate_ms >= 0.1) reps = std::max(reps, static_cast<int>(std::ceil(kTimedMinMs / estimate_ms)));
  const double window_ms = reps * estimate_ms;
  std::atomic<bool> sampling{true};
  double mhz_sum = 0.0, w_sum = 0.0;
  int samples = 0;
  std::thread sampler([&] {
    std::this_thread::sleep_for(std::chrono::microseconds(static_cast<long long>(window_ms * 500.0)));
    while (sampling.load() && g_nvml != nullptr) {
      unsigned mhz = 0, mw = 0;
      if (nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS &&
          nvmlDeviceGetPowerUsage(g_nvml, &mw) == NVML_SUCCESS) {
        mhz_sum += mhz;
        w_sum += mw / 1000.0;
        ++samples;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  BENCH_CUDA(cudaEventRecord(e0));
  for (int r = 0; r < reps; ++r) run();
  BENCH_CUDA(cudaEventRecord(e1));
  Timing t;
  BENCH_CUDA(cudaEventSynchronize(e1));
  sampling = false;
  sampler.join();
  if (samples == 0 && g_nvml != nullptr) {  // a window too short to sample: one reading after it
    unsigned mhz = 0, mw = 0;
    nvmlDeviceGetClockInfo(g_nvml, NVML_CLOCK_SM, &mhz);
    nvmlDeviceGetPowerUsage(g_nvml, &mw);
    mhz_sum = mhz;
    w_sum = mw / 1000.0;
    samples = 1;
  }
  if (samples > 0) {
    t.mhz = static_cast<unsigned>(mhz_sum / samples + 0.5);
    t.watts = w_sum / samples;
  }
  BENCH_CUDA(cudaGetLastError());
  float ms = 0.0f;
  BENCH_CUDA(cudaEventElapsedTime(&ms, e0, e1));
  t.ms = ms / reps;
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return t;
}

__global__ void fill_kernel(__nv_bfloat16 *p, std::size_t n, uint32_t seed, float scale, float offset) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B9u + seed;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    const float u = static_cast<float>((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24)) - 510.0f;
    p[i] = __float2bfloat16(u * (scale / 147.8f) + offset * static_cast<float>((i % 128) & 7));
  }
}

// BF16 q, k, v of `tokens` rows, DiT-like magnitudes.
struct Inputs {
  __nv_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr;
  explicit Inputs(int tokens) {
    const std::size_t n = static_cast<std::size_t>(tokens) * kHD;
    BENCH_CUDA(cudaMalloc(&q, n * 2));
    BENCH_CUDA(cudaMalloc(&k, n * 2));
    BENCH_CUDA(cudaMalloc(&v, n * 2));
    fill_kernel<<<4096, 256>>>(q, n, 1, 2.0f, 0.0f);
    fill_kernel<<<4096, 256>>>(k, n, 2, 1.0f, 0.5f);
    fill_kernel<<<4096, 256>>>(v, n, 3, 2.0f, 0.25f);
    BENCH_CUDA(cudaDeviceSynchronize());
  }
  ~Inputs() {
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
  }
  const __nv_bfloat16 *row(const __nv_bfloat16 *p, int t) const { return p + static_cast<std::size_t>(t) * kHD; }
};

void *dalloc(uint64_t bytes) {
  void *p = nullptr;
  BENCH_CUDA(cudaMalloc(&p, bytes));
  return p;
}

void report(const char *what, int q, int keys, bool causal, const Timing &t) {
  const double ops = 4.0 * q * static_cast<double>(keys) * 128.0 * 32.0 / (causal ? 2.0 : 1.0);
  std::printf("  %-46s %6d x %6d  %8.3f ms  %6.1f TOPS  %5u MHz  %5.0f W\n", what, q, keys, t.ms, ops / (t.ms * 1e-3) / 1e12,
              t.mhz, t.watts);
}

int arg_int(int argc, char **argv, const char *name, int fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atoi(argv[i + 1]);
  }
  return fallback;
}

}  // namespace

int main(int argc, char **argv) {
  const int reps = arg_int(argc, argv, "--reps", 20);
  const int warmup_ms = arg_int(argc, argv, "--warmup-ms", 300);
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (nvmlInit_v2() == NVML_SUCCESS) nvmlDeviceGetHandleByIndex_v2(0, &g_nvml);
  std::printf("%s, %d SMs; %d timed launches per case\n", prop.name, prop.multiProcessorCount, reps);

  constexpr int kPrefix10 = 41203;  // 10 condition images: test_image_attn.cu's layout
  const int prefixes[3] = {200, 4276, kPrefix10};
  const int images[3] = {0, 1, 10};
  const int max_target = 16384;
  Inputs in(kPrefix10 + max_target);
  void *kv_prefix = dalloc(ignis_image_attn_kv8_bytes(kPrefix10));
  void *kv_target = dalloc(ignis_image_attn_kv8_bytes(max_target));
  void *kv_self = dalloc(ignis_image_attn_kv8_bytes(32768));
  void *q8 = dalloc(ignis_image_attn_q8_bytes(32768));
  void *o = dalloc(static_cast<std::size_t>(32768) * kHD * 2);

  // --self N: that one self-attention case only (for a profiler).
  const int self_only = arg_int(argc, argv, "--self", 0);
  if (self_only > 0 && self_only <= 32768) {
    BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, self_only, kHD, nullptr, 0, kv_self, nullptr));
    BENCH_RC(ignis_image_attn_quantize_q(in.q, self_only, kHD, q8, nullptr));
    const Timing t = time_it(
        [&] {
          BENCH_RC(ignis_image_attn(q8, self_only, kv_self, self_only, self_only, nullptr, 0, 0, kScale, o, kHD, 0, nullptr));
        },
        reps, warmup_ms);
    report("self-attention, non-causal", self_only, self_only, false, t);
    return 0;
  }

  std::printf("target over [prefix ; target]                       queries x   keys\n");
  double step_ms[3] = {};
  for (int target : {4096, 16384}) {
    for (int c = 0; c < 3; ++c) {
      const int p = prefixes[c];
      BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, p, kHD, nullptr, 0, kv_prefix, nullptr));
      BENCH_RC(ignis_image_attn_store_kv(in.row(in.k, p), in.row(in.v, p), target, kHD, kv_prefix, p, kv_target, nullptr));
      BENCH_RC(ignis_image_attn_quantize_q(in.row(in.q, p), target, kHD, q8, nullptr));
      const Timing t = time_it(
          [&] {
            BENCH_RC(ignis_image_attn(q8, target, kv_prefix, p, p, kv_target, target, 0, kScale, o, kHD, 0, nullptr));
          },
          reps, warmup_ms);
      char what[96];
      std::snprintf(what, sizeof what, "%s, %d condition image(s)", target == 4096 ? "1024²" : "2048²", images[c]);
      report(what, target, p + target, false, t);
      if (target == 4096) step_ms[c] = t.ms;
    }
  }

  std::printf("self-attention, one run (the bar's geometry, 32 heads)\n");
  for (int n : {4096, 8192, 16384, 32768}) {
    BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, n, kHD, nullptr, 0, kv_self, nullptr));
    BENCH_RC(ignis_image_attn_quantize_q(in.q, n, kHD, q8, nullptr));
    const Timing t = time_it(
        [&] { BENCH_RC(ignis_image_attn(q8, n, kv_self, n, n, nullptr, 0, 0, kScale, o, kHD, 0, nullptr)); }, reps,
        warmup_ms);
    char what[64];
    std::snprintf(what, sizeof what, "N = %d, non-causal", n);
    report(what, n, n, false, t);
  }
  for (int n : {4096, 16384}) {
    BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, n, kHD, nullptr, 0, kv_self, nullptr));
    BENCH_RC(ignis_image_attn_quantize_q(in.q, n, kHD, q8, nullptr));
    const Timing t = time_it(
        [&] { BENCH_RC(ignis_image_attn(q8, n, kv_self, n, n, nullptr, 0, 1, kScale, o, kHD, 0, nullptr)); }, reps,
        warmup_ms);
    char what[64];
    std::snprintf(what, sizeof what, "N = %d, causal", n);
    report(what, n, n, true, t);
  }

  std::printf("the prefix's calls (step 1), over a 10-image prefix\n");
  BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, kPrefix10, kHD, nullptr, 0, kv_prefix, nullptr));
  {
    const int start = 36947, end = 41043;  // image 10
    BENCH_RC(ignis_image_attn_quantize_q(in.row(in.q, start), end - start, kHD, q8, nullptr));
    const Timing t = time_it(
        [&] {
          BENCH_RC(ignis_image_attn(q8, end - start, kv_prefix, kPrefix10, end, nullptr, 0, 0, kScale, o, kHD, 0, nullptr));
        },
        reps, warmup_ms);
    report("image block 10 over its truncated keys", end - start, end, false, t);
  }
  {
    const int start = 41043, end = kPrefix10;  // the last text segment
    BENCH_RC(ignis_image_attn_quantize_q(in.row(in.q, start), end - start, kHD, q8, nullptr));
    const Timing t = time_it(
        [&] {
          BENCH_RC(ignis_image_attn(q8, end - start, kv_prefix, kPrefix10, end, nullptr, 0, 1, kScale, o, kHD, 0, nullptr));
        },
        reps, warmup_ms);
    // Causal bottom-right over 41,203 keys is nearly full: count its work as non-causal.
    report("last text segment, causal bottom-right", end - start, end, false, t);
  }

  std::printf("the quantizers                                            tokens\n");
  const Timing tq = time_it([&] { BENCH_RC(ignis_image_attn_quantize_q(in.q, 4096, kHD, q8, nullptr)); }, reps, warmup_ms);
  std::printf("  %-46s %6d  %8.3f ms  %5u MHz  %5.0f W\n", "quantize Q", 4096, tq.ms, tq.mhz, tq.watts);
  BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, kPrefix10, kHD, nullptr, 0, kv_prefix, nullptr));
  const Timing tt = time_it(
      [&] {
        BENCH_RC(ignis_image_attn_store_kv(in.row(in.k, kPrefix10), in.row(in.v, kPrefix10), 4096, kHD, kv_prefix,
                                           kPrefix10, kv_target, nullptr));
      },
      reps, warmup_ms);
  std::printf("  %-46s %6d  %8.3f ms  %5u MHz  %5.0f W\n", "store the target's run (the prefix's mean)", 4096, tt.ms, tt.mhz,
              tt.watts);
  const Timing tp = time_it(
      [&] { BENCH_RC(ignis_image_attn_store_kv(in.k, in.v, kPrefix10, kHD, nullptr, 0, kv_prefix, nullptr)); }, reps,
      warmup_ms);
  std::printf("  %-46s %6d  %8.3f ms  %5u MHz  %5.0f W\n", "store a 10-image prefix's run", kPrefix10, tp.ms, tp.mhz,
              tp.watts);

  std::printf("per denoising step at 1024² (32 layers: the target call, Q and the target's run)\n");
  for (int c = 0; c < 3; ++c) {
    const double per_layer = step_ms[c] + tq.ms + tt.ms;
    std::printf("  %2d condition image(s): %7.2f ms per step, of which the quantizers %.1f%%\n", images[c],
                32.0 * per_layer, 100.0 * (tq.ms + tt.ms) / per_layer);
  }
  cudaFree(kv_prefix);
  cudaFree(kv_target);
  cudaFree(kv_self);
  cudaFree(q8);
  cudaFree(o);
  return 0;
}
