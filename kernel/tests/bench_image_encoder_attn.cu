// The encoder attention's speed (spec image-model/02 acceptance 5 "TOPS", GitHub #319) -- OURS.
// TOPS = 4 * N * N * 128 * 32 / time (QK^T and PV, two operations per multiply-add; half that for a
// causal call), against FlashAttention-2's 5090 bar of ~208-215 TOPS for BF16, head size 128,
// non-causal (SageAttention's 5090_sageattn2++.png, 4K-32K).
//
//   causal      the text encoder's call: one sequence of N tokens, 32 query heads over 8 KV
//               heads, read from the fused QKV row (stride 6144), from a t2i prompt (~300) to
//               the longest 10-image edit prompt (41,000);
//   non-causal  N = 4096 .. 32768: the bar's geometry (32 heads, no batch);
//   encoder     the 36 layers' attention at the encoder's lengths, from the causal times.
// Each case warms for --warmup-ms, then times --reps launches (at least a second of them) with
// events; the SM clock and the board power are NVML's means over the window's second half.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself
// (GPU lock, timing mode).
//
//   ignis_image_encoder_attn_bench [--reps N] [--warmup-ms N] [--n N [--non-causal]]
//
// --n N times that one case only (for a profiler).

#include "ignis_image_encoder_attn.h"

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
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, ignis_image_encoder_attn_last_error());      \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

namespace {

constexpr int kHD = 4096, kQkv = 6144, kKOff = 4096, kVOff = 5120, kLayers = 36;
const float kScale = 1.0f / std::sqrt(128.0f);

struct Timing {
  double ms = 0.0;  // per launch
  unsigned mhz = 0;
  double watts = 0.0;
};

nvmlDevice_t g_nvml = nullptr;

// Warm `run` for warmup_ms, then time `reps` calls, repeated until they span at least
// kTimedMinMs (for calls of 0.1 ms or more). NVML's board power is a running average over about a
// second and its clock reading lags too, so a thread samples both every millisecond over the
// window's second half only and the mean is reported.
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

// BF16 of roughly unit RMS (q and k as RMSNorm leaves them, v as it comes).
__global__ void fill_kernel(__nv_bfloat16 *p, std::size_t n, uint32_t seed) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B9u + seed;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    const float u = static_cast<float>((x & 255u) + ((x >> 8) & 255u) + ((x >> 16) & 255u) + (x >> 24)) - 510.0f;
    p[i] = __float2bfloat16(u * (1.0f / 147.8f));
  }
}

void report(const char *what, int n, bool causal, const Timing &t) {
  const double ops = 4.0 * n * static_cast<double>(n) * 128.0 * 32.0 / (causal ? 2.0 : 1.0);
  std::printf("  %-34s %6d  %8.3f ms  %6.1f TOPS  %5u MHz  %5.0f W\n", what, n, t.ms, ops / (t.ms * 1e-3) / 1e12, t.mhz,
              t.watts);
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
  bool non_causal_only = false;
  for (int i = 1; i < argc; ++i) non_causal_only = non_causal_only || std::strcmp(argv[i], "--non-causal") == 0;
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (nvmlInit_v2() == NVML_SUCCESS) nvmlDeviceGetHandleByIndex_v2(0, &g_nvml);
  std::printf("%s, %d SMs; %d timed launches per case\n", prop.name, prop.multiProcessorCount, reps);

  constexpr int kMax = 41000;
  const std::size_t n_qkv = static_cast<std::size_t>(kMax) * kQkv;
  __nv_bfloat16 *qkv = nullptr, *o = nullptr;
  BENCH_CUDA(cudaMalloc(&qkv, n_qkv * 2));
  BENCH_CUDA(cudaMalloc(&o, static_cast<std::size_t>(kMax) * kHD * 2));
  fill_kernel<<<4096, 256>>>(qkv, n_qkv, 1);
  BENCH_CUDA(cudaDeviceSynchronize());
  auto call = [&](int n, bool causal) {
    BENCH_RC(ignis_image_encoder_attn(qkv, qkv + kKOff, qkv + kVOff, n, kQkv, kQkv, causal ? 1 : 0, kScale, o, kHD, 0,
                                      nullptr));
  };

  // --n N: that one case only (for a profiler).
  const int only = arg_int(argc, argv, "--n", 0);
  if (only > 0 && only <= kMax) {
    const Timing t = time_it([&] { call(only, !non_causal_only); }, reps, warmup_ms);
    report(non_causal_only ? "non-causal" : "causal", only, !non_causal_only, t);
    return 0;
  }

  std::printf("causal, one sequence (the encoder's call)              N\n");
  const int lengths[] = {300, 2048, 4096, 12400, 16384, 32768, kMax};
  double ms_at[sizeof lengths / sizeof lengths[0]] = {};
  for (std::size_t i = 0; i < sizeof lengths / sizeof lengths[0]; ++i) {
    const Timing t = time_it([&] { call(lengths[i], true); }, reps, warmup_ms);
    report("causal", lengths[i], true, t);
    ms_at[i] = t.ms;
  }
  std::printf("non-causal (the bar's geometry, 32 heads)\n");
  for (int n : {4096, 8192, 16384, 32768}) {
    const Timing t = time_it([&] { call(n, false); }, reps, warmup_ms);
    report("non-causal", n, false, t);
  }
  std::printf("the encoder's 36 attention layers\n");
  const char *names[] = {"t2i, short prompt", "t2i, 2,000-token prompt", "", "10-image edit, 2,000-token prompt", "", "",
                         "the longest 10-image prompt"};
  for (std::size_t i : {0, 1, 3, 6}) {
    std::printf("  %-34s %6d  %8.2f ms\n", names[i], lengths[i], kLayers * ms_at[i]);
  }
  cudaFree(qkv);
  cudaFree(o);
  return 0;
}
