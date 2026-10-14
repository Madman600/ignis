// The VAE ops' speed (spec image-model/02 acceptance 6 "Speed", GitHub #319) -- OURS.
//
//   conv      the convolution's TFLOPS (2 * pixels * out * k * k * in / time) at every distinct
//             decoder and encoder shape of a 1024² and a 2048² image, against the BF16 bar
//             (~210-240 TFLOPS); then a whole decode's and a whole encode's convolutions, each
//             shape's time times the number of times the VAE runs it (the dead time_convs out).
//   attn      the mid-block attention at the encoder's D = 768 and the decoder's D = 1152, 4096
//             and 16,384 tokens, for three row tiles, with the workspace each needs.
//   shortcut  the largest DupUp and AvgDown (memory-bound; GB/s of reads plus writes).
// Each case warms for --warmup-ms, then times --reps launches (at least a second of them) with
// events; the SM clock and the board power are NVML's means over the window's second half.
//
// A timing is a finding, not a pass/fail: a tool, not a CTest. Run it with the card to itself
// (GPU lock, timing mode).
//
//   ignis_image_vae_bench [--reps N] [--warmup-ms N] [--cases conv,attn,shortcut]

#include "ignis_image_vae.h"

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
#include <string>
#include <thread>
#include <tuple>

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
      std::fprintf(stderr, "FATAL: %s: %s\n", #expr, ignis_image_vae_last_error());               \
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

// Warm `run` for warmup_ms, then time `reps` calls, repeated until they span at least
// kTimedMinMs (for calls of 0.1 ms or more); NVML sampled every millisecond over the window's
// second half (bench_image_attn.cu's method: its power reading is a ~1 s running average).
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
  if (samples == 0 && g_nvml != nullptr) {
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

__global__ void fill_kernel(__nv_bfloat16 *p, std::size_t n, uint32_t seed, float amplitude) {
  for (std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    uint32_t x = static_cast<uint32_t>(i) * 0x9E3779B9u + seed;
    x ^= x >> 16;
    x *= 0x7FEB352Du;
    x ^= x >> 15;
    x *= 0x846CA68Bu;
    x ^= x >> 16;
    p[i] = __float2bfloat16((static_cast<float>(x >> 8) * 0x1p-23f - 1.0f) * amplitude);
  }
}

// A VAE convolution: kind, upsample, channels, the stage (image / div) and how many times one
// decode or encode runs it.
struct Shape {
  const char *name;
  uint32_t kind, upsample;
  int cin, cout, div, times;
};

constexpr uint32_t k3 = IGNIS_IMAGE_VAE_CONV_3X3, kS2 = IGNIS_IMAGE_VAE_CONV_3X3_S2, k1 = IGNIS_IMAGE_VAE_CONV_1X1;

const Shape kDecoder[] = {
    {"post_quant_conv", k1, 0, 64, 64, 16, 1},       {"conv_in", k3, 0, 64, 1152, 16, 1},
    {"mid to_qkv", k1, 0, 1152, 3456, 16, 1},        {"mid proj", k1, 0, 1152, 1152, 16, 1},
    {"mid + up0 resnets", k3, 0, 1152, 1152, 16, 10}, {"up0 upsample", k3, 1, 1152, 1152, 16, 1},
    {"up1 resnets", k3, 0, 1152, 1152, 8, 6},         {"up1 upsample", k3, 1, 1152, 1152, 8, 1},
    {"up2 conv1", k3, 0, 1152, 576, 4, 1},            {"up2 conv_shortcut", k1, 0, 1152, 576, 4, 1},
    {"up2 resnets", k3, 0, 576, 576, 4, 5},           {"up2 upsample", k3, 1, 576, 576, 4, 1},
    {"up3 conv1", k3, 0, 576, 288, 2, 1},             {"up3 conv_shortcut", k1, 0, 576, 288, 2, 1},
    {"up3 resnets", k3, 0, 288, 288, 2, 5},           {"up3 upsample", k3, 1, 288, 288, 2, 1},
    {"up4 conv1", k3, 0, 288, 144, 1, 1},             {"up4 conv_shortcut", k1, 0, 288, 144, 1, 1},
    {"up4 resnets", k3, 0, 144, 144, 1, 5},           {"conv_out", k3, 0, 144, 4, 1, 1},
};

const Shape kEncoder[] = {
    {"conv_in", k3, 0, 4, 96, 1, 1},                  {"down0 resnets", k3, 0, 96, 96, 1, 4},
    {"down0 downsample", kS2, 0, 96, 96, 1, 1},       {"down1 conv1", k3, 0, 96, 192, 2, 1},
    {"down1 conv_shortcut", k1, 0, 96, 192, 2, 1},    {"down1 resnets", k3, 0, 192, 192, 2, 3},
    {"down1 downsample", kS2, 0, 192, 192, 2, 1},     {"down2 conv1", k3, 0, 192, 384, 4, 1},
    {"down2 conv_shortcut", k1, 0, 192, 384, 4, 1},   {"down2 resnets", k3, 0, 384, 384, 4, 3},
    {"down2 downsample", kS2, 0, 384, 384, 4, 1},     {"down3 conv1", k3, 0, 384, 768, 8, 1},
    {"down3 conv_shortcut", k1, 0, 384, 768, 8, 1},   {"down3 resnets", k3, 0, 768, 768, 8, 3},
    {"down3 downsample", kS2, 0, 768, 768, 8, 1},     {"down4 + mid resnets", k3, 0, 768, 768, 16, 8},
    {"mid to_qkv", k1, 0, 768, 2304, 16, 1},          {"mid proj", k1, 0, 768, 768, 16, 1},
    {"conv_out", k3, 0, 768, 128, 16, 1},             {"quant_conv", k1, 0, 128, 128, 16, 1},
};

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
  std::string cases = "conv,attn,shortcut";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  cudaDeviceProp prop{};
  BENCH_CUDA(cudaGetDeviceProperties(&prop, 0));
  if (nvmlInit_v2() == NVML_SUCCESS) nvmlDeviceGetHandleByIndex_v2(0, &g_nvml);
  std::printf("%s, %d SMs; %d timed launches per case\n", prop.name, prop.multiProcessorCount, reps);

  // Buffers for the largest call: 2048² x 288 in (up3's upsample reads 1024² x 288), 2048² x 288 out.
  const std::size_t max_elems = static_cast<std::size_t>(2048) * 2048 * 288;
  __nv_bfloat16 *x = nullptr, *y = nullptr, *w = nullptr, *bias = nullptr;
  BENCH_CUDA(cudaMalloc(&x, max_elems * 2));
  BENCH_CUDA(cudaMalloc(&y, max_elems * 2));
  BENCH_CUDA(cudaMalloc(&w, static_cast<std::size_t>(3456) * 1152 * 9 * 2));
  BENCH_CUDA(cudaMalloc(&bias, 4096 * 2));
  fill_kernel<<<4096, 256>>>(x, max_elems, 1, 2.0f);
  fill_kernel<<<4096, 256>>>(w, static_cast<std::size_t>(3456) * 1152 * 9, 2, 0.02f);
  fill_kernel<<<16, 256>>>(bias, 4096, 3, 0.25f);
  BENCH_CUDA(cudaDeviceSynchronize());

  if (want("conv")) {
    for (const auto &[label, shapes, count] :
         {std::make_tuple("decoder", kDecoder, sizeof kDecoder / sizeof kDecoder[0]),
          std::make_tuple("encoder", kEncoder, sizeof kEncoder / sizeof kEncoder[0])}) {
      for (int size : {1024, 2048}) {
        std::printf("%s convolutions, %dx%d image                    in -> out    ms   TFLOPS   MHz     W\n", label, size,
                    size);
        double total_ms = 0.0, total_flop = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
          const Shape &s = shapes[i];
          const int h = size / s.div, k = s.kind == k1 ? 1 : 3;
          const int out_h = s.kind == kS2 ? h / 2 : h << s.upsample;
          const Timing t = time_it(
              [&] {
                BENCH_RC(ignis_image_vae_conv(s.kind, s.upsample, x, h, h, s.cin, 0, h, w, bias, s.cout, nullptr, y, 0,
                                              out_h, 0, nullptr));
              },
              reps, warmup_ms);
          const double flop = 2.0 * out_h * static_cast<double>(out_h) * s.cout * k * k * s.cin;
          std::printf("  %-22s %4d^2 %5d -> %-5d x%-2d %7.3f  %6.1f  %5u  %4.0f\n", s.name, h, s.cin, s.cout, s.times, t.ms,
                      flop / (t.ms * 1e-3) / 1e12, t.mhz, t.watts);
          total_ms += s.times * t.ms;
          total_flop += s.times * flop;
        }
        std::printf("  the whole %s's convolutions: %.1f ms, %.2f TFLOP, %.1f TFLOPS\n", label, total_ms, total_flop / 1e12,
                    total_flop / (total_ms * 1e-3) / 1e12);
      }
    }
  }

  if (want("attn")) {
    std::printf("mid-block attention                  tokens   row tile  workspace    ms   TFLOPS   MHz     W\n");
    for (int dim : {768, 1152}) {
      for (int tokens : {4096, 16384}) {
        for (int tile : {1024, 2048, 4096}) {
          if (tile > tokens) continue;
          const uint64_t ws_bytes = ignis_image_vae_attn_workspace_bytes(tokens, dim, tile);
          void *ws = nullptr;
          BENCH_CUDA(cudaMalloc(&ws, ws_bytes));
          const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
          const Timing t = time_it(
              [&] { BENCH_RC(ignis_image_vae_attn(x, tokens, dim, scale, tile, ws, ws_bytes, y, 0, nullptr)); }, reps,
              warmup_ms);
          const double flop = 4.0 * tokens * static_cast<double>(tokens) * dim;
          std::printf("  D %-4d                         %8d   %8d  %6.0f MiB  %6.3f  %6.1f  %5u  %4.0f\n", dim, tokens, tile,
                      ws_bytes / 1048576.0, t.ms, flop / (t.ms * 1e-3) / 1e12, t.mhz, t.watts);
          BENCH_CUDA(cudaFree(ws));
        }
      }
    }
  }

  if (want("shortcut")) {
    std::printf("shortcuts (2048² image)                                   ms    GB/s\n");
    struct Sc {
      const char *name;
      uint32_t kind;
      int cin, cout, ft, fs, h;
    };
    for (const Sc &s : {Sc{"dec up3 DupUp 576 -> 288", IGNIS_IMAGE_VAE_DUP_UP, 576, 288, 1, 2, 1024},
                        Sc{"dec up2 DupUp 1152 -> 576", IGNIS_IMAGE_VAE_DUP_UP, 1152, 576, 2, 2, 512},
                        Sc{"enc down0 AvgDown 96 -> 96", IGNIS_IMAGE_VAE_AVG_DOWN, 96, 96, 1, 2, 2048},
                        Sc{"enc down1 AvgDown 96 -> 192", IGNIS_IMAGE_VAE_AVG_DOWN, 96, 192, 2, 2, 1024}}) {
      const int out_h = s.kind == IGNIS_IMAGE_VAE_DUP_UP ? s.h * s.fs : s.h / s.fs;
      const Timing t = time_it(
          [&] { BENCH_RC(ignis_image_vae_shortcut(s.kind, x, s.h, s.h, s.cin, s.cout, s.ft, s.fs, y, 1, 0, nullptr)); },
          reps, warmup_ms);
      const double bytes = 2.0 * (static_cast<double>(s.h) * s.h * s.cin + 2.0 * out_h * static_cast<double>(out_h) * s.cout);
      std::printf("  %-32s accumulate  %7.3f  %6.0f\n", s.name, t.ms, bytes / (t.ms * 1e-3) / 1e9);
    }
  }
  cudaFree(x);
  cudaFree(y);
  cudaFree(w);
  cudaFree(bias);
  return 0;
}
