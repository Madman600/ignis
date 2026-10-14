// The image model's VAE ops (spec image-model/02 acceptance 6, GitHub #319) -- OURS. The ops are
// kernel/include/ignis_image_vae.h; every reference is our own fp64 CPU code, except the
// shortcut fixture's, which holds PyTorch's own outputs (kernel/tests/fixtures/image_vae).
//
//   conv       every VAE convolution shape of spec 01's conversion.json (its vae_convs: the
//              distinct kind / upsample / channel / stage combinations, below), at a 1024² and a
//              2048² image, and five of them at the 2752 x 1536 preset both ways round (H != W).
//              The weights go through ignis_image_vae_pack_conv_weight first (checked against the
//              packing restated here, bit for bit). Each call is checked three ways:
//     mechanics  against an fp64 conv2d of the BF16 operands at sampled output pixels, every
//                output channel, with the bound derived below. The output is fp32 for this check.
//     BF16       the same call with BF16 output must be the fp32 output rounded to BF16, bit for
//                bit, over the whole output (counted on the device).
//     NaN        x sits between rows of NaN and the outputs start as NaN: a read outside the
//                image or an unwritten output shows as a NaN, counted over the whole output.
//   bands      a 3 x 3 call, the upsampling one and the stride-2 one computed as three uneven
//              bands, each from its own copy of the rows it needs (between NaN rows), equal the
//              whole-image call bit for bit; a window missing a halo row is refused.
//   shortcuts  AvgDown and DupUp: the fixture's PyTorch outputs, bit for bit in fp32 and within
//              one BF16 rounding in BF16 (the restatement here is checked against the fixture
//              too, and the upsample's nearest-exact restatement); then the ops at every real
//              configuration and stage size against the restatement, bit for bit, and the
//              accumulating form against y + RN(shortcut).
//   attn       the mid-block attention at D = 768 and 1152 over 64 x 64 and 128 x 128 latents,
//              the 172 x 96 preset and small or odd token counts: against fp64 attention that
//              replays the kernel's softmax (bound below), against plain fp64 attention (relative
//              RMS error <= --max-rel-rms, default 2^-8), BF16 == RN(fp32), and two row tiles
//              (one leaving a partial last tile) bit for bit. qkv sits before rows of NaN, and the
//              workspace and the outputs start as NaN, so the padding the kernels must write as
//              zeros is checked too.
//   refusals   each op's argument checks.
//
// The conv bound (bench_mma_peak's BF16 probes, P4 = 25: each of an HMMA's 16 products and its
// accumulator loses at most 2^-25 of the largest addend, and its result is one fp32 rounding).
// Walking the reduction in the kernel's order (taps, then input channels, in k16 steps), with
// S_i the exact partial sum after step i,
//   E_i = E_{i-1} + 19 * 2^-25 * (|S_{i-1}| + E_{i-1} + sum_{step i} |x w|)
// then the bias add and the residual add each add one fp32 rounding of the running value; the
// whole bound times --bound-factor F (default 1).
//
// The attention bound replays the kernel per key in k16 steps, as the encoder attention's test
// does: s_j = q . k_j exact in fp64 with the S GEMM's error e_j (the conv bound over d); the
// kernel's max within em = max e_j of the replay's; the exponent within dx_j of
// x_j = (s_j - m) c; P'_j between the BF16 values of 2^(x_j -+ dx_j)(1 -+ 2^-20) (ex2.approx);
// PV per k16 step as above plus the width of each P interval times |v|; the row sum of the BF16
// P with (n8 / 256 + 12) fp32 roundings (each thread's columns, the warp butterfly, the 8 warps);
// then 1 / D and the product, one rounding each, and the quotient
// |dO| <= (|dN| + |O| |dD|) / (D - |dD|).
//
//   ignis_kernel_image_vae_test [--bound-factor F] [--max-rel-rms R]
//                               [--cases conv1024,conv2048,nonsquare,bands,shortcuts,attn,refusals]
//
// ADR 0006 / docs/agents/testing.md: no SKIP_RETURN_CODE; a missing GPU fails.

#include "ignis_image_vae.h"
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

constexpr double kMma = 19.0 * 0x1p-25;  // per k16 step: 16 products + the accumulator, + one rounding
constexpr double kExp2Rel = 0x1p-20;     // ex2.approx.f32, 2 ulp (PTX ISA) with margin
constexpr int kPoisonRows = 2;           // NaN rows around a conv input
constexpr int kAttnPoisonRows = 8;       // NaN rows after qkv

struct Options {
  double bound_factor = 1.0;
  double max_rel_rms = 0x1p-8;
};

#define VAE_RC(expr)                                                                               \
  do {                                                                                             \
    const int32_t rc_ = (expr);                                                                    \
    if (rc_ != 0) {                                                                                \
      std::fprintf(stderr, "FATAL: %s returned %d: %s\n", #expr, rc_, ignis_image_vae_last_error()); \
      std::exit(EXIT_FAILURE);                                                                     \
    }                                                                                              \
  } while (0)

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

// ---- device helpers ---------------------------------------------------------------------------

__host__ __device__ inline uint32_t mix32(uint32_t x) {  // moe_fixture.h's lowbias32
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// A BF16 value in [-amplitude, amplitude): moe_fixture.h's hash_uniform, rounded to BF16, on the
// host and on the device alike (every step is exact or one fp32 rounding to nearest even).
__host__ __device__ inline uint16_t hash_bf16(uint32_t stream, uint64_t i, float amplitude) {
  const uint32_t h = mix32(static_cast<uint32_t>(i) * 0x9E3779B9u + stream * 0x85EBCA6Bu);
  const float v = (static_cast<float>(h >> 8) * 0x1p-23f - 1.0f) * amplitude;
#ifdef __CUDA_ARCH__
  uint32_t b = __float_as_uint(v);
#else
  uint32_t b;
  std::memcpy(&b, &v, 4);
#endif
  b += 0x7FFFu + ((b >> 16) & 1u);
  return static_cast<uint16_t>(b >> 16);
}

__global__ void fill_hash_kernel(uint16_t *p, std::size_t n, uint32_t stream, uint64_t first, float amplitude) {
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    p[e] = hash_bf16(stream, first + e, amplitude);
  }
}

void fill_hash(void *p, std::size_t n, uint32_t stream, uint64_t first, float amplitude) {
  fill_hash_kernel<<<2048, 256>>>(static_cast<uint16_t *>(p), n, stream, first, amplitude);
  MOE_CUDA(cudaGetLastError());
}

__device__ unsigned long long g_count;

// Counts outputs whose BF16 is not the fp32 rounded (or either is NaN), and fp32 NaNs.
__global__ void bf16_vs_f32_kernel(const float *y32, const uint16_t *y16, std::size_t n, unsigned long long *nan32) {
  unsigned long long diff = 0, nan = 0;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const float v = y32[e];
    const uint16_t r = __bfloat16_as_ushort(__float2bfloat16_rn(v));
    nan += v != v;
    diff += v != v || r != y16[e];
  }
  if (diff != 0) atomicAdd(&g_count, diff);
  if (nan != 0) atomicAdd(nan32, nan);
}

// Counts differing 16-bit words.
__global__ void diff_u16_kernel(const uint16_t *a, const uint16_t *b, std::size_t n) {
  unsigned long long diff = 0;
  for (std::size_t e = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; e < n;
       e += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    diff += a[e] != b[e];
  }
  if (diff != 0) atomicAdd(&g_count, diff);
}

struct BfCheck {
  unsigned long long mismatched = 0, nan32 = 0;
};

BfCheck bf16_vs_f32(const void *y32, const void *y16, std::size_t n) {
  DeviceBytes d_nan(8);
  const unsigned long long zero = 0;
  MOE_CUDA(cudaMemcpyToSymbol(g_count, &zero, 8));
  MOE_CUDA(cudaMemcpy(d_nan.p, &zero, 8, cudaMemcpyHostToDevice));
  bf16_vs_f32_kernel<<<2048, 256>>>(static_cast<const float *>(y32), static_cast<const uint16_t *>(y16), n,
                                    d_nan.as<unsigned long long>());
  MOE_CUDA(cudaDeviceSynchronize());
  BfCheck r;
  MOE_CUDA(cudaMemcpyFromSymbol(&r.mismatched, g_count, 8));
  MOE_CUDA(cudaMemcpy(&r.nan32, d_nan.p, 8, cudaMemcpyDeviceToHost));
  return r;
}

unsigned long long diff_u16(const void *a, const void *b, std::size_t n) {
  const unsigned long long zero = 0;
  MOE_CUDA(cudaMemcpyToSymbol(g_count, &zero, 8));
  diff_u16_kernel<<<2048, 256>>>(static_cast<const uint16_t *>(a), static_cast<const uint16_t *>(b), n);
  MOE_CUDA(cudaDeviceSynchronize());
  unsigned long long r = 0;
  MOE_CUDA(cudaMemcpyFromSymbol(&r, g_count, 8));
  return r;
}

// The smallest odd stride >= (count - 1) / (want - 1), rounded up, coprime with count: i * stride
// mod count for i < want are distinct and reach the end of the range.
long long span_stride(long long count, int want) {
  long long stride = std::max(1LL, (count - 1 + want - 2) / (want - 1));
  if (stride % 2 == 0) ++stride;
  while (std::gcd(stride, count) != 1) stride += 2;
  return stride;
}

double arg_double(int argc, char **argv, const char *name, double fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return std::atof(argv[i + 1]);
  }
  return fallback;
}

// ---- the convolution --------------------------------------------------------------------------

// A distinct VAE convolution (conversion.json's vae_convs, deduplicated): its kind, upsample,
// channels and the stage it runs at, as image size / div (the input's resolution). `residual`
// marks the ones whose output a residual joins in the VAE (the fused add is tested there).
// time_conv: the 1 x 1 temporal convolutions, which never run for one frame (diffusers' resample
// skips them on the first chunk); covered as shapes all the same.
struct ConvShape {
  const char *name;
  uint32_t kind, upsample;
  int cin, cout, div;
  bool residual;
};

constexpr uint32_t k3 = IGNIS_IMAGE_VAE_CONV_3X3, kS2 = IGNIS_IMAGE_VAE_CONV_3X3_S2, k1 = IGNIS_IMAGE_VAE_CONV_1X1;

const ConvShape kShapes[] = {
    // encoder (base 96, mults 1 2 4 8 8)
    {"enc conv_in", k3, 0, 4, 96, 1, false},
    {"enc down0 resnet", k3, 0, 96, 96, 1, true},
    {"enc down0 downsample", kS2, 0, 96, 96, 1, false},
    {"enc down1 conv1", k3, 0, 96, 192, 2, false},
    {"enc down1 resnet", k3, 0, 192, 192, 2, true},
    {"enc down1 conv_shortcut", k1, 0, 96, 192, 2, false},
    {"enc down1 downsample", kS2, 0, 192, 192, 2, false},
    {"enc down1 time_conv", k1, 0, 192, 192, 4, false},
    {"enc down2 conv1", k3, 0, 192, 384, 4, false},
    {"enc down2 resnet", k3, 0, 384, 384, 4, true},
    {"enc down2 conv_shortcut", k1, 0, 192, 384, 4, false},
    {"enc down2 downsample", kS2, 0, 384, 384, 4, false},
    {"enc down2 time_conv", k1, 0, 384, 384, 8, false},
    {"enc down3 conv1", k3, 0, 384, 768, 8, false},
    {"enc down3 resnet", k3, 0, 768, 768, 8, true},
    {"enc down3 conv_shortcut", k1, 0, 384, 768, 8, false},
    {"enc down3 downsample", kS2, 0, 768, 768, 8, false},
    {"enc down3 time_conv", k1, 0, 768, 768, 16, false},
    {"enc down4/mid resnet", k3, 0, 768, 768, 16, true},
    {"enc mid to_qkv", k1, 0, 768, 2304, 16, false},
    {"enc mid proj", k1, 0, 768, 768, 16, true},
    {"enc conv_out", k3, 0, 768, 128, 16, false},
    {"quant_conv", k1, 0, 128, 128, 16, false},
    // decoder (base 144)
    {"post_quant_conv", k1, 0, 64, 64, 16, false},
    {"dec conv_in", k3, 0, 64, 1152, 16, false},
    {"dec mid to_qkv", k1, 0, 1152, 3456, 16, false},
    {"dec mid proj", k1, 0, 1152, 1152, 16, true},
    {"dec mid/up0 resnet", k3, 0, 1152, 1152, 16, true},
    {"dec up0 upsample", k3, 1, 1152, 1152, 16, false},
    {"dec up0 time_conv", k1, 0, 1152, 2304, 16, false},
    {"dec up1 resnet", k3, 0, 1152, 1152, 8, true},
    {"dec up1 upsample", k3, 1, 1152, 1152, 8, false},
    {"dec up1 time_conv", k1, 0, 1152, 2304, 8, false},
    {"dec up2 conv1", k3, 0, 1152, 576, 4, false},
    {"dec up2 conv_shortcut", k1, 0, 1152, 576, 4, false},
    {"dec up2 resnet", k3, 0, 576, 576, 4, true},
    {"dec up2 upsample", k3, 1, 576, 576, 4, false},
    {"dec up2 time_conv", k1, 0, 576, 1152, 4, false},
    {"dec up3 conv1", k3, 0, 576, 288, 2, false},
    {"dec up3 conv_shortcut", k1, 0, 576, 288, 2, false},
    {"dec up3 resnet", k3, 0, 288, 288, 2, true},
    {"dec up3 upsample", k3, 1, 288, 288, 2, false},
    {"dec up4 conv1", k3, 0, 288, 144, 1, false},
    {"dec up4 conv_shortcut", k1, 0, 288, 144, 1, false},
    {"dec up4 resnet", k3, 0, 144, 144, 1, true},
    {"dec conv_out", k3, 0, 144, 4, 1, false},
};

int ksize_of(uint32_t kind) { return kind == k1 ? 1 : 3; }
int stride_of(uint32_t kind) { return kind == kS2 ? 2 : 1; }
int pad_of(uint32_t kind) { return kind == k3 ? 1 : 0; }

// One convolution's operands: x by the hash (stream 11, NHWC index), the weights and bias on the
// host, the residual by the hash (stream 13).
struct ConvCase {
  ConvShape shape;
  int h = 0, w = 0, out_h = 0, out_w = 0, taps = 0, k_total = 0;
  std::vector<uint16_t> w_oihw, w_packed, bias;
  uint16_t x_at(int iy, int ix, int ci) const {
    return hash_bf16(11, (static_cast<uint64_t>(iy) * w + ix) * shape.cin + ci, 2.0f);
  }
  uint16_t res_at(long long m, int n) const {
    return hash_bf16(13, static_cast<uint64_t>(m) * shape.cout + n, 4.0f);
  }
};

ConvCase make_case(const ConvShape &s, int image_h, int image_w) {
  ConvCase c;
  c.shape = s;
  c.h = image_h / s.div;
  c.w = image_w / s.div;
  const int st = stride_of(s.kind);
  c.out_h = st == 2 ? c.h / 2 : c.h << s.upsample;
  c.out_w = st == 2 ? c.w / 2 : c.w << s.upsample;
  const int k = ksize_of(s.kind);
  c.taps = k * k;
  c.k_total = c.taps * s.cin;
  // Weights at about 1 / sqrt(K) (an output of order 1), bias small.
  const float amp = 1.7f / std::sqrt(static_cast<float>(c.k_total));
  c.w_oihw.resize(static_cast<std::size_t>(s.cout) * c.k_total);
  for (std::size_t i = 0; i < c.w_oihw.size(); ++i) c.w_oihw[i] = hash_bf16(12, i, amp);
  c.w_packed.resize(c.w_oihw.size());
  for (int o = 0; o < s.cout; ++o) {
    for (int ci = 0; ci < s.cin; ++ci) {
      for (int t = 0; t < c.taps; ++t) {
        c.w_packed[(static_cast<std::size_t>(o) * c.taps + t) * s.cin + ci] =
            c.w_oihw[(static_cast<std::size_t>(o) * s.cin + ci) * c.taps + t];
      }
    }
  }
  c.bias.resize(s.cout);
  for (int o = 0; o < s.cout; ++o) c.bias[o] = hash_bf16(14, o, 0.25f);
  return c;
}

// The reference for output pixel (oy, ox), every output channel: the exact sum in the kernel's
// order with its bound, then the bias and residual adds.
void conv_ref_pixel(const ConvCase &c, int oy, int ox, bool residual, double F, std::vector<double> &out,
                    std::vector<double> &bound) {
  const ConvShape &s = c.shape;
  const int k = ksize_of(s.kind), st = stride_of(s.kind), pad = pad_of(s.kind);
  const int hl = c.h << s.upsample, wl = c.w << s.upsample;
  std::vector<float> a(c.k_total, 0.0f);
  for (int t = 0; t < c.taps; ++t) {
    const int iy = oy * st - pad + t / k, ix = ox * st - pad + t % k;
    if (iy < 0 || iy >= hl || ix < 0 || ix >= wl) continue;
    for (int ci = 0; ci < s.cin; ++ci) a[t * s.cin + ci] = bf16_to_f32(c.x_at(iy >> s.upsample, ix >> s.upsample, ci));
  }
  const long long m = static_cast<long long>(oy) * c.out_w + ox;
  out.assign(s.cout, 0.0);
  bound.assign(s.cout, 0.0);
  std::vector<float> wr(c.k_total);
  for (int n = 0; n < s.cout; ++n) {
    for (int kk = 0; kk < c.k_total; ++kk) wr[kk] = bf16_to_f32(c.w_packed[static_cast<std::size_t>(n) * c.k_total + kk]);
    double acc = 0.0, err = 0.0;
    for (int s0 = 0; s0 < c.k_total; s0 += 16) {
      double step = 0.0, step_abs = 0.0;
      for (int kk = s0; kk < std::min(c.k_total, s0 + 16); ++kk) {
        const double p = static_cast<double>(a[kk]) * wr[kk];
        step += p;
        step_abs += std::fabs(p);
      }
      err += kMma * (std::fabs(acc) + err + step_abs);
      acc += step;
    }
    double v = acc, e = err + 0x1p-48 * std::fabs(acc);
    v += bf16_to_f32(c.bias[n]);
    e += (std::fabs(v) + e) * 0x1p-24;
    if (residual) {
      v += bf16_to_f32(c.res_at(m, n));
      e += (std::fabs(v) + e) * 0x1p-24;
    }
    out[n] = v;
    bound[n] = F * e + 1e-30;
  }
}

// The sampled output pixels: the corners and edge midpoints, then i * stride mod pixels for i <
// 128 (every residue of a 128-pixel tile when 128 divides the count), plus the last pixel.
std::vector<long long> sample_pixels(int out_h, int out_w) {
  const long long total = static_cast<long long>(out_h) * out_w;
  std::vector<long long> px;
  auto add = [&](long long m) {
    if (m >= 0 && m < total && std::find(px.begin(), px.end(), m) == px.end()) px.push_back(m);
  };
  for (int y : {0, out_h / 2, out_h - 1}) {
    for (int x : {0, out_w / 2, out_w - 1}) add(static_cast<long long>(y) * out_w + x);
  }
  const int want = 128;
  if (total <= want) {
    for (long long m = 0; m < total; ++m) add(m);
  } else {
    const long long stride = span_stride(total, want);
    for (int i = 0; i < want; ++i) add(i * stride % total);
  }
  add(total - 1);
  return px;
}

// Device buffers of one call: x between kPoisonRows of NaN, the packed weight with a NaN row
// after it, bias, the residual, and both outputs (NaN until written).
struct ConvDevice {
  DeviceBytes x, w_raw, w, bias, res, y32, y16;
  const void *x_at = nullptr;
  ConvDevice(const ConvCase &c, bool residual)
      : x(static_cast<std::size_t>(c.h + 2 * kPoisonRows) * c.w * c.shape.cin * 2),
        w_raw(c.w_oihw.size() * 2), w((c.w_packed.size() + c.k_total) * 2), bias(c.bias.size() * 2),
        res(residual ? static_cast<std::size_t>(c.out_h) * c.out_w * c.shape.cout * 2 : 0),
        y32(static_cast<std::size_t>(c.out_h) * c.out_w * c.shape.cout * 4),
        y16(static_cast<std::size_t>(c.out_h) * c.out_w * c.shape.cout * 2) {
    const std::size_t row = static_cast<std::size_t>(c.w) * c.shape.cin;
    MOE_CUDA(cudaMemset(x.p, 0xFF, x.bytes));
    x_at = static_cast<uint16_t *>(x.p) + kPoisonRows * row;
    fill_hash(const_cast<void *>(x_at), static_cast<std::size_t>(c.h) * row, 11, 0, 2.0f);
    upload(w_raw, c.w_oihw);
    MOE_CUDA(cudaMemset(w.p, 0xFF, w.bytes));
    upload(bias, c.bias);
    if (residual) fill_hash(res.p, res.bytes / 2, 13, 0, 4.0f);
    MOE_CUDA(cudaMemset(y32.p, 0xFF, y32.bytes));
    MOE_CUDA(cudaMemset(y16.p, 0xFF, y16.bytes));
  }
};

void run_conv(const Options &o, const ConvShape &s, int image_h, int image_w) {
  const ConvCase c = make_case(s, image_h, image_w);
  const bool residual = s.residual;
  ConvDevice d(c, residual);
  const std::string what = std::string(s.name) + " " + std::to_string(c.shape.cin) + "->" +
                           std::to_string(c.shape.cout) + " at " + std::to_string(c.h) + "x" + std::to_string(c.w);
  VAE_RC(ignis_image_vae_pack_conv_weight(d.w_raw.p, s.cout, s.cin, ksize_of(s.kind), d.w.p, nullptr));
  auto conv = [&](void *y, uint32_t f32) {
    return ignis_image_vae_conv(s.kind, s.upsample, d.x_at, c.h, c.w, s.cin, 0, c.h, d.w.p, d.bias.p, s.cout,
                                residual ? d.res.p : nullptr, y, 0, c.out_h, f32, nullptr);
  };
  if (conv(d.y32.p, 1) != 0 || conv(d.y16.p, 0) != 0) {
    std::fprintf(stderr, "FATAL: ignis_image_vae_conv (%s): %s\n", what.c_str(), ignis_image_vae_last_error());
    std::exit(EXIT_FAILURE);
  }
  MOE_CUDA(cudaDeviceSynchronize());
  const auto packed = download<uint16_t>(d.w.p, c.w_packed.size());
  const bool pack_ok = std::memcmp(packed.data(), c.w_packed.data(), packed.size() * 2) == 0;
  const std::size_t outputs = static_cast<std::size_t>(c.out_h) * c.out_w * s.cout;
  const BfCheck bc = bf16_vs_f32(d.y32.p, d.y16.p, outputs);

  const std::vector<long long> px = sample_pixels(c.out_h, c.out_w);
  std::vector<int> over(px.size(), 0);
  std::vector<double> worst(px.size(), 0.0);
  std::vector<std::vector<double>> tight(px.size());
  parallel_for(static_cast<int>(px.size()), [&](int i) {
    const long long m = px[i];
    std::vector<double> ref, bound;
    conv_ref_pixel(c, static_cast<int>(m / c.out_w), static_cast<int>(m % c.out_w), residual, o.bound_factor, ref, bound);
    std::vector<float> got(s.cout);
    MOE_CUDA(cudaMemcpy(got.data(), static_cast<const float *>(d.y32.p) + static_cast<std::size_t>(m) * s.cout,
                        s.cout * 4, cudaMemcpyDeviceToHost));
    for (int n = 0; n < s.cout; ++n) {
      const double err = std::fabs(got[n] - ref[n]);
      if (!(err <= bound[n])) ++over[i];
      worst[i] = std::isnan(err) ? 1e300 : std::max(worst[i], err / bound[n]);
      if (ref[n] != 0.0) tight[i].push_back(bound[n] / std::fabs(ref[n]));
    }
  });
  int total_over = 0;
  double worst_all = 0.0;
  std::vector<double> all;
  for (std::size_t i = 0; i < px.size(); ++i) {
    total_over += over[i];
    worst_all = std::max(worst_all, worst[i]);
    all.insert(all.end(), tight[i].begin(), tight[i].end());
  }
  std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());
  std::printf("  %-52s %zu px: worst %.3f of bound (bound/|y| median %.1e), %d over; BF16 = RN(fp32) %s; %llu NaN\n",
              what.c_str(), px.size(), worst_all, all[all.size() / 2], total_over,
              bc.mismatched == 0 ? "bit for bit" : "NO", bc.nan32);
  check(pack_ok, what + ": the packed weight is the restated packing, bit for bit");
  check(total_over == 0, what + ": within the mechanics bound of fp64");
  check(bc.mismatched == 0, what + ": the BF16 output is the fp32 output rounded, bit for bit");
  check(bc.nan32 == 0, what + ": no NaN output (nothing read outside the image, every output written)");
}

// The rows [lo, hi) of the input that output rows [y0, y1) read, restated here (not the leaf's
// helper): the taps' rows in the (upsampled) grid, clamped, halved back when upsampling.
void band_input_rows(const ConvShape &s, int h, int y0, int y1, int &lo, int &hi) {
  const int k = ksize_of(s.kind), st = stride_of(s.kind), pad = pad_of(s.kind), hl = h << s.upsample;
  const int first = std::max(0, y0 * st - pad), last = std::min(hl - 1, (y1 - 1) * st - pad + k - 1);
  lo = first >> s.upsample;
  hi = (last >> s.upsample) + 1;
}

void run_bands(const ConvShape &s, int image_h, int image_w) {
  const ConvCase c = make_case(s, image_h, image_w);
  ConvDevice d(c, s.residual);
  const std::string what = std::string(s.name) + " at " + std::to_string(c.h) + "x" + std::to_string(c.w);
  VAE_RC(ignis_image_vae_pack_conv_weight(d.w_raw.p, s.cout, s.cin, ksize_of(s.kind), d.w.p, nullptr));
  VAE_RC(ignis_image_vae_conv(s.kind, s.upsample, d.x_at, c.h, c.w, s.cin, 0, c.h, d.w.p, d.bias.p, s.cout,
                              s.residual ? d.res.p : nullptr, d.y16.p, 0, c.out_h, 0, nullptr));
  if (s.residual) {
    // residual == y (BF16 output): the in-place add equals the out-of-place one, bit for bit.
    DeviceBytes inplace(d.res.bytes);
    MOE_CUDA(cudaMemcpy(inplace.p, d.res.p, d.res.bytes, cudaMemcpyDeviceToDevice));
    VAE_RC(ignis_image_vae_conv(s.kind, s.upsample, d.x_at, c.h, c.w, s.cin, 0, c.h, d.w.p, d.bias.p, s.cout, inplace.p,
                                inplace.p, 0, c.out_h, 0, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    const unsigned long long dd = diff_u16(d.y16.p, inplace.p, static_cast<std::size_t>(c.out_h) * c.out_w * s.cout);
    check(dd == 0, what + ": the in-place residual (residual == y) equals the out-of-place add, bit for bit");
  }
  // Three uneven bands of output rows, each from a copy of only the input rows it reads.
  const int cuts[4] = {0, c.out_h / 3 + 1, 2 * c.out_h / 3 - 3, c.out_h};
  const std::size_t in_row = static_cast<std::size_t>(c.w) * s.cin, out_row = static_cast<std::size_t>(c.out_w) * s.cout;
  DeviceBytes banded(d.y16.bytes);
  MOE_CUDA(cudaMemset(banded.p, 0xFF, banded.bytes));
  int refused = 0;
  for (int b = 0; b < 3; ++b) {
    int lo = 0, hi = 0;
    band_input_rows(s, c.h, cuts[b], cuts[b + 1], lo, hi);
    DeviceBytes xb(static_cast<std::size_t>(hi - lo + 2 * kPoisonRows) * in_row * 2);
    MOE_CUDA(cudaMemset(xb.p, 0xFF, xb.bytes));
    uint16_t *xb_at = static_cast<uint16_t *>(xb.p) + kPoisonRows * in_row;
    MOE_CUDA(cudaMemcpy(xb_at, static_cast<const uint16_t *>(d.x_at) + lo * in_row, (hi - lo) * in_row * 2,
                        cudaMemcpyDeviceToDevice));
    const uint16_t *res = s.residual ? static_cast<const uint16_t *>(d.res.p) + cuts[b] * out_row : nullptr;
    uint16_t *yb = static_cast<uint16_t *>(banded.p) + cuts[b] * out_row;
    VAE_RC(ignis_image_vae_conv(s.kind, s.upsample, xb_at, c.h, c.w, s.cin, lo, hi - lo, d.w.p, d.bias.p, s.cout, res,
                                yb, cuts[b], cuts[b + 1] - cuts[b], 0, nullptr));
    // One row short of the halo (when the band has one to lose) is refused.
    if (b > 0 && lo > 0) {
      refused += ignis_image_vae_conv(s.kind, s.upsample, xb_at + in_row, c.h, c.w, s.cin, lo + 1, hi - lo - 1, d.w.p,
                                      d.bias.p, s.cout, res, yb, cuts[b], cuts[b + 1] - cuts[b], 0, nullptr) != 0;
    }
  }
  MOE_CUDA(cudaDeviceSynchronize());
  const unsigned long long diff = diff_u16(d.y16.p, banded.p, static_cast<std::size_t>(c.out_h) * out_row);
  std::printf("  %-44s bands [%d, %d, %d, %d): %llu outputs differ from the whole-image call; a short halo refused %d/2\n",
              what.c_str(), cuts[0], cuts[1], cuts[2], cuts[3], diff, refused);
  check(diff == 0, what + ": the banded call equals the whole-image call, bit for bit");
  check(refused == 2, what + ": a window missing a halo row is refused");
}

// The upsample: the fused gather against the restated nearest-exact 2x (x[y >> 1][x >> 1],
// proven against PyTorch in the shortcut fixture) materialized and convolved without it.
void run_upsample_vs_materialized(const ConvShape &s, int image_h, int image_w) {
  const ConvCase c = make_case(s, image_h, image_w);
  ConvDevice d(c, false);
  VAE_RC(ignis_image_vae_pack_conv_weight(d.w_raw.p, s.cout, s.cin, 3, d.w.p, nullptr));
  VAE_RC(ignis_image_vae_conv(k3, 1, d.x_at, c.h, c.w, s.cin, 0, c.h, d.w.p, d.bias.p, s.cout, nullptr, d.y32.p, 0,
                              c.out_h, 1, nullptr));
  const auto x = download<uint16_t>(d.x_at, static_cast<std::size_t>(c.h) * c.w * s.cin);
  std::vector<uint16_t> up(static_cast<std::size_t>(c.out_h) * c.out_w * s.cin);
  for (int y = 0; y < c.out_h; ++y) {
    for (int xx = 0; xx < c.out_w; ++xx) {
      std::memcpy(&up[(static_cast<std::size_t>(y) * c.out_w + xx) * s.cin],
                  &x[(static_cast<std::size_t>(y >> 1) * c.w + (xx >> 1)) * s.cin], s.cin * 2);
    }
  }
  DeviceBytes d_up(up.size() * 2), y2(d.y32.bytes);
  upload(d_up, up);
  VAE_RC(ignis_image_vae_conv(k3, 0, d_up.p, c.out_h, c.out_w, s.cin, 0, c.out_h, d.w.p, d.bias.p, s.cout, nullptr,
                              y2.p, 0, c.out_h, 1, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  const unsigned long long diff = diff_u16(d.y32.p, y2.p, d.y32.bytes / 2);
  std::printf("  %-44s fused upsample vs materialized: %llu half-words differ\n", s.name, diff);
  check(diff == 0, std::string(s.name) + ": the fused upsample equals the materialized nearest-exact 2x, bit for bit");
}

// ---- the shortcuts ----------------------------------------------------------------------------

struct ShortcutGeo {
  int kind, cin, cout, ft, fs, h, w;
  int out_h() const { return kind == 0 ? h / fs : h * fs; }
  int out_w() const { return kind == 0 ? w / fs : w * fs; }
};

// The restated definitions (the header's), on NHWC floats; fp32 arithmetic as the op's.
std::vector<float> shortcut_ref(const ShortcutGeo &g, const std::vector<float> &x) {
  const int oh = g.out_h(), ow = g.out_w(), factor = g.ft * g.fs * g.fs;
  std::vector<float> y(static_cast<std::size_t>(oh) * ow * g.cout);
  for (int Y = 0; Y < oh; ++Y) {
    for (int X = 0; X < ow; ++X) {
      for (int o = 0; o < g.cout; ++o) {
        float v;
        if (g.kind == 0) {
          const int group = g.cin * factor / g.cout;
          float sum = 0.0f;
          for (int gg = 0; gg < group; ++gg) {
            const int j = o * group + gg, c = j / factor, rem = j % factor;
            const int t = rem / (g.fs * g.fs), sh = (rem % (g.fs * g.fs)) / g.fs, sw = rem % g.fs;
            sum += t == g.ft - 1 ? x[(static_cast<std::size_t>(Y * g.fs + sh) * g.w + X * g.fs + sw) * g.cin + c] : 0.0f;
          }
          v = sum / static_cast<float>(group);
        } else {
          const int repeats = g.cout * factor / g.cin, sh = Y % g.fs, sw = X % g.fs;
          const int c = (o * factor + (g.ft - 1) * g.fs * g.fs + sh * g.fs + sw) / repeats;
          v = x[(static_cast<std::size_t>(Y / g.fs) * g.w + X / g.fs) * g.cin + c];
        }
        y[(static_cast<std::size_t>(Y) * ow + X) * g.cout + o] = v;
      }
    }
  }
  return y;
}

uint32_t bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}

// RN to BF16 that keeps NaN, signed zeros and subnormals (f32_to_bf16's carry form is exact for
// finite values, subnormals included).
uint16_t rn_bf16(float v) { return f32_to_bf16(v); }

void run_shortcut_fixture() {
#ifndef IGNIS_IMAGE_VAE_FIXTURE_PATH
#error "IGNIS_IMAGE_VAE_FIXTURE_PATH must name kernel/tests/fixtures/image_vae/shortcuts.bin"
#endif
  const auto fx = read_fixture(IGNIS_IMAGE_VAE_FIXTURE_PATH);
  std::vector<std::string> names;
  for (const auto &kv : fx) {
    const std::string &n = kv.first;
    if (n.size() > 9 && n.compare(n.size() - 9, 9, ".geometry") == 0) names.push_back(n.substr(0, n.size() - 9));
  }
  check(names.size() == 9, "the fixture holds the 9 cases");
  for (const std::string &name : names) {
    const auto geo = need(fx, name + ".geometry").as<int32_t>();
    const ShortcutGeo g{geo[0], geo[1], geo[2], geo[3], geo[4], geo[5], geo[6]};
    const auto x = need(fx, name + ".x").as<float>();
    const auto y = need(fx, name + ".y").as<float>();
    const auto y16 = need(fx, name + ".y_bf16").as<uint16_t>();
    std::vector<float> x16(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) x16[i] = bf16_to_f32(rn_bf16(x[i]));
    if (g.kind == 2) {
      // The upsample: the restated nearest-exact 2x is PyTorch's, bit for bit.
      long long diff = 0;
      for (int Y = 0; Y < 2 * g.h; ++Y) {
        for (int X = 0; X < 2 * g.w; ++X) {
          for (int c = 0; c < g.cin; ++c) {
            const std::size_t at = (static_cast<std::size_t>(Y) * 2 * g.w + X) * g.cin + c;
            const std::size_t from = (static_cast<std::size_t>(Y >> 1) * g.w + (X >> 1)) * g.cin + c;
            diff += bits(y[at]) != bits(x[from]) || y16[at] != rn_bf16(x16[from]);
          }
        }
      }
      std::printf("  %-22s restated nearest-exact 2x vs PyTorch: %lld differ\n", name.c_str(), diff);
      check(diff == 0, name + ": x[y >> 1][x >> 1] is PyTorch's nearest-exact 2x, bit for bit");
      continue;
    }
    // The restatement against PyTorch, then the op against PyTorch.
    const auto ref = shortcut_ref(g, x), ref16 = shortcut_ref(g, x16);
    long long restated = 0;
    for (std::size_t i = 0; i < y.size(); ++i) restated += bits(ref[i]) != bits(y[i]) || rn_bf16(ref16[i]) != y16[i];
    DeviceBytes dx(x.size() * 4), dy(y.size() * 4), dx16(x.size() * 2), dy16(y.size() * 2);
    upload(dx, x);
    std::vector<uint16_t> xb(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xb[i] = rn_bf16(x[i]);
    upload(dx16, xb);
    MOE_CUDA(cudaMemset(dy.p, 0xFF, dy.bytes));
    MOE_CUDA(cudaMemset(dy16.p, 0xFF, dy16.bytes));
    VAE_RC(ignis_image_vae_shortcut(g.kind, dx.p, g.h, g.w, g.cin, g.cout, g.ft, g.fs, dy.p, 0, 1, nullptr));
    VAE_RC(ignis_image_vae_shortcut(g.kind, dx16.p, g.h, g.w, g.cin, g.cout, g.ft, g.fs, dy16.p, 0, 0, nullptr));
    MOE_CUDA(cudaDeviceSynchronize());
    const auto got = download<float>(dy.p, y.size());
    const auto got16 = download<uint16_t>(dy16.p, y.size());
    long long f32_diff = 0, bf16_diff = 0, bf16_far = 0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      f32_diff += bits(got[i]) != bits(y[i]);
      bf16_diff += got16[i] != y16[i];
      bf16_far += std::abs(static_cast<int>(got16[i]) - static_cast<int>(y16[i])) > 1;
    }
    std::printf("  %-22s restated vs PyTorch %lld differ; op vs PyTorch: fp32 %lld differ, BF16 %lld differ (%lld by more "
                "than one rounding), of %zu\n",
                name.c_str(), restated, f32_diff, bf16_diff, bf16_far, y.size());
    check(restated == 0, name + ": the restated definition is PyTorch's (fp32 bit for bit, BF16 its rounding)");
    check(f32_diff == 0, name + ": fp32 output is PyTorch's, bit for bit");
    check(bf16_far == 0, name + ": BF16 output within one BF16 rounding of PyTorch's");
    check(bf16_diff == 0, name + ": BF16 output is PyTorch's, bit for bit");
  }
}

// The real configurations at a stage size: the op against the restatement, bit for bit, on six
// sampled output rows (whole rows: every channel and column), each restated from only the input
// rows it reads; then the accumulating form against RN(y + RN(shortcut)).
void run_shortcut_real(const ShortcutGeo &g, const char *name) {
  const std::size_t in_row = static_cast<std::size_t>(g.w) * g.cin, in_n = in_row * g.h;
  const std::size_t out_row = static_cast<std::size_t>(g.out_w()) * g.cout, out_n = out_row * g.out_h();
  DeviceBytes dx(in_n * 2), dy(out_n * 2), dacc(out_n * 2);
  fill_hash(dx.p, in_n, 21, 0, 3.0f);
  fill_hash(dacc.p, out_n, 22, 0, 3.0f);
  MOE_CUDA(cudaMemset(dy.p, 0xFF, dy.bytes));
  std::vector<int> sampled;
  for (int Y : {0, 1, g.out_h() / 2, g.out_h() / 2 + 1, g.out_h() - 2, g.out_h() - 1}) {
    if (Y >= 0 && Y < g.out_h() && std::find(sampled.begin(), sampled.end(), Y) == sampled.end()) sampled.push_back(Y);
  }
  std::vector<std::vector<uint16_t>> acc0;
  for (int Y : sampled) acc0.push_back(download<uint16_t>(static_cast<const uint16_t *>(dacc.p) + Y * out_row, out_row));
  VAE_RC(ignis_image_vae_shortcut(g.kind, dx.p, g.h, g.w, g.cin, g.cout, g.ft, g.fs, dy.p, 0, 0, nullptr));
  VAE_RC(ignis_image_vae_shortcut(g.kind, dx.p, g.h, g.w, g.cin, g.cout, g.ft, g.fs, dacc.p, 1, 0, nullptr));
  MOE_CUDA(cudaDeviceSynchronize());
  long long diff = 0, acc_diff = 0;
  for (std::size_t k = 0; k < sampled.size(); ++k) {
    const int Y = sampled[k];
    // The input rows output row Y reads: AvgDown's factor_s rows from Y * factor_s, DupUp's row
    // Y / factor_s (where it is row Y % factor_s of the output).
    ShortcutGeo mg = g;
    mg.h = g.kind == 0 ? g.fs : 1;
    const int in0 = g.kind == 0 ? Y * g.fs : Y / g.fs, local = g.kind == 0 ? 0 : Y % g.fs;
    const auto xr = download<uint16_t>(static_cast<const uint16_t *>(dx.p) + in0 * in_row, mg.h * in_row);
    std::vector<float> xf(xr.size());
    for (std::size_t i = 0; i < xr.size(); ++i) xf[i] = bf16_to_f32(xr[i]);
    const auto ref = shortcut_ref(mg, xf);
    const auto got = download<uint16_t>(static_cast<const uint16_t *>(dy.p) + Y * out_row, out_row);
    const auto acc = download<uint16_t>(static_cast<const uint16_t *>(dacc.p) + Y * out_row, out_row);
    for (std::size_t i = 0; i < out_row; ++i) {
      const uint16_t want = rn_bf16(ref[local * out_row + i]);
      diff += got[i] != want;
      acc_diff += acc[i] != rn_bf16(bf16_to_f32(acc0[k][i]) + bf16_to_f32(want));
    }
  }
  std::printf("  %-30s %dx%dx%d -> %dx%dx%d: %lld differ from the restatement, accumulating %lld differ\n", name, g.h,
              g.w, g.cin, g.out_h(), g.out_w(), g.cout, diff, acc_diff);
  check(diff == 0, std::string(name) + ": the shortcut is the restated definition, bit for bit");
  check(acc_diff == 0, std::string(name) + ": the accumulating form is RN(y + RN(shortcut)), bit for bit");
}

// ---- the attention ----------------------------------------------------------------------------

struct AttnCase {
  int tokens = 0, dim = 0, grid_w = 0;
  float scale = 0.0f;
  std::vector<uint16_t> qkv;  // [tokens][3 dim]
};

// q and k: noise plus a positional component (cosines of the latent position at per-channel
// frequencies), so a query's scores peak near its own position; each query's amplitude varies
// from flat to sharp. v: noise with per-channel offsets.
AttnCase make_attn(int tokens, int dim, int grid_w) {
  AttnCase a;
  a.tokens = tokens;
  a.dim = dim;
  a.grid_w = grid_w;
  a.scale = 1.0f / std::sqrt(static_cast<float>(dim));
  a.qkv.resize(static_cast<std::size_t>(tokens) * 3 * dim);
  std::vector<float> wy(dim), wx(dim), ph(dim), vo(dim);
  for (int d = 0; d < dim; ++d) {
    wy[d] = 0.6f * std::fabs(hash_uniform(31, d, 1.0f));
    wx[d] = 0.6f * std::fabs(hash_uniform(32, d, 1.0f));
    ph[d] = hash_uniform(33, d, 3.14159f);
    vo[d] = hash_uniform(34, d, 1.5f);
  }
  parallel_for(tokens, [&](int t) {
    const float y = static_cast<float>(t / grid_w), x = static_cast<float>(t % grid_w);
    const float amp = 1.3f * std::fabs(hash_uniform(35, t, 1.0f));
    uint16_t *row = &a.qkv[static_cast<std::size_t>(t) * 3 * dim];
    for (int d = 0; d < dim; ++d) {
      const float pos = std::cos(wy[d] * y + wx[d] * x + ph[d]);
      const uint64_t i = static_cast<uint64_t>(t) * dim + d;
      row[d] = f32_to_bf16(0.7f * hash_uniform(36, i, 1.7f) + amp * pos);
      row[dim + d] = f32_to_bf16(0.7f * hash_uniform(37, i, 1.7f) + 1.1f * pos);
      row[2 * dim + d] = f32_to_bf16(hash_uniform(38, i, 1.7f) + vo[d]);
    }
  });
  return a;
}

double kernel_p(double p) {  // the BF16 value nearest to p >= 0; 0 below 2^-126 (ex2.approx.ftz)
  if (p < 0x1p-126) return 0.0;
  int e = 0;
  std::frexp(p, &e);
  const double step = std::ldexp(1.0, e - 8);
  return std::nearbyint(p / step) * step;
}

void attn_ref_row(const AttnCase &a, int i, double F, std::vector<double> &out, std::vector<double> &bound,
                  std::vector<double> &exact) {
  const int n = a.tokens, D = a.dim, ld = 3 * D;
  const double c = static_cast<float>(static_cast<double>(a.scale) * 1.4426950408889634);
  auto at = [&](int t, int off, int d) { return static_cast<double>(bf16_to_f32(a.qkv[static_cast<std::size_t>(t) * ld + off + d])); };
  std::vector<double> q(D), s(n), e(n);
  for (int d = 0; d < D; ++d) q[d] = at(i, 0, d);
  double m = -std::numeric_limits<double>::infinity(), em = 0.0;
  for (int j = 0; j < n; ++j) {
    const uint16_t *kr = &a.qkv[static_cast<std::size_t>(j) * ld + D];
    double acc = 0.0, err = 0.0;
    for (int s0 = 0; s0 < D; s0 += 16) {
      double step = 0.0, step_abs = 0.0;
      for (int d = s0; d < s0 + 16; ++d) {
        const double p = q[d] * bf16_to_f32(kr[d]);
        step += p;
        step_abs += std::fabs(p);
      }
      err += kMma * (std::fabs(acc) + err + step_abs);
      acc += step;
    }
    s[j] = acc;
    e[j] = err + 0x1p-48 * std::fabs(acc);
    m = std::max(m, acc);
    em = std::max(em, e[j]);
  }
  // Plain fp64 attention.
  exact.assign(D, 0.0);
  double sum = 0.0;
  for (int j = 0; j < n; ++j) {
    const double p = std::exp((s[j] - m) * static_cast<double>(a.scale));
    sum += p;
    for (int d = 0; d < D; ++d) exact[d] += p * at(j, 2 * D, d);
  }
  for (int d = 0; d < D; ++d) exact[d] /= sum;
  // The replay: P per key, PV in k16 steps of keys, the row sum.
  std::vector<double> N(D, 0.0), EN(D, 0.0), step(D), step_abs(D), flip(D);
  double Dsum = 0.0, flip_d = 0.0;
  for (int j0 = 0; j0 < n; j0 += 16) {
    std::fill(step.begin(), step.end(), 0.0);
    std::fill(step_abs.begin(), step_abs.end(), 0.0);
    std::fill(flip.begin(), flip.end(), 0.0);
    for (int j = j0; j < std::min(n, j0 + 16); ++j) {
      const double x = (s[j] - m) * c;
      const double dx = c * (e[j] + em) * (1.0 + 0x1p-22) + 0x1p-23 * (std::fabs(m * c) + std::fabs(x)) + 0x1p-60;
      const double p = kernel_p(std::exp2(x));
      const double width = kernel_p(std::exp2(x + dx) * (1.0 + kExp2Rel)) - kernel_p(std::exp2(x - dx) * (1.0 - kExp2Rel));
      Dsum += p;
      flip_d += width;
      for (int d = 0; d < D; ++d) {
        const double v = at(j, 2 * D, d), pv = p * v;
        step[d] += pv;
        step_abs[d] += std::fabs(pv);
        if (width > 0.0) flip[d] += width * std::fabs(v);
      }
    }
    for (int d = 0; d < D; ++d) {
      EN[d] += kMma * (std::fabs(N[d]) + EN[d] + step_abs[d] + flip[d]) + flip[d];
      N[d] += step[d];
    }
  }
  const int n8 = (n + 7) / 8 * 8;
  const double ED = ((n8 + 255) / 256 + 12) * 0x1p-24 * (Dsum + flip_d) + flip_d;
  out.assign(D, 0.0);
  bound.assign(D, 0.0);
  for (int d = 0; d < D; ++d) {
    const double o = N[d] / Dsum;
    out[d] = o;
    bound[d] = F * ((EN[d] + std::fabs(o) * ED) / (Dsum - ED) + std::fabs(o) * 2.0 * 0x1p-24) + 1e-30;
  }
}

void run_attn(const Options &o, int grid_h, int grid_w, int dim, int tile_a, int tile_b) {
  const int tokens = grid_h * grid_w;
  const AttnCase a = make_attn(tokens, dim, grid_w);
  const std::string what = "D " + std::to_string(dim) + ", " + std::to_string(grid_h) + "x" + std::to_string(grid_w) +
                           " (" + std::to_string(tokens) + " tokens), row tiles " + std::to_string(tile_a) + "/" +
                           std::to_string(tile_b);
  const std::size_t row = static_cast<std::size_t>(3) * dim;
  DeviceBytes d_qkv((tokens + kAttnPoisonRows) * row * 2);
  MOE_CUDA(cudaMemset(d_qkv.p, 0xFF, d_qkv.bytes));
  upload(d_qkv, a.qkv);
  const uint64_t ws_bytes = std::max(ignis_image_vae_attn_workspace_bytes(tokens, dim, tile_a),
                                     ignis_image_vae_attn_workspace_bytes(tokens, dim, tile_b));
  DeviceBytes ws(ws_bytes), o32(static_cast<std::size_t>(tokens) * dim * 4), o32b(o32.bytes),
      o16(static_cast<std::size_t>(tokens) * dim * 2);
  auto attn = [&](uint32_t tile, void *out, uint32_t f32) {
    MOE_CUDA(cudaMemset(ws.p, 0xFF, ws.bytes));
    MOE_CUDA(cudaMemset(out, 0xFF, f32 ? o32.bytes : o16.bytes));
    if (ignis_image_vae_attn(d_qkv.p, tokens, dim, a.scale, tile, ws.p, ws.bytes, out, f32, nullptr) != 0) {
      std::fprintf(stderr, "FATAL: ignis_image_vae_attn (%s): %s\n", what.c_str(), ignis_image_vae_last_error());
      std::exit(EXIT_FAILURE);
    }
  };
  attn(tile_a, o32.p, 1);
  attn(tile_a, o16.p, 0);
  attn(tile_b, o32b.p, 1);
  MOE_CUDA(cudaDeviceSynchronize());
  const std::size_t outs = static_cast<std::size_t>(tokens) * dim;
  const BfCheck bc = bf16_vs_f32(o32.p, o16.p, outs);
  const unsigned long long tiles_diff = diff_u16(o32.p, o32b.p, outs * 2);
  const auto y = download<float>(o32.p, outs);

  // Rows: 64 at an odd coprime stride (reaching the end), the first and last, and both sides of
  // each tile_a boundary up to 3.
  std::vector<int> rows;
  auto add = [&](int r) {
    if (r >= 0 && r < tokens && std::find(rows.begin(), rows.end(), r) == rows.end()) rows.push_back(r);
  };
  if (tokens <= 64) {
    for (int r = 0; r < tokens; ++r) add(r);
  } else {
    const long long stride = span_stride(tokens, 64);
    for (int i = 0; i < 64; ++i) add(static_cast<int>(i * stride % tokens));
  }
  add(0);
  add(tokens - 1);
  for (int b = 1; b <= 3; ++b) {
    add(b * tile_a - 1);
    add(b * tile_a);
  }
  std::vector<int> over(rows.size(), 0);
  std::vector<double> worst(rows.size(), 0.0), ee(rows.size(), 0.0), ww(rows.size(), 0.0);
  std::vector<std::vector<double>> tight(rows.size());
  parallel_for(static_cast<int>(rows.size()), [&](int k) {
    std::vector<double> ref, bound, exact;
    attn_ref_row(a, rows[k], o.bound_factor, ref, bound, exact);
    for (int d = 0; d < dim; ++d) {
      const double got = y[static_cast<std::size_t>(rows[k]) * dim + d];
      const double err = std::fabs(got - ref[d]);
      if (!(err <= bound[d])) ++over[k];
      worst[k] = std::isnan(err) ? 1e300 : std::max(worst[k], err / bound[d]);
      if (ref[d] != 0.0) tight[k].push_back(bound[d] / std::fabs(ref[d]));
      ee[k] += (got - exact[d]) * (got - exact[d]);
      ww[k] += exact[d] * exact[d];
    }
  });
  int total_over = 0;
  double worst_all = 0.0, e2 = 0.0, w2 = 0.0;
  std::vector<double> all;
  for (std::size_t k = 0; k < rows.size(); ++k) {
    total_over += over[k];
    worst_all = std::max(worst_all, worst[k]);
    e2 += ee[k];
    w2 += ww[k];
    all.insert(all.end(), tight[k].begin(), tight[k].end());
  }
  std::nth_element(all.begin(), all.begin() + all.size() / 2, all.end());
  const double rel = std::isnan(e2) ? 1e300 : std::sqrt(e2 / w2);
  std::printf("  %-52s %zu rows: worst %.3f of bound (bound/|O| median %.1e), %d over; rel RMS vs fp64 %.2e; "
              "BF16 = RN(fp32) %s; tiles %s; %llu NaN\n",
              what.c_str(), rows.size(), worst_all, all[all.size() / 2], total_over, rel,
              bc.mismatched == 0 ? "bit for bit" : "NO", tiles_diff == 0 ? "bit for bit" : "DIFFER", bc.nan32);
  check(total_over == 0, what + ": within the mechanics bound of fp64");
  check(rel <= o.max_rel_rms, what + ": relative RMS error against fp64 <= --max-rel-rms");
  check(bc.mismatched == 0, what + ": the BF16 output is the fp32 output rounded, bit for bit");
  check(tiles_diff == 0, what + ": the two row tiles give the same output, bit for bit");
  check(bc.nan32 == 0, what + ": no NaN output");
}

// ---- refusals ---------------------------------------------------------------------------------

void refusals() {
  DeviceBytes buf(1 << 22), out(1 << 22);
  const uint16_t *x = buf.as<uint16_t>();
  auto conv = [&](uint32_t kind, uint32_t up, const void *xx, uint32_t h, uint32_t w, uint32_t cin, uint32_t x_row0,
                  uint32_t x_rows, uint32_t cout, void *y, uint32_t y_row0, uint32_t y_rows) {
    return ignis_image_vae_conv(kind, up, xx, h, w, cin, x_row0, x_rows, x, nullptr, cout, nullptr, y, y_row0, y_rows, 0,
                                nullptr);
  };
  check(conv(k3, 0, x, 8, 8, 16, 0, 8, 16, out.p, 0, 8) == 0, "a valid 3 x 3 call runs");
  check(conv(k3, 0, nullptr, 8, 8, 16, 0, 8, 16, out.p, 0, 8) != 0, "NULL x refused");
  check(conv(3, 0, x, 8, 8, 16, 0, 8, 16, out.p, 0, 8) != 0, "an unknown kind refused");
  check(conv(kS2, 1, x, 8, 8, 16, 0, 8, 16, out.p, 0, 4) != 0, "upsample with the stride-2 kind refused");
  check(conv(k3, 2, x, 8, 8, 16, 0, 8, 16, out.p, 0, 8) != 0, "upsample 2 refused");
  check(conv(kS2, 0, x, 7, 8, 16, 0, 7, 16, out.p, 0, 3) != 0, "an odd height with the stride-2 kind refused");
  check(conv(k3, 0, x, 8, 8, 12, 0, 8, 16, out.p, 0, 8) != 0, "in_channels 12 refused");
  check(conv(k3, 0, x, 8, 8, 16, 0, 8, 15, out.p, 0, 8) != 0, "odd out_channels refused");
  check(conv(k3, 0, x + 4, 8, 8, 16, 0, 8, 16, out.p, 0, 8) != 0, "x off 16 bytes refused");
  check(conv(k3, 0, x, 8, 8, 16, 0, 8, 16, static_cast<char *>(out.p) + 4, 0, 8) != 0, "y off 8 bytes refused");
  check(conv(k3, 0, x, 8, 8, 16, 0, 8, 16, out.p, 0, 0) != 0, "no output rows refused");
  check(conv(k3, 0, x, 8, 8, 16, 0, 8, 16, out.p, 4, 5) != 0, "output rows past the image refused");
  check(conv(k3, 0, x, 8, 8, 16, 1, 7, 16, out.p, 0, 8) != 0, "x missing an input row refused");
  check(conv(k3, 0, x, 8, 8, 16, 2, 6, 16, out.p, 3, 3) == 0, "x holding exactly the halo runs");
  check(conv(k3, 0, x, 8, 8, 16, 3, 5, 16, out.p, 3, 3) != 0, "x missing the top halo row refused");
  check(conv(k3, 0, x, 8, 8, 16, 2, 4, 16, out.p, 3, 3) != 0, "x missing the bottom halo row refused");
  check(conv(k3, 0, x, 8, 8, 16, 0, 9, 16, out.p, 0, 8) != 0, "x rows past the image refused");
  check(conv(k3, 0, x, 40000, 8, 16, 0, 8, 16, out.p, 0, 8) != 0, "a height above 32768 refused");
  check(ignis_image_vae_pack_conv_weight(x, 16, 16, 2, out.p, nullptr) != 0, "a kernel of 2 refused");
  auto sc = [&](uint32_t kind, uint32_t h, uint32_t w, uint32_t cin, uint32_t cout, uint32_t ft, uint32_t fs) {
    return ignis_image_vae_shortcut(kind, x, h, w, cin, cout, ft, fs, out.p, 0, 0, nullptr);
  };
  check(sc(IGNIS_IMAGE_VAE_AVG_DOWN, 4, 4, 96, 192, 2, 2) == 0, "a valid AvgDown runs");
  check(sc(IGNIS_IMAGE_VAE_AVG_DOWN, 4, 4, 96, 192, 3, 2) != 0, "factor_t 3 refused");
  check(sc(IGNIS_IMAGE_VAE_AVG_DOWN, 4, 4, 96, 100, 2, 2) != 0, "AvgDown channels that do not divide refused");
  check(sc(IGNIS_IMAGE_VAE_AVG_DOWN, 5, 4, 96, 192, 2, 2) != 0, "AvgDown on an odd height refused");
  check(sc(IGNIS_IMAGE_VAE_DUP_UP, 4, 4, 96, 100, 2, 2) != 0, "DupUp channels that do not divide refused");
  check(sc(2, 4, 4, 96, 96, 1, 2) != 0, "an unknown shortcut kind refused");
  check(ignis_image_vae_shortcut(0, reinterpret_cast<const char *>(x) + 1, 4, 4, 96, 192, 2, 2, out.p, 0, 0, nullptr) != 0,
        "a misaligned shortcut x refused");
  const uint64_t need = ignis_image_vae_attn_workspace_bytes(64, 64, 64);
  check(need > 0 && ignis_image_vae_attn_workspace_bytes(64, 12, 64) == 0, "the workspace query refuses dim 12");
  auto at = [&](const void *q, uint32_t tokens, uint32_t dim, float scale, uint32_t tile, uint64_t bytes) {
    return ignis_image_vae_attn(q, tokens, dim, scale, tile, out.p, bytes, buf.as<uint16_t>() + (1 << 20), 0, nullptr);
  };
  check(at(x, 64, 64, 0.125f, 64, need) == 0, "a valid attention call runs");
  check(at(x, 0, 64, 0.125f, 64, need) != 0, "zero tokens refused");
  check(at(x, 64, 12, 0.125f, 64, need) != 0, "dim 12 refused");
  check(at(x, 64, 64, 0.125f, 0, need) != 0, "row_tile 0 refused");
  check(at(x, 64, 64, 0.125f, 64, need - 256) != 0, "a short workspace refused");
  check(at(x, 64, 64, 0.0f, 64, need) != 0, "a zero scale refused");
  check(at(x, 64, 64, std::numeric_limits<float>::quiet_NaN(), 64, need) != 0, "a NaN scale refused");
  check(at(x + 4, 64, 64, 0.125f, 64, need) != 0, "qkv off 16 bytes refused");
  check(ignis_image_vae_attn(x, 64, 64, 0.125f, 64, static_cast<char *>(out.p) + 128, need, buf.as<uint16_t>() + (1 << 20),
                             0, nullptr) != 0,
        "a workspace off 256 bytes refused");
  const std::string message = ignis_image_vae_last_error();
  check(message.find("ignis_image_vae_attn") != std::string::npos && message.find("workspace") != std::string::npos,
        "a refusal names the op and the argument: " + message);
  MOE_CUDA(cudaDeviceSynchronize());
}

}  // namespace

int main(int argc, char **argv) {
  Options o;
  o.bound_factor = arg_double(argc, argv, "--bound-factor", o.bound_factor);
  o.max_rel_rms = arg_double(argc, argv, "--max-rel-rms", o.max_rel_rms);
  int devices = 0;
  MOE_CUDA(cudaGetDeviceCount(&devices));
  std::string cases = "conv1024,conv2048,nonsquare,bands,shortcuts,attn,refusals";
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--cases") == 0) cases = argv[i + 1];
  }
  auto want = [&](const char *name) { return ("," + cases + ",").find("," + std::string(name) + ",") != std::string::npos; };
  std::printf("VAE ops (bound factor %.2f, max rel RMS %.2e; cases %s)\n", o.bound_factor, o.max_rel_rms, cases.c_str());
  for (int size : {1024, 2048}) {
    if (!want(size == 1024 ? "conv1024" : "conv2048")) continue;
    std::printf("convolutions of a %dx%d image\n", size, size);
    for (const ConvShape &s : kShapes) run_conv(o, s, size, size);
  }
  if (want("nonsquare")) {
    // The 2752 x 1536 preset (latent 172 x 96), both ways round: a swapped height and width shows.
    std::printf("convolutions of the 2752 x 1536 preset, both orientations\n");
    for (const char *name : {"dec conv_in", "dec up0 upsample", "enc down3 downsample", "enc down1 conv_shortcut",
                             "dec conv_out"}) {
      for (const ConvShape &s : kShapes) {
        if (std::strcmp(s.name, name) != 0) continue;
        run_conv(o, s, 1536, 2752);
        run_conv(o, s, 2752, 1536);
      }
    }
  }
  if (want("bands")) {
    std::printf("bands\n");
    for (const char *name : {"dec up4 resnet", "dec up2 upsample", "enc down2 downsample"}) {
      for (const ConvShape &s : kShapes) {
        if (std::strcmp(s.name, name) == 0) {
          run_bands(s, 1024, 1024);
          run_bands(s, 1536, 2752);
        }
      }
    }
    for (const ConvShape &s : kShapes) {
      if (std::strcmp(s.name, "dec up3 upsample") == 0) run_upsample_vs_materialized(s, 1024, 1024);
    }
  }
  if (want("shortcuts")) {
    std::printf("shortcuts: PyTorch's outputs (fixture)\n");
    run_shortcut_fixture();
    for (int size : {1024, 2048}) {
      std::printf("shortcuts at a %dx%d image\n", size, size);
      run_shortcut_real({0, 96, 96, 1, 2, size, size}, "enc down0 AvgDown");
      run_shortcut_real({0, 96, 192, 2, 2, size / 2, size / 2}, "enc down1 AvgDown");
      run_shortcut_real({0, 192, 384, 2, 2, size / 4, size / 4}, "enc down2 AvgDown");
      run_shortcut_real({0, 384, 768, 2, 2, size / 8, size / 8}, "enc down3 AvgDown");
      run_shortcut_real({0, 768, 768, 1, 1, size / 16, size / 16}, "enc down4 AvgDown (identity)");
      run_shortcut_real({1, 1152, 1152, 2, 2, size / 16, size / 16}, "dec up0 DupUp");
      run_shortcut_real({1, 1152, 1152, 2, 2, size / 8, size / 8}, "dec up1 DupUp");
      run_shortcut_real({1, 1152, 576, 2, 2, size / 4, size / 4}, "dec up2 DupUp");
      run_shortcut_real({1, 576, 288, 1, 2, size / 2, size / 2}, "dec up3 DupUp");
    }
    run_shortcut_real({0, 192, 384, 2, 2, 384, 688}, "enc down2 AvgDown, 1536x2752");
    run_shortcut_real({1, 1152, 576, 2, 2, 192, 344}, "dec up2 DupUp, 1536x2752");
  }
  if (want("attn")) {
    std::printf("mid-block attention\n");
    for (int dim : {768, 1152}) {
      run_attn(o, 1, 1, dim, 1024, 1000);
      run_attn(o, 1, 7, dim, 4, 3);
      run_attn(o, 63, 65, dim, 1024, 1000);
      run_attn(o, 64, 64, dim, 1024, 1000);
      run_attn(o, 128, 128, dim, 2048, 1000);
    }
    run_attn(o, 96, 172, 1152, 2048, 1000);
  }
  if (want("refusals")) refusals();
  if (g_failed != 0) {
    std::fprintf(stderr, "test_image_vae: %d failure(s)\n", g_failed);
    return 1;
  }
  std::printf("test_image_vae: OK\n");
  return 0;
}
