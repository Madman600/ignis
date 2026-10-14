# 02 — the image kernels: FP8 GEMM, 8-bit attention, encoder attention, convolution, fused ops

GitHub: #319. ADR 0050, ADR 0051. Feature: `spec.md`. Survey:
[kernels for the image model on the 5090](../../findings/2026-10-13-image-model-kernels-on-the-5090.md).

## Problem

The image model needs kernels ignis does not have:
- **GEMM:** a GEMM with FP8 activations; `ignis_fp8_linear` is W8A16.
- **Attention:** non-causal attention at head size 128 over tens of thousands
  of keys; causal GQA 32/8 at head size 128 for the encoder.
- **VAE:** convolution and upsampling.
- **DiT ops:** the DiT's modulation, its 3-axis RoPE and its gates.

## Solution

Each kernel is our own or an adapted op (ADR 0051), under `kernel/src/image/`,
each with a reference test at the image model's real geometry.

| need | kernel | source | tier |
|---|---|---|---|
| linears | FP8 W8A8 GEMM: activations quantized per token to E4M3, weights per output row, FP32 accumulate, row × column scale epilogue | extends `ignis_fp8_linear`'s `mma.sync` path. CUTLASS 2.x sm80 multistage and vLLM's `scaled_mm_sm120` as reading | our own |
| DiT attention | SageAttention2++: Q/K INT8 with K smoothing, V FP8 per channel, PV FP8 with FP16 accumulate folded into FP32, D=128, 32 heads | `thu-ml/SageAttention`, `csrc/qattn` sm89/sm120 path, torch removed (Apache-2.0) | adapted |
| encoder attention | FlashAttention-2 forward, bf16, causal, GQA 32/8, D=128 | `Dao-AILab/flash-attention` `csrc/flash_attn/src`: structure and softmax forms, raw PTX (BSD-3) | adapted |
| VAE convolution | implicit-GEMM conv2d, bf16, the shapes in spec 01's `conversion.json` (3×3 stride 1 pad 1, 3×3 stride 2 after a (0,1,0,1) pad, 1×1) | written from our FP8 GEMM; CUTLASS 2.x's implicit-GEMM fprop and ggml's `conv2d_implicit_gemm` (BSD-3 / MIT) were allowed, neither was used | our own |
| VAE mid-block attention | one head, D = 768 (encoder) / 1152 (decoder), over the latent grid: row-tiled QKᵀ, softmax, PV on the bf16 GEMM, with a fixed workspace | — | our own |
| DiT elementwise | LayerNorm without affine × (1 + scale), with the per-token choice of the t row or the t=0 row; `x + tanh(gate) · y`; QK RMSNorm + 3-axis RoPE (complex form, adjacent pairs); SwiGLU; GELU-tanh; zero-centred RMSNorm (w + 1); the norms and activations can emit the FP8 GEMM's E4M3 input | written new: the existing norm/RoPE ops are vendored with the text model's layouts. FlashInfer `fused_dit_layernorm.cuh` and `fused_qk_rmsnorm_rope.cuh` (Apache-2.0) were allowed, neither was used | our own |
| VAE elementwise | channel RMS norm with its SiLU fused (AC 7); the nearest-exact 2× upsample (fused into the convolution's gather) and the AvgDown / DupUp shortcuts (AC 6) | — | our own |
| sampler | Euler update `x += (σ_next − σ) · v` | — | moved to #321: it is Rust (spec 04), not a kernel |

**The GEMM's inner instruction is measured, not assumed.** One single source
reports that FP8 `mma.sync kind::mxf8f6f4` with unit block scales runs at the
FP16-accumulate rate (838 class) while accumulating in FP32 (419 class). The
first task is a microbenchmark of three loops on the card: BF16 m16n8k16,
FP8 `kind::f8f6f4` with FP32 accumulate, and `kind::mxf8f6f4` with unit
scales. The GEMM uses the fastest FP32-accumulating instruction. The kernel
already builds for `120a` (`kernel/CMakeLists.txt`), so no architecture change
is needed.

**The condition prefix in 8 bits.**
- **Storage.** In the first denoising step the DiT computes the condition
  prefix's keys and values once and stores them already quantized: K in INT8
  with its block scales, V in FP8 with its per-channel scales, and the
  prefix's K mean.
- **Smoothing K.** SageAttention subtracts a mean of K over the keys, and a
  cached prefix plus a target per step would give two means. Subtracting the
  prefix's mean from the target's K too is exact: the softmax does not change
  when one constant is added to every score of a query.
- **Scaling V.** The prefix's and the target's V have different per-channel
  scales. When the key loop crosses from prefix to target, the FP32
  accumulator is rescaled per channel, which is also exact.
- **Size.** About 256 KiB per token: 32 layers × (K + V) × 4096 × 1 byte, plus
  the scales. That is about 1 GiB per 1024² condition image.

**The block-causal first step needs no custom mask.** The sequence is text
segments interleaved with condition-image blocks, then the target. Each
segment is one call that reads the keys from the start up to the end of that
segment:
- **image blocks:** non-causal over the truncated keys;
- **text segments:** causal **aligned bottom-right**, that is
  `ones(len, start) ‖ tril(len)`. These segments are tiny. If the adapted
  SageAttention causal mode does not align bottom-right when the query is
  shorter than the keys, they go to a plain path;
- **target:** non-causal over everything.

In steps 2-8 only the target's queries run, over [prefix ; target].

## Acceptance

1. The microbenchmark reports TFLOPS on the 5090 for the three loops above,
   under the GPU lock, written in this spec's "As built". The GEMM's
   instruction choice cites it.
2. The FP8 W8A8 GEMM:
   - **Mechanics.** It matches an fp64 GEMM of the *dequantized* operands to a
     tight bound, at every DiT and encoder linear shape:
     - M from 1 to the prefix of 10 condition images;
     - K and N in {64, 4096, 6144, 12288, 24576}, as the artifact's shapes
       require.
   - **Quantization error.** Its error against the unquantized bf16 product is
     reported.
   - **Speed.** Its time at the DiT's 1024² shapes is reported as TFLOPS
     against the literature bar (~690-740).
3. Per-token activation quantization is fused into the producer of each GEMM
   input (norm/modulation epilogue, SwiGLU, attention output), or is a
   separate kernel whose cost is reported.
4. The 8-bit attention:
   - **Mechanics.** It matches fp64 attention computed on the same quantized,
     smoothed operands to a tight bound.
   - **Quantization error.** Against fp64 attention on unquantized inputs it
     reaches cosine similarity ≥ 0.99 per head, at the real geometry.
   - **Cases covered:**
     - the target over [8-bit prefix ; target] for 0, 1 and 10 condition
       images;
     - the image-segment calls over truncated keys;
     - the text-segment calls with bottom-right causality.
   - **Prefix handling.** The target's call over [stored prefix run ; target
     run] matches fp64 attention on exactly those stored operands within the
     mechanics bound, for 0, 1 and 10 condition images. The prefix run is
     quantized alone in step 1, with its own K mean and V scales. The target
     run is stored with the prefix's K mean and its own V scales. Literal
     equality with quantizing [prefix ; target] at once cannot hold, because
     that changes the mean, the V scales and the 64-key blocks. So the seams
     are tested directly:
     - **shared K mean:** the target run's mean is the prefix's, byte for
       byte, and its K codes restate from that mean bit for bit;
     - **accumulator rescale:** a negative control, the reference computed
       without the rescale on inputs whose target V spread differs from the
       prefix's, fails on most outputs;
     - **storage:** the prefix run is byte-identical after every call that
       read it;
     - **cost of the split:** its error against fp64 on the unquantized
       inputs is reported beside that of one run of [prefix ; target], not
       gated. This covers 0, 1 and 10 images, including inputs whose prefix
       and target K means differ strongly.
   - **Speed.** TOPS reported against ~620-640.
5. The encoder attention (adapted FlashAttention-2):
   - matches fp64 causal GQA 32/8 attention at D=128 for sequences from 1 to
     the longest 10-image edit prompt;
   - its TOPS are reported against ~210.
6. The convolution:
   - matches an fp64 conv2d at every VAE shape in `conversion.json`, at 1024²
     and 2048²;
   - the VAE mid-block attention matches fp64 at both D;
   - the upsample and shortcuts match their PyTorch definitions bit for bit
     in fp32 and within bf16 rounding in bf16.
7. The elementwise ops each match an fp64 reference, including:
   - the per-token t / t=0 modulation choice;
   - the RoPE table against diffusers' `pos_embed` for a t2i layout and an
     edit layout (from spec 01's recordings).
8. Each adapted op names its upstream repo, commit, path and licence in its
   header, and `kernel/NOTICE` lists it. No file comes from CUTLASS 3.x or from
   FlashInfer's `csrc/fmha_v2`.
9. Every test above runs in the GPU profile (`--ignored`). Each fp64
   reference is our own CPU code, so this spec does not wait for 01's
   recordings, except the RoPE comparison in AC 7.

## As built

**Status per acceptance criterion** (2026-10-14, branch `image-kernels-319`;
the sections below have the numbers):
1. **Done.** The instruction table below; the GEMM takes `kind::mxf8f6f4`
   with unit scales (911 TFLOPS with FP32 accumulation, twice the plain form).
2. **Done.** 0 outputs over the bound at every linear shape, 1 to 41,000
   tokens; quantization error reported (3.6-3.8% relative RMS on synthetic
   inputs); 495-536 TFLOPS, about 72% of the bar (reported, not tuned further).
3. **Done.** The norms and activations emit the E4M3 pair in their own pass
   (AC 7), bit for bit with `ignis_image_quantize_tokens`. The attention
   output's quantization stays the separate kernel: 0.012 ms per 1024²
   block, 0.3% of it.
4. **Done.** 0 outputs over the bound in every call; cosine ≥ 0.99 per head
   on DiT-like inputs and on the DiT's real Q, K and V (worst 0.99796, both
   gated); the
   prefix seams tested; 489-573 TOPS against 618-643. The synthetic 20×
   outlier arm (0.918-0.96) is report-only, by the coordinator's call.
5. **Done.** 0 outputs over a replay bound from 1 to 41,000 tokens; 196-207
   TOPS against ~210.
6. **Done.** Every VAE convolution shape at 1024² and 2048², the mid-block
   attention at both D, and the upsample and shortcuts (bit for bit against
   PyTorch); 167-186 TFLOPS on the 3 × 3 calls.
7. **Done.** Every elementwise op within a bound that tracks its roundings,
   the per-token t / t=0 row bit for bit with a negative control, and the
   RoPE table within one ulp of diffusers' `pos_embed` for the t2i and edit
   layouts.
8. **Done.** The adapted ops (SageAttention for AC 4, FlashAttention-2 for
   AC 5, FlashInfer's helpers through SageAttention) are named in their
   headers and `kernel/NOTICE`; everything else is our own. Nothing comes
   from CUTLASS 3.x or FlashInfer's `csrc/fmha_v2`.
9. **Done.** Five CTests at the image model's geometry, plus two that read
   data outside the repo (the real Q/K/V probe, the RoPE oracle), which print
   SKIP without it outside the profile and fail inside it; five Rust `--ignored` tests drive every binding. The
   benches are tools.

### The instruction microbenchmark (AC 1)

`kernel/tests/bench_mma_peak.cu` (`ignis_mma_peak_bench`, a tool, not a
CTest). Register-resident operands, 680 blocks × 256 threads, eight
independent accumulator chains per warp, 300 ms warm-up per loop, ten timed
launches. Block 0 reads `clock64` and `%globaltimer` around its loop, so each
rate carries the SM clock it ran at.

Measured 2026-10-13 on the RTX 5090 (driver 596.36, CUDA 13.1, power limit
575 W), under the GPU lock:

| loop | SASS | TFLOPS | SM clock | TFLOPS/GHz | % of nominal per clock |
|---|---|---|---|---|---|
| BF16 m16n8k16, FP32 accumulate | `HMMA.16816.F32.BF16` | 250.6 | 2898 MHz | 86.5 | 99% of 209.5 @ 2407 |
| FP8 `kind::f8f6f4`, FP32 accumulate | `QMMA.16832.F32.E4M3.E4M3` | 489.3 | 2850 MHz | 171.7 | 99% of 419 @ 2407 |
| FP8 `kind::mxf8f6f4`, unit UE8M0 scales, FP32 accumulate | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | **911.4** | 2638 MHz | **345.5** | 99% of **838** @ 2407 |
| FP8 `kind::f8f6f4`, FP16 accumulate (reference) | `QMMA.16832.F16.E4M3.E4M3` | 946.5 | 2722 MHz | 347.7 | 100% of 838 @ 2407 |

- **The block-scaled form runs at twice the plain form's rate**: 2.01×
  per clock (1.86× in raw TFLOPS, because the card drops its clock at the
  higher power). It reaches the FP16-accumulate rate while accumulating in
  FP32: the survey's single source is confirmed on this card.
- **Same results.** On random operands the block-scaled form with unit
  scales gives the plain form's results bit for bit, and both give fp64's
  exactly on operands whose sums fit fp32.
- **Accumulation precision.** Probes of one instruction (C = 2^8 plus a
  product 2^(8−e); two products; one large and 31 small products) are exact
  up to e = 23 for all three FP32-accumulating forms, as fp32 is. With
  C = 2^8 plus 32 products of 2^(8−e) the result stays exact to e = 25 and
  drops the products at e = 26: within one instruction, addends more than
  2^25 below the largest are truncated. That is close to fp32 and far from
  Hopper's ~14-bit FP8 accumulation, so the GEMM's mechanics bound is an
  fp32-class bound per instruction.

**The GEMM's inner instruction:** `mma.sync.m16n8k32.kind::mxf8f6f4
.block_scale.scale_vec::1X` with every block scale 2^0
(`kernel/src/image/image_mma.cuh`, `mma_e4m3_mx_unit_f32`). The per-token
and per-row scales are applied in the epilogue.

**For the 8-bit attention (AC 4)** the bench gained two loops, run again
with the others on 2026-10-13 under the GPU lock (timing mode):

| loop | SASS | TOPS | SM clock | TOPS/GHz |
|---|---|---|---|---|
| BF16 m16n8k16 (again) | `HMMA.16816.F32.BF16` | 249.8 | 2897 MHz | 86.2 |
| FP8 `kind::f8f6f4`, FP32 accumulate (again) | `QMMA.16832.F32.E4M3.E4M3` | 495.6 | 2865 MHz | 173.0 |
| FP8 `kind::mxf8f6f4`, unit scales (again) | `QMMA.SF.16832.F32.E4M3.E4M3.E8` | 908.2 | 2674 MHz | 339.6 |
| FP8 FP16 accumulate (again) | `QMMA.16832.F16.E4M3.E4M3` | 941.2 | 2733 MHz | 344.4 |
| INT8 m16n8k32, INT32 accumulate | `IMMA.16832.S8.S8` | **977.8** | 2836 MHz | **344.8** |
| INT8 and `mxf8f6f4` alternating | both | 893.5 | 2768 MHz | 322.8 |

- INT8 runs at the 838 class, the same rate per clock as `mxf8f6f4` with
  unit scales. One instruction is exact (checked against the integer
  product, including 32 products of −128 × −128 on top of 2^31 − 1 − 2^19).
- Alternating the two, as the attention does, keeps 95% of that rate per
  clock.
- **So the attention takes INT8 for QK^T and `mxf8f6f4` with FP32
  accumulation for PV.** SageAttention2++'s FP16-accumulate PV buys nothing
  here, and it would cost precision: its V scale is capped at 2.25 so that
  the FP16 accumulator cannot overflow. With FP32 accumulation, V uses
  E4M3's whole range (448).

### The FP8 W8A8 GEMM and the activation quantization (AC 2, AC 3)

**Where.**
- ABI: `kernel/include/ignis_image_gemm.h`: `ignis_image_gemm_prepare`,
  `ignis_image_quantize_tokens`, `ignis_image_gemm_w8a8`,
  `ignis_image_gemm_last_error`.
- Code: `kernel/src/image/` (`fp8_gemm.cu`, `fp8_gemm_kernel.cuh`,
  `fp8_quant.cu`, `fp8_quant.cuh`, `image_mma.cuh`, `image_common.*`). All our
  own (ADR 0051 tier 1); CUTLASS 2.x sm80 and vLLM's `scaled_mm_sm120` were
  read, nothing was copied, so `kernel/NOTICE` has nothing to add.
- The leaf instantiates the default tile configuration only. The sweep's
  configurations are built into the test and the bench alone
  (`kernel/tests/image_gemm_sweep.{h,cu}`, from the same kernel template).
- Tests: the CTest `ignis_kernel_image_gemm_test`
  (`kernel/tests/test_image_gemm.cu`) and the Rust `--ignored` test
  `crates/core/tests/image_gemm_gpu.rs` (bindings in
  `crates/core/src/image_gemm.rs`). Speed: `ignis_image_gemm_bench`
  (`kernel/tests/bench_image_gemm.cu`, a tool).

**The activation contract (AC 3).** Activations reach the GEMM as E4M3FN
codes `[tokens][cols]` plus fp32 `scales[tokens]`:
- `scales[t] = amax_t / 448`;
- `codes[t][c] = E4M3(x[t][c] · (448 / amax_t))`, rounded to nearest even
  and saturated, each step one fp32 operation;
- a row whose amax is below 448 · 2^-126 = 7 · 2^-120 gets scale 1 and zero
  codes. That floor is the smallest amax whose scale is a normal fp32;
  below it 448 / amax can overflow, and 0 · Inf would make NaN codes. An
  all-zero row is one such row;
- a NaN element gets a NaN code, and an infinite one makes the scale Inf:
  a non-finite x never dequantizes to a finite value (tested, with rows
  at, below and above the floor).

`ignis_image_quantize_tokens` is the standalone producer.
`fp8_quant.cuh` holds the device steps, so a norm, modulation, SwiGLU or
attention-output epilogue can emit the same pair and drop the separate pass.
The norms and activations of AC 7 emit the pair in their own pass ("The
elementwise ops (AC 7)" below); the attention output still goes through the
standalone kernel, whose cost is in the speed table below.

**The kernel.**
- Weights: the row-scale-v1 payload as it is.
- Instruction: `kind::mxf8f6f4` with unit scales.
- Tile: 128 tokens × 256 output rows, 8 warps of 64 × 64, 64 code bytes per
  stage, 4 cp.async stages (96 KiB). Shared rows are XOR-swizzled.
  Fragments are double-buffered across k32 steps.
- Raster: groups of 8 token tiles.
- Epilogue: `(acc · scales[t]) · wscale[r]`, BF16 or fp32.
- No atomics: each output is computed by one fixed sequence, so a second run
  agrees bit for bit (tested).
- Shapes taken: `cols` a multiple of 64, `rows` a multiple of 16, any number
  of tokens.

**Mechanics (AC 2).** The reference is our own fp64 GEMM of the dequantized
operands: the quantizer's codes and scales, and the payload's codes and BF16
row scales. Shapes are the real linears (spec.md's model table), rows × cols:
- 4096 × 64: `img_in`;
- 64 × 4096: `proj_out`;
- 4096 × 4096: q/k/v/out, `txt_in`, encoder q/o;
- 1024 × 4096: encoder k/v;
- 6144 × 4096: encoder q/k/v fused;
- 12288 × 4096: SwiGLU gate or up, DiT q/k/v fused;
- 24576 × 4096: gate and up fused;
- 4096 × 12288: down.

Every shape runs at 1, 2, 17, 129, 4096, 16,512 and 41,000 tokens (a
10-image prefix). At large token counts the check samples 161 token rows and
385 output columns, i · stride mod count plus the last one. The stride is
the smallest odd number ≥ (count − 1) / (want − 1), rounded up, that is
coprime with the count. So the samples are distinct and reach the end of the
range: every output-row tile is compared, and every token tile up to 16,512
tokens (at 41,000 tokens the 161 rows hit 161 of 321 tiles). The rows
cover every residue mod 128 and the columns every residue mod 256. Before
the review fix the stride was rounded down, which left the end of the range
unsampled: 4096 output rows reached 3447, so tile 14 was never compared.

**The encoder's longest prompt.** It is 10 × 1024 vision slots plus the
template and the prompt text: about 12,400 tokens with spec 03's longest
tested prompt (2,000 tokens), inside 16,512. The prompt text has no cap of
its own, though. So the encoder-only shapes (1024 × 4096, 6144 × 4096) also
run at 41,000 tokens: the encoder's sequence is never longer than the DiT
prefix it becomes, since each slot becomes 4 prefix tokens.

The bound comes from the M1 probes. Per instruction, its 32 products and its
accumulator input each lose at most 2^-25 of the largest addend, and its
result one fp32 rounding. The test walks each output's k32 steps in the
kernel's order. With `S_i` the exact fp64 prefix sum after step `i`:

`E_i = E_{i−1} + F · 35 · 2^-25 · (|S_{i−1}| + E_{i−1} + Σ_{step i}|p|)`

`|y − ref| ≤ E_last · |scales[t] · wscale[r]| + |ref| · (2^-23 + 2^-8 for BF16)`

`F` is `--bound-factor`, default 1. The accumulation part of the bound over
|ref| has these medians:

| K | per-k32 bound (now) | whole-row Σ\|terms\| bound (before, review's simulation) |
|---|---|---|
| 64 | 4.9e-6 | 1.1e-5 |
| 4096 | 1.27e-4 | 4.9e-3 |
| 12288 | 3.4e-4 | 1.9e-2 |

**Result:** 0 outputs over the bound, at every shape and token count.
- fp32 outputs use 3-14% of the bound.
- BF16 outputs use 86-99.5%; that is BF16's own half ulp.
- The six other tile configurations pass the same check at the edge shapes.
- Run time: about 2 minutes.

**Quantization error**, against the unquantized product. The reference is
fp64 of BF16 x times BF16 W, with W quantized per row as the converter does
(layout.md §6.1: the row scale is amax / 448 rounded **up** to the next
BF16). Shapes are 4096 tokens × (4096 × 4096) and (4096 × 12288):

| x | W8A8 relative RMS | cosine | weights only |
|---|---|---|---|
| Gaussian | 3.77-3.78% | 0.99929 | 2.64-2.66% |
| Gaussian, 8 channels at 20× | 3.61-3.66% | 0.99933-0.99935 | 2.65-2.67% |

The inputs are synthetic. The error on real DiT activations comes with spec
01's recordings and spec 04's end-to-end test. The CTest gates only on
`--max-quant-rel-rms` (default 0.1), as a sanity check.

**Speed (2026-10-13, 4096 target tokens = 1024², BF16 out, GPU lock in
timing mode, card power limit 575 W).** NVML read the SM clock and board
power halfway through each run.

| linear | rows × cols | ms | TFLOPS | SM clock | board |
|---|---|---|---|---|---|
| to_q / to_k / to_v / to_out | 4096 × 4096 | 0.273 | 504 | 2902 MHz | 229 W |
| SwiGLU gate or up | 12288 × 4096 | 0.769 | 536 | 2677 MHz | 419 W |
| SwiGLU down | 4096 × 12288 | 0.797 | 517 | 2527 MHz | 535 W |
| q/k/v fused | 12288 × 4096 | 0.770 | 536 | 2527 MHz | 535 W |
| gate+up fused | 24576 × 4096 | 1.589 | 519 | 2550 MHz | 579 W |
| `img_in` | 4096 × 64 | 0.037 | 58 | | |
| `proj_out` | 64 × 4096 | 0.062 | 35 | | |

**One DiT block** (separate q, k, v, out, gate, up, down, plus their four
quantizations): 3.608 ms, **495 TFLOPS**. The quantization is 3.4% of that.
The 32 blocks' linears take 115.5 ms per denoising step at 1024².

**Quantization alone:**
- 4096 tokens × 4096: 0.012 ms. The data fits in L2 (4.2 TB/s).
- 4096 tokens × 12288: 0.087 ms (1.75 TB/s, DRAM-class).
- At 16,512 tokens: 1.5-1.7 TB/s, and 5.0% of a block (measured with
  configuration 0, whose block took 15.2 ms there).

**The tile sweep.** `--sweep`, block TFLOPS at 1024²:

| config | tile | block TFLOPS |
|---|---|---|
| 0 | 128 × 128, BK 64, 4 stages, 2 × 4 warps | 465 |
| 1 | 128 × 128, BK 64, 3 stages, 2 CTAs/SM | 469 |
| 2 | 128 × 128, BK 128, 3 stages | 491 |
| 3 | 128 × 256, BK 64, 3 stages | 476 |
| 4 | 256 × 128, BK 64, 3 stages | 464 |
| 5 | 128 × 128, BK 64, 3 stages, 2 × 2 warps, 2 CTAs/SM | 490 |
| 6 | 128 × 256, BK 64, 4 stages | **492** (default) |

**Against the literature bar (~690-740): 495-536 TFLOPS, about 72% of it.**
Speed is a target, not a gate, so tuning stopped after the sweep. What
would close the gap, largest first:
1. **The main loop.** On the large shapes it runs at about 61-65% of the
   clock-adjusted peak (345.5 TFLOPS/GHz × 2.5-2.7 GHz). Not diagnosed: no
   Nsight Compute run was made. Candidates:
   - L2 traffic: a 128 × 256 tile brings 171 FLOP per byte, about 3 TB/s
     from L2 now and about 5 TB/s at peak;
   - barrier and ldmatrix latency with 8 warps per SM;
   - the epilogue's 4-byte stores.

   The next step is one ncu run on the 12288 × 4096 shape. A TMA-fed,
   warp-specialized pipeline is the structural answer if latency is the
   cause.
2. **Wave quantization.** At 4096 tokens a 4096-row output has 512 tiles,
   which is 3.01 waves of 170 SMs, so it takes 4 waves. Those shapes run at
   ~50% of the clock peak, at 2.9 GHz and 229 W. A stream-K split of the
   last wave would recover most of it, for 4 of a block's 7 linears.
3. **Power.** The card is limited to 575 W; the literature's numbers are at
   600 W. The wide GEMMs reach 575-580 W and drop to 2.0-2.6 GHz.
4. **Fusing the quantization** into the producers: 3.4% at 1024² and about
   5% at 16,512 tokens. Done for the norms and SwiGLU in AC 7; the
   attention output's (one of the four) remains.

### The 8-bit DiT attention (AC 4)

**Where.**
- ABI: `kernel/include/ignis_image_attn.h`: `ignis_image_attn_q8_bytes`,
  `ignis_image_attn_kv8_bytes`, `ignis_image_attn_quantize_q`,
  `ignis_image_attn_store_kv`, `ignis_image_attn`,
  `ignis_image_attn_last_error`.
- Code: `kernel/src/image/attn8.cu`, an adapted op (ADR 0051 tier 2). It
  adapts `thu-ml/SageAttention` d1a57a5 `csrc/qattn` (the sm89/sm120 path),
  whose smem, cp.async and mma helpers are themselves adapted from FlashInfer
  v0.1.5; all Apache-2.0. The file header and `kernel/NOTICE` give the
  chain. Those helpers now live in `attn_smem.cuh`, shared with the encoder
  attention (AC 5). The quantizers (`attn8_quant.cu`) and the layout
  (`attn8_layout.cuh`) are our own.
- The upstream was fetched as data into this worktree's
  `.scratch/upstream/sageattention`, pinned at d1a57a5. Nothing of it was
  built or run.
- Tests: the CTest `ignis_kernel_image_attn_test`
  (`kernel/tests/test_image_attn.cu`) and the Rust `--ignored` test
  `crates/core/tests/image_attn_gpu.rs` (bindings in
  `crates/core/src/image_attn.rs`). Real Q, K and V: the CTest
  `ignis_kernel_image_attn_qkv_probe_test` (the same file, `--cases
  qkv-probe`; "Real Q, K and V" below). Speed: `ignis_image_attn_bench`
  (`kernel/tests/bench_image_attn.cu`, a tool).

**The instructions** (the bench table above): INT8 m16n8k32 for QK^T
(exact in int32) and `kind::mxf8f6f4` with unit scales and FP32
accumulation for PV. Both run at the 838 class.

**The scheme.** SageAttention2's, with four changes:
- Q is INT8 per 32-row block (per warp), K is INT8 per 64-key block after
  subtracting a K mean, and V is E4M3 per channel.
- **PV accumulates in FP32**, at the same rate as SA2++'s FP16
  accumulation. So V's scale is amax / 448, not SA2++'s amax / 2.25, and
  there is no FP16 overflow to guard.
- **The denominator is the row sum of the same E4M3 P that multiplies V**,
  as an MMA against ones; SA2 sums the fp32 P before rounding. So O is a
  convex combination of V's rows whatever P's rounding, and V's channel
  offset does not amplify P's rounding error. The offset still costs V's
  own E4M3 resolution: the test's channels of spread 0.25 and offset ±2
  lose about 3 bits. V is not smoothed (the coordinator's call, to revisit
  only if spec 04's data needs it). With this denominator, smoothing V by a
  shared per-channel mean would be exact up to D's rounding. The MMA count
  rises by 3%.
- **Two K/V runs per call.** When the key loop crosses from run A (the
  stored prefix) to run B (the target), the fp32 accumulator is rescaled
  per channel by vscale_A / vscale_B.
- **Causal is aligned bottom-right** (query i sees keys 0 .. i + keys −
  queries), masked on every block the diagonal crosses at any offset. So
  the text segments need no plain path.

**The ABI for spec 04's leaf.** A K/V *run* is one byte buffer whose layout
the header documents: K codes, V codes (transposed and permuted), K block
scales, V channel scales, the K mean. Per layer it costs 8 KiB per token
(plus 2 bytes of K scales), padded to 64 tokens, and a fixed 32 KiB. That is
about 1 GiB per 1024² condition image over 32 layers, as spec'd.
- **Step 1** (`encode_condition`), per layer:
  - `store_kv` of the prefix's K and V with its own mean: the stored prefix,
    kept for the job;
  - then per segment, `quantize_q` of the segment's rows and one
    `ignis_image_attn` over run A = the prefix truncated at the segment's
    end: causal for text, not for images.
- **Every denoising step**, per layer:
  - `store_kv` of the target's K and V with `mean_from` = the prefix run, so
    every key is smoothed with the same mean;
  - `quantize_q` of the target;
  - one call over A = the whole prefix and B = the target's run.
- Every call takes a stream, allocates nothing, is graph-capturable and is
  deterministic. Q, K, V and O are addressed with a row stride, so the
  fused QKV row can be read as it is, after the QK norm and RoPE applied to
  it in place.
- `scale` must be finite and > 0 (refused otherwise). A negative scale
  would turn max(S) · c into the row's minimum, and 0 gives NaN on masked
  keys.
- **Two preconditions the op cannot check:**
  - run B must have been stored with `mean_from` = run A. The kernel never
    reads the means, so a run B smoothed with another mean gives silently
    wrong scores;
  - `mean_from_tokens` must be run A's exact token count. It locates the
    mean in the run's layout, and a wrong count silently reads another
    region. Spec 04's prefix handle carries the count.
- The ops are asynchronous and carry no synchronization. Spec 04's Rust
  driver orders the `CudaDevice` load-stream copies (noise upload,
  velocity copy-out) against them itself, with `stream_sync(null)`; the
  bindings' docs say so.

**Mechanics.** The reference is our own fp64 code on the same quantized,
smoothed operands (the q8 buffer and the runs, read back). It replays the
kernel's softmax in the kernel's block order with the same fp32 operations:
- the score scale `(qscale · kscale) · fl(scale · log2 e)`;
- the running max `max(m, fma(max S, c, −8.807))`;
- the exponent argument `fma(S, c, −m)`.

S is exact in int32. What it cannot replay it bounds:
- **P's E4M3 code.** ex2.approx is within 2 ulp of the correctly rounded
  2^x (PTX ISA); the test takes 2^-20. A key whose 2^x · (1 ± 2^-20)
  straddles a rounding midpoint adds that ulp · |v| to the numerator's
  error and that ulp to the denominator's.
- **The rescales** when the max rises: ex2 of fl32(m_old − m_new) (the
  subtraction replayed as the kernel rounds it), plus one rounding.
- **PV and the row sum:** per k32 step, 35 · 2^-25 · (|acc| + the error so
  far + Σ|P v|), as in the GEMM.
- **The run boundary** (two roundings) and **the end** (three).
- **The quotient:** |ΔO| ≤ (|ΔN| + |O| · |ΔD|) / (D − |ΔD|).

`--bound-factor` multiplies the whole bound (default 1). The output is fp32
for this check. The same call with BF16 output must equal the fp32 output
rounded to BF16, bit for bit, which also shows the call is deterministic.

**Quantization error**, against fp64 attention on the unquantized BF16
inputs. The gate is cosine per head ≥ `--min-cosine` (default 0.99); the
centred cosine and the relative RMS error are reported beside it. The
inputs are synthetic but DiT-shaped:
- K has a per-channel mean of ±3, shifted per token kind (text, image,
  target);
- V has per-channel scales of 0.25-4 and offsets of ±2, and the target's
  spread is 2.5× the prefix's;
- Q runs at a per-head temperature from flat to peaked.

Three harder profiles follow below the main table: a heavy tail, outlier
channels, and prefix and target K means far apart. Real activations follow
them ("Real Q, K and V"); spec 04's end-to-end test has the last word.

**The cases** (`ignis_kernel_image_attn_test`, about 75 s with the fp64
references on 16 threads). Each check covers 64 query rows per head (all
rows of a smaller call), i · stride + h mod the call's rows, plus the
call's last row. The stride is the smallest odd number ≥ (rows − 1) / 63,
rounded up, that is coprime with the row count. So the rows are distinct
and spread to the end of the call, and the head's shift makes the 32 heads
together cover every residue mod 128. The last row is added because the
shift alone stops short: at 16,384 queries a head's rows end at
16,182 + h, which never compares the last 128-query CTA. Before the first
review fix the stride was rounded down: 160 queries were checked at rows
0-94 only. The prefix layouts are:
- 0 images: text 200;
- 1 image: text 20, image 4096, text 160;
- 10 images: text 20, then image 4096 and text 7 ten times, the last text
  160 (41,203 tokens).

The target is 4096 tokens (1024²), and once 16,384 (2048²). The table gives
the worst error over the bound, the bound's median over |O|, and the worst
head's cosine (centred) and relative RMS error:

| call | queries × keys | worst / bound | bound / \|O\| | cosine (centred) | rel. RMS |
|---|---|---|---|---|---|
| target over [prefix ; target], 0 images | 4096 × 4296 | 0.81 | 1.5e-4 | 0.99951 (0.99830) | 3.2% |
| target over [prefix ; target], 1 image | 4096 × 8372 | 0.70 | 2.9e-4 | 0.99973 (0.99805) | 2.3% |
| target over [prefix ; target], 10 images | 4096 × 45,299 | 0.53 | 1.5e-3 | 0.99990 (0.99812) | 1.4% |
| target over [prefix ; target], 0 images, 2048² | 16,384 × 16,584 | 0.66 | 5.6e-4 | 0.99959 (0.99834) | 2.9% |
| image block 1 of 1 (truncated keys) | 4096 × 4116 | 0.94 | 1.4e-4 | 0.99985 (0.99803) | 1.7% |
| image blocks 1 and 10 of 10 | 4096 × 4116 / 41,043 | 0.85 / 0.42 | 1.4e-4 / 1.4e-3 | ≥ 0.99978 (0.99798) | ≤ 2.1% |
| text segments, causal bottom-right | 7-200 × 20-41,203 | ≤ 0.98 | 4.5e-6-1.4e-3 | ≥ 0.99954 (0.99781) | ≤ 3.0% |
| small odd sizes (partial blocks, both runs, causal over two runs) | 1-200 × 1-207 | ≤ 0.04 | ≤ 2.1e-5 | ≥ 0.99950 (0.99825) | ≤ 3.2% |

- **0 outputs over the bound**, in every call.
- The BF16 output equals the fp32 output rounded, bit for bit, in every call.
- The quantizers equal their restated definitions bit for bit, padding
  included. The prefix's K mean is within the fp32-sum error of fp64; the
  target's run carries the prefix's mean, byte for byte.
- **Edge blocks** of Q, K and V check the floors: blocks of zeros and of
  1e-37 get scale 1 and zero codes, and blocks at the floor (127 · 2^-126
  for Q and K, 448 · 2^-126 for V) get scale 2^-126. A regression to
  `amax > 0` fails there.
- A worst near 1 is one E4M3 rounding flip of P, which the bound adds in
  full. The median shows how tight the bound is: from 5e-6 to 1.5e-3 of
  |O|, growing with the number of k32 steps. That is 10-1000× below the
  quantization error.
- **What only the small cases pin.** At 4K keys and more, the bound (1e-4
  to 1.5e-3 of |O|) is 10-100× one key's weight (~2e-5 at 45K keys). So a
  one-key mask error or an `a_keys` off by one is caught only by the small
  cases, whose bound is ≤ 2.1e-5 of |O|: one row and one key, the truncated
  prefix, causal over two runs.

**Prefix handling** (AC 4's text above):
- **The two-run call is checked against fp64 of the operands the split path
  stores**: the prefix's run (its own mean and V scales) and the target's
  run (the prefix's mean, its own V scales), concatenated. That is the
  target rows above. A lost boundary rescale or a second mean would be an
  O(1) error there.
- **The shared mean:** the target run's mean is the prefix's byte for byte,
  and its K codes restate from that mean bit for bit.
- **Negative control:** the same bound computed without the rescale fails
  36,735 of 36,736 outputs (1 image, where the target's V spread is 2.5× the
  prefix's).
- **Storage:** the prefix run is byte for byte as stored after every call
  that read it, in every layout and profile.
- **The cost of the split.** One run of [prefix ; target] has its own mean
  and V scales; the split is the leaf's layout. Both are checked against
  their own operands (0 over) and compared to fp64 on the unquantized
  inputs. Worst head's cosine (centred) and relative RMS:

| inputs | images | split | one run | split vs one run |
|---|---|---|---|---|
| DiT-like | 0 | 0.99951 (0.99830), 3.2% | 0.99956 (0.99828), 3.0% | 1.7% |
| DiT-like | 1 | 0.99973 (0.99805), 2.3% | 0.99970 (0.99808), 2.4% | 1.2% |
| DiT-like | 10 | 0.99990 (0.99812), 1.4% | 0.99988 (0.99812), 1.6% | 0.5% |
| DiT-like, 2048² | 0 | 0.99959 (0.99834), 2.9% | 0.99966 (0.99831), 2.6% | 1.3% |
| K means apart | 0 | 0.99729 (0.99476), 7.8% | 0.99916 (0.99808), 4.1% | 5.0% |
| K means apart | 1 | 0.99767 (0.99490), 6.9% | 0.99871 (0.99765), 5.1% | 3.9% |
| K means apart | 10 | 0.99800 (0.99391), 6.3% | 0.99816 (0.99585), 6.1% | 3.2% |

  "K means apart" puts an offset of 10σ in four channel pairs (d, d + 64)
  per head, at a per-head angle for the prefix, rotated by 2 rad for the
  target (as RoPE rotates a pair's mean differently at text and target
  positions). The prefix's mean is then ~17 off in those channels of every
  target key.
  - On DiT-like inputs the split stays within 0.3 points of one run's
    relative RMS error (3.2% against 3.0% at 0 images, 2.9% against 2.6%
    at 2048²).
  - With the means apart it costs most at 0 images (the review's M2). There,
    a ~200-token text mean smooths 4096 target keys, and the relative RMS
    error is 7.8% against 4.1% for one run; the cosine still clears 0.99.
  - This arm is a measurement, not a gate. Spec 04's Seams name the exact
    fallback if real data shows target-K error: the target run's own mean,
    plus q · (m_t − m_p) on run B's exponents.

**Harder inputs (the review's M3).** gauss() is an Irwin-Hall sum, bounded
at ±3.45σ. Two profiles add what stretches a block's amax, each at 0, 1 and
10 images (split / one run):
- **A heavy tail** on q, k and v (one value in 256 at 6×, up to ~20σ):
  cosine ≥ 0.99659 (centred 0.99561), relative RMS ≤ 8.3%. Gated at 0.99,
  passes.
- **Outlier channels**, the GEMM test's "8 channels at 20×": q's channel 37
  and k's channel 91 in heads 0, 4, …, 28, Gaussian.
  - Mechanics: 0 over the bound.
  - Accuracy: the worst head is always one of those 8. Its cosine is 0.918
    / 0.908 at 0 images, 0.939 / 0.953 at 1 and 0.962 / 0.948 at 10, with
    relative RMS 29-43%. **Below AC 4's 0.99.**
  - Cause: a 20× channel sets the 32-row Q block's or the 64-key K block's
    INT8 amax. The other 127 channels keep ~2 levels per σ, so the scores
    carry noise of ~0.1 × the head's temperature. K's mean smoothing does
    not help.
  - Status: the arm is reported, not gated. An exact per-channel Q/K
    migration scale (q_d / s_d, k_d · s_d, fixed per layer and head) is the
    known remedy. The DiT's own Q, K and V do not need it (below), so the
    arm stays report-only.

**Real Q, K and V (the probe, 2026-10-13).** Inputs:
- Q, K and V as the DiT computes them, after QK-norm and RoPE, from a plain
  bf16 diffusers run of spec 01's t2i and edit cases at 1024²: the oracle's
  prompts, noise and condition image, snapshot fc13ba1.
- Steps 0 and 4 (step 4 reads step 0's cached prefix) and blocks 0, 15 and
  31. The prefix is text 31 for t2i, and text 8, image 4096, text 18 for
  edit; the target is 4096 tokens.

They go through spec 04's calls: the prefix run stored alone, the target run
with the prefix's mean, the target over both, and step 0's segment calls.
Beside them runs the target over [prefix ; target] stored as one run. That
is 36 calls.

Where it lives:
- The dump is `F:/ai/models/ignis-image-2.1-turbo/qkv-probe` (1.5 GB).
  `dump_qkv.py`, `provenance.json` and the statistics (`stats.json`, from
  `qkv_stats.py`) sit beside it.
- The test is `ignis_kernel_image_attn_qkv_probe_test`, 31 s. It reads the
  directory from `--qkv-probe`, else `$IGNIS_IMAGE_QKV_PROBE`, else that
  path.
- Without the dump it prints SKIP and passes, unless `IGNIS_GPU_PROFILE=1`:
  in the profile a missing dump fails the test (a skip is never green there,
  `docs/agents/testing.md`). The data is outside the repo, so only a machine
  that has it can run the profile.

Per head, against fp64 on the unquantized inputs:

| calls | heads | cosine min / p1 / median | centred min / p1 / median | rel. RMS median / max |
|---|---|---|---|---|
| target over [prefix ; target], the split | 384 | 0.99878 / 0.99911 / 0.99984 | 0.98515 / 0.99105 / 0.99975 | 1.8% / 5.4% |
| the same, one run | 384 | 0.99879 / 0.99908 / 0.99983 | 0.98284 / 0.99199 / 0.99974 | 1.9% / 5.3% |
| prefix segments (step 0) | 384 | 0.99796 / 0.99888 / 0.99967 | −0.147 / 0.000 / 0.99953 | 2.6% / 8.9% |

Re-measured 2026-10-14, after the review fix that adds each call's last row
to the rows checked: the cosines moved in the fourth decimal, and the
low-spread heads below (whose centred cosine is noise) from 11 to 15.

- Mechanics: 0 outputs over the bound in all 36 calls (worst 0.998 of
  it, on a 31-key text segment at block 31; the 4096-row target calls reach
  0.983 at most). A call with 31 keys has few terms, so its bound is mostly
  the P-flip term, which is tight by construction: an output sitting where
  P's BF16 rounding can go either way reaches it. The terms were not
  measured one by one. A new dump may therefore land at 0.99-1.0 on such a
  segment; read that as the flip term, and look at the centred and
  fp64-on-unquantized numbers for a real regression. The BF16 output equals the fp32 output rounded in every call. Every
  run and q8 buffer matches its definition bit for bit.
- **No head is below 0.99, and that is gated:** the test fails a head under
  `--min-cosine` (0.99) on the real data, so it works as a regression test.
  The centred cosine stays reported.
- **No head is below 0.99.** The worst is 0.99796, an 8-token text segment
  (edit, block 15, head 13). On the target calls it is 0.99878 (t2i,
  step 4, block 0, head 28).
- **The centred cosine falls below 0.99 only where the output barely varies
  from one query to the next.** The test reports that share as *spread*:
  the reference's norm with each channel's mean over the rows removed,
  over its whole norm.
  - 15 heads of the edit image block at block 0 have a spread of 0.00-2.2%
    of |O|, and their centred cosine falls as low as −0.15.
  - In 11 of them the spread is ≤ 0.15%, below one BF16 rounding of O (2^-9
    ≈ 0.2%), so the bf16 pipeline cannot hold that part either. Their
    error is the usual 2-3.4% of |O|.
  - Three heads with a spread of 7-13% sit at 0.985-0.989: head 7 of that
    block, head 30 of the image block at block 31, and head 3 of block 0
    on the target calls.
  - Which of these matter is for spec 04's end-to-end check.
- **The split costs next to nothing.** The target's K mean sits up to
  27σ from the prefix's in single channels (block 31; ≤ 6.8σ RMS over
  channels), including t2i's 31-token prefix. In every call the split's
  worst head is within 0.0003 of one run's cosine. Spec 04's K-mean fallback is
  not needed here.
- **Real Q has strong outlier channels; K's are milder.** Q is taken over
  the target rows, K over both runs after the prefix's mean.
  - Q's top channel RMS is up to 23.6× the head's median, and 23-32 heads
    of 32 are above 4×.
  - A 32-row Q block's amax is a median 24× the median channel RMS at
    block 0, about 5 INT8 levels per σ. Head 3 sits at 53-57× (blocks up
    to 68×). There channels 0 and 1, one RoPE pair at 0.7-0.85 of each
    other's RMS, set 66-86% of the blocks' amax.
  - At blocks 15 and 31 the median is 6.4-7.3×, and the worst heads are at
    14-19×. There one channel (104/105, 70) sets every block, and its
    partner carries only 0.08-0.18 of its RMS.
  - K stays ≤ 12× in channel RMS, with a block amax of 6-10× the median.
    The worst head is at 24× (blocks up to 31×: edit, block 0, head 7, the
    pair 72/73 equally).
- **Verdict: no migration.** Even with these outliers every head clears
  0.99 with margin. The synthetic arm, with 20× outliers in unrelated Q and
  K channels of flat Gaussian heads, is harsher than the DiT's own
  activations, so it stays report-only. If a fix were ever needed, Q's
  pattern points to exact scales per RoPE pair.

**Speed** (2026-10-13, `ignis_image_attn_bench`, GPU lock in timing mode,
card power limit 575 W). The SM clock and board power are NVML's means over
the second half of a window of at least 1 s. TOPS = 4 · queries · keys · 128
· 32 / time:

| call | queries × keys | ms | TOPS | SM clock | board |
|---|---|---|---|---|---|
| target, 1024², 0 images | 4096 × 4296 | 0.589 | 489 | 2650 MHz | 397 W |
| target, 1024², 1 image | 4096 × 8372 | 1.115 | 504 | 2606 MHz | 576 W |
| target, 1024², 10 images | 4096 × 45,299 | 5.813 | 523 | 2604 MHz | 574 W |
| target, 2048², 0 images | 16,384 × 16,584 | 7.952 | 560 | 2464 MHz | 579 W |
| target, 2048², 1 image | 16,384 × 20,660 | 9.866 | 562 | 2450 MHz | 576 W |
| target, 2048², 10 images | 16,384 × 57,587 | 27.13 | 570 | 2444 MHz | 575 W |
| self, N = 4096 | 4096 × 4096 | 0.549 | 500 | 2616 MHz | 570 W |
| self, N = 8192 | 8192 × 8192 | 2.048 | 537 | 2501 MHz | 578 W |
| self, N = 16,384 | 16,384 × 16,384 | 7.866 | 559 | 2422 MHz | 576 W |
| self, N = 32,768 | 32,768 × 32,768 | 30.71 | 573 | 2412 MHz | 576 W |
| self, N = 16,384, causal (half the work) | 16,384 × 16,384 | 4.135 | 532 | 2445 MHz | 580 W |
| image block 10 of 10, truncated keys | 4096 × 41,043 | 5.311 | 519 | 2576 MHz | 571 W |
| last text segment, causal | 160 × 41,203 | 0.879 | 123 | 2908 MHz | 470 W |

The quantizers at 1024²:
- `quantize_q`, 4096 tokens: 0.013 ms;
- the target's run: 0.080 ms;
- a 10-image prefix's run: 1.91 ms, once per job and layer.

A denoising step's 32 attention layers at 1024² take 21.8 ms with 0
condition images, 38.7 ms with 1 and 189 ms with 10. The quantizers are
13.5%, 7.6% and 1.6% of that.

**Against the bar (SA2++ on the 5090: 618-643 TOPS from 4K to 32K, the
survey's figure): 500-573 TOPS on the bar's own geometry (81% at 4K to 89%
at 32K, taking the bar's low and high ends), 560-570 at 2048² and 489-523 at
1024².**
- **Per clock.** Nsight Compute (clocks locked at 2.01 GHz, N = 16,384)
  puts the tensor pipe 72% busy: 233 TOPS/GHz against the 322.8 the bench
  measures for alternating INT8 and FP8 instructions. The stalls are the
  saturated math pipe and fixed-latency waits (28% each), then barriers
  (9%). The card sits at the 575 W limit and 2.4-2.6 GHz.
- **What tuning bought.** The first version ran at 458-472 TOPS. Taking a
  per-element branch out of the plain block's score conversion moved
  `branch_resolving` from 11% of the stall samples to 0.4%, worth +23%.
  The two-stage K/V ring and skipping the rescale by 1 measured neutral;
  they stay, being exact.
- **What would close the rest**, largest first:
  1. **1024² wave quantization.** 4096 queries × 32 heads are 1024 CTAs on
     340 slots, 3.01 waves. A split-KV tail or a persistent scheduler
     would recover about 10% there. 2048² is 12.05 waves.
  2. **Short causal segments** (160 × 41K: 64 CTAs) run at 123 TOPS. They
     are step 1 only, 28 ms per job at 10 images. Split-KV is the fix.
  3. **The prefix store** (1.9 ms for 41K tokens) reads K and V twice
     (stats, then codes) with 16-byte rows. It is once per job per layer.
  4. **Fusing the target's quantization** into the QKV GEMM's epilogue
     (spec 04): 7.6-13.5% of a step at 1024² with 0-1 images.

Tuning stopped here: speed is a target, not a gate.

**AC 8.** `attn8.cu`'s header and `kernel/NOTICE` (a new "Adapted ops (ADR
0051)" section) name SageAttention d1a57a5's files and FlashInfer v0.1.5,
under Apache-2.0. Nothing comes from CUTLASS 3.x or FlashInfer's
`csrc/fmha_v2`.

**AC 9.** The CTest runs in the profile's kernel stage. The Rust `--ignored`
test drives the bindings: quantize, store two runs, three calls checked
against fp64 of the unquantized inputs, and a refusal. The bench is a tool.

### The encoder attention (AC 5)

**Where.**
- ABI: `kernel/include/ignis_image_encoder_attn.h`:
  `ignis_image_encoder_attn`, `ignis_image_encoder_attn_last_error`.
- Code: `kernel/src/image/encoder_attn.cu`, an adapted op (ADR 0051 tier 2)
  of FlashAttention-2's forward (`Dao-AILab/flash-attention` v2.8.3,
  060c918, BSD-3-Clause).
  - FA2 is CuTe code, and CuTe is CUTLASS 3.x, so no FA2 code is compiled.
  - The kernel takes FA2's structure and its operation forms:
    `compute_attn_1rowblock`, `softmax.h`, `mask.h` and the tile for
    consumer SMs.
  - The data movement and the MMAs are written out as `mma.sync`,
    `ldmatrix` and `cp.async`.
  - The shared-memory helpers moved from `attn8.cu` to `attn_smem.cuh`
    (FlashInfer through SageAttention, Apache-2.0); both attentions
    include it.
- The upstream was fetched as data into this worktree's
  `.scratch/upstream/flash-attention`, pinned at 060c918. The checkout is
  sparse: `csrc/flash_attn/src`, `LICENSE` and `AUTHORS`. Nothing of it was
  built or run.
- Tests: the CTest `ignis_kernel_image_encoder_attn_test`
  (`kernel/tests/test_image_encoder_attn.cu`) and the Rust `--ignored` test
  `crates/core/tests/image_encoder_attn_gpu.rs` (bindings in
  `crates/core/src/image_encoder_attn.rs`). Speed:
  `ignis_image_encoder_attn_bench` (`kernel/tests/bench_image_encoder_attn.cu`,
  a tool).

**One sequence per call.** The encoder runs one prompt per job:
- Turbo has no CFG and no negative prompt (spec.md);
- the condition images' vision tokens sit inside the prompt's one causal
  sequence (spec 03).

So the op takes one sequence, with as many queries as keys, and no batch or
segments. It also takes non-causal calls, which the bench uses for the bar's
geometry.

**The ABI.**
- Q, K and V are BF16 base pointers with row strides. The fused QKV row
  [tokens][6144] is read in place, after the QK norm and mrope applied to
  it in place: q at column 0, k at 4096, v at 5120.
- GQA: query head h reads KV head h / 4.
- O is BF16 for the leaf, or fp32 for the test.
- The call takes a stream, allocates nothing, and is graph-capturable and
  deterministic.
- It needs no prepare call: 48 KiB of shared memory is within the default
  limit.

**The kernel.**
- **Tile:** 64 queries × 64 keys, 4 warps of 16 rows: FA2's causal tile for
  consumer SMs (sm86/sm89). FA2's own table sends sm120 to its datacenter
  tile, 128 × 64, which needs 64 KiB. This one uses 48 KiB and 182
  registers, so two CTAs fit an SM.
- **Pipeline:** FA2's.
  - Single K and V buffers. V of block n loads during QK^T, and K of block
    n − 1 during the softmax and PV. Two barriers per block.
  - Key blocks are walked from the last down. Masked blocks (the diagonal,
    a partial last block) take their own path behind one warp-uniform
    branch.
  - Causal CTAs launch heaviest first.
- **Softmax:** FA2's operation forms: the running max kept in raw scores,
  one FMA for the exponent, ex2.approx. Two changes:
  - the row sum adds the BF16-rounded P that multiply V, not FA2's fp32 P.
    So O is a convex combination of V's rows whatever the rounding, as in
    the 8-bit attention;
  - a row whose max did not move is rescaled by exactly 1.
- **Past the sequence:** tiles are zero-filled, never read. FA2 clears only
  V's first block.

**Mechanics (AC 5).** The reference is our own fp64 code. It replays the
kernel per 64-key block, in the kernel's order. s_j = q · k_j is exact in
fp64, since BF16 products are exact; the kernel's fp32 S'_j is not, and the
test bounds what follows from that:
- **S.** Per k16 step: 19 · 2^-25 · (|partial| + error + Σ|q_d k_d|). This
  is the BF16 accumulation probe of the instruction bench (P4 = 25): each of
  an instruction's 16 products and its accumulator loses at most 2^-25 of
  the largest addend, and the result is one fp32 rounding.
- **The max and the exponent.** The kernel's max m' is within em, the
  largest S error seen, of the replay's m. Its exponent fl(S'c − fl(m'c))
  is within dx_j = c (e_j + em)(1 + 2^-22) + 2^-23 (|mc| + |x_j|) of
  x_j = (s_j − m) c.
- **P.** P'_j = BF16(ex2.approx(x')), and ex2 is within 2^-20 of 2^x. So
  P'_j lies between the BF16 values of 2^(x_j ∓ dx_j)(1 ∓ 2^-20). The replay
  takes BF16(2^x_j). Where that interval holds two BF16 values, the
  numerator's error gains the width times |v| and the denominator's the
  width.
- **Rescales.** One happens when the max rises, or when it comes within
  2 em of rising, since the kernel may then rescale where the replay does
  not. Each adds ex2's error, the argument's two roundings and the
  products' one.
- **PV:** 19 · 2^-25 per k16 step.
- **The row sum:** 16 · 2^-24 of the sum per block (16 terms per lane), then
  the quad's two adds.
- **The end:** two roundings, then the quotient.

`--bound-factor` multiplies the whole bound (default 1). The same call with
BF16 output must equal the fp32 output rounded, bit for bit. Against plain
fp64 attention on the BF16 inputs, the relative RMS error per head is gated
at `--max-rel-rms` (default 2^-8, BF16's unit roundoff).

**The inputs** are what spec 03's text model hands the attention, laid out
as the fused QKV row:
- **q and k:** RMSNorm'd per head with weights shared by the heads (Qwen3's
  q_norm / k_norm), then RoPE (rotate-half, θ 5e6). Four low-frequency
  outlier channels make some heads sharp and the first token a sink.
- **V:** Gaussian, with per-channel scales and offsets.
- **Zero queries:** head 7 has one at every 97th token, so those rows
  attend uniformly.

**Guards.**
- Each call reads a device copy of the first `tokens` rows followed by 64
  rows of NaN: a kernel that multiplied V rows past the sequence would make
  NaN outputs. The guard covers V only: a K row past the sequence is masked
  to −inf whatever it holds, and a Q row past it is never stored.
- Outputs start as NaN.

**Rows checked:** 64 per head, as in the 8-bit attention's test: i · stride
+ h mod the row count, the stride odd and coprime with the count. So the rows
reach the end of the call and the 32 heads together cover every residue mod
128. Plus rows 0, 5, 63, 64, 127, 128 and the last.

The table gives the worst error over the bound, the bound's median over
|O|, and the worst head's relative RMS error against plain fp64 (fp32
output):

| call | tokens | worst / bound | bound / \|O\| | rel. RMS vs fp64 |
|---|---|---|---|---|
| causal, the smallest | 1, 2, 7 | ≤ 0.095 | 1.8e-6 | ≤ 6.7e-4 |
| causal, tile edges | 63, 64, 65, 127, 128, 129 | ≤ 0.989 | 2.9e-6-1.1e-5 | ≤ 5.8e-4 |
| causal, tile edges, a t2i prompt | 191, 192, 193, 300 | ≤ 0.994 | 2.7e-5-4.8e-5 | ≤ 5.7e-4 |
| causal, a 2,000-token prompt | 2,048 | 0.913 | 2.1e-4 | 3.8e-4 |
| causal | 4,097 | 0.971 | 2.7e-4 | 3.5e-4 |
| causal, 10 images and a 2,000-token prompt | 12,400 | 0.740 | 4.3e-4 | 3.0e-4 |
| causal, the longest 10-image prompt | 41,000 | 0.740 | 8.3e-4 | 2.9e-4 |
| non-causal | 1, 65, 300, 4,097 | ≤ 0.989 | 1.8e-6-3.7e-4 | ≤ 6.2e-4 |

- **0 outputs over the bound**, in every call.
- The BF16 output equals the fp32 output rounded, bit for bit, in every
  call.
- **The bound is tight.** With `--bound-factor 0.5`, every causal length
  from 63 up has 32-411 outputs over it, so the bound is within 2× of the
  worst error at every length. A worst near 1 fits one P rounded to the
  neighbouring BF16 value, which the bound adds in full.
- The median grows with the number of k16 steps, as the 8-bit attention's
  does: 8.3e-4 of |O| at 41,000 keys, below BF16's resolution.
- **What only the shorter calls pin.** One key's weight is about 1/N of
  |O|. Against the bound's median that is 4.9e-4 against 2.1e-4 at 2,048
  tokens (caught), 2.4e-4 against 2.7e-4 at 4,097 (borderline) and 8e-5
  against 4.3e-4 at 12,400 (not caught). So a one-key mask error, a causal
  limit off by one, is pinned up to about 2K tokens: by the small calls
  (1-300 tokens, bound ≤ 4.8e-5 of |O|, every tile edge) and the 2,048 one.

**Speed** (2026-10-13, `ignis_image_encoder_attn_bench`, GPU lock in timing
mode, card power limit 575 W). The SM clock and board power are NVML's
means over the second half of a window of at least 1 s. TOPS = 4 · N² · 128
· 32 / time, half of that for a causal call:

| call | N | ms | TOPS | SM clock | board |
|---|---|---|---|---|---|
| causal | 300 | 0.013 | 57 | | |
| causal | 2,048 | 0.182 | 189 | 2704 MHz | 501 W |
| causal | 4,096 | 0.679 | 202 | 2637 MHz | 575 W |
| causal | 12,400 | 6.093 | 207 | 2608 MHz | 575 W |
| causal | 16,384 | 10.71 | 205 | 2614 MHz | 577 W |
| causal | 32,768 | 42.70 | 206 | 2587 MHz | 575 W |
| causal | 41,000 | 66.40 | 207 | 2598 MHz | 575 W |
| non-causal | 4,096 | 1.400 | 196 | 2683 MHz | 573 W |
| non-causal | 8,192 | 5.466 | 201 | 2640 MHz | 577 W |
| non-causal | 16,384 | 21.51 | 205 | 2610 MHz | 575 W |
| non-causal | 32,768 | 85.65 | 205 | 2587 MHz | 576 W |

The encoder's 36 attention layers take:
- 0.47 ms for a 300-token t2i prompt;
- 6.6 ms for a 2,048-token one;
- 219 ms for 10 images and a 2,000-token prompt (12,400 tokens);
- 2.39 s at 41,000 tokens.

**Against the bar (FA2 on the 5090: 208-215 TOPS from 4K to 32K,
non-causal): 196-205 non-causal at the same lengths, 94-96% of it like for
like. Causal calls run at 202-207 from 4K up.**
- The card sits at its 575 W limit (the literature's numbers are at 600 W)
  and 2.6 GHz. That is ~80 TOPS/GHz, 92% of the BF16 MMA's 86.5 per clock
  (the instruction table above).
- A short prompt is latency-bound: 300 tokens are 32 heads × 5 query tiles,
  0.013 ms.
- Tuning stopped at the first version: one tile, measured once, near the
  bar and at the power limit.

**AC 8.** `encoder_attn.cu`'s header and `kernel/NOTICE` ("Adapted ops (ADR
0051)") name FlashAttention v2.8.3 (060c9188), the five files and the
BSD-3-Clause licence; NOTICE carries the licence's text. `attn_smem.cuh`'s
header gives FlashInfer through SageAttention (Apache-2.0). Nothing comes
from CUTLASS 3.x (FA2's CuTe is not taken) or from FlashInfer's
`csrc/fmha_v2`.

**AC 9.** The CTest runs in the profile's kernel stage. The Rust `--ignored`
test drives the bindings, each call against fp64 computed in Rust:
- causal and non-causal calls;
- fp32 and BF16 output, into a strided O;
- Q, K and V from the fused row and from separate buffers;
- a refusal, which comes back with the leaf's message.

The bench is a tool.

### The VAE ops (AC 6)

**Where.**
- ABI: `kernel/include/ignis_image_vae.h`: `ignis_image_vae_pack_conv_weight`,
  `ignis_image_vae_conv`, `ignis_image_vae_shortcut`,
  `ignis_image_vae_attn_workspace_bytes`, `ignis_image_vae_attn`,
  `ignis_image_vae_last_error`.
- Code: `kernel/src/image/vae_conv.cu` (with `vae_conv.cuh`, the GEMM the
  attention shares), `vae_shortcut.cu`, `vae_attn.cu`. All our own (ADR 0051
  tier 1). The convolution is written from the FP8 GEMM above; no
  convolution source (CUTLASS 2.x's fprop, ggml's `conv2d_implicit_gemm`)
  was read or copied. It includes `attn_smem.cuh`'s shared-memory helpers
  (FlashInfer through SageAttention, Apache-2.0), and `kernel/NOTICE`'s
  entry for them now names it.
- Tests: the CTest `ignis_kernel_image_vae_test`
  (`kernel/tests/test_image_vae.cu`, about 13 s) with the fixture
  `kernel/tests/fixtures/image_vae/` (`record.py`, `shortcuts.bin`,
  `provenance.json`), and the Rust `--ignored` test
  `crates/core/tests/image_vae_gpu.rs` (bindings in
  `crates/core/src/image_vae.rs`). Speed: `ignis_image_vae_bench`
  (`kernel/tests/bench_image_vae.cu`, a tool).

**Layout.** Activations are one image, BF16, NHWC. A band of whole rows is
one contiguous range, which is what spec 04's banded 2048² decode needs, and
the per-pixel channel norm (AC 7) reads one contiguous row. In diffusers'
`AutoencoderKLQwenImage21` every causal 3D convolution of a single frame is
already a 2D one (`QwenImage21CausalConv3d` is an `nn.Conv2d`), so nothing
temporal is left.

**The convolution.**
- **An implicit GEMM:** rows are output pixels, columns output channels. The
  reduction runs over the taps (ky, kx) in order and, within a tap, over the
  input channels in order, in k16 MMA steps (BF16 `mma.sync` m16n8k16, fp32
  accumulation).
- **Tile:** 128 pixels × 96 channels, 8 warps of 32 × 48, 32 BF16 of the
  reduction per stage, 3 cp.async stages. That is 42 KiB, so no prepare call
  and two CTAs an SM, at 99-102 registers. 96 divides every VAE channel count
  but 4, 64, 128 and 144.
- **The gather:** each thread keeps its pixels' receptive-field corners and
  walks its (tap, channel) position one stage at a time, loading 16-byte
  pieces of 8 channels. The RGBA input of the encoder's `conv_in` (4
  channels) takes 8-byte pieces. Pieces in the padding, past the image or
  past the reduction are zero-filled and never read.
- **The kinds:** 3 × 3 stride 1 pad 1; 3 × 3 stride 2 after the (0, 1, 0, 1)
  pad, which is the right column and the bottom row; and 1 × 1.
- **The upsample is fused.** With `upsample` the gather reads pixel
  (iy >> 1, ix >> 1) for position (iy, ix) of the 2× grid: PyTorch's
  nearest-exact 2×, never materialized. At 2048² the up3 resample's
  upsampled input would be 2048² × 288 BF16, 2.4 GB.
- **Epilogue:** the bias and then the residual are added in fp32, one
  rounding each, then one rounding to the output type (BF16 for the leaf,
  fp32 for the test).
  - **This differs from diffusers.** diffusers' BF16 path rounds the
    convolution (and, unless cuDNN fuses it, the bias add) to BF16 before
    it adds the residual (`x + h`) in BF16; the fused epilogue skips those
    intermediate roundings.
  - Policy for the leaf: fewer BF16 roundings, toward the fp32 recording.
    Spec 04's reference PNG is diffusers' fp32 decode, so skipping the
    intermediate roundings moves ignis toward it, not away. The expected
    PSNR floor is diffusers' own BF16 decode against that fp32 PNG, which
    spec 04 measures (#321, or with the recorder); a miss of 40 dB can be
    told from BF16's inherent loss only against that number.
- **Weights** are packed once at load, from PyTorch's [out][in][ky][kx] (the
  artifact's BF16 tensors as they are) to [out][ky][kx][in]
  (`ignis_image_vae_pack_conv_weight`).
- **Row windows for bands:**
  - x holds input rows [x_row0, x_row0 + x_rows) and the call writes output
    rows [y_row0, y_row0 + y_rows).
  - Padding is decided by the global row, so a band whose x holds its halo
    computes exactly what the whole image does.
  - A window missing a row the outputs read is refused, naming both ranges.
- **Deterministic:** each output's reduction order does not depend on the
  tile, the call or the band.

**Mechanics.** The reference is our own fp64 conv2d of the BF16 operands,
walked in the kernel's reduction order. The bound is the GEMM's, per k16 step
(the instruction bench's BF16 probes, P4 = 25):
`E_i = E_{i−1} + 19 · 2^-25 · (|S_{i−1}| + E_{i−1} + Σ_step |x w|)`, then one
fp32 rounding for the bias add and one for the residual add.
- **Shapes:** spec 01's `conversion.json` lists 88 VAE convolutions. Grouped
  by kind, upsample, channels and stage, they are 46 distinct calls, each run
  at a 1024² and a 2048² image.
- **The preset:** five calls also run at the 2752 × 1536 preset, in both
  orientations, so a swapped height and width shows. They are `conv_in`,
  the up0 resample, a stride-2 downsample, a 1 × 1 shortcut and `conv_out`
  at 2752 × 1536.
- **The `time_conv`s:** the six 1 × 1 `time_conv` weights never run for
  one frame (diffusers' resample skips them on the first chunk). They are
  covered as shapes all the same.
- **Pixels checked:** each call compares the corners, the edge midpoints and
  128 pixels at an odd stride coprime with the pixel count, with every output
  channel. The 128 cover every position of a 128-pixel tile. The weights go
  through the pack op, checked against the packing restated in the test, bit
  for bit.
- **Guards:**
  - x sits between rows of NaN, the packed weight has a NaN row after it,
    and the outputs start as NaN;
  - NaN outputs are counted over the whole output on the device, as are
    BF16 outputs that differ from the fp32 ones rounded.

Results, over all 102 calls:
- **0 outputs over the bound.** The worst error is 0.05-0.08 of the bound.
  The bound over |y| has a median of 4.6e-6 to 3.1e-4, growing with the
  reduction length.
- **BF16 == RN(fp32)** over every output, bit for bit, and **0 NaN**.
- **Why 0.05-0.08 and not near 1.** The bound is a worst case per step,
  while the real errors random-walk, as in the GEMM's test (3-14%).
  - With `--bound-factor 0.05`, every one of the 46 shapes at 1024² has
    outputs over it, so the bound is within 20× of the worst error.
  - One wrong term (one channel of one tap) is about 1/√K of |y|, 1% at
    K = 10,368, which is 30-2000× the bound.

**Bands and the upsample.**
- **Bands:** three calls are computed as three uneven bands of output rows,
  each from a copy of only the input rows it reads (between NaN rows):
  - the 144-channel resnet with its residual;
  - the 576-channel upsampling resample;
  - the 384-channel stride-2 downsample.

  At a 1024² image and at the preset, the banded output equals the
  whole-image call bit for bit. In each, the two bands with a top halo are
  refused when x lacks one halo row.
- **The fused upsample** (the 288-channel resample at 1024²) equals the
  same convolution over the input upsampled by the restated nearest-exact
  2×, bit for bit. That restatement is PyTorch's (the fixture below).

**The shortcuts** (`QwenImage21AvgDown3D` and `QwenImage21DupUp3D` for one
frame; the header has the index formulas).
- **One frame:**
  - With factor_t = 2 the frame is the last slot: AvgDown's front padding
    puts a zero frame first, and DupUp's `first_chunk` slice keeps the last.
  - So for AvgDown(96 → 192), output channel 2c is the mean of zeros and
    2c + 1 the 2 × 2 mean of channel c.
  - DupUp copies one source channel per sub-pixel.
- **The arithmetic:** AvgDown's mean is an fp32 sum from 0 in group order,
  then one division by the group. In BF16 the shortcut is rounded once.
  With `accumulate` the op adds it into y as diffusers' BF16 add does:
  RN(y + RN(s)).
- **The fixture is PyTorch's own outputs.** `record.py` runs diffusers'
  modules (torch 2.13.0+cu130, diffusers 0.41.0, CPU, one thread) at every
  channel configuration the VAE uses:
  - AvgDown 96 → 96, 96 → 192, 192 → 384, 384 → 768 and the identity
    768 → 768;
  - DupUp 1152 → 1152, 1152 → 576 and 576 → 288;
  - the upsample.

  The configurations are on small spatial sizes. Inputs span a wide range,
  with signed zeros and subnormals, so a different summation order would
  show.
  - The test's restatement equals PyTorch's fp32 outputs, and its BF16
    rounding equals PyTorch's BF16 outputs, bit for bit.
  - The op does too: fp32 bit for bit, and BF16 bit for bit (the gate is one
    BF16 rounding).
- **At the real stage sizes** (the 9 configurations at 1024² and 2048², two
  at the preset), the op equals the restatement bit for bit on six whole
  output rows per call. The accumulating form equals RN(y + RN(s)) bit for
  bit.

**The mid-block attention.**
- **Three passes per tile** of `row_tile` query rows, on a fixed workspace:
  - S = Q Kᵀ in fp32: the convolution's GEMM as a 1 × 1 convolution over
    the q rows, with the k rows as weight rows, both at stride 3·dim;
  - a softmax kernel, one CTA per row:
    - P = BF16(2^(S · c − fl(max S · c))), with c = fl32(scale · log2 e),
      one FMA and ex2.approx;
    - zeros past the tokens;
    - 1 / D, where D is the fp32 sum of the BF16 P in a fixed order: each
      thread's columns in order, a warp butterfly, then the 8 warps in
      order;
  - O = (P V) · (1 / D): the GEMM again, against Vᵀ (transposed once per
    call, zero-padded to n8 = tokens rounded up to 8), with 1 / D as a row
    scale in the epilogue.
- **Determinism:** a row's output does not depend on the tile it falls in.
- **Workspace:** Vᵀ [dim][n8] BF16, S [row_tile][n8] fp32, P [row_tile][n8]
  BF16 and 1 / D, each 256-byte aligned. The leaf picks `row_tile`; the
  bench below gives the trade.
- **Mechanics:** the replay of the encoder attention's test, without the
  online max:
  - S per k16 step as above;
  - the kernel's max within em of the replay's;
  - P's BF16 interval from ex2's 2^-20, its width counted where it holds two
    values;
  - PV per k16 step;
  - the row sum with ceil(n8 / 256) + 12 fp32 roundings;
  - 1 / D and the product, one rounding each.
- **Guards:** qkv sits before rows of NaN, and the workspace and the outputs
  start as NaN, so the zero padding the kernels must write is checked.

| D | latent (tokens) | row tiles | worst / bound | bound / \|O\| | rel. RMS vs fp64 |
|---|---|---|---|---|---|
| 768 | 1 × 1, 1 × 7 | 1024 / 1000, 4 / 3 | ≤ 0.14 | 1.5e-6-2.9e-6 | ≤ 3.3e-4 |
| 768 | 63 × 65 (4095), 64 × 64 | 1024 / 1000 | 0.46 / 0.35 | 6.5e-4 / 8.0e-4 | ≤ 2.9e-4 |
| 768 | 128 × 128 (16,384) | 2048 / 1000 | 0.42 | 1.1e-3 | 3.0e-4 |
| 1152 | 1 × 1, 1 × 7 | 1024 / 1000, 4 / 3 | ≤ 0.15 | 1.5e-6-1.7e-6 | ≤ 4.2e-4 |
| 1152 | 63 × 65 (4095), 64 × 64 | 1024 / 1000 | 0.27 / 0.25 | 1.4e-3 | ≤ 3.2e-4 |
| 1152 | 128 × 128 (16,384) | 2048 / 1000 | 0.40 | 1.8e-3 | 3.4e-4 |
| 1152 | 96 × 172 (16,512, the preset) | 2048 / 1000 | 0.41 | 1.5e-3 | 3.2e-4 |

- **0 outputs over the bound**, the BF16 output equals the fp32 output
  rounded, and the two row tiles (one leaving a partial last tile) give the
  same output, bit for bit, in every call. 0 NaN.
- With `--bound-factor 0.5` the worst reaches 0.92 of the halved bound.
- Relative RMS error against plain fp64 is ≤ 4.2e-4 (gate `--max-rel-rms`,
  default 2^-8).

**Speed** (2026-10-14, `ignis_image_vae_bench`, GPU lock in timing mode, card
power limit 575 W; SM clock and board power are NVML's means over the second
half of a window of at least 1 s). TFLOPS = 2 · output pixels · out · k² · in
/ time, BF16 out.

| call (decoder, 2048²) | at | in → out | ms | TFLOPS | SM clock | board |
|---|---|---|---|---|---|---|
| mid / up0 resnet | 128² | 1152 → 1152 | 2.435 | 161 | 2705 MHz | 571 W |
| up1 resnet | 256² | 1152 → 1152 | 8.618 | 182 | 2636 MHz | 575 W |
| up1 resample (upsample) | 256² → 512² | 1152 → 1152 | 33.81 | 185 | 2629 MHz | 576 W |
| up2 resnet | 512² | 576 → 576 | 8.607 | 182 | 2607 MHz | 575 W |
| up2 resample (upsample) | 512² → 1024² | 576 → 576 | 34.04 | 184 | 2617 MHz | 575 W |
| up3 resnet | 1024² | 288 → 288 | 9.359 | 167 | 2670 MHz | 571 W |
| up3 resample (upsample) | 1024² → 2048² | 288 → 288 | 35.98 | 174 | 2644 MHz | 577 W |
| up4 conv1 | 2048² | 288 → 144 | 21.83 | 143 | 2599 MHz | 578 W |
| up4 resnet | 2048² | 144 → 144 | 11.29 | 139 | 2580 MHz | 575 W |
| up4 conv_shortcut (1 × 1) | 2048² | 288 → 144 | 3.102 | 112 | 2235 MHz | 575 W |
| conv_out | 2048² | 144 → 4 | 5.144 | 8.5 | 2797 MHz | 575 W |

Each shape's time, times the number of times one pass runs it (the
`time_conv`s out):

| pass | image | convolutions | TFLOP | TFLOPS |
|---|---|---|---|---|
| decode | 1024² | 104.2 ms | 16.95 | 163 |
| decode | 2048² | 405.8 ms | 67.82 | 167 |
| encode | 1024² | 18.8 ms | 3.11 | 166 |
| encode | 2048² | 70.2 ms | 12.43 | 177 |

The mid-block attention, with each row tile's workspace:

| D | tokens | row tile 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 768 (encode) | 4096 | 0.791 ms, 30 MiB | 0.464 ms, 54 MiB | 0.387 ms, 102 MiB |
| 768 (encode) | 16,384 | 10.38 ms, 120 MiB | 7.373 ms, 216 MiB | 6.761 ms, 408 MiB |
| 1152 (decode) | 4096 | 0.835 ms, 33 MiB | 0.670 ms, 57 MiB | 0.558 ms, 105 MiB |
| 1152 (decode) | 16,384 | 11.58 ms, 132 MiB | 10.63 ms, 228 MiB | 9.309 ms, 420 MiB |

That is 122-139 TFLOPS (4 · tokens² · dim / time) with a 4096-row tile. A
decode's attention is under 3% of its convolutions.

The shortcuts are memory-bound at 550-860 GB/s. The largest, up3's DupUp into
2048² × 288 with the accumulating add, takes 7.0 ms.

**Against the bar (BF16 on the 5090, ~210-240 TFLOPS): 167-186 TFLOPS on
the 3 × 3 calls of 192 channels and more at 256² and larger, 70-89% of
it.**
- **Per clock.** The card runs at its 575 W limit and 2.6 GHz. That is ~70
  TFLOPS/GHz, 81% of the BF16 MMA's 86.5 per clock (the instruction table
  above).
- **A whole decode's convolutions** run at 163-167 TFLOPS: 104 ms at 1024²
  and 406 ms at 2048².
- **What would close the rest**, largest first:
  1. **The 144-channel stage** (up4: 20% of a 2048² decode's convolution
     time) runs at 132-143 TFLOPS. The 96-wide tile pads 144
     channels to 192; a 144-wide (or 48-wide) tile configuration would
     recover about 25% there.
  2. **`conv_out`'s 4 channels** fill 4 of 96 columns: 5.1 ms at 2048²
     (1.3%). A narrow tile, or SIMT, is the fix.
  3. **The 1 × 1 convolutions** (K = in channels, a short reduction) run at
     93-168 TFLOPS where they are large enough to measure (the latent-sized
     ones are launch-bound), and the RGBA `conv_in` (K = 36 in a 64-wide
     reduction) at 46.
- Tuning stopped at the first version: one tile configuration, measured once.

**Memory.** The convolution and the shortcuts allocate nothing and need no
workspace. The attention's workspace is the table above: 132-420 MiB at
2048² for D = 1152, and 33-105 MiB at 1024². The widest decoder activations
at 2048² are 2048² × 288 BF16, 2.4 GB: up3's resample output and up4's
input. 2048² × 144 is 1.2 GB. The banded decode (spec 04) bounds them.

**AC 8.** Nothing is adapted in the VAE ops beyond `attn_smem.cuh`'s helpers,
already listed. `kernel/NOTICE`'s entry for those helpers now names the
convolution. Nothing comes from CUTLASS 3.x or FlashInfer's `csrc/fmha_v2`.

**AC 9.** The CTest runs in the profile's kernel stage. The Rust `--ignored`
test drives every binding:
- the pack and the three convolution kinds, with the upsample, RGBA input,
  bias, residual, fp32 and BF16 output, and a band, against fp64;
- AvgDown in fp32 and DupUp accumulating in BF16, bit for bit;
- the attention in fp32 and BF16, against fp64;
- refusals that come back with the leaf's message.

The bench is a tool.

### The elementwise ops (AC 7)

**Where.**
- ABI: `kernel/include/ignis_image_dit.h`: `ignis_image_dit_rope_positions`
  (host), `ignis_image_dit_rope_table`, `ignis_image_dit_modulated_norm`,
  `ignis_image_dit_rms_norm`, `ignis_image_dit_gated_residual`,
  `ignis_image_dit_qk_norm_rope`, `ignis_image_dit_act` (SiLU, GELU-tanh,
  SwiGLU), `ignis_image_dit_last_error`; and `ignis_image_vae_rms_norm` in
  `ignis_image_vae.h`.
- Code: `kernel/src/image/dit_ops.cu` and `vae_norm.cu`. All our own (ADR
  0051 tier 1).
  - The leaf's existing norm and RoPE ops are vendored ninfer code
    (`kernel/vendor`, never edited) with the text model's layouts
    (rotate-half RoPE, one position per token), so these are new.
  - FlashInfer's `fused_dit_layernorm.cuh` and `fused_qk_rmsnorm_rope.cuh`
    were not read or copied.
- Tests: the CTests `ignis_kernel_image_elementwise_test`
  (`kernel/tests/test_image_elementwise.cu`, about 1 s) and
  `ignis_kernel_image_elementwise_rope_oracle_test` (the same file, `--cases
  rope-oracle`), and the Rust `--ignored` test
  `crates/core/tests/image_elementwise_gpu.rs` (bindings in
  `crates/core/src/image_dit.rs`, and `rms_norm` in `image_vae.rs`). Speed:
  `ignis_image_elementwise_bench` (`kernel/tests/bench_image_elementwise.cu`,
  a tool).

**What diffusers 0.41.0 does** (`transformer_qwenimage21.py`), and where
these ops differ:
- **No shift.** `_modulate` (lines 604-613) is `x * (1 + scale)` with
  `scale, gate = mod_params.chunk(2)`, and `norm_out`
  (`QwenImage21AdaLayerNormContinuous`) is "scale only". The shared
  modulation's four chunks are [mod1.scale, mod1.gate, mod2.scale,
  mod2.gate]. So the op is LayerNorm × (1 + s), without a shift.
- **The gate is tanh(gate) · y** (lines 641 and 644:
  `hidden_states + img_gate1.tanh() * attn_output`).
- **The row choice** (`_select_modulation_rows`): text and condition-image
  tokens take the t = 0 row, the target's tokens the t row. The target block
  is the sequence's last, so a call takes both rows and `first_target`. The
  leaf's prefix pass reads only t = 0, a denoising step only t.
- **RoPE**: the complex product of adjacent pairs (`apply_rotary_emb_qwen`
  with `use_real=False`), positions from `QwenImage21Rope.forward`, and the
  table from `rope_params`, computed in fp32 on the CPU at init.
- **One rounding instead of diffusers' BF16 intermediates.** Each op works
  in fp32 from its BF16 inputs and rounds once to the output.
  - diffusers' BF16 path rounds the norm's output, `1 + scale`, `tanh(gate)`
    and the product before the residual add, the normed q and k before RoPE
    (and RMSNorm's x · r before the weight), and SiLU's output before
    SwiGLU's product.
  - Each is one BF16 rounding, up to 2^-8 of its value; `1 + scale`'s
    carries over whole to the modulated value.
  - The VAE norm skips three BF16 roundings (the normalized x, × √C, × γ)
    and SiLU's.
  - So spec 04's teacher-forced comparison sees these ops' rounding, not
    diffusers'.
- The modulation rows are fp32 (the GEMM's fp32 output; diffusers holds them
  in BF16). The norms' weights are BF16, as stored.

**The ops** (the header has the definitions):
- **Row ops** (the two norms and the activations): one CTA per token, each
  thread on the row's 8-element chunks, sums in a fixed order (each thread's
  elements, a warp butterfly, then the 8 warps).
- **The modulated norm**: the row is staged in shared memory, and the
  variance takes two passes (the mean, then the centred sum of squares). The
  one-pass E[x²] − μ² cancels on a row with a large mean; the test has rows
  at −96 ± 0.75.
- **The zero-centred RMSNorm**: (w + 1) in fp32. `eps` is an argument
  (`txt_in` passes 1e-6; the class default is 1e-5).
- **The gated residual**: one fma per element, `fma(tanhf(g), y, x)`, BF16,
  in place.
- **The QK norm with RoPE**: one warp per (token, head), q and k in one
  call. It rewrites the fused QKV row in place (q at column 0, k at 4096, v
  untouched), the addressing the 8-bit attention reads.
- **The activations**: SiLU = x / (1 + e^-x). GELU-tanh is computed as x /
  (1 + e^-2z), which equals 0.5 x (1 + tanh z) without its cancellation at
  negative x. SwiGLU reads separate gate and up rows, or the fused
  [tokens][24576] output with one row stride.
- **The VAE norm**: one warp per NHWC pixel row; fl32(√C), as PyTorch takes
  the Python float; the norm clamped at 1e-12; SiLU fused; y may be x.
- Every rounding step is an `_rn` intrinsic, so nvcc's FMA contraction
  cannot change the sequence the bound follows. `expf`, `tanhf` and `rsqrtf`
  are CUDA's accurate functions (no fast math); sqrt and division are IEEE.

**The fused quantization (AC 3).** The norms and activations take an E4M3
output: the BF16 output is staged in shared memory, then quantized with
`fp8_quant.cuh`'s steps. The result is `ignis_image_quantize_tokens` of the
BF16 output, bit for bit (tested over every output).
- **Fused** for the inputs of q/k/v and gate/up (the two modulated norms),
  `proj_out` (norm_out), `txt_in`'s two linears (the zero-centred norm,
  GELU), down (SwiGLU) and the timestep path's linears (SiLU).
- **Not fused**: the attention output into `to_out` (AC 4's kernel; the
  separate kernel costs 0.3% of a block), `img_in`'s latents and the
  timestep sinusoid into `linear_1` (both quantized by the separate kernel
  on [T][64] and [1][256] rows: negligible).

**RoPE.**
- **Positions**: a host call, from the blocks' starts and sizes. diffusers
  finds the starts in the image-token mask; the leaf knows them. A text
  token advances the position on all three axes. A block takes frame =
  the position, height and width on a grid centred on zero, then advances
  the position by max(h, w). Positions outside [−1024, 8192), diffusers'
  table rows, are refused: beyond them diffusers' indexing would read the
  negative rows or fail. So are a block taller or wider than the sequence,
  or starting past it, checked before the height × width product (a u32 pair
  could overflow it and write out of bounds). #321 maps these refusals to a
  400, never a 500.
- **The row ops' width**: cols at most 24560, not 24576: the row in shared
  memory (2 B per column) plus the 32-byte reduction buffer must fit the
  48 KiB a launch gets without an opt-in. The test launches at the cap.
- **Aliasing**: `out` may be `x` (gated residual) and `y` may be `x` (the VAE
  norm) for BF16 output only; an fp32 row is twice as wide.
- **Frequencies**: PyTorch's fp32 values, f_i = fl32(1 / fl32(10000^fl32(2i
  / d))).
  - The exponent is rounded first, which puts f up to 5 ulp from
    fl32(10000^(−2i / d)).
  - torch's `powf` differs from that formula at one pair: d = 56, i = 11
    (0x1.b7886ep-6, one ulp above it).
  - The 64 values are constants in `dit_ops.cu`; the test restates the
    formula and the exception.
- **The table**: fl32(position · f), as torch's outer product rounds it,
  then cos and sin in fp64 rounded to fp32.
- **The bound is one ulp per element.** The angles are bit-identical by
  construction. Ours are then correctly rounded, and torch's CPU fp32 cos and
  sin are within one ulp.
- **Results:**
  - The table op against the host's table at every position of
    [−1024, 8192), on every axis and pair (1,179,648 values): equal, bit
    for bit.
  - The positions op against diffusers' forward restated in its mask form,
    for seven layouts: equal. They are t2i, edit, adjacent blocks, an image
    first, text only, trailing text, and ten condition images with the
    172 × 96 preset (57,505 tokens). Also a layout worked by hand, and the
    refusals.
  - **Against the oracle's `rope.npy`** (spec 01's recordings):

    | case | tokens | equal | at one ulp | worse |
    |---|---|---|---|---|
    | t2i (text 31, target 64 × 64) | 4127 | 527,739 | 517 (0.10%) | 0 |
    | edit (text 8, image 64 × 64, text 18, target 64 × 64) | 8218 | 1,046,782 | 5,122 (0.49%) | 0 |

    The largest difference is 2^-24.
  - The comparison sees the constants: with the (56, 11) frequency one ulp
    lower, diffusers' own module puts 558 elements of the edit table more
    than one ulp from `rope.npy`.

**Mechanics.** The reference is our own fp64 code that follows each kernel's
operations in its order and tracks a bound on its error:
- every fp32 operation adds 2^-24 of its result;
- `expf`, `tanhf` and `rsqrtf` add 2 ulp (CUDA Math API);
- an input's error goes through the operation's own sensitivity;
- a sum in a fixed tree adds γ_d of Σ|terms|, with d each thread's element
  count plus 5 butterfly levels plus 8 warps.

The inputs' rows mix profiles: plain, tiny, × 40, an offset, a large offset
with a small spread (the mean cancels), outlier channels, zero rows and
constant rows.

| op | calls | worst / bound |
|---|---|---|
| modulated norm | 1, 7, 4127 (prefix 31), 8218 (prefix 4122), 16,512 tokens × 4096 | 0.04-0.10 |
| zero-centred RMSNorm | 1, 31, 2048 × 4096 | 0.10-0.18 |
| SiLU | 2, 33 × 4096 | 0.72-0.82 |
| GELU-tanh | 31, 2048 × 4096 | 0.80 |
| SwiGLU | 1, 4127 × 12288 separate; 4096, 16,512 fused | 0.59-0.84 |
| gated residual | 1, 7, 4127, 16,512 × 4096 | 0.84-0.99 |
| QK norm + RoPE | 1, 4127 (t2i), 8218 (edit), 16,584 tokens × 64 heads | 0.26-0.36 |
| VAE norm, with and without SiLU | 4096 pixels × every channel count (96-1152), 1 pixel, 1024² × 144, 128² × 1152 | 0.12-0.30 |

- **0 outputs over the bound.** BF16 equals the fp32 output rounded over
  every output; the in-place forms equal the out-of-place ones; the E4M3
  output equals the standalone quantization of the BF16 one; 0 NaN (the
  outputs start as NaN); v untouched.
- **The row choice.** In [prefix ; target] calls the target's rows equal the
  all-t call's and the prefix's the all-t0 call's, bit for bit. The
  reference with the two rows swapped puts 98-99.9% of the checked outputs
  over the bound (the negative control).
- **Tightness.** With `--bound-factor 0.05` every op has calls over the
  bound (all but the two one-token LayerNorms), so each bound is within 20×
  of the worst error. The residual and the activations sit at 0.6-0.99:
  their bounds are one rounding plus CUDA's 2-ulp transcendentals.

**Speed** (2026-10-14, `ignis_image_elementwise_bench`, GPU lock in timing
mode, zero inputs; reads plus writes over time). At 2048² every op runs at
1.5-1.66 TB/s, 84-93% of the card's 1.79 TB/s; at 1024² part of the data
stays in L2.

| op | 1024² (4096 tokens) | 2048² (16,384) |
|---|---|---|
| modulated norm, E4M3 out | 0.019 ms | 0.121 ms |
| modulated norm BF16, then the separate quantization | 0.029 ms | 0.295 ms |
| gated residual, in place | 0.017 ms | 0.256 ms |
| QK norm + RoPE, in place | 0.032 ms | 0.357 ms |
| SwiGLU from the fused gate / up, E4M3 out | 0.150 ms | 0.622 ms |
| SwiGLU BF16, then the separate quantization | 0.279 ms | 1.160 ms |
| **a step**: 32 × (2 norms, 2 residual adds, QK, SwiGLU) + norm_out | **8.1 ms** (12.9 unfused) | **55.6 ms** (84.1 unfused) |

- The fused quantization saves 37% (1024²) and 34% (2048²) of a step's
  elementwise time. At 1024² the 8.1 ms sit beside the linears' 115.5 ms
  (AC 2) and the attention's 21.8 ms (AC 4).
- The RoPE table: 0.013 ms for 4127 tokens, 0.123 ms for 57,505; once per
  job.
- The VAE norm with SiLU at a 2048² decode's stage sizes: 0.03 ms (128² ×
  1152) to 3.15 ms (2048² × 288), 1.63 ms at 2048² × 144; 1.5 TB/s.
- Not tuned. The next saving would be folding each residual add into the
  norm after it (one read of x less); it was not built.

**Defaults taken** (the owner may revisit):
1. The modulation rows are fp32; diffusers holds them in BF16.
2. The residual stream is BF16, as in diffusers' BF16 path.
3. The E4M3 output is the quantization of the BF16 output, not of the fp32
   value, so the seam is tested bit for bit.
4. RoPE positions outside [−1024, 8192) are refused.
5. GELU-tanh is computed as x · σ(2z).
6. The QK norm takes q and k in one call, in place, with 32 heads of 128.

**AC 8.** Nothing is adapted: the headers say OURS, and `kernel/NOTICE` has
nothing to add.

**AC 9.** The CTests run in the profile's kernel stage. The oracle case
prints SKIP without the oracle (`--oracle DIR`, `$IGNIS_IMAGE_ORACLE`, or
`F:/ai/models/ignis-image-2.1-turbo/oracle`) and fails under
`IGNIS_GPU_PROFILE=1`, as the real Q/K/V probe does; the Rust test's oracle
part goes through `gpu_profile::skip_or_fail`. The Rust `--ignored` test drives every binding:
- the positions worked by hand and the table, and the oracle's t2i table
  when it is there;
- the modulated norm with its row choice in fp32, BF16 and E4M3 (against
  `quantize_tokens`);
- the RMSNorm, the residual (in place), the QK norm in place on a fused row
  (v untouched), the three activations and the VAE norm with SiLU, against
  fp64;
- refusals that come back with the leaf's message.

The bench is a tool.
