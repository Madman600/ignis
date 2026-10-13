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
| encoder attention | FlashAttention-2 forward, bf16, causal, GQA 32/8, D=128 | `Dao-AILab/flash-attention` `csrc/flash_attn/src`, c10 and dropout stubbed (BSD-3) | adapted |
| VAE convolution | implicit-GEMM conv2d, bf16, the shapes in spec 01's `conversion.json` (3×3 stride 1 pad 1, 3×3 stride 2 after a (0,1,0,1) pad, 1×1) | CUTLASS 2.x Ampere implicit-GEMM fprop and ggml `conv2d_implicit_gemm` as reading or source (BSD-3 / MIT) | ours or adapted |
| VAE mid-block attention | one head, D = 768 (encoder) / 1152 (decoder), over the latent grid: row-tiled QKᵀ, softmax, PV on the bf16 GEMM, with a fixed workspace | — | our own |
| DiT elementwise | LayerNorm without affine × (1 + scale), with the per-token choice of the t row or the t=0 row; `x + tanh(gate) · y`; QK RMSNorm + 3-axis RoPE (complex form, adjacent pairs); SwiGLU; GELU-tanh; zero-centred RMSNorm (w + 1) | our existing norm/RoPE ops extended. FlashInfer `fused_dit_layernorm.cuh` and `fused_qk_rmsnorm_rope.cuh` as reading or source (Apache-2.0) | ours or adapted |
| VAE elementwise | channel RMS norm, SiLU, nearest-exact 2× upsample, the AvgDown / DupUp shortcuts | — | our own |
| sampler | Euler update `x += (σ_next − σ) · v` on the device | — | our own |

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
   - **Prefix handling.** The prefix stored in step 1 and read in step 2 gives
     the same result as quantizing [prefix ; target] at once within the
     mechanics bound. This tests the shared K mean and the accumulator
     rescale.
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

### The FP8 W8A8 GEMM and the activation quantization (AC 2, AC 3)

**Where.**
- ABI: `kernel/include/ignis_image_gemm.h`: `ignis_image_gemm_prepare`,
  `ignis_image_quantize_tokens`, `ignis_image_gemm_w8a8`,
  `ignis_image_gemm_last_error`.
- Code: `kernel/src/image/` (`fp8_gemm.cu`, `fp8_gemm_internal.h`,
  `fp8_quant.cu`, `fp8_quant.cuh`, `image_mma.cuh`, `image_common.*`). All our
  own (ADR 0051 tier 1); CUTLASS 2.x sm80 and vLLM's `scaled_mm_sm120` were
  read, nothing was copied, so `kernel/NOTICE` has nothing to add.
- Tests: the CTest `ignis_kernel_image_gemm_test`
  (`kernel/tests/test_image_gemm.cu`) and the Rust `--ignored` test
  `crates/core/tests/image_gemm_gpu.rs` (bindings in
  `crates/core/src/image_gemm.rs`). Speed: `ignis_image_gemm_bench`
  (`kernel/tests/bench_image_gemm.cu`, a tool).

**The activation contract (AC 3).** Activations reach the GEMM as E4M3FN
codes `[tokens][cols]` plus fp32 `scales[tokens]`:
- `scales[t] = amax_t / 448`, or 1 for an all-zero row;
- `codes[t][c] = E4M3(x[t][c] · (448 / amax_t))`, rounded to nearest even
  and saturated, each step one fp32 operation.

`ignis_image_quantize_tokens` is the standalone producer.
`fp8_quant.cuh` holds the device steps, so a norm, modulation, SwiGLU or
attention-output epilogue can emit the same pair and drop the separate pass.
For now it is a separate kernel, whose cost is in the speed table below.

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

Every shape runs at 1, 2, 17, 129, 4096 and 16,512 tokens, and the DiT's
shapes also at 41,000 (a 10-image prefix). At large token counts the check
samples 161 rows covering every residue mod 128 and 385 columns covering
every residue mod 256.

The bound comes from the M1 probes. Per instruction, its 32 products and its
accumulator input each lose at most 2^-25 of the largest addend, and its
result one fp32 rounding:

`|y − ref| ≤ F · ⌈cols/32⌉ · 35 · 2^-25 · Σ|terms| + |ref| · (2^-23 + 2^-8 for BF16)`

`F` is `--bound-factor`, default 1.

**Result:** 0 outputs over the bound, at every shape and token count.
- fp32 outputs use 0.1-10% of the bound.
- BF16 outputs use 27-99%; that is BF16's own half ulp.
- The six other tile configurations pass the same check at the edge shapes.
- Run time: about 2 minutes.

**Quantization error**, against the unquantized product. The reference is
fp64 of BF16 x times BF16 W, with W quantized per row as a converter would.
Shapes are 4096 tokens × (4096 × 4096) and (4096 × 12288):

| x | W8A8 relative RMS | cosine | weights only |
|---|---|---|---|
| Gaussian | 3.75-3.81% | 0.99927-0.99930 | 2.66-2.67% |
| Gaussian, 8 channels at 20× | 3.64-3.67% | 0.99933-0.99934 | 2.65-2.70% |

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
   5% at 16,512 tokens.
