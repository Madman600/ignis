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
