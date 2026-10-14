# 04 — the image leaf: the DiT, the condition prefix, the sampler, the VAE, and the end-to-end check

GitHub: #321. ADR 0050, ADR 0009 (Step ABI). Feature:
`spec.md`. Builds on 01, 02 and 03.

## Problem

With an artifact, kernels and an encoder, nothing yet makes an image:
- no DiT forward;
- no condition prefix;
- no sampler or VAE;
- no Step ABI calls a server can drive.

The only defence against a consistent but wrong implementation is a
whole-model comparison with diffusers, as ADR 0043 found for Flash-Next.

## Solution

A leaf program for the family `Qwen-Image-2.1` (`kernel/src/image/`), driven
from Rust one call at a time (ADR 0009):

| call | does |
|---|---|
| `encode_condition` | the encoder (spec 03); the VAE encode of each condition image; the DiT over the **prefix tokens only**, with the t=0 modulation, storing the condition prefix's keys and values in 8 bits (spec 02). Returns a prefix handle |
| `denoise_step` | one denoising step over the **target tokens only**: `img_in`, 32 blocks with modulation at σ, attention over [prefix ; target], `norm_out`, `proj_out`, then the Euler update on the device. The velocity can be copied out for tests |
| `vae_decode` | latents × std + mean, the decoder, clamp, RGBA bytes to the host |
| `vae_encode` | RGBA → latents, (z − mean) / std; used by `encode_condition`, exposed for tests |
| `release_condition` | frees a prefix |

The prefix never depends on the target, for three reasons:
- its queries never see the target's keys;
- its modulation is fixed at t = 0;
- its RoPE positions depend only on the tokens before them, because the
  target block comes last (diffusers' `QwenImage21Rope`).

So it is computed once in
`encode_condition` and reused by every denoising step **of every image the
job makes** (`n` images, one prefix). diffusers instead computes it inside its
first denoising step; the result is the same up to rounding.

## Seams

- **Load and plan.** The plan has three parts:
  - the weights (DiT 7.1 GB and encoder 7.0 GB in FP8, the vision tower and
    VAE, about 16 GB in all);
  - one sequential arena, sized for the largest preset (2400 × 1792, 16,800
    target tokens) plus the largest prefix segment;
  - the **condition budget**, which is whatever the VRAM plan leaves, counted
    in condition-prefix tokens of ~256 KiB.

  The plan line prints all three. `image.condition_budget_tokens` (spec 05)
  can only lower the budget. `encode_condition` refuses a prefix above the
  budget: the server scales images down first, so a refusal is a bug.
- **The joint sequence and RoPE.**
  - **Prefix tokens:** the encoder's outputs at text positions go through
    `txt_in`. Each image-pad slot becomes 2 × 2 latent tokens of its
    condition image, through `img_in`.
  - **Target tokens:** they are appended after the prefix.
  - **Positions:** the 3-axis RoPE positions follow diffusers' `pos_embed`
    for the layout. The block and segment structure follows
    `build_token_metadata` and `_qwenimage21_prefix_segments`.
- **The 8-bit attention's K mean** (spec 02 AC 4). Each step stores the
  target's K/V run smoothed with the prefix's K mean. That is exact, but the
  target's INT8 blocks lose resolution when its keys sit far from that mean;
  with 0 condition images the prefix is ~200 text tokens. Spec 02's "As
  built" measures the cost on synthetic inputs. If this leaf's real data
  shows target-K error, the exact alternative is:
  - the target run gets its own K mean m_t;
  - the attention adds q · (m_t − m_p) per query, times the softmax scale,
    to the exponents of run B's keys, so every key is scored as against
    m_p.

  It is not built; the end-to-end check here decides.
- **Modulation.** The timestep embedder (256 sinusoidal features, then an MLP)
  gives `temb` at σ, and at 0 for the prefix. One shared linear produces the
  four 4096-wide chunks, scale and gate for attention and for the MLP, that
  every block uses.
- **The sampler** (Rust):
  - the sigmas are the artifact's `sample_sigmas` followed by a terminal 0;
  - the DiT's timestep is the scheduler's divided by `num_train_timesteps`,
    as the pipeline's `timestep / 1000` computes it on the GPU:
    fl(fl(1000σ) · fl(1/1000)), within an ulp of σ (321e);
  - the update is `x += (σ_next − σ) · v`.
  For this checkpoint the scheduler config has `use_dynamic_shifting: false`
  and `shift: 1.0`, so the schedule is the same at every size. The loader
  refuses an artifact whose scheduler config says otherwise.
- **Noise:**
  - normally a standard normal tensor (1, 64, H/16, W/16) from ignis's RNG
    and the job's seed, drawn on the host and uploaded;
  - in tests, read from spec 01's noise file.
- **The VAE decode at 2K.** The decoder's widest activations at 2048² are
  2.4 GB each (288 ch). After the mid block, which is global, the decoder runs in
  horizontal bands with exact halos. Its norms are per pixel, so banding
  changes nothing.
- **Cancellation.** Between any two calls. The server releases the prefix of
  a cancelled job.

## Acceptance

1. The image family loads from the artifact. The plan line shows weights,
   arena and condition budget. A load whose weights and arena do not fit the
   card is refused, naming the shortfall.
2. **Sampler:** the timesteps equal the recorded ones. Given the recorded
   velocity, the Euler update reproduces the recorded output latents in fp32.
3. **t2i, step by step,** at 1024² and at 2048²: fed each step's recorded
   input latents (teacher-forced), `denoise_step`'s velocity has a relative L2
   error within spec 01's `thresholds.json` bound at every one of the 8 steps.
   The 1024² case is the first milestone.
4. **Edit, step by step:** the same holds for the edit case, with the
   condition prefix built by `encode_condition`. In ACs 3 and 4,
   `denoise_step` reads ignis's own encoder's output via `encode_condition`,
   never the recorded fp32 `prompt_embeds`.
5. The DiT block outputs at blocks 0, 15 and 31 are written beside the
   recorded ones as a diagnostic report, without a threshold: on a failure of
   AC 3 or 4 they show where the error comes from.
6. **VAE decode** of the recorded final latents gives the recorded PNG at
   ≥ 40 dB PSNR per channel, alpha included. The banded decode equals the
   unbanded one bit for bit at 1024². A 2048² decode runs within the arena.
7. **VAE encode** of the edit case's condition image matches the recorded
   condition latents within a stated bf16 bound.
8. A job with `n` = 3 computes one prefix and three images. Releasing it, and
   cancelling a job between denoising steps, leaves the leaf's device memory
   where it was before the job.
9. **Free run, for the eye:** each case run end to end from its noise file
   produces a PNG. It is saved next to diffusers' fp32 PNG and linked in "As
   built", so the owner can look.
10. **One timing run, at the end:** t2i at 1024² and 2048²; edit with 1 and
    4 condition images at 2048².
    - **Per-phase times:** encoder, prefix, each denoising step, VAE decode.
    - **Per kernel class:** throughput against the literature bars (FP8 GEMM
      ~690-740 TFLOPS, 8-bit attention ~620-640 TOPS, bf16 ~210-240).
    - **The whole DiT loop:** against ~1.1 s at 1024² and ~6 s at 2048².

    Written in "As built" and in a finding. It is information, not a gate.
11. ACs 3-10 run once, at the end of the phase, under the GPU lock. The bound
    in ACs 3-4 is a bug detector, not a gate (ADR 0050). A step over it goes
    to the owner with AC 5's report, and the ticket is not closed on it
    without the owner.

## As built (321a, 2026-10-14: ACs 1, 2, 3 at 1024², 5; the DiT, the t2i prefix, the sampler)

Status: **AC 1 done** (load, plan line, refusal naming the shortfall). **AC 2
done** (bit for bit). **AC 3 at 1024² done**: all 8 steps within the bound,
at the sim yardstick's level. **AC 5** written for t2i (below). AC 3 at
2048², AC 4, 6-10 are not done. AC 11 (the end-of-phase run) still applies
to all of them.

- **Where.**
  - The DiT's program: ABI `kernel/include/ignis_image_transformer.h`, code
    `kernel/src/image/transformer.cu`, ours (ADR 0051 tier 1). It has
    `_modulation`, `_txt_in`, `_img_in`, `_blocks` and `_head`, plus
    `_workspace_bytes` (about 60 KiB a token).
    - `_blocks` runs blocks `[first, end)` in place on the fp32 residual
      stream. With `target` 0 it is the prefix pass: it writes block b's
      K/V run and runs one attention call per segment. With `target` 1 it
      is a target pass over the prefix's runs.
    - `_head` is norm_out, proj_out, then the Euler update on the device.
    - Its three ops of its own are exposed for the tests:
      `_residual_norm` (the gated residual fused with the modulated
      LayerNorm, E4M3 from fp32), `_linear_bf16` (the timestep path's BF16
      linears) and `_quantize_f32`.
    - Everything else is spec 02's ops: the GEMM, the 8-bit attention, the
      QK norm with RoPE, the RoPE table, SwiGLU, GELU and the zero-centred
      RMSNorm.
  - Rust: `crates/core/src/image_transformer.rs` (`TransformerConfig::parse`,
    the bindings, `TransformerWeights` over the full bind) and
    `crates/core/src/image_leaf.rs`. The latter holds:
    - `Sampler` and `euler_step`;
    - `ImagePlan`;
    - with `cuda`, `ImageLeaf`, whose calls are `load`, `encode_condition`,
      `start_image`, `set_latents`, `denoise_step`, `latents`, `velocity`
      and `release_condition`, plus `encode_condition_tapped` and
      `denoise_step_tapped`, which copy the residual stream out after given
      blocks.

    Errors are `ImageLeafError::{Refused, Failed, Plan}`. A refusal (a RoPE
    position, the condition budget, the arena, a token id) is the server's
    400, never its 500. `qwen_image::TRANSFORMER_CONFIG` and `IMAGE_MODEL`
    name the two resources.
  - Tests:
    - the CTest `ignis_kernel_image_transformer_test`;
    - the Rust `--ignored` test `crates/core/tests/image_leaf_gpu.rs`
      (ACs 1, 2, 3, 5 on the real artifact);
    - CPU tests: `image_leaf::tests` (the sampler, the scheduler refusals,
      the Euler rounding, the plan) and `image_transformer::tests` (the
      config).
- **The sequence.**
  - `encode_condition` runs four things:
    - the text encoder (spec 03) on the prompt's ids and mrope positions;
    - `txt_in` on its rows `[drop_idx, L)`;
    - the prefix's RoPE table from `ignis_image_dit_rope_positions`;
    - the modulation at σ = 0, then the prefix pass over the 32 blocks.

    The condition prefix is one allocation of 32 runs of `kv8_bytes(P)`,
    each 256-byte aligned. It comes from the condition budget, not the
    arena. For t2i the prefix is the text rows alone: one causal segment
    `[0, P)`.
  - `start_image` uploads the noise (packed, `[y·w + x][c]`). It computes
    the joint sequence's positions with the target block at P, and the
    table of the target's rows.
  - `denoise_step(step)` runs, in order: the modulation at σ_step, img_in,
    the target pass, the head with the Euler update from σ_step to
    σ_step+1. The velocity stays in the image slot, readable.
- **The plan (AC 1), as the test printed it.** `image plan: weights 14.40
  GiB, arena 1.23 GiB, condition budget 48448 tokens (11.83 GiB); budget
  27.48 GiB of 28.98 GiB free (derived)`. The load took 8.2 s.
  - Weights: the full bind, every tensor of the artifact, the VAE's
    included.
  - Arena: the image slot (latents, velocity, target positions and table,
    the modulation row; 17 MiB at the maximum), then one sequential region.
    The region is sized for a denoising step at 16,800 target tokens: the
    fp32 residual stream plus the DiT's workspace.
  - Condition budget: the largest prefix whose 32 runs fit what the weights
    and arena leave, about 256 KiB a token.
  - An 8 GiB explicit budget is refused before anything is placed: `short
    by 8196481032 bytes (7.63 GiB)`.
- **AC 2.**
  - The sampler's sigmas and timesteps equal `sigmas.npy` and
    `timesteps.npy` bit for bit.
  - For each of the 8 steps, `euler_step` of the recorded input latents by
    the recorded velocity is the recorded output bit for bit. That is
    dt = fl(σ' − σ), then fl(x + fl(dt·v)), with no FMA.
  - In each step of AC 3, the device's update equals the host's on the
    leaf's own velocity, bit for bit.
- **AC 3 at 1024² (t2i), teacher-forced.** The velocity's relative L2, in %:

  | step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | **ignis** | **4.72** | **5.29** | **4.76** | **3.57** | **3.19** | **3.02** | **2.88** | **3.58** |
  | bound | 11.01 | 22.47 | 11.78 | 7.04 | 6.28 | 6.04 | 5.75 | 7.06 |
  | sim | 5.51 | 7.12 | 4.09 | 3.52 | 3.14 | 3.02 | 2.88 | 3.53 |
  | bf16 | 1.17 | 4.49 | 2.36 | 1.22 | 0.76 | 0.73 | 0.73 | 0.67 |

  - ignis is within 0.7 points of the sim yardstick at every step, and
    under it at steps 0-1. The 8-bit attention, the 8-bit prefix and the
    BF16 intermediates add almost nothing to the W8A8 linears' own error.
    The K-mean alternative (above) is not needed for t2i, and the block-0
    head-3 caveat (319d) costs nothing visible end to end.
  - The DiT reads ignis's own encoder output (the recorded `input_ids` and
    `mrope_position_ids` through spec 03's forward), never the recorded
    `prompt_embeds`. The packed noise file equals step 0's input, checked.
- **AC 5 (t2i), relative L2 of the residual stream after the block, in %.**
  These are diagnostic numbers with no threshold.

  | after block | 0 | 15 | 31 |
  |---|---|---|---|
  | prefix rows (encode_condition vs step 0's first 31 rows) | 1.09 | 2.39 | 2.59 |
  | target rows, step 0 | 1.13 | 8.96 | 4.75 |
  | target rows, step 1 | 1.06 | 10.42 | 6.23 |

  The target's error peaks in the middle of the stack and falls by block
  31. No step is over its bound, so nothing was diagnosed further.
- **Timing (information only, not AC 10).** In the test, warm, steps 2-7:
  144 ms a step at 1024², so 1.15 s for the 8 steps against the ~1.1 s bar.
  Steps 0-1 took 228 ms, with 3 taps each, which copy 64 MiB out per tap.
  `encode_condition` took 62 ms with its 3 taps.
- **The CTest**, all against fp64 that tracks the kernels' roundings, or
  bit for bit:
  - the gated residual with the modulated LayerNorm, 1, 45 and 2,049 ×
    4096: worst 0.131 of the bound; the sums equal the host's fma; the E4M3
    output equals the host's quantization;
  - the BF16 linear at `[4096×256]`, `[4096×4096]` and `[16384×4096]`:
    worst 0.036;
  - the fp32 quantization, bit for bit;
  - the timestep chain at σ = 0, 0.978453 and 0.414568: 3.2e-7 relative L2
    against an fp64 restatement of diffusers'; cos and sin swapped read
    1.2 (the negative control);
  - the program on synthetic FP8 weights (2 blocks, a 13-token prefix of
    text | 2×3 image | text, a 70-token target):
    - txt_in and img_in are row-independent bit for bit; img_in reads
      0.007 of its bound against fp64;
    - both passes split by block equal the whole bit for bit, runs
      included, and a second run is identical;
    - another prefix changes every target value;
    - the Euler update equals the host's.
- **Defaults taken** (the owner may revisit):
  1. **The residual stream is fp32**, as in the fp32 pipeline and the
     encoder (spec 03). 319g's default was BF16; this follows the rule of
     fewer BF16 roundings.
     - img_in, txt_in's out_layer, to_out and the MLP's down write fp32.
     - The gated add `x = fma(tanh g, y, x)` is fused with the next
       modulated LayerNorm, which quantizes its fp32 output to E4M3
       directly.
     - tanh(gate) is taken once per call, not once per token.
  2. q, k, v, the attention output, the MLP's gate and up, and txt_in's
     hidden are BF16, because the ops that read them take BF16.
  3. txt_in rounds the encoder's fp32 rows to BF16 once for its
     zero-centred norm (spec 02's op). The latents are quantized to E4M3
     from fp32.
  4. The timestep path is BF16 weights × fp32 with fp32 accumulation in a
     fixed order (`_linear_bf16`), and the modulation rows are fp32.
     - The sinusoid's frequencies are torch's: fl32(fl32(−ln 10⁴ · i) /
       128) on the host, then exp correctly rounded.
     - The angles fl32(fl32(1000 σ) · f) and CUDA's cosf/sinf are computed
       on the device, as torch's CUDA path does.
  5. **The DiT's timestep is σ**, not `dit_timesteps` (1 ulp off at steps
     1, 2, 3 and 6). Superseded by 321e: the DiT reads `dit_timesteps`.
  6. **The prefix pass runs whole blocks**, block 31's MLP included,
     though only the K/V runs are kept. That is 1/32 of the pass, and it
     keeps the prefix's rows available for AC 5.
  7. **One image slot.** A job's images run one after another; the VAE
     decode will read the slot's latents.
  8. **The plan.**
     - Free memory is NVML's at load, with the context already counted.
       The headroom is the caller's `VramMode`: the test used the server's
       1.5 GiB default.
     - The arena is the step phase at `MAX_TARGET_TOKENS` = 16,800 (2400 ×
       1792). The prefix pass and the step never run together, so they
       share it.
     - `image.condition_budget_tokens` is a cap (`ImageLeafOptions`).
  9. **Refusals.**
     - The scheduler config: not `FlowMatchEulerDiscreteScheduler`;
       dynamic shifting; a shift other than 1; `shift_terminal`; inverted,
       Karras, exponential or beta sigmas; stochastic sampling; sigmas not
       decreasing within (0, 1].
     - The transformer config: dims other than the inventory's; a patch
       size other than 1; RoPE axes other than (16, 56, 56);
       `causal_condition` off.
- **For 321b (2048², edit, the VAE encode).**
  - **2048²** should need no code: `start_image(.., 128, 128, ..)` fits the
    arena (16,384 ≤ 16,800). Run the test's loop on `t2i-2k`.
  - **Edit's prefix:**
    - Add the condition images to `encode_condition`: the tower and
      DeepStack through the encoder's hooks (320c's order), the VAE encode,
      `img_in` on the condition latents.
    - Lay the prefix out in joint order. Text rows go to text segments,
      each image-pad slot becomes its 2×2 latent tokens, and the segments
      come from `build_token_metadata` and `_qwenimage21_prefix_segments`.
    - Fill `Condition.blocks` with each image block's (start, (h, w)):
      `start_image`'s RoPE positions already read it.
  - **The arena for large prefixes.** The encode phase must fit the
    sequential region (1.23 GiB). Otherwise `encode_condition` is refused,
    naming both numbers.
    - A 1024² condition image (P ≈ 4.1K) needs about 0.3 GiB.
    - Prefixes above about 16K tokens (four or more 1,230-token images)
      will not fit. They need a token-chunked prefix pass (the K/V runs
      need every token before V's per-channel scale, so chunk the GEMMs
      and the MLP, not the run) or a larger region.
  - **The VAE.** Its phases (the decode bands, the encode) will share the
    sequential region; `EncodePhase` and `step_bytes` are the pattern.
- **Gates.**
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124, with no flake (residency_trace
    passed).
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2869 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server --features cuda --tests` and `-p
    ignis-core --features cuda --tests`: ok, with only the old warnings.
  - `image_leaf_gpu` under the profile: 1/1.

## As built (321b, 2026-10-14: ACs 3 at 2048², 4, 5 for 2K and edit, 7; the condition images, the chunked prefix pass)

Status: **AC 3 at 2048² done**, all 8 steps within the bound. **AC 4 done**,
all 8 steps within the bound. **AC 5** written for t2i-2k and edit (below).
**AC 7 done**. ACs 6, 8, 9 and 10 are not done; AC 11 still applies.

- **Where.**
  - `crates/core/src/image_autoencoder.rs` (new): the VAE encoder, driven op
    by op over spec 02's VAE ops. It holds `VaeConfig::parse` (refuses a
    non-residual VAE, a patch size above 1, attention in the down blocks, a
    ladder other than the inventory's), `VaeEncoderWeights`,
    `encode_workspace_bytes`, `encode`, `input_bf16` and `check_size`.
  - `crates/core/src/image_leaf.rs`:
    - CPU: `LatentNorm` (`latents_mean`, `latents_std`) and `PrefixLayout`
      (below);
    - device: `ConditionImage` { grid, pixels, width, height, rgba };
      `ConditionText` gains `image_pad_rows`; `encode_condition(text,
      images)` and `encode_condition_tapped(text, images, taps)`;
      `vae_encode(width, height, rgba)`; `Condition::segments`, `blocks`
      and `prefix_chunk`.
  - The kernel leaf: `ignis_image_transformer_prefix_blocks` and
    `_prefix_kv_bytes` (the chunked prefix pass), with Rust's
    `transformer::prefix_blocks` and `prefix_kv_bytes`.
  - `qwen_image::VAE_CONFIG` names the resource.
  - Tests:
    - `crates/core/tests/image_leaf_gpu.rs`: four tests, t2i at 1024²
      (with AC 1), t2i-2k, edit (AC 4), and the VAE encode (AC 7);
    - `crates/server/tests/image_leaf_ten_images_gpu.rs`: ten condition
      images through `EncoderFrontend`;
    - the CTest's program case: the chunked pass against the whole one;
    - CPU: `image_autoencoder::tests` and `image_leaf::tests`.
- **The condition prefix with images** (`PrefixLayout`, diffusers'
  `forward`, `build_token_metadata` and `_qwenimage21_prefix_segments`).
  - The encoder's rows from `drop_idx` on, in order. A text row is one
    prefix token through `txt_in`; each image's pad rows become its block of
    latent tokens through `img_in`, 4 a pad, in the VAE's raster order.
  - Each text run is a causal segment, each image block a bidirectional
    one. Two images with no text between them stay two blocks.
  - The edit case's segments equal `case.json`'s, checked:
    [0, 8) text, [8, 4104) image, [4104, 4122) text.
  - Refused (400): pads that do not add up to the images' (height x width
    / 4 each), an image whose pads are not consecutive, a grid that is not
    the image's pixels in 16-pixel patches, one frame.
- **`encode_condition`'s order, in the arena's sequential region.**
  1. Each image's VAE encode. Its normalized latents stay at the region's
     start until img_in; the VAE's own phase runs after them and is
     reused by the next image.
  2. The tower on all the images (spec 03's `vision_call`).
  3. The text encoder with the hooks: the merger's rows over the pads,
     DeepStack after the first text layers (320c's order).
  4. txt_in per text run and img_in per image, at their prefix rows, in
     pieces of at most the pass's rows (both are row-independent, bit for
     bit).
  5. The RoPE table, the t = 0 modulation, the prefix pass.
- **The chunked prefix pass.**
  - Whole when its workspace fits what the region leaves after the
    prefix's residual stream and table. Otherwise in chunks: the largest
    multiple of 128 whose workspace fits.
  - Per block:
    - phase A, chunk by chunk: img_norm1, K (with its norm and RoPE) and
      V, into a BF16 K/V buffer for the whole prefix; then one store of
      the block's run. A run's K mean and V scales are over all its tokens,
      so #319's run semantics are kept as they are;
    - phase B, chunk by chunk of whole segments: img_norm1 again, Q, the
      attention, to_out, the MLP. A segment longer than the chunk is cut
      every `chunk` queries, so its pieces' quantized-query blocks and
      query tiles fall as in one call; a causal piece's keys end at its
      own end.
  - The CTest: 834 tokens (text | image | text | image | 300 text), chunks
    of 128 and 384, whole and one block a call. x and both runs equal the
    whole pass bit for bit.
  - **Ten 1024² images** (through `EncoderFrontend`): 10,335 encoder
    tokens, a 41,041-token prefix (10.03 GiB of runs) in chunks of 7,040.
    `encode_condition` took 6.1 s (the first call: ten VAE encodes, the
    tower on 40,960 patches, the encoder, the prefix pass). A 64 x 64 step
    over it took 306 ms, with a finite velocity.
- **The plan line now reads** `condition budget 45824 tokens (11.89 GiB)`.
  The budget counts the chunked pass's BF16 K/V (16 KiB a token) beside
  the 32 runs, so a token is about 272 KiB. It was 48,640 on the same free
  memory.
- **AC 3 at 2048² (t2i-2k), teacher-forced.** The velocity's relative L2,
  in %:

  | step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | **ignis** | **4.97** | **5.45** | **3.19** | **2.92** | **2.77** | **2.70** | **2.70** | **3.44** |
  | bound | 10.99 | 23.49 | 9.54 | 5.90 | 5.57 | 5.39 | 5.36 | 6.79 |
  | sim | 4.68 | 5.65 | 3.19 | 2.95 | 2.79 | 2.70 | 2.68 | 3.40 |
  | bf16 | 2.20 | 4.70 | 1.91 | 1.18 | 0.58 | 0.63 | 0.62 | 0.60 |

  No code was needed: `start_image(.., 128, 128, ..)` fits the arena.
- **AC 4 (edit), teacher-forced.** The condition prefix comes from
  `encode_condition` (the recorded sequence and tower input, the VAE input
  from the recorded bytes). The velocity's relative L2, in %:

  | step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | **ignis** | **6.49** | **5.90** | **2.99** | **2.89** | **2.74** | **2.66** | **2.65** | **3.83** |
  | bound | 13.79 | 12.53 | 7.42 | 5.97 | 5.48 | 5.31 | 5.29 | 7.63 |
  | sim | 6.89 | 6.27 | 3.09 | 2.99 | 2.74 | 2.65 | 2.65 | 3.81 |
  | bf16 | 2.08 | 1.78 | 1.48 | 1.16 | 0.69 | 0.61 | 0.64 | 0.73 |

  Every step is at the sim yardstick's level, and under it at steps 0-3.
  As for t2i, the K-mean alternative (above) is not needed.
- **AC 5, relative L2 of the residual stream after the block, in %**
  (diagnostic, no threshold):

  | after block | 0 | 15 | 31 |
  |---|---|---|---|
  | t2i-2k prefix rows | 1.09 | 2.39 | 2.59 |
  | t2i-2k target, step 0 | 1.20 | 8.97 | 4.45 |
  | t2i-2k target, step 1 | 1.25 | 10.25 | 5.24 |
  | edit prefix rows | 1.18 | 11.27 | 11.04 |
  | of them, the image block's | 0.71 | 26.30 | 12.42 |
  | of them, the text rows | 2.70 | 3.38 | 3.95 |
  | edit target, step 0 | 1.09 | 13.75 | 6.81 |
  | edit target, step 1 | 1.04 | 13.82 | 6.48 |

  - The condition image's rows peak at block 15, as the target's do, and
    the velocity is at the sim's level.
  - No step is over its bound, so nothing was diagnosed further. Whether
    the image block's 26% at block 15 is the W8A8 linears' own is open
    (the sim has no block dumps); it is here for information.
- **AC 7, the VAE encode.** The edit case's condition image (1024², its
  bytes from `condition_vae_input.npy`) against the recorded fp32
  `condition_latents`:
  - **1.07%**, 0.73 x bf16's own 1.48%, in 27 ms (the first call).
  - **The bound is `thresholds.json`'s `condition_latents.bound`,
    7.38%:** 5 x bf16's error, the v4 rule every quantity follows. The sim
    is 0 here: its VAE is the fp32 one on the same BF16 weights.
  - The encoder rounds less often than diffusers' BF16 path (the fused
    epilogues, one rounding per norm), so bf16's own error is the level to
    expect; it came in under it.
- **Timing (information only, not AC 10).** Warm steps: 733 ms at 2048²
  (5.9 s for the 8, against the ~6 s bar) and 163 ms for edit at 1024²
  (over its 4,122-token prefix). Edit's `encode_condition` took 542 ms
  with 3 taps.
- **Defaults taken** (the owner may revisit):
  1. **The chunked pass holds the prefix's BF16 K/V** in an allocation
     from the condition budget, freed when `encode_condition` returns. It
     costs about 6% of the budget's tokens. The alternative, two passes
     that gather the K mean and V scales across chunks, would recompute K
     and V (about 15% of the pass) and add entry points to the 8-bit
     attention; it was not built.
  2. **In the chunked pass the MLP's add is not fused with the next
     block's img_norm1**: phases A and B each recompute img_norm1 from x
     (the same values).
  3. **The VAE encoder is Rust over spec 02's ops, not a kernel program.**
     - The 3x3 weights are packed per call into a scratch region: no
       packed copy is kept and the plan does not change.
     - The activations use four slots of the widest activation (768 bytes
       a pixel, 805 MB at 1024²).
     - Only quant_conv's first 64 output rows (the posterior's mean) are
       computed, in fp32.
  4. **The downsampler adds the AvgDown shortcut as its convolution's
     residual**: one BF16 rounding fewer than diffusers.
  5. **The VAE's input** is the bytes' `2 (u / 255) - 1` in f32, rounded to
     BF16. The normalization `(z - mean) / std` is done on the host in fp32
     (two roundings, the pipeline's), so the latents make one trip through
     the host per image (1 MiB at 1024²).
  6. **`encode_condition` checks its own prefix against the budget, not
     the sum of the live conditions.**
  7. **The arena is unchanged.** The ten-image phases fit its 1.22 GiB
     region; the tower and encoder phase needs about 1.15 GiB of it.
- **For 321c.**
  - **The banded decode** goes in `image_autoencoder.rs`, beside the
    encoder: the `Run` slots and the pack scratch are the pattern, and
    `VaePhase` is the arena's.
    - `LatentNorm` holds the mean and std. The decode's `z x std + mean` is
      not written yet.
    - The decode reads the image slot's fp32 latents (packed `[h x w][64]`).
  - **n = 3, release and cancel:** a `Condition` is independent of the
    image slot. `release_condition` frees its runs; the chunked pass's K/V
    never outlives `encode_condition`.
  - **The runner adapter** builds `ConditionImage` from `EncoderFrontend`'s
    `VisionPatches` and the RGBA at the condition size (sides multiples of
    32, so the grid is the pixels in 16-pixel patches).
- **Gates.**
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124, with no flake.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2874 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server --features cuda --tests` and `-p
    ignis-core --features cuda --tests`: ok, with only the old warnings.
  - Under the profile: `image_leaf_gpu` 4/4 and `image_leaf_ten_images_gpu`
    1/1.

## As built (321c, 2026-10-14: ACs 6, 8, 9; the VAE decode, n images, release and cancel, the free runs)

Status: **AC 6 done**, every case at least 50 dB per channel, about 3 dB
above diffusers' own bf16 decode. **AC 8 done**, measured. **AC 9 done**, the
PNGs are linked below. Only AC 10 (the timing run) is not done; AC 11 still
applies.

- **Where.**
  - `crates/core/src/image_autoencoder.rs` gains the decoder:
    - `VaeDecoderWeights`;
    - `decode`, with `decode_workspace_bytes`;
    - the band planner `band_rows` and `bands`, with `BandRows` and
      `BlockRows`;
    - the host tail `latents_bf16` and `rgba8`.

    `VaeConfig::parse` now also checks `decoder_base_dim`: absent means
    diffusers' default, 144; null means `base_dim`.
  - `crates/core/src/image_leaf.rs` adds:
    - `LatentNorm::denormalize`;
    - `ImageLeaf::vae_decode()`, which turns the started image's latents
      into RGBA8 bytes `[16 h][16 w][4]`;
    - `vae_decode_band_rows()`, the rows a band takes;
    - `vae_decode_in_bands(band_rows)`, for tests and diagnosis;
    - `device_free_bytes()`.

    The image slot now remembers its (h, w).
  - Tests:
    - three new GPU tests in `crates/core/tests/image_leaf_gpu.rs`, which
      now holds 7: `the_vae_decode_of_the_final_latents_is_the_recorded_png`,
      `a_job_of_three_images_and_a_cancelled_job_leave_the_device_memory_where_it_was`,
      `each_case_runs_free_from_its_noise_file`;
    - CPU tests: the band plan worked by hand, the bands' tiling, the
      workspace, the bytes against diffusers' own tail, the BF16 rounding,
      the denormalization and the config.
  - No kernel change: the decoder is Rust over spec 02's ops.
- **The decode.**
  - **The global phase**, on the whole latent grid: `post_quant_conv`,
    `conv_in`, then the mid block. Its attention is global. The mid
    block's output (1152 ch) stays whole while the bands run.
  - **The band phase**: the 5 up blocks, `norm_out`, SiLU and `conv_out`,
    in horizontal bands of image rows. Each band starts from the mid
    block's output and works backward from `conv_out`'s halo:
    - a 3×3 conv needs one row on each side;
    - the upsampler needs `[(y0 − 1) >> 1, (y1 >> 1) + 1)` of the rows
      before it;
    - an upsampled block's output is rounded out to even rows, so the
      DupUp maps whole rows.

    The convolution pads by the global row, so a band is the whole image's
    rows bit for bit.
  - **Slots.** Four slots of the widest activation of any band, as the
    encoder's `Run`, plus the mid output and the pack scratch.
    - At 2048² in 224-row bands, a slot is block 3's output: 240 rows ×
      2048 × 288 ch, 283 MB.
    - Unbanded at 1024², the widest is 604 MB, so the 1.23 GiB region
      cannot hold the unbanded decode.
  - **Band size.** The leaf takes the most rows (a multiple of 16) whose
    phase (input + fp32 image + workspace) fits the arena's region: 512
    rows at 1024², 224 at 2048².
  - **The halo's recompute**, in conv FLOPs of the banded stages: 1.08×
    at 1024², 1.33× at 2048².
- **AC 6.** The input is each case's recorded `step7/latents_out`, through
  `vae_decode`. The reference is the recorded `image.png`, diffusers' fp32
  decode. The floor is diffusers' own bf16 decode of the same latents
  through the pipeline's tail (bf16 VAE, the latents and their
  denormalization in bf16). It ran in the venv, diffusers 0.41.0 / torch
  2.13, on the RTX 5090, script `.scratch/321c/bf16_floor.py` in the worktree
  `image-kernels-319` (not tracked).

  PSNR per channel, in dB:

  | case | ignis R / G / B / A | diffusers bf16 R / G / B / A |
  |---|---|---|
  | t2i 1024² | **50.76 / 50.05 / 52.36 / 65.89** | 47.71 / 46.99 / 49.12 / 60.30 |
  | edit 1024² | **55.14 / 56.06 / 55.28 / 63.05** | 52.03 / 52.77 / 52.99 / 55.35 |
  | t2i-2k 2048² | **52.20 / 51.65 / 53.01 / 65.98** | 49.15 / 48.57 / 50.00 / 59.25 |

  - **Banded against unbanded.** At 1024², the arena's decode in 512-row
    bands equals the unbanded one bit for bit: 0 differing bytes. The
    unbanded one ran in a workspace allocated for the call.
  - **2048² runs in the arena.** `vae_decode` allocates nothing: the free
    memory is the same before and after. Its 224-row bands equal a decode
    in 112-row bands bit for bit.
- **AC 8, measured with `cudaMemGetInfo` after the stream drains.**
  - **The first job.** One step and a decode leave 2 MiB less free after
    its release. That is CUDA's lazy loading of the kernels the process
    launched for the first time: with `CUDA_MODULE_LOADING=EAGER` the same
    test reads 0, both jobs equal to the byte. It is a once-per-process
    cost, not the job's. The test runs that first job before it measures.
  - **n = 3.** One `encode_condition` (the t2i prefix: 18.0 MiB held) and
    three images from noises A, B, A, each 8 steps then `vae_decode`.
    - The third image is the first's bytes exactly, so the prefix outlived
      the image between them unchanged; the second differs.
    - After `release_condition` the free memory is the before-job value,
      15,706,619,904 bytes, to the byte.
  - **Cancel.** A job released after its third step: the free memory is
    back to the same value, to the byte.
- **AC 9, the free runs.** Each case from its noise file: `encode_condition`,
  the 8 steps on the leaf's own latents, `vae_decode`. The PNGs are in
  `F:/ai/models/ignis-image-2.1-turbo/freerun/<case>/`, `ignis.png` beside
  `diffusers-fp32.png`:
  - [t2i](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/t2i/ignis.png)
    (beside [diffusers](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/t2i/diffusers-fp32.png));
  - [t2i-2k](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/t2i-2k/ignis.png)
    (beside [diffusers](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/t2i-2k/diffusers-fp32.png));
  - [edit](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/ignis.png)
    (beside [diffusers](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/diffusers-fp32.png)).

  What they look like, by eye:
  - **t2i and t2i-2k.** The same picture as diffusers': the fox, the
    "ignis" sign legible, the trees, the dawn sky or the falling snow, the
    tail. They differ only in the strokes: an ear's outline, the sign's
    tilt at 2K.
  - **edit.** Where diffusers' alpha is opaque, the same glass circle,
    square and triangle, to about 4-6 levels of RGB.
    - The transparent background differs. ignis keeps a faint trace of the
      condition image's white-to-blue gradient band, in hidden RGB with
      alpha up to 18/255. diffusers' alpha there is at most 4.
    - Viewers that ignore alpha show it as stripes. Over white it is a
      barely visible grey veil.
    - That background is why the edit's RGB PSNR against diffusers reads
      16.6-25.8 dB. Over white it reads 35.1-36.4 dB.
  - **Distance from diffusers' fp32 run.** The final latents are 8.72%
    (t2i), 7.36% (2K) and 7.53% (edit) from it. The PSNR against its
    image is t2i 26.6-27.1, 2K 26.5-29.1 and edit (alpha) 37.9 dB. This
    is information only: a free run drifts, as the bf16 yardstick's would.
- **Timing (information only, not AC 10).**
  - `vae_decode` takes 133-170 ms at 1024² and 643-725 ms at 2048²; spec
    02's convolutions alone were 104 and 406 ms.
  - The difference is the halo's recompute (1.33× at 2K), the 3×3
    weights packed once per band, and the fp32 image's trip to the host.
  - Free run, warm, t2i: `encode_condition` 60 ms, 8 steps 1.18 s; 2K: 8
    steps 5.9 s; edit: `encode_condition` 524 ms, 8 steps 1.32 s.
- **Defaults taken** (the owner may revisit):
  1. **The denormalization is on the host.** It is `z · std + mean` in
     fp32 with two roundings, as the pipeline's fp32 path, then one
     rounding to BF16 for `post_quant_conv`. diffusers' bf16 path rounds
     the latents to bf16 and denormalizes in bf16.
  2. **`conv_out` writes fp32.** The pipeline's tail is on the host, in
     f32: the clamp to [−1, 1], `x · 0.5 + 0.5` clamped to [0, 1], then
     `(x · 255)` rounded half to even (numpy) to bytes. The expected bytes
     of the CPU test came from diffusers' own tail.
  3. **The upsampler adds the DupUp shortcut as its residual**, in place:
     one rounding, where diffusers' BF16 path rounds the convolution
     first. The DupUp is written into the output's slot first.
  4. **Four uniform slots** of the band's widest activation, the encoder's
     `Run`. One slot size serves the mid block too, and the attention's
     row tile is halved until its workspace fits a slot.
  5. **Bands are the most 16-row multiples that fit.** They are not
     balanced, so the last band is shorter.
  6. **The 3×3 weights are packed per call, so per band.** That is about
     0.5 GB of BF16 read and written per band, under a millisecond each.
  7. **`vae_decode_in_bands`** allocates its workspace outside the plan
     when the phase does not fit the region. It exists only for tests;
     `vae_decode` never allocates.
  8. **The free-run noise** is spec 01's noise file. For AC 8's second
     image, it is the edit case's noise, also 64 × 64.
- **For the runner adapter (after #322).**
  - A job:
    - `encode_condition` once;
    - per image, `start_image` (noise from the job's seed), the steps,
      `vae_decode`;
    - `release_condition` at the end, or on cancel between any two calls.
  - `vae_decode` returns RGBA8 bytes ready for PNG. The alpha is the
    VAE's fourth channel: transparent backgrounds are real, so never drop
    alpha.
  - A refusal (`ImageLeafError::Refused`) is a 400. `vae_decode` before
    `start_image` is one; a decode too wide for the arena would be one,
    but the presets fit.
- **For the timing agent (AC 10).**
  - `vae_decode` is one host trip in (the latents, 1-4 MiB) and one out
    (the fp32 image, 16-64 MiB), around the device work.
  - Run one decode before you time: the first launches load kernel
    modules (2 MiB, lazy).
  - The halo's recompute and the per-band weight packing are the obvious
    gaps at 2K. A larger region, or bands that carry halo rows forward
    instead of recomputing them, would cut them. Not tuned (owner rule).
- **Gates.**
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124, with no flake.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2880 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server --features cuda --tests` and `-p
    ignis-core --features cuda --tests`: ok, with only the old warnings.
  - Under the profile: `image_leaf_gpu` 7/7. AC 7's encode still reads
    1.074%: the encoder's `conv` now goes through the windowed `conv_rows`.
  - Not rerun: the server's `image_leaf_ten_images_gpu`. What it calls that
    this change touched (that convolution wrapper, the image slot's shape)
    is covered by `image_leaf_gpu`'s ACs 3, 4 and 7.

## As built (321d, 2026-10-14: the runner adapter, the leaf end to end through HTTP)

Status: the image model serves spec 05's routes on this leaf, loaded by the
server's loader like a text model (start, switch, idle unload's reload). No
AC of this spec changes status: AC 10 (timing) is open, AC 11 applies. The
branch merged `image-serving-322` first (d7447b4; the conflicts were additive:
`NOTICE` and the two module lists).

- **Where.**
  - `crates/server/src/images/leaf.rs`: `LeafRunner` (with `cuda`), the
    `ImageRunner` over `ImageLeaf` and spec 03's `EncoderFrontend`;
    `load.rs`'s `cuda_image_runner` loads it in place of the stub. Its CPU
    helpers: `max_target_cells`, the latents' packing, the step lookup, the
    error map.
  - `crates/server/src/image_encoder_input.rs`: `EncoderFrontend::count`
    (`ConditionCount`) and `ImageProcessor::keeps`; `image_prompt.rs`:
    `ImagePrompt::rows`.
  - The leaf: `ImageLeaf::condition_fits`; `release_condition` clears the
    image slot; NVML is read for `CudaDevice::ordinal()`; `ImageLeafError`
    lives outside the `cuda` module; `Condition` is `#[must_use]`.
  - `ignis_core::image::ImageError::Refused`, which the routes answer
    `400 image_refused`.
- **The count** (review-321ab I3: "a refusal is a bug").
  `condition_tokens(prompt, sizes)` is `EncoderFrontend::count`'s prefix: the
  raw template tokenized once with one pad a slot, its `rows − n − drop_idx`
  text rows, plus each image's latent cells. That is `PrefixLayout`'s count,
  the one `encode_condition` refuses by:
  - a CPU test checks it against the front end's own input laid out by
    `PrefixLayout`;
  - on the real tokenizer it gives the oracle's prefixes, 31 (t2i) and
    4,122 (edit), and 41,041 for ten 1024² images, the ten-image test's.

  Within the plan's budget it counts `budget + 1` when:
  - the arena has no room for the condition (`ImageLeaf::condition_fits`:
    `EncodePhase` from the shapes alone, the VAE encode, the tower and
    encoder, the prefix);
  - or a condition size is one the processor would not keep
    (`ImageProcessor::keeps`: `smart_resize` changes it or refuses it; off the
    32-pixel grid, under its pixel floor, past 200:1).

  So `budget::fit` scales the images for those limits too, and
  `encode_condition` refuses them, 400, before any image is prepared.
  (321f: superseded. The count carries a reason instead: the arena's
  shortfall is no room, scaled for; a size the processor would not keep is
  refused, `400 image_refused`. See "As built (321f)".)
  Measured on an oversubscribed 40 GiB budget (93,888 tokens): ten 13120x96
  images, the worst condition size (49,200 cells, 1,230 merged tokens each),
  count 93,889 for the arena, and the fit lands on 12800x64. On this card's
  own budget (45,184 tokens) the budget binds first.
- **The largest target, one source** (review-321ab I2). `max_target_cells`
  is the largest of `image.sizes` in latent cells: 16,800 (2400x1792) by
  default. It sets both `ImageLeafOptions.max_target_tokens` (the arena) and
  `ImagePlan.max_target_cells` (the routes' `takes`). A larger size added by
  a live `PATCH` is served after the next load. (321f: within
  `image.max_target_cells`, a new field; a size past it is not served.)
- **The calls.**
  - `encode_condition`: the count (`Refused`, or `OverBudget` past the
    plan's budget), `EncoderFrontend::input`, then the leaf's encode. Its
    `Condition` goes into the runner's map at once; nothing fallible sits
    between.
  - `denoise_step`: the job's channel-major latents go up packed,
    `[y·w + x][c]`. An image starts (`start_image`, its RoPE table) at its
    first step over a prefix, tracked as (prefix, size); later steps
    `set_latents`. The step is the sampler's whose (σ, σ_next) are the job's,
    bit for bit: the plan copies the sampler's sigmas. The latents come back
    channel-major, and the velocity when asked.
  - `vae_decode`: the job's latents into the slot of the image a step
    started, then the leaf's decode. Another size is a server bug: `Leaf`,
    a 500.
- **Errors.** The leaf's `Refused` and every front-end refusal are
  `ImageError::Refused`: `400 invalid_request_error`, code `image_refused`,
  in the leaf's words, also as the stream's `error` event. `Failed` is
  `Leaf`, `500 engine_error`. `Plan` is the load's refusal, naming the
  shortfall. Over HTTP, a 9,000-word prompt answers 400: "a text position
  falls outside [-1024, 8192)".
- **Release on every path** (review-321c I2, review-321ab I4). The runner
  holds each prefix from its encode on. `release_condition` frees it, and
  `Drop for LeafRunner` frees every one still held before the leaf drops.
  That path is real: a switch or an idle unload drops the runner between
  two calls of a running job, and the job's own release then finds no
  runner.
  - Measured: a model shut down holding the edit's prefix (1,042 MiB)
    leaves the free memory where a clean unload left it, +0.0 MiB.
  - With the `Drop` disabled the same test reads −1,042 MiB: red, then
    green.
  - `Condition` gets no `Drop` of its own: freeing needs the leaf's device.
- **Step 0 done here.**
  - Review-321c L3: `vae_decode` and `latents` before `start_image` and
    after `release_condition` are `Refused`, asserted in AC 8's GPU test.
    The release clears the slot because its RoPE table was laid out after
    that prefix.
  - Review-321ab L3: NVML is read for the device the leaf was given. The
    server still runs device 0, as the text models do.
- **Through HTTP** (the 5090, `ArtifactLoader` on the default options,
  `crates/server/tests/image_runner_gpu.rs`):
  - the load takes 8.5-11.0 s; the condition budget is 45,184 tokens;
  - a 1024² generation (encode, 8 steps, decode, PNG) takes 1.70 s, and the
    same seed again gives the same bytes;
  - a 1024² edit with the oracle's 1024² condition image takes 2.25 s, the
    same bytes twice;
  - the PNGs are `F:/ai/models/ignis-image-2.1-turbo/freerun/http/t2i-seed7.png`
    (the fox and a legible "ignis" sign) and `edit-seed3.png` (the glass
    shapes, a transparent background whose hidden RGB keeps the condition's
    band: review-321c M1, open).
- **Defaults taken** (the owner may revisit):
  1. **The arena follows `image.sizes`.** It is sized for the largest size at
     load, with no field of its own: fewer and smaller sizes give a smaller
     arena and a larger condition budget.
  2. **`image.condition_budget_tokens` is not the leaf's load cap**
     (`condition_budget_cap: None`). The routes apply it per request, so a
     `PATCH` of it holds live.
  3. **Past the arena or the processor, the count is `budget + 1`**, not an
     error of its own. A fit refused there names a token count one past the
     budget. (321f: replaced by `ConditionUnfit`.)
  4. **The latents cross to the host every step**, the trait's contract,
     with no skip for an unchanged upload: about 4 MiB each way at 2K,
     against 0.7 s steps.
  5. **Device 0**, as the text models. `LeafRunner` is `Send` under
     `ImageModel`'s one lock; the leaf's FFI selects the device on each call.
  6. **The front end is read by the load**, not by `prepare_model`. A
     broken image artifact is refused by the load, and a switch then puts
     the old model back.
- **Left for the next agent** (`.scratch/orchestra/step0-321d.md`): M1
  (chunked against whole on real weights), L1 (the timestep), L2 (a budget
  floor), L4 (a non-square VAE encode), L5, I1 (§Seams' 16,512 line), and
  review-321c's M1 (the edit's background), L1, L2, L4, I1 and I3; then AC
  10's timing run.
  - One corner for spec 05's fit: the processor's 200:1 limit is not
    monotone in the area, and `budget::fit` reads the floor first. Ten
    16380x100 panoramas at the default 512 floor round to 6560x32, past
    200:1, and the request is refused `400 condition_budget_exceeded`,
    though 12800x64 would serve (the GPU test fits from 640). (321f: fixed;
    the GPU test fits from 512.)
- **Gates.**
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2964 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server --features cuda --tests` and `-p
    ignis-core --features cuda --tests`: ok, with only the old warnings.
  - Under the profile, the image GPU tests 19/19: core's `image_attn_gpu`,
    `image_elementwise_gpu`, `image_encoder_attn_gpu`, `image_encoder_gpu`
    (2), `image_gemm_gpu`, `image_vae_gpu`, `image_leaf_gpu` (7); the
    server's `image_encoder_edit_gpu` (2), `image_leaf_ten_images_gpu`,
    `image_runner_gpu` (2).
  - The 27B's vision tests 7/7 under the profile (`vision_load_gpu`,
    `vision_canary_gpu`, `vision_dflash2_gpu`, `vision_mixed_load_gpu`,
    `vision_prefix_reuse_gpu`, `classify_vision_readout_gpu`,
    `classify_pointing_gpu`).
  - `npm test` in `web/`: 855 passed.

## As built (321e, 2026-10-14: the close-out, ACs 10 and 11; the edit's background)

Status, one line per AC (the AC 11 run, below, on dbc760e):

1. **Done.** The load prints the plan line (condition budget 44,096 tokens
   in this run). An 8 GiB budget is refused naming the shortfall. A plan
   that leaves less than `image.min_condition_budget_tokens` is refused too
   (321e).
2. **Done.** The sigmas, the timesteps and now the DiT's timesteps
   (`dit_timesteps.npy`) equal the recordings bit for bit. The host's Euler
   update and the device's are bit for bit.
3. **Done.** All 16 steps at 1024² and 2048² are within their bound, at
   most 51% of it.
4. **Done.** All 8 steps are within their bound, at most 53% of it (step
   1).
5. **Written** (the AC 11 run's report is below). The edit's image rows at
   26.3% after block 15 are the simulated W8A8's own: 26.2%.
6. **Done.** 50.8-56.1 dB per channel, alpha 63.1-66.0 dB. The banded
   decode equals the unbanded one at 1024². At 2048², 224- and 208-row bands
   agree bit for bit. Every default size decodes within the arena.
7. **Done.** 1.074% against a bound of 7.376%. A 1024x768 crop encodes
   finite.
8. **Done**, to the byte: n = 3, and the cancel after 3 steps. Under
   `CUDA_MODULE_LOADING=EAGER` the first job leaves 0 bytes too.
9. **Done.** The AC 11 run rewrote the PNGs, now with the DiT's timestep.
   Added: a free run at 2752x1536.
10. **Done**, for information: below, and in
    [the finding](../../findings/2026-10-14-image-model-on-the-5090.md).
11. **Done.** ACs 3-10 ran once under the GPU lock in timing mode on dbc760e,
    with nothing else on the card or the CPU. No step is over its bound, so
    nothing goes to the owner from ACs 3-4. The edit's background (below)
    goes to the owner as information.

- **The step-0 items**
  - **Review-321ab L1, the DiT's timestep.** It is
    `Sampler::dit_timestep`: fl(fl(1000σ) · fl(1/1000)). That is torch's
    `timestep / 1000` on the GPU, a product by the reciprocal. It equals
    the recorded `dit_timesteps` bit for bit; σ and a true division by 1000
    are each 1 ulp off at steps 1, 2, 3 and 6 (CPU test). Only those steps'
    velocities moved against 321a/321b's tables. Steps 0, 4, 5 and 7 read
    as before, so the move is the timestep's, not a kernel's:
    - edit step 1: 5.90 → **6.60%**, now above the sim's 6.27% (bound
      12.53%). 321b's "under the sim at steps 0-3" no longer holds for
      step 1;
    - t2i-2k step 1: 5.45 → 5.19% (sim 5.65%);
    - t2i steps 1-3: 5.29 / 4.76 / 3.57 → 5.19 / 4.68 / 3.48%;
    - the rest within 0.1 point.
  - **Review-321ab M1, chunked against whole on real weights.**
    `ImageLeafOptions::prefix_chunk` is a test knob, and the server passes
    None. In chunks of 1,024, the edit's image block is cut in four. The
    prefix's rows after blocks 0, 15 and 31 equal the whole pass's bit for
    bit, and so do step 0's target rows and velocity over the chunked
    prefix (`the_chunked_prefix_pass_equals_the_whole_one_on_the_edit`).
  - **Review-321ab L2, a minimum condition budget.** The field is
    `image.min_condition_budget_tokens`, default 8,191
    (`DEFAULT_MIN_CONDITION_BUDGET_TOKENS`), 0..=2^20, 0 taking any.
    - Why 8,191: it is the longest text-to-image prefix the DiT's
      positions admit. The text takes 0 .. P, and the target's block starts
      at P < 8,192.
    - A plan whose budget, before the cap, is below it is refused with
      `ImagePlanError::BelowMinCondition`. The message names the tokens, the
      bytes short and the way out. The CPU test that enshrined a 0-token
      load now does so only with a minimum of 0.
  - **Review-321ab L4, review-321c L2: non-square.**
    - A 1024x768 crop of the condition image encodes to 48x64 finite
      latents. Its top 24 rows are 6.1% from the square's, and 21.2% from
      the square's rows one lower (the control).
    - `decode_band_rows` (host-side) says each default size decodes in the
      arena the load sizes for 16,800 cells: bands of 160 rows (2752x1536)
      to 512 (1024²). The server's test checks all eight.
    - 2752x1536 runs free from the server's noise (seed 7, the t2i prompt):
      8 steps in 5.93 s, the decode 682 ms in the arena, nothing allocated.
      The picture is the fox and a legible sign,
      [ignis.png](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/t2i-2752x1536/ignis.png).
      It is the first non-square target run on the GPU.
  - **Review-321c L1, L4.** The 2K bit-equal check uses 208-row bands
    against the planned 224: lcm 2,912 > 2,048, asserted, so no band edge is
    shared. `AC6_MIN_PSNR_DB` names AC 6's 40 dB.
  - **Review-321c I1.** The EAGER run of AC 8 is kept in the worktree at
    `.scratch/321e/ac8-eager.log`, untracked.
  - **Review-321c I3.** `image_vision::to_bf16` is the leaf's one host
    rounding, for the patches, the VAE's input and the decoder's latents. A
    NaN becomes torch's 0x7FC0. Before, a NaN became a zero, or overflowed
    in debug.
  - **Review-321ab L5, I1, nit.** The VAE header says the caller packs the
    3x3 weights. §Seams' largest preset is 2400x1792. The unused
    `ImageLeaf::vision()` is gone.
- **The edit's background (review-321c M1): W8A8's, not a bug.** diffusers
  was run free on the edit case from its noise file, as `bf16_floor.py`
  runs, but end to end. Two precisions:
  - the bf16 pipeline;
  - the simulated W8A8: `record.py`'s yardstick sim, fp32 compute on the
    artifact's stored values.

  The scripts are `.scratch/321e/ghost*.py` in this worktree, untracked.
  The mask is review-321c's. It takes the pixels where diffusers' fp32
  alpha is at most 4 and that lie at least 8 px from any pixel above 4:
  249,856 in the condition's band (rows ≥ 780), 423,126 above it.

  | run | band alpha max / p99.9 / mean | band row mean, top to bottom | hidden RGB in the band | alpha above the band, max | final latents from fp32 |
  |---|---|---|---|---|---|
  | diffusers fp32 (the oracle) | 4 / 2 / 1.00 | 1.0 1.0 1.0 1.0 | 164, 57, 208 | 2 | 0 |
  | diffusers bf16 | 5 / 3 / 0.85 | 0.8 1.0 0.9 0.8 | 161, 58, 205 | 2 | 1.37% |
  | **simulated W8A8** | **59 / 58 / 13.39** | **8.8 12.0 15.6 17.1** | 73, 85, 156 | 14 | 10.92% |
  | ignis (AC 11 run) | 18 / 18 / 4.88 | 3.1 3.9 5.5 7.1 | 97, 77, 159 | 5 | 7.83% |

  Teacher-forced at step 0, relative L2 against the recorded fp32 dumps,
  in %:

  | | image rows, blocks 0 / 15 / 31 | target, blocks 0 / 15 / 31 | velocity |
  |---|---|---|---|
  | bf16 | 0.38 / 16.88 / 5.81 | 0.44 / 5.40 / 2.70 | 2.08 |
  | simulated W8A8 | 0.62 / **26.21** / 12.46 | 1.04 / 13.72 / 6.92 | 6.89 |
  | ignis | 0.71 / **26.30** / 12.42 | 1.09 / 13.75 / 6.81 | 6.49 |

  - **The verdict.** bf16 keeps the background clean. The W8A8 model, run
    in fp32 on the artifact's own values, leaks the band three times as
    much as ignis does. So the ghost is the FP8 quantization's, and ignis's
    fewer roundings make it smaller, not larger.
  - **321ab's open item is closed.** The image rows' 26% at block 15 is the
    sim's 26.21% to a tenth of a point. Block 31 and the target agree as
    closely.
  - **What was not done.** Nothing was tuned. A lever, if the owner wants
    the background clean, would be a finer precision on the condition
    image's path (its img_in and early blocks). It is not built.
  - **For the owner's eye**, in
    `F:/ai/models/ignis-image-2.1-turbo/freerun/edit/`:
    [ignis](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/ignis.png),
    [bf16](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/diffusers-bf16.png),
    [sim](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/diffusers-sim.png),
    [fp32](file:///F:/ai/models/ignis-image-2.1-turbo/freerun/edit/diffusers-fp32.png).
- **AC 10, the timing run.** It ran on dbc760e under the GPU lock in
  timing mode, with no build or other GPU client running. The board sat at
  its 575 W limit, SM clocks 2.4-2.9 GHz. The test is
  `crates/server/tests/image_leaf_timing_gpu.rs` (`IGNIS_IMAGE_TIMING=1`).
  Each job is timed on its second run (the first warms), with the calls the
  server makes.

  | job | prefix tokens | encode_condition, ms (VAE / tower / text encoder / txt_in+img_in / prefix pass) | a step, ms (min-max) | the DiT loop | vae_decode | the job |
  |---|---|---|---|---|---|---|
  | t2i 1024² | 31 | 42 (0 / 0.7 / 21.6 / 0.2 / 19.3) | 145-152 | **1.20 s** (bar ~1.1 s) | 134 ms | 1.37 s |
  | t2i 2048² | 31 | 42 (0 / 0.7 / 21.6 / 0.2 / 19.2) | 747-802 | **6.11 s** (bar ~6 s) | 646 ms | 6.79 s |
  | edit, 1 image, 2048² | 4,122 | 469 (37 / 219 / 41 / 0.4 / 157) | 809-816 | 6.50 s | 646 ms | 7.62 s |
  | edit, 4 images, 2048² | 16,428 (chunks of 15,488) | 2,162 (150 / 1,040 / 163 / 0.9 / 749) | 982-991 | 7.91 s | 645 ms | 10.72 s |

  - **The tower's column includes the encoder's input upload**, its
    embedding rows. For text to image, that upload is the whole 0.7 ms.
  - **Run to run.** The dry run 40 minutes earlier read 1.18 s and 5.94 s
    for the DiT loop. AC 9's free runs read 1.18 s and 5.91 s. That puts the
    noise at about ±3%.
  - **Per kernel class**, from the kernel leaf's benches at the leaf's
    shapes, run in the same hold:

    | class | bench case | throughput | literature bar | of the bar |
    |---|---|---|---|---|
    | FP8 W8A8 GEMM | a DiT block's 7 linears + 4 quantizations, 4,096 / 16,384 tokens | 498 / 507 TFLOPS (each GEMM alone 479-517 / 511-556) | 690-740 | 67-73% (each 65-81%) |
    | 8-bit attention | the target call at 1024², no condition image | 496 TOPS | 620-640 | 78-80% |
    | | at 2048², 0 / 1 / 10 condition images | 564 / 567 / 575 TOPS | | 88-93% |
    | BF16 (the VAE's convolutions) | the decoder at 1024² / 2048², the encoder at 1024² | 173 / 177 / 169 TFLOPS (the best stages 190-197) | 210-240 | 70-84% |

  - **A step, rebuilt from the benches** (ms: GEMM + attention target
    calls + elementwise, against the median step measured):
    - t2i 1024²: 114.8 + 21.6 (with its quantizers) + 8.2 = 144.6, against
      150.0;
    - t2i 2048²: 450.9 + 252.6 + 55.6 = 759.1, against 758.0;
    - edit with 1 image, 2048²: 450.9 + 312.8 + 55.6 = 819.3, against
      813.2;
    - edit with 4 images, 2048²: 450.9 + about 493 (interpolated between
      the bench's 1 and 10 images) + 55.6 = about 999, against 988.6.

    At 2048² a step is 59% FP8 GEMM, 33% attention and 7% elementwise.
    The linears at the GEMM bar would take the step to about 630 ms and
    the loop to about 5.0 s.
  - **The VAE decode** at 2048² takes 646 ms against its convolutions'
    383 ms alone. The halo's recompute (1.33x, 321c) accounts for about
    509 ms of that. The rest is the per-band weight packing, the norms, the
    mid block and the image's trip to the host.
  - **The edit's encode is the tower's.** The tower is 47-48% of
    `encode_condition` at 1 and at 4 images, at about 22 TFLOPS effective
    (spec 03's 320c). The prefix pass is 33-35%.
  - Information, not a gate. Nothing was tuned.
- **Defaults taken** (the owner may revisit):
  1. **The DiT's timestep is the recorded one**, the product by the
     reciprocal. Review-321ab wrote fl(fl(1000σ)/1000), but a true division
     is 1 ulp off the recording at four steps.
  2. **`image.min_condition_budget_tokens` is `[Visible, Patchable]` and
     not `ReloadRequired`, like `image.sizes`.** The load reads it, so a
     change holds from the next load. A reload on its change would reload
     whatever model is loaded, a text model included, for an image-only
     field. The minimum is checked against the planned budget before the
     cap.
  3. **AC 10's "edit with 4 condition images at 2048²"** is four 1024²
     condition images and a 2048² target. The server encodes a condition
     image at about 1024² of area, and four 2048² conditions would be
     65,536 tokens, past this card's budget.
  4. **The timed encode drains the stream between phases**
     (`encode_condition_timed`); the server's call does not. Each drain
     costs microseconds against phases of milliseconds.
  5. **The kernel classes come from the benches**, standalone at the
     leaf's shapes, not from inside the leaf: nsys hangs on Windows. Their
     sum lands within 1-4% of each measured step.
  6. **The timing test runs only under `IGNIS_IMAGE_TIMING=1`**, so the
     GPU sweep does not spend a minute on it.
- **For the live run (spec 05 AC 14)** and the next agent:
  - Through HTTP, a 1024² generation takes 1.64 s and an edit 2.21 s
    (`image_runner_gpu`, this run).
  - The background of an edit that asks to keep its transparency shows the
    condition's band faintly, alpha ≤ 18 (above). The sim says that is
    W8A8's.
  - Still open, from 322g: a runner that panicked keeps serving. With the
    real leaf, a panic may leave device state behind, and the adapter could
    then fail the model. (321f: done.)
  - Still open, from 321d: spec 05's fit corner (ten 16380x100 panoramas).
    (321f: fixed.)
- **Gates** (on dbc760e; the docs commit follows):
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2973 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server --features cuda --tests` and `-p
    ignis-core --features cuda --tests`: ok, with only the old warnings.
  - Under the profile, the image GPU tests 23/23: core's `image_attn_gpu`,
    `image_elementwise_gpu`, `image_encoder_attn_gpu`, `image_encoder_gpu`
    (2), `image_gemm_gpu`, `image_vae_gpu`, `image_leaf_gpu` (9); the
    server's `image_encoder_edit_gpu` (2), `image_leaf_ten_images_gpu`,
    `image_runner_gpu` (3), `image_leaf_timing_gpu`.
  - The 27B's vision tests 7/7 under the profile.
  - `npm test` in `web/`: 860 passed.

## As built (321f, 2026-10-14: the review-321d close-out)

Status: no AC changes status. ACs 1-11 stay done as 321e left them. AC 1's
plan line is now the message of a named event (L3, below). The work is
review-321d's F1, F2, L1-L3 and its two information items
(`.scratch/orchestra/reviews/review-321d.md`), and 322g's note on a runner
that panics. Commits 05b6c7c..HEAD on `image-leaf-321`.

- **F1: the count carries a reason.** `ImageRunner::condition_tokens`
  returns `Result<u64, ConditionUnfit>` (`ignis_core::image`). The magic
  `budget + 1` (321d's default 3) is gone. The three outcomes:
  - `Ok`: the prefix's tokens, also past the budget (the fit scales for
    that);
  - `NoRoom(why)`: within the budget, but past a limit beside it, the
    arena (`ImageLeaf::condition_fits`) or the RoPE table (L1). Smaller
    images may fit, so `budget::fit` scales them;
  - `Refused(why)`: a size the processor would not keep
    (`ImageProcessor::keeps`: past 200:1, off the grid, under its floor), a
    prompt the template refuses, or a limit beside the budget with no image
    to scale. The answer is `400 image_refused` in those words, before
    anything is encoded. A 250:1 source now reads "condition image 0 is
    16192x64: the encoder's processor refuses it (... at most 200:1)", not
    a budget message with the wrong advice.
- **The fit (spec 05).** `budget::fit` returns `NotFitted`:
  - `OverBudget` and `NoRoom { floor, why }` are `400
    condition_budget_exceeded`; `Refused` is `400 image_refused`.
  - A refusal at 1024² is final: those are diffusers' own sizes.
  - Below 1024², the fit bisects without reading the floor first. It
    remembers the largest side it saw fit, and a refused size counts as one
    that does not fit.
  - So 321d's corner is fixed. Ten 16380x100 panoramas over a 30,000-token
    budget land on 11968x64 (CPU test), where the floor's own 6560x32 is
    past 200:1. On the 40 GiB budget the GPU test fits them from the default
    512 floor.
  - Where 200:1 is not monotone in the area, the fit may land below the
    largest side that fits, never on one that does not.
- **L1: the RoPE table.** The count runs the kernel's own rule
  (`image_dit::rope_positions`, host-side) over the prefix laid out in a
  fixed order: text first, then each image's block, then a one-cell target
  (`target_rope_layout`).
  - The order moves no position the rule checks: a text token moves the
    position by one, an image block by its longer side.
  - A target's shape moves only its own rows. `image.sizes` keeps its sides
    within the table (F2).
  - Past the table, the count is `NoRoom` with images and `Refused` with
    none. So a 9,000-word prompt is refused at the fit, before the encode.
  - GPU: five 12800x64 images take 4,000 positions. With ~4,300 more words
    the count is `NoRoom`, and the fit scales the images until the target
    has a position.
- **F2: the arena's cap.** `image.max_target_cells` is a new field:
  - latent cells, 1..=2^20, default 16,800 (`MAX_TARGET_TOKENS`);
  - `[Visible, Patchable]`, read by the load as `image.sizes` is.

  `max_target_cells(sizes, cap)` is the largest size within the cap, or the
  cap when none is. An oversized `image.sizes` entry no longer grows the
  arena or refuses the next load, an idle unload's reload included. It is
  not served, and `ignis.image.sizes_not_served` fires at load.
  `image.sizes` refuses a side past 16,384 pixels (`MAX_IMAGE_SIDE`): at
  most 2^20 cells, and a target's RoPE block within the table. Without
  `cuda` the fake plans what the leaf would, so a CPU test covers the
  warning.
- **L2.** The `Send` comment says the runner is sound for device 0 only.
  Kernel launches and the GEMM's `require_prepared` run on the thread's
  current device.
- **L3.** `images::leaf::log_plan` emits the load's plan as one
  `ignis.image.plan` event. Its fields are `mode`, `free_at_start_bytes`,
  `budget_bytes`, `weights_bytes`, `arena_bytes`, `condition_budget_tokens`,
  `condition_budget_bytes`, `max_target_cells` and `load_s`; its message is
  the plan's line. The leaf's own unnamed line is gone; core's GPU tests
  print the line themselves.
- **Information items.**
  - One on-white blend (`image_encoder_input::blend_on_white`) serves the
    opaque answer and the encoder's copy. It is checked against the
    nearest-byte formula for every colour at every alpha.
  - `image_runner_gpu` adds n = 2 through HTTP: image 1 of seed 7 is image 0
    of seed 8, byte for byte.
- **322g's note: a runner that panics fails its model.**
  - `job::spawn` catches the panic on its blocking thread. The prefix was
    already released as the job unwound.
  - The queue closes, so no job behind it reaches the runner, and the
    runner is dropped. Both happen before the job stops counting. The
    request still gets `500 engine_error`.
  - `model_switch::unload_failed` then unloads the model as the idle clock
    does. It acts only while that model is still the one serving, under the
    switch's slot. The status reads `unloaded`, `ignis.model.unloaded`
    carries the reason, and the next request loads the model again
    (`model_kind.rs`).
  - Without a switcher (tests only) the model stays closed: `503
    model_stopped`.
- **Defaults taken** (the owner may revisit):
  1. **A refusal at 1024² is final**, even where a smaller common area
     would pass 200:1: diffusers refuses those sizes too.
  2. **No room even at the floor is `condition_budget_exceeded`**, with the
     runner's words, not `image_refused`. The remedy is the budget's: fewer
     or smaller images, or a shorter prompt.
  3. **The fit bisects** and does not scan every side.
  4. **With no `image.sizes` entry within the cap, the arena is the cap.**
  5. **A runner that panics unloads its model** (status `unloaded`, not
     `failed`), so the next request loads it again. The job's
     `ignis.image.job_panicked` names the panic.
  6. **The RoPE check's target block is one cell.** `MAX_IMAGE_SIDE` keeps
     every served target's rows within the table.
- **For the live run (spec 05 AC 14).**
  - Build first: the server running holds its exe, so stop, `make build`,
    start.
  - Through HTTP on this run: the load takes 8.6 s (condition budget 43,840
    tokens); a 1024² generation 1.71 s, n = 2 3.37 s; a 1024² edit 2.24 s.
    From 321e's AC 10, a 2-image edit at 2048² should land between 7.6 s
    (one image) and 10.7 s (four).
  - The plan is the `ignis.image.plan` line at load.
  - A condition image past 200:1 answers `400 image_refused`. If a job
    panics, the model unloads, and the next request loads it again
    (~9 s).
- **Gates** (the GPU runs and the cuda check on 31ed17b; the workspace and
  `npm test` on 23b8d14, the same code but for one `cuda`-only test's
  tolerance):
  - The CTest under the profile (`kernel/build.ps1 -Test`,
    `IGNIS_GPU_PROFILE=1`): 124/124.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2980 passed, 0 failed,
    6 ignored.
  - `cargo check -p ignis-server -p ignis-core --features cuda --tests`:
    ok, with only the old warnings.
  - Under the profile, the image GPU tests 23/23: core's 16 (as 321e's)
    and the server's `image_encoder_edit_gpu` (2),
    `image_leaf_ten_images_gpu`, `image_runner_gpu` (3),
    `image_leaf_timing_gpu`.
    - `image_runner_gpu`'s runner-drop check failed once at +84.6 MiB
      against a 64 MiB tolerance. Run alone, it passed at -46.5 MiB. NVML's
      free memory includes the desktop's own use, so the tolerance is now
      256 MiB; a prefix left behind reads about -1,042 MiB.
  - The 27B's vision tests 7/7 under the profile.
  - `npm test` in `web/`: 860 passed. `mk/flags-selftest.sh`: ok.
