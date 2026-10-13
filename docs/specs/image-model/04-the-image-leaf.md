# 04 — the image leaf: the DiT, the condition prefix, the sampler, the VAE, and the end-to-end check

GitHub: (on the owner's OK). ADR 0050, ADR 0009 (Step ABI). Feature:
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
  - one sequential arena, sized for the largest preset (2752 × 1536, 16,512
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
- **Modulation.** The timestep embedder (256 sinusoidal features, then an MLP)
  gives `temb` at σ, and at 0 for the prefix. One shared linear produces the
  four 4096-wide chunks, scale and gate for attention and for the MLP, that
  every block uses.
- **The sampler** (Rust):
  - the sigmas are the artifact's `sample_sigmas` followed by a terminal 0;
  - the DiT's timestep is σ;
  - the update is `x += (σ_next − σ) · v`.
  For this checkpoint the scheduler config has `use_dynamic_shifting: false`
  and `shift: 1.0`, so the schedule is the same at every size. The loader
  refuses an artifact whose scheduler config says otherwise.
- **Noise:**
  - normally a standard normal tensor (1, 64, H/16, W/16) from ignis's RNG
    and the job's seed, drawn on the host and uploaded;
  - in tests, read from spec 01's noise file.
- **The VAE decode at 2K.** The decoder's widest activations at 2048² are
  ~1.2 GB each. After the mid block, which is global, the decoder runs in
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
   condition prefix built by `encode_condition`.
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
