# 01 — the image artifact: the converter, the published release, and the diffusers recordings

GitHub: (on the owner's OK). ADR 0050, ADR 0051. Feature: `spec.md`.

## Problem

ignis loads `.ninfer` containers. Qwen-Image-2.1-Turbo ships as diffusers
safetensors in bf16 (32.5 GB): a Qwen3-VL text encoder, the DiT and the VAE.
Nothing in the repo converts it. Without a recording of what diffusers
computes, specs 03 and 04 have nothing to be checked against.

## Solution

An offline Python tool, `tools/image-converter/`, does three jobs:
1. It turns the pinned checkpoint into one `.ninfer` v2 artifact of family
   `Qwen-Image-2.1`.
2. It records diffusers in fp32 on three fixed cases. These recordings are the
   oracle for specs 03 and 04.
3. It records diffusers in bf16, teacher-forced, as the yardstick for the
   end-to-end tolerance.

The artifact is published on Hugging Face as `gpillon/ignis-image-2.1-turbo`.

## Seams

- **The container.** `.ninfer` v2 as `docs/specs/flash-next/layout.md`
  defines it. The header family is `Qwen-Image-2.1`, and `crates/artifact`
  reads it as a new `ModelFamily` variant (`QwenImage21`,
  `crates/core/src/compute.rs`) with its own object list. The binder requires
  every object the family declares and refuses an artifact that lacks one, by
  name.
- **What is stored, and how:**

  | part | encoding |
  |---|---|
  | DiT block linears (QKV, out, SwiGLU gate/up, down), `img_in`, `txt_in`'s two linears, `proj_out` | FP8 E4M3, one scale per output row |
  | text encoder decoder-layer linears (36 layers) | FP8 E4M3, one scale per output row |
  | DiT shared modulation (4096→16384), timestep embedder, all norms | bf16 |
  | token embedding (151936 × 4096) | bf16. It is read on the host and only the prompt's rows are uploaded |
  | vision tower + merger + the three DeepStack mergers | the dtypes the 27B's vision path reads (`kernel/src/vision_encode.cu`) |
  | VAE encoder and decoder, `quant_conv`, `post_quant_conv` | bf16 |
  | `frontend/tokenizer.json` | the checkpoint's Qwen3-VL tokenizer, as the 27B's artifact carries its own |
  | metadata | the processor's image config, `sample_sigmas`, the scheduler config, `latents_mean`/`latents_std`, the system prompt and both templates verbatim, the checkpoint id and revision |

  Not stored: `lm_head`, and the text model's final norm, whose input is the
  output we use.
- **`conversion.json`** beside the artifact, like Flash-Next's:
  - every tensor's source name, encoding and FP8 round-trip relative error;
  - the VAE's convolution shapes (kernel, stride, padding, channels), which
    spec 02 implements;
  - the checkpoint revision;
  - the tool's own commit.
- **The recorder** (`tools/image-converter/record.py`):
  - **Setup.** diffusers `QwenImage21Pipeline` on the 5090, with sequential
    CPU offload (fp32 does not fit in VRAM: the DiT alone is ~28.5 GB, the
    encoder ~33 GB). TF32 is off for both matmul and cuDNN. `use_kv_cache=True`
    is the layout ignis computes. Batch 1. Noise is read from a file the
    recorder writes from a fixed numpy seed.
  - **Three cases.**
    - **t2i:** one fixed prompt at 1024².
    - **t2i-2k:** the same prompt at 2048², the size of the checkpoint's
      presets.
    - **edit:** one fixed prompt with one RGBA condition image (a
      checked-in test image with real transparency) at 1024².
  - **What it saves per case:**
    - the token ids, `drop_idx` and `image_pad_mask`;
    - the encoder's last hidden state before the final norm;
    - the condition latents;
    - the processor's `pixel_values` and `image_grid_thw`;
    - the encoder's mrope position ids;
    - the DiT's RoPE table and its segment boundaries;
    - the timesteps;
    - for each denoising step, the input latents, the predicted velocity and
      the output latents;
    - the outputs of DiT blocks 0, 15 and 31 in denoising steps 1 and 2;
    - the decoded RGBA image as PNG.
  - **The traps it must reproduce, each named in a comment:**
    - the prompt is built from the raw template string, not
      `apply_chat_template`;
    - `drop_idx` is the length of the system message tokenized through the
      chat template;
    - an empty prompt becomes `" "`;
    - the encoder output is taken **before** the final norm. diffusers'
      forward hook does this; transformers ≥ 5 otherwise returns the normed
      state;
    - the encoder sees each condition image composited on white, while the
      VAE reads all four channels.
- **The bf16 yardstick** (same tool). The same pipeline in bf16,
  **teacher-forced**:
  - at each denoising step i, the recorded fp32 input latents go in;
  - the bf16 velocity is compared with the fp32 velocity of that step.

  It writes the per-step relative L2 error and the end-to-end threshold,
  **5 × that error per step**, to `thresholds.json` before ignis ever runs
  (spec 04). A free-running comparison would measure the sampler amplifying
  one ULP, not the error: the pipeline's docstring says even the cache on/off
  switch gives "visibly distinct samples" in reduced precision.
- **Publication.** The Hugging Face repo `gpillon/ignis-image-2.1-turbo`
  holds:
  - the artifact and `conversion.json`;
  - the checkpoint's `LICENSE` (Qwen Research License);
  - a card that says "Built with Qwen", states non-commercial use only, links
    the source checkpoint and points to ignis.

  It is created private, made public after the same privacy scan as
  Flash-Next's release (no local paths, no user names in any file), and
  tagged at a pinned revision. The built-in catalog entry is spec 05's.

## Acceptance

1. `tools/image-converter/convert.py` converts the checkpoint at revision
   `fc13ba1986fcb905854dc8de2f52eee395f49b5b` into one `.ninfer` v2 file of
   family `Qwen-Image-2.1`. The table above holds tensor by tensor. It writes
   `conversion.json` with the fields listed, and refuses a checkpoint whose
   revision or tensor list differs from the pinned one, naming the difference.
2. No tensor has a NaN or an infinity after conversion. Every FP8 tensor's
   round-trip error is in `conversion.json`. The conversion fails on a tensor
   whose error is above a bound given on the command line (default from the
   27B's FP8 linears).
3. `crates/artifact` binds the artifact. `ModelFamily::QwenImage21` is read
   from the header. A missing object is refused by name. A Rust test binds a
   tiny synthetic artifact (2 DiT blocks, 2 encoder layers, a small VAE) that
   the converter writes from a synthetic checkpoint.
4. The converter's own tests (pytest, CPU) cover FP8 row scaling, the tensor
   map and the refusal paths on the synthetic checkpoint.
5. The recorder writes the three cases with every field listed, under
   `F:/ai/models/ignis-image-2.1-turbo/oracle/`, with the noise file and a
   manifest that pins the versions of torch, diffusers and transformers. It
   asserts TF32 off and `use_kv_cache=True`. Each trap above has a test or an
   assertion.
6. `thresholds.json` holds the bf16 teacher-forced per-step error, the 5×
   threshold for each case, and the encoder's bf16 error. It is written in
   the same session as the fp32 recording and never edited after ignis
   results exist.
7. The Hugging Face repo exists with the files and card above, passed the
   privacy scan, and is pinned at a revision. Its sha256s are recorded for
   spec 05's catalog entry.
8. The GPU runs (recordings) took the shared GPU lock. Their wall times are
   written in this spec's "As built".
