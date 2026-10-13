# 01 — the image artifact: the converter, the published release, and the diffusers recordings

GitHub: #318. ADR 0050, ADR 0051. Feature: `spec.md`.

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

## As built (2026-10-13, branch `image-artifact-318`: AC 1-4)

Where the implementation chose what the sections above leave open. AC 5, 6
and 8 (the recorder) are in the next section; AC 7 (publication) is not done.

- **Who writes the container.** `tools/image-converter/convert.py run`
  writes work units (`frontend`, `text_encoder`, `vision`, `transformer`,
  `vae`: `tensors.json`, one payload file per tensor, `DONE`, as
  `docs/specs/flash-next/layout.md` §2 and §6), then runs the packer
  (`ignis-artifact-pack --family qwen-image`, `Family::QwenImage21`), which
  assembles the container through the crate's writer and merges
  `converter.json` into `<artifact>.conversion.json`. Python never writes
  the container, as for Flash-Next.
- **Identity.** The header names the family through its `model_id`:
  `ignis-image-2.1-turbo`, which is also the id it is served under by default
  (ADR 0050: "Qwen" is never its primary name); `weights_id`
  `fp8rows-bf16-fc13ba1`. `ModelFamily::QwenImage21` (`name()`
  `Qwen-Image-2.1`) is read from it by `ModelFamily::of_model_id`.
- **Object names.** `text_encoder/` + the checkpoint name without
  `model.language_model.`; `transformer/` and `vae/` + the checkpoint name;
  the vision tower under the 27B artifact's own names (`vision/layers/{b}/...`,
  `vision/merger/...`) and the DeepStack mergers as `vision/deepstack/{i}/...`.
  `frontend/` holds the tokenizer, `tokenizer_config.json`, the chat
  template, the checkpoint's configs (`model_index.json`,
  `processor_config.json`, `scheduler_config.json`,
  `text_encoder_config.json`, `text_encoder_generation_config.json`,
  `transformer_config.json`, `vae_config.json`) and `image_model.json`
  (sample sigmas, scheduler, image processor, `latents_mean`/`latents_std`,
  the system prompt and both templates verbatim from diffusers 0.41.0, the
  source revision).
- **Encodings the table leaves open.** `norm_out.linear` (`norm_out`'s
  modulation from the timestep embedding) is bf16. The vision tower takes
  the 27B's formats: Q6 patch embedding stored `[1152, 1536]`, Q4 qkv/fc1,
  Q5 output/fc2, W8 merger and DeepStack mergers, all `row-split-k128-v1`,
  bf16 biases, norms and positions; the encoder is adapted from the
  reference converter (gpillon/ninfer `a00648c`, Apache-2.0) and bit-exact
  with it. The token embedding is bound on the host.
- **The pin.** `tools/image-converter/pin.json`: every file the converter
  reads with its size and sha256 (all 16 matched the Hub's LFS oids), every
  tensor's dtype and shape from the safetensors headers. The revision check
  is the snapshot directory's name plus those hashes and the tensor list.
- **The FP8 bound.** `--fp8-bound` defaults to 0.03: the 27B's FP8 linears
  (`unsloth/Qwen3.8-27B-NVFP4@7d6f8d4` against `Qwen/Qwen3.8-27B@1d4bf0f`,
  ten tensors, `tools/image-converter/fp8_bound.py`) measured 2.659-2.663%
  relative Frobenius error, rounded up.
- **The artifact.** `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`,
  16,523,373,824 bytes, 1,294 objects: 1,283 tensors (480 FP8, 13.96 GB; 686
  bf16, 2.13 GB; 54 Q4, 54 Q5, 1 Q6, 8 W8, 0.41 GB) and 11 resources.
  - sha256 `ea5f971917d31db13edeedf84b99699d6903e846f8d877ffef523f0f6215dfd4`;
  - its `.conversion.json` (618,239 bytes), sha256
    `a60d7e46a5575977417b6f14ee82317891a53c5388ab6ac2bb9bb3c4b5f1bd0f`;
  - converted by commit `1e1ef04` (clean), torch 2.13.0+cu130, diffusers
    0.41.0, transformers 5.17.0.
- **Numbers.** No NaN or infinity in any source tensor or decoded payload.
  FP8 round-trip error 2.591-2.649% (mean 2.644%; the worst, text encoder
  layer 13's `v_proj`), all under the bound. The vision tower's grouped
  formats: Q4 10.4-16.2% (the worst `vision/layers/0/attention/qkv`), Q5
  5.0-6.7%, Q6 2.5%, W8 0.54%. `conversion.json` records 88 VAE
  convolutions.
- **Wall times (CPU only).** Download 351 s; the pin's hashes 63 s; the
  conversion 443 s (hashing included, beside a workspace build); the pack
  79 s; the bind 1.1 s (`crates/artifact/tests/qwen_image.rs`
  `the_real_artifact_binds`).
- **For publication (AC 7).** `converter.command` in the sidecar holds local
  paths, and `frontend/vae_config.json` (verbatim) carries the upstream
  `_name_or_path` `/cpfs02/user/...`: the privacy scan will flag both. The
  `.pack.lock`, `.pack-state.json` and `work/` beside the artifact are not
  for publication.

## As built (2026-10-13, branch `image-artifact-318`: AC 5, 6, 8, the recordings)

- **The recordings.** `F:/ai/models/ignis-image-2.1-turbo/oracle/`, 2.89 GB,
  from `tools/image-converter/record.py run` at commit `7075b12` (clean),
  session `2026-10-13T18:08:17Z`:
  - one directory per case (`t2i`, `t2i-2k`, `edit`): every field of the
    list above as `.npy` (little-endian, C order, masks as uint8), indexed
    with shapes and dtypes in `case.json`; the decoded `image.png`; for edit,
    the condition image as `condition.png`;
  - `thresholds.json` (written once: the recorder refuses an existing one);
  - `manifest.json`, written last: torch 2.13.0+cu130, diffusers 0.41.0,
    transformers 5.17.0 (and the rest), the TF32 flags, the attention
    check, wall times and peaks, and `record.FIELDS`, which describes each
    file. `run.log` is the run's output.
- **The cases.** `PROMPT` (a fox holding a sign that reads "ignis") at 1024²
  (seed 1) and 2048² (seed 2); `EDIT_PROMPT` (the shapes in glass, keep the
  transparent background) at 1024² (seed 3) on
  `tools/image-converter/fixtures/edit-condition.png`: procedural
  (`record.edit_image(1024)`, a test checks the pixels), 7.6 kB, alpha 0,
  partial and 255, non-white under alpha 0. At 1024² the pipeline's resize is
  the identity.
- **Layouts the spec leaves open.**
  - The noise is `(1, 64, H/16, W/16)` float32,
    `numpy.random.default_rng(seed).standard_normal`; the pipeline reads it
    packed, `packed[0, y*w + x, c] = noise[0, c, y, x]`. Latents and
    velocities are packed, `(1, N, 64)`.
  - Steps are 0-based: the spec's "steps 1 and 2" are `step0` (the prefill,
    kv mode `extract`, the whole joint sequence) and `step1` (the first
    `cached` step, the target rows only). Block outputs are
    `step{0,1}/block{00,15,31}.npy`.
  - The RoPE table is `rope.npy`, `(S, 64, 2)`: (cos, sin) as the real and
    imaginary parts of diffusers' complex table. The segments, `img_shapes`
    and the kv modes per step are in `case.json`; `image_ids`,
    `target_token_mask` and `joint_image_mask` are `.npy`.
  - `encoder_hidden_states` is the whole sequence before the final norm;
    `prompt_embeds` is it without the first `drop_idx` (14) rows, what the
    DiT reads. `condition_vae_input` is the VAE's exact input (RGBA in
    [-1, 1]).
- **Memory: fp32 arithmetic on bf16 weights**, instead of the sequential CPU
  offload the Setup names (the fp32 pipeline is ~65 GB, host RAM 64 GB):
  - the token embedding, the 36 text layers, `lm_head` and the 32 DiT blocks
    hold the checkpoint's bf16 and are upcast to fp32 inside their own
    forward; everything else is fp32;
  - each call is split at `encode_prompt`: an encoder phase (every prompt,
    the encoder on the GPU, the DiT in host RAM), then the encoder is
    dropped and a DiT phase replays those outputs up to the final latents,
    then the DiT is dropped and a decode phase runs the pipeline's own
    decode tail (`decode_image`) with the VAE alone;
  - the VAE's single-frame causal cache keeps nothing (`_NoRetain`): it
    holds a copy of every convolution's input that one frame never reads;
  - the 2048² decode runs on the host (fp32, CPU, 33.5 GiB peak working
    set): on the GPU it ran out of the 24 GiB cap, and 4× the 1024²
    decode's 6.7 GiB of activations is ~28 GiB. The other decodes run on the
    GPU.
  - `test_record.py` shows each of these bit-identical to
    `from_pretrained(torch_dtype=float32)` and one call per case, on a tiny
    checkpoint built around the pinned processor; the host decode differs
    from a GPU one by rounding only.
- **Traps.** Each is asserted on the run (`_check_prompt_traps`,
  `_check_denoise_traps`) and pinned by a test in `test_record.py`. The raw
  template and `apply_chat_template` differ only with an image (`<image1>`
  before `<|vision_start|>`).
- **Precision checks.** TF32 off through the legacy flags (torch 2.13 refuses
  to read them after a mix with the new `fp32_precision` API), asserted
  through both: `allow_tf32` False for matmul and cuDNN,
  `float32_matmul_precision` highest. The DiT's attention runs under
  `sdpa_kernel(EFFICIENT_ATTENTION)` only, which raises instead of falling
  back to the math kernel; a self-check against the math kernel gave a
  1.07e-6 max difference. `use_kv_cache=True` is asserted from the kv modes
  (`extract`, then `cached` ×7).
- **thresholds.json** (relative L2 over the whole tensor, the bf16 pipeline
  teacher-forced, threshold = 5 × the error):

  | case | bf16 velocity error per step, % | encoder, % |
  |---|---|---|
  | t2i | 1.17 4.49 2.36 1.22 0.76 0.73 0.73 0.67 | 8.78 |
  | t2i-2k | 2.20 4.70 1.91 1.18 0.58 0.63 0.62 0.60 | 8.78 |
  | edit | 2.08 1.78 1.48 1.16 0.69 0.61 0.64 0.73 | 39.2 |

  The edit case's condition latents: 1.48%. Its encoder error is over every
  row of `prompt_embeds`, 97% of whose squared norm is the 1024 image-pad
  rows, which the DiT discards; the vision path alone carries 16% in bf16
  (below). Its 5× bound, 196%, cannot fail spec 03's edit test, DeepStack
  off included: a bound over the text rows needs a bf16 baseline the
  recording does not hold (the owner's call).
- **The vision tower's formats (informational, `vision_probe.py`).** The
  edit case's 4096 patches through the tower, the merger and the DeepStack
  mergers on the CPU, against the checkpoint's weights in fp32 (relative L2;
  all 117 grouped payloads matched the artifact record's sha256s):

  | weights / compute | merger | DeepStack 8, 16, 24 |
  |---|---|---|
  | the artifact's Q6/Q4/Q5/W8, fp32 | 41.8% | 12.6%, 21.5%, 35.1% |
  | the checkpoint's, bf16 | 15.9% | 1.6%, 4.4%, 5.5% |

  The merger's bf16 error is 3× the last DeepStack tap's, so the merger's
  whole-tensor error is likely outlier-dominated; the formats still cost
  2.6-8× bf16.
- **Wall times (AC 8)**, one session under the GPU lock (2026-10-13
  20:08-20:17), 494 s in all: load 30 s; encoder phase 0.3, 0.1, 1.0 s in
  fp32 and 0.2, 0.1, 0.4 s in bf16 (t2i, t2i-2k, edit); encoder to DiT 13 s;
  DiT fp32 11.7, 69.8, 16.1 s; DiT bf16 3.7, 23.4, 5.2 s; decode 0.7 s
  (GPU), 316 s (host), 1.4 s (GPU). Peak VRAM of the process (GiB, under a
  24 GiB cap, the desktop's ~2.3 aside): encoder 19.8, 19.8, 21.3; DiT 16.8,
  20.2, 22.0. Four earlier holds (19:47-20:01) failed and taught the
  shape above: the cap's device index, the encoder's `lm_head` cast-back
  (OOM), a forward pre-hook that returned the module, and the 2048² decode
  (OOM).
- **The venv** gains torchvision 0.28.0+cu130: the checkpoint's processor
  (`Qwen2VLImageProcessorFast`, `Qwen3VLVideoProcessor`) does not build
  without it.
