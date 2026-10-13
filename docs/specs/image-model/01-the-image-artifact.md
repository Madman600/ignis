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
3. It measures two yardsticks against that oracle, teacher-forced:
   diffusers in bf16, and the pipeline computing in fp32 on the artifact's
   own values with ignis's W8A8 quantization simulated. The end-to-end
   tolerance comes from both.

The artifact is published on Hugging Face as `gpillon/ignis-image-2.1-turbo`.

## Seams

- **The container.** `.ninfer` v2 as `docs/specs/flash-next/layout.md`
  defines it. The header names the family `Qwen-Image-2.1` through its
  `model_id` (`ignis-image-2.1-turbo`; the header has no family field), and
  `crates/artifact` reads it as a new `ModelFamily` variant (`QwenImage21`,
  `crates/core/src/compute.rs`) with its own object list. The binder requires
  every object the family declares and refuses an artifact that lacks one, by
  name.
- **What is stored, and how:**

  | part | encoding |
  |---|---|
  | DiT block linears (QKV, out, SwiGLU gate/up, down), `img_in`, `txt_in`'s two linears, `proj_out` | FP8 E4M3, one scale per output row |
  | text encoder decoder-layer linears (36 layers) | FP8 E4M3, one scale per output row |
  | DiT shared modulation (4096→16384), `norm_out.linear` (`norm_out`'s modulation from the timestep embedding), timestep embedder, all norms | bf16 |
  | token embedding (151936 × 4096) | bf16. It is read on the host and only the prompt's rows are uploaded |
  | vision tower + merger + the three DeepStack mergers | W8 (`W8G32_F16S`, `row-split-k128-v1`) for every matrix, bf16 biases, norms and positions, under the 27B's object names |
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
- **The yardsticks** (same tool, `record.py yardstick`, against the recorded
  oracle). Both run the pipeline **teacher-forced**:
  - at each denoising step i, the recorded fp32 input latents go in;
  - the step's velocity is compared with the fp32 velocity of that step.

  The two pipelines:
  - **bf16**: diffusers in bf16 on the checkpoint's weights. It has no
    quantization error, so a bound from it alone would fail every step of a
    correct ignis.
  - **simulated W8A8 (sim)**: the pipeline computing in fp32 on the
    artifact's stored values. Each linear the artifact stores as FP8 (480,
    the text encoder's and the DiT's) uses its payload, E4M3 codes × bf16
    row scale, and fake-quantizes its input per token to E4M3 with scale
    amax/448, with the semantics of spec 02's kernel. The vision tower uses
    its W8 payloads, and every other tensor its bf16. Attention is not
    simulated.

  The metrics are relative L2 against fp32:
  - the velocity and the condition latents over the whole tensor;
  - the encoder over its **text rows** only: the positions after
    `drop_idx` whose `image_pad_mask` is 0. The DiT replaces the image-pad
    rows with the condition latents, so their error never reaches it.

  The **bound** ignis is held to is **max(2 × sim error, 5 × bf16 error)**,
  per step, for the encoder and for the condition latents. Both factors are
  flags with these defaults. `thresholds.json` carries both errors, the
  bound, each quantity's metric, the factors, and the oracle's and the
  yardstick's session ids. It is written before ignis ever runs (spec 04).

  A free-running comparison would measure the sampler amplifying one ULP,
  not the error: the pipeline's docstring says even the cache on/off switch
  gives "visibly distinct samples" in reduced precision.
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
6. `thresholds.json` holds, for each case:
   - the teacher-forced per-step velocity error of both yardsticks (bf16
     and simulated W8A8);
   - their encoder error over the text rows, and their condition-latent
     error;
   - the bound max(2 × sim, 5 × bf16) for each of these.

   It also holds the metrics, the factors and the oracle's session id. It is
   written against the recorded oracle before any ignis result exists, and
   never edited after.
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
  modulation from the timestep embedding) is bf16. The vision tower is W8
  G32 `row-split-k128-v1` for every matrix (the patch embedding stored
  `[1152, 1536]`), bf16 biases, norms and positions. The first conversion
  kept the 27B's formats (Q6 patch embedding, Q4 qkv/fc1, Q5 output/fc2, W8
  mergers): its Q4 tensors carried 10.4-16.2% weight error and the merger's
  output 41.8% (the probe below), and the owner made the tower W8
  throughout (2026-10-13). The row-split encoder is adapted from the
  reference converter (gpillon/ninfer `a00648c`, Apache-2.0) and bit-exact
  with it. The token embedding is bound on the host.
- **The W8 tower needs a kernel change.** Every tower linear in
  `kernel/src/vision_encode.cu` goes through ninfer's `ops::linear`. Its W8
  arm, `select_w8_a16_launch`
  (`kernel/vendor/src/ops/linear/w8/w8_dispatch.cpp`), is a finite registry
  of exact (N, K) problems and throws outside it. It registers K 4608 with
  N 2048, 4608 and 5120 (the mergers' fc1 `[4608, 4608]` among them), but
  none of these:
  - the tower's `[3456, 1152]` qkv, `[1152, 1152]` output, `[4304, 1152]`
    fc1 and `[1152, 4304]` fc2 (stored K 4352);
  - the `[1152, 1536]` patch embedding;
  - the image merger's and DeepStack mergers' fc2 `[4096, 4608]`, missing
    under the old formats too.

  The W8 MMA and SIMT kernels look shape-generic (a non-"Full" path takes a
  padded K, as Q5 does at K 4304 today). The change is therefore new
  dispatch arms, plus `linear_workspace_capacity_bytes`, which calls
  `select_w8_launch`, plus their test cases: a recorded vendored patch, for
  spec 03 (or 02).
- **The pin.** `tools/image-converter/pin.json`: every file the converter
  reads with its size and sha256 (all 16 matched the Hub's LFS oids), every
  tensor's dtype and shape from the safetensors headers. The revision check
  is the snapshot directory's name plus those hashes and the tensor list.
- **The FP8 bound.** `--fp8-bound` defaults to 0.03: the 27B's FP8 linears
  (`unsloth/Qwen3.8-27B-NVFP4@7d6f8d4` against `Qwen/Qwen3.8-27B@1d4bf0f`,
  ten tensors, `tools/image-converter/fp8_bound.py`) measured 2.659-2.663%
  relative Frobenius error, rounded up.
- **The artifact (the W8 tower).**
  `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer` is
  16,721,644,544 bytes, 1,294 objects: 1,283 tensors and 11 resources. The
  tensors are 480 FP8 (13.96 GB), 686 bf16 (2.13 GB) and 117 W8 (0.61 GB).
  - sha256 `4205dd1932f0801804b0a544fe1340697e4e0a0dea4a276996bcb981006050af`.
  - Its `.conversion.json` is 618,040 bytes, sha256
    `b96d32efc29102c8d2fcef139b544b0f24758459a250e2c79fc854f569bc13ce`.
  - It was converted from `e1f2351`'s tree plus the W8 recipe, later
    committed as `6500add`. The sidecar records `e1f2351`, dirty, with
    torch 2.13.0+cu130, diffusers 0.41.0 and transformers 5.17.0.
  - The 1,174 payloads it shares with the first conversion are
    byte-identical by sha256. That conversion was commit `1e1ef04`, clean;
    its artifact was 16,523,373,824 bytes, sha256 `ea5f9719...`, and is
    deleted.
  - `vision_probe.py --conversion` re-encoded all 117 W8 payloads from
    `6500add` to the recorded sha256s.
- **Numbers.** No source tensor or decoded payload holds a NaN or an
  infinity. The FP8 round-trip error is 2.591-2.649% (mean 2.644%; the worst
  is text encoder layer 13's `v_proj`), all under the bound. The W8 tower's
  is 0.51-0.78%; the worst is `vision/layers/0/attention/qkv`, which was
  16.2% in Q4. `conversion.json` records 88 VAE convolutions.
- **Wall times (CPU only).**
  - Download 351 s; the pin's hashes 63 s.
  - The first conversion took 443 s, hashing included, beside a workspace
    build; its pack 79 s.
  - The W8 reconversion took 345 s, hashing included; its pack 78 s; the
    sha256 56 s.
  - The bind took 1.1 s (`crates/artifact/tests/qwen_image.rs`
    `the_real_artifact_binds`).
- **For publication (AC 7).** `converter.command` in the sidecar holds local
  paths, and `frontend/vae_config.json` (verbatim) carries the upstream
  `_name_or_path` `/cpfs02/user/...`: the privacy scan will flag both. The
  `.pack.lock`, `.pack-state.json` and `work/` beside the artifact are not
  for publication.

## As built (2026-10-13, branch `image-artifact-318`: AC 5, 6, 8, the recordings and the yardsticks)

- **The recordings.** `F:/ai/models/ignis-image-2.1-turbo/oracle/`, 2.89 GB,
  from `tools/image-converter/record.py run` at commit `7075b12` (clean),
  session `2026-10-13T18:08:17Z`:
  - one directory per case (`t2i`, `t2i-2k`, `edit`): every field of the
    list above as `.npy` (little-endian, C order, masks as uint8), indexed
    with shapes and dtypes in `case.json`; the decoded `image.png`; for edit,
    the condition image as `condition.png`;
  - `thresholds.json`: the recorder's bf16-only file (written once: the
    recorder refuses an existing one), since replaced by the yardstick
    session's bound (below);
  - `yardstick-report.json`: that session's report;
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
- **thresholds.json** (schema v2) comes from `record.py yardstick`, one
  session under the GPU lock: session `2026-10-13T20:43:05Z`, commit
  `eee2433` (dirty: the README's uncommitted text only), on oracle session
  `2026-10-13T18:08:17Z`. It is the bound max(2 × sim, 5 × bf16), with
  relative L2, both yardsticks teacher-forced, and the encoder over its
  text rows. The recorder's bf16-only file is kept as
  `thresholds.2026-10-13T180817Z.json`.

  | case | | velocity per step, % | encoder (text rows), % |
  |---|---|---|---|
  | t2i | bf16 | 1.17 4.49 2.36 1.22 0.76 0.73 0.73 0.67 | 8.78 |
  | | sim | 5.51 7.12 4.09 3.52 3.14 3.02 2.88 3.53 | 17.85 |
  | | **bound** | 11.01 22.47 11.78 7.04 6.28 6.04 5.75 7.06 | 43.88 |
  | t2i-2k | bf16 | 2.20 4.70 1.91 1.18 0.58 0.63 0.62 0.60 | 8.78 |
  | | sim | 4.68 5.65 3.19 2.95 2.79 2.70 2.68 3.40 | 17.85 |
  | | **bound** | 10.99 23.49 9.54 5.90 5.57 5.39 5.36 6.79 | 43.88 |
  | edit | bf16 | 2.08 1.78 1.48 1.16 0.69 0.61 0.64 0.73 | 6.56 |
  | | sim | 6.89 6.27 3.09 2.99 2.74 2.65 2.65 3.81 | 15.16 |
  | | **bound** | 13.78 12.53 7.42 5.97 5.48 5.31 5.29 7.63 | 32.82 |

  - **Condition latents (edit):** bf16 1.48%, sim exactly 0 (the VAE is the
    artifact's bf16, computing in fp32 as the oracle does), bound 7.38%.
  - **Which factor sets the bound:** 5 × bf16 at the first steps (step 1 of
    both t2i cases), 2 × sim everywhere else.
  - **Checks in `oracle/yardstick-report.json`:**
    - every one of the artifact's 1,283 stored tensors was checked against
      its sidecar sha256: the 480 FP8 and 117 W8 payloads read from the
      container, and the 686 bf16 tensors from the loaded models;
    - the fp32 encoder replayed in the session has error exactly 0 against
      the oracle;
    - the bf16 velocity errors are bit-identical to the recorder's.
  - **Where the encoder error lies.**
    - t2i: one row, the first after `drop_idx`, carries 95% of the bf16
      squared error and 70% of the sim one.
    - edit, whole tensor: the 1024 image-pad rows carry 99.9% (bf16) and
      99.3% (sim) of it (39.2% and 31.3% whole).
  - **Spec 03 AC 4's premise does not hold under this bound.** The edit
    encoder in fp32 with DeepStack off has a text-row error of 20.7%. That
    is above the sim error (15.2%) but below the bound (32.8%), so "with
    DeepStack off, the same test fails" cannot discriminate as written (the
    owner's call).
- **The vision tower's formats (informational, `vision_probe.py`).** The
  edit case's 4096 patches go through the tower, the merger and the
  DeepStack mergers on the CPU. They are compared with the checkpoint's
  weights in fp32 (relative L2). All 117 grouped payloads matched the
  artifact record's sha256s.

  | weights / compute | merger | DeepStack 8, 16, 24 |
  |---|---|---|
  | the first artifact's Q6/Q4/Q5/W8, fp32 | 41.8% | 12.6%, 21.5%, 35.1% |
  | **the artifact's W8, fp32** | **4.0%** | **1.1%, 1.7%, 3.3%** |
  | the checkpoint's, bf16 | 15.9% | 1.6%, 4.4%, 5.5% |

  The merger's bf16 error is 3× the last DeepStack tap's, so the merger's
  whole-tensor error is likely outlier-dominated. The W8 tower in fp32 is
  closer to fp32 than a bf16 tower is, at every output.
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
- **Wall times of the yardstick session (AC 8)**, under the GPU lock
  (2026-10-13 22:43-22:47), 227 s in all:
  - load 26.5 s;
  - encoders 0.6, 0.2, 0.7 s in bf16 and 0.2, 0.2, 0.8 s in sim (t2i,
    t2i-2k, edit);
  - the sim swaps 24.9 s (encoder) and 20.2 s (DiT), reading and hashing
    the payloads;
  - encoder to DiT 13.1 s;
  - DiT 3.6, 23.6, 5.2 s in bf16 and 12.5, 73.8, 16.8 s in sim.

  Peak VRAM of the process (GiB): encoder 17.6 (bf16) and 14.9 (sim), DiT
  17.1 (bf16) and 14.9 (sim). The sim holds the FP8 codes at rest and
  dequantizes one linear at a time.
- **The venv** gains torchvision 0.28.0+cu130: the checkpoint's processor
  (`Qwen2VLImageProcessorFast`, `Qwen3VLVideoProcessor`) does not build
  without it.
