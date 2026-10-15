# 03 — the text encoder: Qwen3-VL-8B's text model, its vision tower and DeepStack

GitHub: #320. ADR 0050. Feature: `spec.md`. Builds on 01
(artifact, recordings) and 02 (FP8 GEMM, encoder attention).

## Problem

The DiT reads the prompt as Qwen3-VL-8B encodes it. Condition images go in
too, as vision context. ignis has the tower's geometry (the 27B's vision path,
`kernel/src/vision_encode.cu`), but not this text model, not DeepStack, and
not the pipeline's exact prompt and image handling. Small slips here, such as
the wrong norm or the wrong template, quietly degrade every image.

## Solution

The image leaf runs the encoder once per image job:
1. **Tokens.** Tokenize the prompt with the artifact's tokenizer into the
   pipeline's raw template (t2i, or ti2i with one
   `<imageN><|vision_start|><|image_pad|><|vision_end|>` per condition image).
2. **Vision.** Run the vision tower on each condition image, composited on
   white, as the Qwen3-VL processor prepares it.
3. **Text model.** Run the 36 layers, adding the DeepStack features after
   layers 0, 1 and 2 at the vision positions.
4. **Output.** Return the last layer's hidden state **before** the final norm,
   without the first `drop_idx` tokens.

The DiT discards the encoder's outputs at the image-pad positions and puts the
condition latents there instead (spec 04). The tower still has to run,
because the text tokens after an image attend to it.

## Seams

- **Tokenizer and template** (Rust, server side): `frontend/tokenizer.json`
  from the artifact, through the binder like the 27B's.
  - The templates and the system prompt are the artifact's metadata,
    verbatim.
  - `drop_idx` is the length of the system message as the chat template
    tokenizes it, computed at load.
  - An empty prompt becomes `" "`.
- **Image preprocessing** (Rust, `crates/server/src/media.rs`):
  - **Size.** Each condition image is first resized to the job's condition
    area, keeping its aspect ratio and rounding each side to a multiple of
    32. The area is 1024² unless the condition budget scales it down (spec
    05).
  - **Encoder copy.** It is then composited on white and prepared as the
    Qwen3-VL processor does, from the processor config in the artifact:
    patch 16, temporal 2, merge 2, mean/std normalization, `image_grid_thw`.
  - **VAE copy.** The VAE reads the same resized image with its alpha
    channel (spec 04).
- **The text model** (leaf, `kernel/src/image/encoder.cu`):
  - 36 decoder layers: RMSNorm, QKV with per-head QK RMSNorm, interleaved
    mrope [24,20,20] at θ 5e6, attention, then a SwiGLU 12288 MLP;
  - FP8 W8A8 GEMMs and the FlashAttention-2 encoder attention from spec 02;
  - the embedding rows gathered on the host and uploaded, with the vision
    features written over the image-pad rows;
  - no final norm and no `lm_head`.
- **The vision tower:** `vision_encode.cu` reused, parametrised instead of
  forked:
  - the merger's output width (4096 here, 5120 on the 27B);
  - the segment bound (10 condition images of up to 1,230 merged tokens each, see
    the defaults);
  - DeepStack: the features after tower layers 8, 16 and 24, each through its
    own merger, which the text model adds after its layers 0, 1 and 2.
- **mrope positions:** the Qwen3-VL rule for text and vision tokens
  (3D positions, the text position continuing after each image), as
  transformers computes them for this processor.

## Acceptance

1. Token ids and `drop_idx` equal the recorded ones for every case of spec 01.
   A CPU test covers an empty prompt, a CJK prompt, a 2,000-token prompt, and
   templates with 1, 2 and 10 condition images.
2. Image preprocessing reproduces the recorded `pixel_values` and
   `image_grid_thw` for the edit case within a stated bound. A CPU test covers
   aspect ratios 1:1, 16:9, 9:16 and a 2-pixel-wide image, and the
   white-composite of an RGBA image whose alpha is 0, 128 and 255.
3. The **t2i case**: the encoder output against the recorded fp32 output has
   an error ≤ the `encoder` bound in spec 01's `thresholds.json`, both
   measured with spec 01's encoder metric over the text rows: the positions
   after `drop_idx` that are not image-pad, from row 1 on, each row's
   relative L2, then their RMS. This is the first milestone. The first row's
   error (`encoder_row0`) is reported beside the yardsticks', in no bound.
4. The **edit case**: with the vision tower and DeepStack on, the error over
   the text rows after the first image pad is ≤ the `encoder_after_vision`
   bound. With DeepStack off, the same test fails; that check proves
   DeepStack matters to the test. In spec 01's yardstick session, the fp32
   encoder with DeepStack off exceeded that bound by only 4% (26.85% against
   25.76%), so the check is narrow, and it is weak: the fail rests on 2 of
   the 18 rows, `<|vision_end|>` (42.0%) and one text row, " the" (60.0%);
   without both, the error is under the bound (95% of it). The error over all the text rows (the
   `encoder` bound) is reported beside it; DeepStack off did not exceed that
   bound (spec 01, As built).
5. The 27B's vision tests pass unchanged after the tower is parametrised.
6. mrope positions equal transformers' for a t2i layout and a 3-image layout
   (CPU test against a recorded table).
7. The encoder's wall time for the t2i case and for a 10-image edit is
   written in "As built", for information.

## As built (320f, 2026-10-14: ACs 1, 2, 6, the front end)

Where the CPU front end chose what the sections above leave open.
- **Where it lives.** `crates/server/src/image_prompt.rs` (tokens and
  template) and `image_encoder_input.rs` (preprocessing, mrope, the whole
  input), not `media.rs`: #322 owns that file, with the RGBA decode and the
  Pillow-exact resize to the condition size. Preprocessing here starts from
  that resized image.
- **The API.** `EncoderFrontend::from_reader(&Reader)` reads
  `frontend/tokenizer.json`, `chat_template.jinja` and `image_model.json`
  (not through `FrontendSet`, whose six resources are the 27B's).
  `input(prompt, &[ConditionImage { width, height, rgba }])` returns an
  `EncoderInput`:
  - `input_ids` (u32, the whole sequence) and `drop_idx`;
  - `image_pad_rows` (u32);
  - `positions` (i32, axis-major (3, L), the oracle's layout);
  - `images`: per condition image its `grid` and f32 `pixel_values`, rows
    of 1536. The 27B's tower reads BF16; the rounding is the uploader's.
    So `EncoderInput.images` holds f32 pixels until then: about 25 MB for
    a 1024² image (4,096 rows), about 250 MB for 10.
- **drop_idx.** `ChatTemplate::render` always adds the generation prompt, so
  the system message is the render less `<|im_start|>assistant\n`: 14
  tokens. The load refuses templates that do not open with it, or that lack
  the prompt's `{}` or the first image slot.
- **Image pads.** The template is tokenized with one pad per slot, then
  each slot's pad is repeated for its image's merged tokens. That equals the
  processor's string expansion, since the pad is a special token split out
  before pre-tokenization; the fixture checks it. A prompt that writes
  `<|image_pad|>` itself is refused: the pipeline would take it for an image.
- **On white.** Pillow's `paste` is `((t >> 8) + t) >> 8` with
  `t = 255·(255 − a) + v·a + 128`, exact for all 65,536 (value, alpha)
  pairs (the recipe asserts it).
- **Size and resample.** `smart_resize` is the 27B's
  (`ignis_artifact::vision::smart_resize`, the same rule: round half to
  even, then scale into [65,536, 16,777,216] pixels). When it changes the
  size, the filter is torchvision's `resize(BICUBIC, antialias=True)` on
  uint8, which runs ATen's integer path: Keys cubic a = −0.5, separable,
  horizontal then vertical through a uint8 intermediate, the f64 weights
  scaled to int16 at the largest precision that fits. The 27B's resize is
  ninfer's float filter, a different one. The filter is ported from PyTorch
  2.13.0's `aten/src/ATen/native/cpu/UpSampleKernel.cpp` (BSD-style, quoted
  in the root `NOTICE`), restated in Rust rather than copied.
- **Normalize.** The processor fuses rescale and normalize:
  `(v − mean/r) / (std/r)` in f32, with `mean/r` and `std/r` taken in f32
  (127.5 here).
- **mrope.** The 27B's `assign_positions`, which is transformers' rule for
  images: text advances all three axes, an image takes (p, p + y, p + x)
  over its merged grid, then p advances by the longer merged side. A pad run
  that is not the next grid's size is refused.
- **Config.** `image_model.json`'s `image_processor` must have the 27B's
  patch geometry (16, temporal 2, merge 2), `resample` 3 and `do_resize`,
  `do_rescale` and `do_normalize` true. The pixel bounds, mean, std and
  rescale factor are read from it. The synthetic fixture's config (patch 4,
  no `size`) is refused.
- **Results: bound 0 everywhere.**
  - Against the oracle, every case of spec 01 (t2i, t2i-2k, edit):
    `input_ids`, `drop_idx`, `mrope_position_ids` and the image-pad rows;
    edit's `image_grid_thw` [1, 64, 64] and `pixel_values` (4096 × 1536),
    all bit for bit.
  - The edit image's RGBA is read back from `condition_vae_input`, and the
    read-back is exact: `2·(u/255) − 1` of the recovered bytes is the
    recording.
  - Against the fixture: the 2-pixel-wide (2 × 64 → 64 × 1472) and odd-size
    (450 × 250 → 448 × 256) resamples are bit for bit, as are 1:1, 16:9 and
    9:16.
- **The fixture.** `crates/server/tests/fixtures/image_encoder_input.json`
  (107 kB) comes from `tools/image-converter/record_encoder_input.py`
  (command in its docstring), run in the venv on the pinned processor: its
  `recipe.versions` record diffusers 0.41.0, transformers 5.17.0, torch
  2.13.0+cu130, torchvision 0.28.0+cu130, Pillow 12.3.0 and numpy 2.5.3, on
  snapshot `fc13ba19` (`Qwen2VLImageProcessor`, CPU capability AVX2). It
  calls the pipeline's own `_get_qwen_prompt_embeds` with a tiny seeded
  text encoder, hooking its inputs and its rotary embedding's positions. It
  covers:
  - prompts: empty, CJK, 2,035 tokens, and 1, 2, 3 and 10 condition images
    (one with an empty prompt);
  - the five preprocessing cases, each as the sha256 of its `pixel_values`;
  - the composite rows for alpha 0, 128 and 255, and the sha256 of the
    whole table.

  Images are a procedural RGBA pattern that the test rebuilds.
- **Tests.** `cargo test -p ignis-server --test image_encoder_input` runs
  the composite, preprocessing and mrope checks everywhere. The token and
  oracle checks skip without the artifact or the oracle
  (`IGNIS_IMAGE_ARTIFACT`, `IGNIS_IMAGE_ORACLE`, else the owner's layout).
  Both modules also carry unit tests on a word-level tokenizer.

## As built (320a, 2026-10-14: AC 3, AC 7 for t2i, the text model on the GPU)

Where the text model chose what the sections above leave open. AC 4 (the
tower and DeepStack on the edit case) and AC 7's 10-image edit are 320b's;
the hooks they fill are below.

- **Where.**
  - ABI `kernel/include/ignis_image_encoder.h`, code
    `kernel/src/image/encoder.cu`, ours (ADR 0051 tier 1).
    `ignis_image_encoder_forward` runs decoder layers
    `[first_layer, end_layer)` in place on the residual stream x, fp32
    `[tokens][4096]`. Beside it: `_workspace_bytes` (about 60 KiB a token,
    0.71 GiB at 12,400 tokens) and three ops of its own, `_residual_norm`,
    `_qk_norm_mrope` and `_rows`. The linears and the attention are spec
    02's ops; the GEMM needs `ignis_image_gemm_prepare` first.
  - Rust: `crates/core/src/image_encoder.rs` (`TextEncoderConfig::parse`,
    `embed_rows`, the bindings, `TextEncoderWeights` over a materialized
    artifact) and `qwen_image::bind_text_encoder`, a scoped bind of the text
    config and the `text_encoder/` tensors.
  - Tests: the CTest `ignis_kernel_image_encoder_test` (the ops and the
    hooks); the Rust `--ignored` `crates/core/tests/image_encoder_gpu.rs`
    (AC 3 on the real artifact, and the bindings); on the CPU, the config's
    parse and the embedding rows (`image_encoder::tests`) and the scoped bind
    (`crates/artifact/tests/qwen_image.rs`).
- **The layer.**
  - RMSNorm to E4M3; q, k and v as three GEMMs on the payloads as stored;
    the QK RMSNorm with the mrope; the causal GQA attention at scale
    fl32(128^-1/2).
  - The attention output quantized, then `o_proj`; the residual add fused
    with the next RMSNorm.
  - Gate and up as two GEMMs, #319's SwiGLU with E4M3 output, `down_proj`;
    the add after a layer's MLP fused with the next layer's input norm.
  - The weights are the artifact's objects where the materializer put
    them, with no fused copy, so spec 04's full bind points the descriptors
    at the same objects.
- **Defaults taken** (the owner may revisit):
  1. The residual stream is fp32, as in the fp32 pipeline the oracle and
     the sim yardstick ran; `o_proj` and `down_proj` write fp32.
  2. The norms quantize their fp32 output to E4M3 directly, with spec 02's
     steps and no BF16 rounding between (the sim quantizes the fp32
     activation).
  3. q and k leave their GEMMs in fp32 and are rounded once to BF16 for the
     attention, after the QK norm and the mrope; v, gate and up are BF16.
  4. **The output is all of x, fp32**: the last layer before the final norm,
     every row. The caller takes the rows from `drop_idx` on, so the
     forward needs no `drop_idx` and a layer range composes. No BF16 output
     was built: `txt_in`'s norm op (spec 02) reads BF16, so #321 adds a cast
     or an fp32 input to it.
  5. **The mrope frequencies** are fl32(1 / fl32(θ^fl32(2i / 128))), each
     step correctly rounded.
     - torch's `powf`, which transformers' `inv_freq` comes from, is one ulp
       up at i = 19 for θ = 5e6 (0x1.504622p-7 against 0x1.50462p-7,
       measured in the venv). That moves an angle by under 4e-5 rad at
       41,000 positions, so the formula stays.
     - The slot's axis is transformers' interleaved rule: h for i % 3 = 1
       and w for i % 3 = 2 below 3 × 20 = 60, t otherwise. Channels d and
       d + 64 are slot d's rotate_half pair.
     - The positions are int32 [3][tokens], rows t, h, w (the oracle's
       `mrope_position_ids`).
  6. **eps, θ and the section come from `frontend/text_encoder_config.json`**,
     read in either key form: the checkpoint's `rope_theta` and
     `rope_scaling`, or transformers 5's `rope_parameters`. Its dimensions
     are checked against the artifact's inventory. A rope that is not the
     default interleaved mrope, or a section that does not cover the head,
     is refused by name.
- **The hooks.** `ignis_image_encoder_vision` holds:
  - the vision rows' sequence positions, device int32 and distinct; feature
    row j goes to row `row_index[j]` (transformers' masked order);
  - `embeds`, BF16 [rows][4096], written over x's rows when the forward
    starts at layer 0;
  - `deepstack[3]`, BF16 [rows][4096] each, added after layers 0, 1 and 2
    (NULL for none).
- **The CTest** checks the ops against fp64 references that track their
  roundings, as spec 02's elementwise test does:
  - **the residual add with the RMSNorm**, 1, 45 and 2,049 tokens: the sums
    are the host's fp32 adds, bit for bit; the norm uses 0.09-0.18 of its
    bound; the E4M3 output is the host's quantization of the fp32 output,
    bit for bit;
  - **the QK norm with the mrope**, 1, 45 and 1,100 tokens, with t, h and w
    positions all different (w up to 40,000): 0.97-0.99 of the bound,
    which the BF16 output's rounding sets.
    - Two wrong layouts fail as negative controls. The axes in chunks
      [24 | 20 | 20] put 63-65% of the outputs over the bound; adjacent
      pairs (2i, 2i + 1) put 89-92% over.
    - The t2i case cannot tell the layouts apart (its three position rows
      are equal), so this is the only check of the axes until AC 4;
  - **the rows**, written over and added, exactly, an index past the
    sequence skipped;
  - **the hooks**, on synthetic FP8 weights (layers A, B, A; 37 tokens, 9
    vision rows):
    - the forward with the embeddings and the three DeepStack adds equals,
      bit for bit, the same additions made on the host between one-layer
      calls. That also checks that a layer range equals its layers run in
      consecutive calls;
    - a second run is identical; without DeepStack every vision value
      changes.
- **AC 3.** The Rust test feeds the recorded `input_ids` and
  `mrope_position_ids` and measures spec 01's encoder metric (text rows from
  row 1 on, per-row relative L2, RMS):

  | case | tokens | encoder | bound | bf16 / sim | encoder_row0 | bf16 / sim |
  |---|---|---|---|---|---|---|
  | t2i | 45 (`drop_idx` 14, 30 text rows) | **9.61%** | 19.50% | 2.03% / 9.75% | 28.77% | 53.72% / 93.86% |
  | t2i-2k | the same prompt | **9.61%** | 19.50% | 2.03% / 9.75% | 28.77% | 53.72% / 93.86% |

  - The error is the sim yardstick's (9.75%): W8A8 quantization, as
    expected, not a bug.
  - **Per layer**, teacher-forced, against transformers' text model run
    layer by layer in fp32 on the CPU, on the checkpoint's bf16 weights
    (`tools/image-converter/ref_layers.py`, run with the venv
    `F:/ai/venvs/ignis-image/Scripts/python.exe ref_layers.py <snapshot>
    <oracle>/t2i <out dir>`; its last layer is within 1.1e-5 of the oracle
    per row, so it reproduces the oracle's computation). The GPU test, with
    `IGNIS_IMAGE_ENCODER_LAYERS` naming `<out dir>`, runs each layer alone:
    it loads the reference's layer l-1 output (the embedding rows for layer
    0) into x, runs layer l, and compares with the reference's layer l
    output. Each figure is that layer's own error, not an accumulation.
    - Text rows from row 1: 1.25-2.4% per layer through layer 33 (2.4% at
      layer 0, 1.25-2.3% in the middle), then 1.9% at layer 34 and
      3.3% at layer 35.
    - Row 0: 1.0-3.4% through layer 16, 0.004-0.22% from layer 17 to 33, then 4.7% at
      layer 34 and 7.2% at layer 35.
    - So the 9.6% of the free-running stack is a per-layer W8A8 error of
      about 1.5-2% that accumulates over 36 layers, with the last layer
      (3.3%, and 7.2% for row 0) the one layer that stands out.
    - An earlier version of this section read the free-running stack's
      per-layer errors and subtracted them to attribute "what a layer adds"
      (3.7 points for layer 35); those errors are cumulative, including
      cancellations and the amplification of earlier error, so the figures
      above replace it.
- **AC 7, t2i.** The GPU lock in normal mode, nothing else on the card; wall
  clock around the forward's launches and a stream sync.
  - **The forward: 21.6-22.0 ms** for 36 layers on 45 tokens, the median
    of 5 warm runs, measured four times (t2i and t2i-2k, two runs of the
    test; min 21.6 ms; a first run 21.9-22.4 ms).
  - The host's embedding rows and uploads: 1-2 ms. The text encoder's
    6.47 GiB materialize in 4.1-4.4 s from a warm page cache.
  - At 45 tokens the GEMM is likely SM-bound (an estimate, not profiled):
    its 128-token tile gives 4 to 48 CTAs on 170 SMs, each computing a full
    tile. A split-K or a narrower token tile would help; neither was built
    (speed is information here).
- **Gates** (5a6790c and this section): `kernel/build.ps1 -Test` with
  `IGNIS_GPU_PROFILE=1`, 121 of 122. The one failure,
  `ignis_kernel_residency_trace_test` (Flash-Next residency, four eviction
  checks; nothing here touches it), passed in three direct runs and in
  `ctest -R residency` (5 of 5). The Rust `image_encoder_gpu` tests pass
  under the profile (2 of 2). `CARGO_INCREMENTAL=0 cargo test --workspace`:
  2,857 passed, 0 failed, 6 ignored. `cargo check -p ignis-server --features
  cuda --tests` passes (one earlier warning, in `openapi.rs`).

## As built (320c, 2026-10-14: ACs 4, 5, 7, the tower and DeepStack)

**Status of #320, one line per acceptance item:**
1. Done (320f): token ids and `drop_idx` bit for bit on every case of
   spec 01, and the CPU tests.
2. Done (320f): the edit case's `pixel_values` and `image_grid_thw` bit for
   bit (bound 0), and the CPU tests.
3. Done (320a): t2i 9.61% against the 19.50% bound.
4. Done (320c): edit 12.78% after the first image pad against the 25.76%
   bound; with DeepStack off 29.68%, over it. 11.70% since the tower's
   K-1152 linears run on MMA (#324, the section at the end).
5. Done (320c): the 27B's vision tests pass unchanged (7 GPU tests under
   the profile, and the CTest).
6. Done (320f): mrope positions bit for bit on the t2i and 3-image tables.
7. Done, for information: t2i 21.6-22.0 ms (320a); a 10-image edit, tower
   2.84 s and text model 0.48 s (320c); tower 0.43 s and text model 0.46 s
   since #324 (324vis).

Where the tower chose what the sections above leave open.
- **Where.**
  - ABI `kernel/include/ignis_image_vision.h`, code
    `kernel/src/vision_encode.cu`: the 27B's tower program, parametrised.
    One call's description (`VisionEncodeCall`) carries the host arrays,
    the merger's output width, its output pages, and the DeepStack
    mergers with their taps and outputs.
  - The 27B's `ignis_media_encode` builds its call with the model's hidden,
    its pool pages and no DeepStack, so it runs the same ops in the same
    order as before. The image model's call has one page, its whole
    output, 4096 wide.
  - `ignis_image_vision_forward` takes the weights as device pointers
    (every matrix W8G32 `row-split-k128-v1`: codes, then the scale plane)
    and up to 10 segments. It writes the merger's output and each DeepStack
    output asked for (NULL: not run), BF16 `[tokens][4096]` each.
    `ignis_image_vision_workspace_bytes` is the 27B's layout: 56.8 MB at
    1,024 merged tokens, 568 MB at 10 × 1,024.
  - Rust: `crates/core/src/image_vision.rs` (`VisionTowerConfig::parse`,
    `vision_call`, the bindings, `VisionWeights` over a materialized
    artifact) and `qwen_image::bind_vision`, a scoped bind of the config and
    the `vision/` tensors.
  - Tests: the CTest `ignis_kernel_image_vision_test`; the Rust `--ignored`
    `crates/server/tests/image_encoder_edit_gpu.rs` (AC 4 and AC 7); on the
    CPU, `image_vision::tests` and the scoped bind
    (`crates/artifact/tests/qwen_image.rs`).
- **DeepStack.** Merger i taps the residual after tower block
  `deepstack_layers[i]` (0-based, transformers' `deepstack_visual_indexes`:
  8, 16 and 24). Its LayerNorm runs over the merged 4608 after the 2×2 view
  (`use_postshuffle_norm`); the merger's runs over each patch's 1152 before
  it. Then fc1, GELU-exact and fc2, as in the merger. Between blocks nothing
  in the workspace is live but the residual, so the DeepStack mergers run in
  the merger's own regions: no extra workspace.
- **The vendor patch** (a90699a, recorded in `kernel/vendor/manifest.json`
  and `VENDOR.md`). The W8 dispatch is a finite registry of (N, K) shapes.
  The patch registers the tower's shapes:
  - qkv `[3456, 1152]`, output `[1152, 1152]` and fc1 `[4304, 1152]` on the
    SIMT routes at every T. The MMA kernels copy eight groups' scales as one
    16-byte `cp.async`, which needs a row of scales (padded K / 16 bytes) to
    be a multiple of 16 bytes. K 1152 gives 72: routed to `mma_r64_c128`,
    they gave wrong outputs at T 17 and a misaligned-address fault at T
    4096. Since #324 they run on SIMT up to T 16, then `mma_r64_c128`, which
    a second recorded patch lets stage such a row (the section at the end).
  - the patch embedding `[1152, 1536]`, fc2 `[1152, 4304]` (stored K 4352)
    and the mergers' fc2 `[4096, 4608]` on SIMT up to T 16, then
    `mma_r64_c128`.

  It is neither an ADR 0031 bottleneck nor an ADR 0037 bug, and its record
  says so: whether registry extensions become a standing reason is the
  owner's call. `ignis_kernel_w8_linear_tests` covers the six shapes from T
  1 up to 49,200 (qkv and the four other patch-rate shapes) or 12,300 (the
  mergers' fc2), ten images of the 1,230-token worst case, and throws without
  the patch.
- **Defaults taken** (the owner may revisit):
  1. **No per-image token bound**, only the 10 segments and 131,072
     patches (32,768 merged tokens, where the mergers' fc2 has no W8 arm; a
     larger call is refused before anything is enqueued). A condition image
     near 1024² at an extreme aspect ratio exceeds 1,024 merged tokens. The
     worst case is **1,230** (96 × 13,120: diffusers' `calculate_dimensions`
     rounds each side to 32, and the processor's `smart_resize` refuses
     ratios over 200:1), so ten images are 12,300 tokens, 49,200 patches and
     about 0.68 GB of tower workspace. The caller sizes its arenas for 1,230
     tokens per condition image (#321, #322) or adds a bound of its own.
  2. **`pixel_values` are rounded to BF16 on the host**, to nearest even
     (`vision_call`), since the 27B's tower reads BF16 patches.
  3. **The DeepStack taps come from the config's `vision_config`**, not
     from the header. Its dimensions are checked against the artifact's
     inventory (depth, widths, patch, merge, positions, GELU-tanh). The
     descriptor also refuses a geometry other than the leaf's, or a head
     count other than 16.
  4. **The arithmetic is the 27B tower's**: BF16 activations and residual,
     W8 weights through the vendored A16 linear.
  5. **The bilinear table coordinates are the 27B's**
     (`vision_item_control`: fl32(i · 47 / (side − 1))), not
     `torch.linspace`'s two-sided formula. The corner weights can differ by
     an ulp; that is inside the tower's error and was not measured apart.
  6. **The text model reads the tower's outputs as they are.** The hook's
     `embeds` and `deepstack[i]` are the tower's own buffers, and its
     `row_index` the image-pad rows in order. Merged token j is the j-th
     2×2 block of the patch rows, so the rows follow the pads.
- **The CTest** (synthetic W8G32 weights, 1.6 s):
  - **The mergers against fp64.** A neutral block (every code and bias 0)
    adds exactly 0 to the residual. With all 27 neutral, x after the tower
    is the patch embedding plus its bias plus the position table. On a 4 ×
    4-patch image whose four patches per token differ in scale, the merger
    and the three DeepStack mergers are within 8.5e-4 to 9.6e-4 relative L2
    of fp64 that rounds where the device stores BF16 (bound 1e-2). With the
    norm placements swapped, each is 0.42 off: the negative control.
  - **The taps.** With block 9 live, DeepStack 0 (after block 8) equals the
    all-neutral run's bit for bit, and DeepStack 1 and 2 do not. With block
    8 live, DeepStack 0 does not. Taps at (9, 16, 24) move it. The merger's
    output is the same without DeepStack, and one DeepStack output asked
    alone equals the full run's.
  - **Segments.** Images of 8 × 10 and 12 × 8 patches in one call equal each
    alone, bit for bit, with a live block whose attention mixes patches. The
    same call as one segment differs.
  - **The refusals**, and the workspace equal to the 27B's layout.
- **AC 4.** The test feeds the edit case's recorded inputs (`input_ids`,
  `mrope_position_ids`, `pixel_values`, `image_grid_thw`, and the pad rows
  where `mm_token_type_ids` is 1). It runs the tower, then the 36 layers:
  the merger's rows are written over the 1,024 pad rows, and the DeepStack
  rows added after layers 0, 1 and 2. Spec 01's metric against the recorded
  fp32 output:

  | DeepStack | after the first pad (18 rows) | bound | encoder (25 text rows) | bound | encoder_row0 |
  |---|---|---|---|---|---|
  | **on** | **12.78%** | 25.76% (sim 12.88, bf16 3.22) | 13.48% | 28.99% (sim 14.49, bf16 3.13) | 28.77% (sim 21.12, bf16 30.42) |
  | off | 29.68% | | 26.43% | | 28.77% |

  - **DeepStack on** sits at the sim yardstick, as t2i does: W8A8
    quantization. Its worst row after the pad is 20.7% (" the", row 1056).
    Since #324 (the K-1152 linears on MMA, another accumulation order):
    11.70% after the pad, 12.76% on the encoder's rows, row 0 unchanged.
  - **DeepStack off is over the bound**, and the test asserts it (spec 03's
    check). It is as weak as spec 01 measured: rows 1046
    (`<|vision_end|>`, 47.6%) and 1056 (" the", 58.9%) carry it. Without
    both, the other 16 rows read 25.16%, under the bound. The text rows
    from row 1 (26.43%) stay under their bound, as in fp32.
  - **The two paths agree.** The same test runs through `EncoderFrontend`
    from `condition.png`, decoded (it equals the VAE input read back, byte
    for byte). Its sequence, positions, pad rows and pixels are the
    recording's. The tower's outputs and the encoder's output equal the
    recorded path's, bit for bit.
- **The tower's stages against transformers** (the diagnosis, printed when
  `IGNIS_IMAGE_VISION_REF` names the references). The references are
  transformers' tower on the CPU in fp32, on the checkpoint's weights and on
  the artifact's W8 weights decoded (`vision_probe.outputs`, through
  `tools/image-converter/ref_vision.py`, run with the same venv:
  `ref_vision.py <snapshot> <oracle> <out dir>`, and `IGNIS_IMAGE_VISION_REF`
  naming `<out dir>`).

  | output | vs fp32 weights | vs W8 weights | spec 01's bf16 yardstick |
  |---|---|---|---|
  | merger | 18.2% (per-row RMS 27.4%, worst row 299%) | 17.7% | 15.9% |
  | DeepStack 0 (block 8) | 2.1% | 1.7% | 1.6% |
  | DeepStack 1 (block 16) | 5.2% | 4.9% | 4.4% |
  | DeepStack 2 (block 24) | 7.0% | 6.1% | 5.5% |

  - The error is BF16 compute, the level of the checkpoint computing in
    bf16. Against the W8 weights it barely drops, so the W8 weights add
    little.
  - The merger's error is outlier-dominated, as spec 01 found: a few rows
    with small norms carry it.
  - No bug was found, and AC 4 holds at the sim level.
- **AC 7** (the GPU lock in normal mode, nothing else on the card; wall
  clock around each stage's launches and a stream sync, the median of 3 warm
  runs):
  - **Ten condition images** (the edit image ten times, synthetic as spec 03
    allows): 10,335 tokens, 10,240 image pads, 40,960 patches in 10
    segments. Front end 80 ms (CPU), **tower 2.84 s, text model 0.48 s**.
  - One image (the edit case, 1,064 tokens): tower 253 ms, text model
    45 ms.
  - The text model's 6.47 GiB and the tower's 0.57 GiB materialize in
    4.7-5.1 s.
  - The tower is 85% of the ten-image encode. The likely cause is its K-1152
    linears on SIMT (qkv, output and fc1: about two thirds of its linear
    work), the route the vendor patch takes because of the MMA's scale
    copy. That is an estimate, not profiled: the tower's linears and
    attention run at about 22 TFLOPS effective.
  - **Since #324** (measured, the section at the end): the tower takes
    0.43 s for ten images and 44 ms for one, against 2.68 s and 226 ms for
    the build before it in the same hold. The text model is unchanged
    (0.46 s and 41 ms), so the tower is now 48% of the ten-image encode.
- **Gates** (this section and its code, 4c7e3a1 to cae718e):
  - `kernel/build.ps1 -Test` with `IGNIS_GPU_PROFILE=1`: 123 of 123.
  - The Rust GPU tests under the profile:
    - `image_encoder_edit_gpu` (2 of 2) and `image_encoder_gpu` (2 of 2);
    - the 27B's vision tests (7 of 7): `vision_canary_gpu`,
      `vision_dflash2_gpu`, `vision_mixed_load_gpu`,
      `vision_prefix_reuse_gpu`, `classify_vision_readout_gpu`,
      `classify_pointing_gpu` and core's `vision_load_gpu`;
    - not run: the two `attn-tap`-feature tests
      (`attention_head_point_gpu`, `attn_tap_hq_consumed_gpu`), which need
      their own build.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2,862 passed, 0 failed,
    6 ignored. `cargo check -p ignis-server --features cuda --tests` passes
    (two earlier warnings, in `openapi.rs` and `classify_pointing_gpu.rs`).
  - AC 5 rests on those tests passing, the canary's teacher-forced floor
    among them. The 27B's embedding was not compared bit for bit before and
    after the change: "the same ops in the same order" holds by
    construction.

## Close-out (320d, 2026-10-14: the reviews' fixes)

The "As built" status lines above stand; this adds what the reviews of 320a and
320c asked for.
- **The diagnosis scripts** are in `tools/image-converter/` (`ref_layers.py`,
  `ref_vision.py`), with the invocations above.
- **Per layer, teacher-forced** (AC 3's diagnosis): see the table's section in
  320a; the earlier cumulative reading is withdrawn.
- **The artifact is identified** by its size and its header's identity (model
  id and weights id) against `thresholds.json`, not by the sha256 it records:
  hashing 16.7 GB on every run is not worth it, and the weights id names the
  checkpoint's revision. A different artifact is a skip (a failure under the
  profile).
- **The per-image ceiling is 1,230 merged tokens**, not 1,032 (the defaults
  above). The vendor patch's tests now reach T 49,200 on the five patch-rate
  shapes and T 12,300 on the mergers' fc2 (the patch record is refreshed). A
  call over 131,072 patches is refused up front, and
  `ignis_image_vision_workspace_bytes` returns 0 for a size the forward
  refuses.
- **Tests added:** the ten-image run's segments equal the one-image run's
  merger and DeepStack rows bit for bit (real weights, T 40,960); the CTest
  taps pin DeepStack 1 and 2 (blocks 16/17 and 24/25); `TextEncoderWeights`
  reads a full `qwen_image::bind`'s handles and `VisionWeights::new` refuses a
  geometry or head count the leaf does not run (`crates/core/tests/
  image_weights_cpu.rs`, on a `CpuDevice`); the text config parse refuses
  `attention_bias: true`.

## #324 item 5 (324vis, 2026-10-15): the tower's K-1152 linears on MMA

- **What changed.** qkv `[3456, 1152]`, output `[1152, 1152]` and fc1
  `[4304, 1152]` now run on SIMT up to T 16, then `mma_r64_c128`, like the
  tower's other W8 arms. Their 72-byte rows of scales reach the MMA kernel
  through a second recorded patch, of
  `kernel/vendor/src/ops/linear/w8/w8_rowsplit_gemm_mma.cuh` (ADR 0031):
  when a row of scales is a multiple of 8 bytes but not of 16, the kernel
  stages its window of eight groups as two 8-byte `cp.async` halves and
  zero-fills a half past the row. Any other row takes the unchanged 16-byte
  copy. The artifact is unchanged.
- **The measurement that opened it.** An nsys capture of the ten-image
  test (`image_encoder_edit_gpu`, the whole test: four ten-image and two
  one-image tower runs) showed `w8_rowsplit_gemm_simt_kernel` as 73% of all
  kernel time (9.99 of 13.63 s). The bound: at T 40,960 the
  three linears are 22.7 TFLOP per tower, about 0.11 s at the bf16 bar of
  ~210 TFLOPS, against about 2.3 s on SIMT.
- **Which layers moved, and by how much** (one block, synthetic weights,
  CUDA events, T 40,960):

  | linear | SIMT r8_c8 | MMA r64_c128 |
  |---|---|---|
  | qkv `[3456, 1152]` | 35.2 ms (9.3 TFLOPS) | 1.81 ms (181 TFLOPS) |
  | output `[1152, 1152]` | 11.9 ms (9.1 TFLOPS) | 0.59 ms (183 TFLOPS) |
  | fc1 `[4304, 1152]` | 39.7 ms (10.2 TFLOPS) | 2.45 ms (166 TFLOPS) |

  Over 27 blocks that is 2.34 s against 0.13 s. The patch embedding, fc2
  and the mergers did not move: they were on MMA already.
- **The tower** (`image_encoder_edit_gpu`, median of 3 warm runs, before
  and after in the same hold): ten images 2.68 s → 0.43 s, one image
  226 ms → 44 ms. The same capture after holds 4.15 s of kernels: the
  text model's W8A8 GEMMs (1.40 s) and attention (0.75 s) lead, then the
  tower's MMA linears (0.92 s) and its attention (0.54 s).
- **Correctness.**
  - `ignis_kernel_w8_linear_tests`: with the dispatch moved but not the
    kernel patch, `[3456, 1152]` fails at T 17 and 64 (wrong outputs) and
    faults at T 128 (misaligned address). With both, every case passes.
    T 64 and 128 are new at K 1152, a partial and a full column tile.
  - Every other W8 problem is bit for bit what it was. The FNV-1a hash of
    the output is equal before and after for the reference's 16 problems
    (the 27B's among them) and the tower's three earlier MMA arms, at 322
    (problem, T) pairs that cover each dispatch arm, and for K 1152 up to
    T 16. Only K 1152 above T 16 changed. The kernel's register counts
    moved by -6 to +8 per instantiation, with no spills, and a timing of
    those problems at 4,096 tokens moved within run noise.
  - AC 4: 11.70% after the first pad (bound 25.76%), 12.76% on the
    encoder's rows (bound 28.99%). The ten-image segments still equal the
    one-image run bit for bit.
- **Default taken.** `mma_r64_c128` from T 17 on all three shapes, as the
  tower's other arms. Below about 130 columns `mma_r32_c128`, or SIMT for
  `[1152, 1152]` at T 17, is up to 1.7x faster. The server never runs the
  tower that narrow: it encodes each condition image at about 1024² of
  area, 4,096 patches, where `mma_r64_c128` is the fastest of the three.
- **Gates** (0dfc97c, the GPU lock in timing mode):
  - `kernel/build.ps1 -Test` with `IGNIS_GPU_PROFILE=1`: 125 of 125.
  - The Rust GPU tests under the profile, 30 tests in 18 binaries, all
    passing: the image tests (`image_attn_gpu`, `image_elementwise_gpu`,
    `image_encoder_attn_gpu`, `image_encoder_gpu`, `image_gemm_gpu`,
    `image_leaf_gpu`, `image_vae_gpu`, `image_encoder_edit_gpu`,
    `image_leaf_ten_images_gpu`, `image_leaf_timing_gpu`,
    `image_runner_gpu`) and the 27B's 7 vision tests.
  - `CARGO_INCREMENTAL=0 cargo test --workspace`: 2,981 passed, 0 failed,
    6 ignored. `cargo check -p ignis-server --features cuda --tests`
    passes, with the earlier warnings only.
