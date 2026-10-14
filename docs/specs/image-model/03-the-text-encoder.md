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
  - the segment bound (10 condition images of up to 1024 merged tokens each);
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
  ninfer's float filter, a different one.
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
  - Against the oracle: `input_ids`, `drop_idx` and `mrope_position_ids`
    for t2i and edit; edit's image-pad rows, `image_grid_thw` [1, 64, 64]
    and `pixel_values` (4096 × 1536), all bit for bit.
  - The edit image's RGBA is read back from `condition_vae_input`, and the
    read-back is exact: `2·(u/255) − 1` of the recovered bytes is the
    recording.
  - Against the fixture: the 2-pixel-wide (2 × 64 → 64 × 1472) and odd-size
    (450 × 250 → 448 × 256) resamples are bit for bit, as are 1:1, 16:9 and
    9:16.
- **The fixture.** `crates/server/tests/fixtures/image_encoder_input.json`
  (107 kB) comes from `tools/image-converter/record_encoder_input.py`
  (command in its docstring), run in the venv on the pinned processor. It
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
