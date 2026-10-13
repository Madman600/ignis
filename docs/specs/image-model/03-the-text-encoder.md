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
   a relative L2 error ≤ the 5× bound in spec 01's `thresholds.json`. This is
   the first milestone.
4. The **edit case**: the same bound holds with the vision tower and
   DeepStack on. With DeepStack off, the same test fails; that check proves
   DeepStack matters to the test.
5. The 27B's vision tests pass unchanged after the tower is parametrised.
6. mrope positions equal transformers' for a t2i layout and a 3-image layout
   (CPU test against a recorded table).
7. The encoder's wall time for the t2i case and for a 10-image edit is
   written in "As built", for information.
