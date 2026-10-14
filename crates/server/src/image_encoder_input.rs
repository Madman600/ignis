//! The image encoder's input (spec image-model/03, ACs 1, 2 and 6): the
//! prompt's tokens, the condition images and the mrope positions, as
//! `QwenImage21Pipeline` hands them to its text encoder.
//!
//! [`EncoderFrontend`] is loaded once from the artifact and builds an
//! [`EncoderInput`] per image job. Every step is the pipeline's, checked bit
//! for bit (`tests/image_encoder_input.rs`) against the fixture
//! `tools/image-converter/record_encoder_input.py` records and spec 01's
//! oracle:
//! - **Tokens:** [`ImagePrompt`], the raw template.
//! - **On white:** a condition image arrives RGBA at its condition size
//!   (`media.rs`); the encoder's copy is composited on white as the pipeline
//!   does it, with Pillow's `paste` and the alpha as mask ([`on_white`]).
//! - **Size:** `smart_resize` on the processor's pixel bounds, the 27B's
//!   ([`ignis_artifact::vision::smart_resize`], the same rule). At a
//!   condition size (sides multiples of 32, at least 256² pixels) it changes
//!   nothing; otherwise the image is resampled as the processor does it,
//!   with torchvision's antialiased bicubic on uint8 ([`resize_bicubic`]),
//!   not the 27B's float filter.
//! - **Patches:** the processor's fused rescale and normalize,
//!   `(v − mean/rescale) / (std/rescale)` in f32, in the 27B's patch order
//!   (2×2 merge blocks; columns channel × temporal × 16 × 16, the frame
//!   repeated). f32, the processor's `pixel_values`: the 27B packs BF16.
//! - **Positions:** the Qwen3-VL mrope rule, the 27B's
//!   ([`assign_positions`]): the text position continues after each image.

use serde_json::Value as JsonValue;

use ignis_artifact::vision::layout::{assign_positions, IMAGE, TEXT};
use ignis_artifact::vision::{smart_resize, Grid, MERGE, PATCH, PATCH_FEATURES, TEMPORAL};
use ignis_artifact::{ChatTemplate, Reader, Tokenizer};
use ignis_core::image::ImageSize;

use crate::image_prompt::ImagePrompt;

/// A condition image as `media.rs` leaves it: RGBA, 8 bits a channel,
/// row-major, at its condition size.
#[derive(Debug, Clone, Copy)]
pub struct ConditionImage<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba: &'a [u8],
}

/// One condition image as the vision tower reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct VisionPatches {
    /// `image_grid_thw`, in patches.
    pub grid: Grid,
    /// `pixel_values`: row-major `[grid.raw_patches(), PATCH_FEATURES]`.
    pub pixels: Vec<f32>,
}

/// What the text encoder reads for one image job.
#[derive(Debug, Clone, PartialEq)]
pub struct EncoderInput {
    /// The whole sequence: the raw template, each image's pads expanded.
    pub input_ids: Vec<u32>,
    /// The leading rows the output loses: the system message.
    pub drop_idx: usize,
    /// The rows of `input_ids` that are image pads, in order: where the
    /// vision features go.
    pub image_pad_rows: Vec<u32>,
    /// Axis-major `[3, L]` mrope positions: temporal, height, width.
    pub positions: Vec<i32>,
    /// The condition images, in prompt order.
    pub images: Vec<VisionPatches>,
}

/// The image encoder's front end: the artifact's prompt and image processor.
pub struct EncoderFrontend {
    prompt: ImagePrompt,
    processor: ImageProcessor,
}

impl EncoderFrontend {
    /// From an image artifact's `frontend/tokenizer.json`,
    /// `frontend/chat_template.jinja` and `frontend/image_model.json`.
    pub fn from_reader(reader: &Reader) -> Result<Self, String> {
        let resource = |name: &str| {
            let object = reader.find(&format!("frontend/{name}")).ok_or_else(|| format!("the image artifact has no frontend/{name}"))?;
            reader.payload_at(object).map(|span| span.data).map_err(|e| e.to_string())
        };
        let model: JsonValue = serde_json::from_slice(resource("image_model.json")?).map_err(|e| format!("image_model.json: {e}"))?;
        let tokenizer = Tokenizer::from_bytes(resource("tokenizer.json")?).map_err(|e| e.to_string())?;
        let chat_template = ChatTemplate::from_bytes(resource("chat_template.jinja")?).map_err(|e| e.to_string())?;
        Ok(Self {
            prompt: ImagePrompt::new(tokenizer, &chat_template, &model["prompt"])?,
            processor: ImageProcessor::from_json(&model["image_processor"])?,
        })
    }

    /// The encoder's input for `prompt` and its condition images, in order
    /// (none for text-to-image).
    pub fn input(&self, prompt: &str, images: &[ConditionImage<'_>]) -> Result<EncoderInput, String> {
        let images = images.iter().map(|&image| self.processor.prepare(image)).collect::<Result<Vec<_>, _>>()?;
        let grids: Vec<Grid> = images.iter().map(|image| image.grid).collect();
        let tokens: Vec<usize> = grids.iter().map(|grid| grid.vision_tokens() as usize).collect();
        let input_ids = self.prompt.tokens(prompt, &tokens)?;
        let pad = self.prompt.image_pad();
        let positions = mrope_positions(&input_ids, pad, &grids)?;
        let image_pad_rows = (0..).zip(&input_ids).filter(|&(_, &id)| id == pad).map(|(row, _)| row).collect();
        Ok(EncoderInput { input_ids, drop_idx: self.prompt.drop_idx(), image_pad_rows, positions, images })
    }

    /// What [`EncoderFrontend::input`] builds for `prompt` with condition
    /// images of `sizes`, counted without preparing an image: the template
    /// tokenized once, and each image a pad per 2 x 2 of its latent cells.
    /// Refused for a size the processor would not keep
    /// ([`ImageProcessor::keeps`]): the image model encodes only those.
    pub fn count(&self, prompt: &str, sizes: &[ImageSize]) -> Result<ConditionCount, String> {
        for (k, size) in sizes.iter().enumerate() {
            self.processor.keeps(*size).map_err(|why| format!("condition image {k} is {size}: {why}"))?;
        }
        let text = self.prompt.rows(prompt, sizes.len())?.saturating_sub(sizes.len()) as u64;
        let cells: u64 = sizes.iter().map(|size| size.latent_cells() as u64).sum();
        let pads = cells / 4;
        Ok(ConditionCount { encoder_rows: text + pads, pads, prefix: text.saturating_sub(self.prompt.drop_idx() as u64) + cells })
    }
}

/// A condition's shape, as [`EncoderFrontend::count`] counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConditionCount {
    /// The encoder's sequence: the template's tokens, each image's pads
    /// expanded.
    pub encoder_rows: u64,
    /// Its image pads, one per merged token of the vision tower.
    pub pads: u64,
    /// The condition prefix (spec image-model/04): the text rows from
    /// `drop_idx` on, and each image's latent cells.
    pub prefix: u64,
}

/// The Qwen3-VL image processor, from the artifact's `image_model.json`
/// (`image_processor`: the checkpoint's `processor_config.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImageProcessor {
    min_pixels: u64,
    max_pixels: u64,
    /// The fused rescale and normalize: `mean / rescale_factor` and
    /// `std / rescale_factor`, as torch computes them in f32.
    mean: [f32; 3],
    std: [f32; 3],
}

impl ImageProcessor {
    /// Read the processor's config. The patch geometry must be the 27B's
    /// tower's, and the steps the ones implemented here: bicubic resize
    /// (`resample` 3), rescale and normalize.
    pub fn from_json(config: &JsonValue) -> Result<Self, String> {
        let invalid = |what: String| format!("image_processor: {what}");
        for (key, expected) in [("patch_size", PATCH), ("temporal_patch_size", TEMPORAL), ("merge_size", MERGE), ("resample", 3)] {
            if config.get(key).and_then(JsonValue::as_u64) != Some(expected as u64) {
                return Err(invalid(format!("{key} must be {expected}")));
            }
        }
        for key in ["do_resize", "do_rescale", "do_normalize"] {
            if config.get(key).and_then(JsonValue::as_bool) != Some(true) {
                return Err(invalid(format!("{key} must be true")));
            }
        }
        let bound = |key: &str| {
            config.get("size").and_then(|size| size.get(key)).and_then(JsonValue::as_u64).filter(|&v| v > 0)
                .ok_or_else(|| invalid(format!("size.{key} must be a positive integer")))
        };
        let positive = |value: &JsonValue| value.as_f64().filter(|&v| v > 0.0);
        let rescale = config.get("rescale_factor").and_then(positive).ok_or_else(|| invalid("rescale_factor must be positive".into()))?;
        // torch.tensor(mean) * (1.0 / rescale_factor): f32, the scalar taken as f32
        let scale = (1.0 / rescale) as f32;
        let triple = |key: &str, at_least: f64| -> Result<[f32; 3], String> {
            match config.get(key).and_then(JsonValue::as_array).map(|values| values.iter().map(JsonValue::as_f64).collect::<Vec<_>>()) {
                Some(values) if values.len() == 3 && values.iter().all(|v| v.is_some_and(|v| v >= at_least)) => {
                    Ok([0, 1, 2].map(|c| values[c].unwrap() as f32 * scale))
                }
                _ => Err(invalid(format!("{key} must be three numbers"))),
            }
        };
        Ok(Self {
            min_pixels: bound("shortest_edge")?,
            max_pixels: bound("longest_edge")?,
            mean: triple("image_mean", f64::MIN)?,
            std: triple("image_std", f64::MIN_POSITIVE)?,
        })
    }

    /// Whether the processor keeps a condition image of `size` as it is:
    /// `smart_resize` leaves it (sides multiples of 32 within the pixel
    /// bounds, at most 200:1). Only then does its every 2 x 2 of latent cells
    /// meet one vision token, as the image model's condition prefix lays them
    /// out (spec image-model/04); Err says what the processor would do.
    pub fn keeps(&self, size: ImageSize) -> Result<(), String> {
        match smart_resize(size.height, size.width, self.min_pixels, self.max_pixels) {
            Ok(kept) if (kept.width, kept.height) == (size.width, size.height) => Ok(()),
            Ok(kept) => Err(format!(
                "the encoder's processor resizes it to {}x{} (it keeps sides that are multiples of 32 within {} to {} pixels), so its \
                 vision tokens would not line up with its latents",
                kept.width, kept.height, self.min_pixels, self.max_pixels
            )),
            Err(e) => Err(format!("the encoder's processor refuses it ({e:?}: at most 200:1)")),
        }
    }

    /// One condition image as the processor prepares it: on white,
    /// `smart_resize`d (resampled only when that changes its size),
    /// normalized and patchified.
    pub fn prepare(&self, image: ConditionImage<'_>) -> Result<VisionPatches, String> {
        let (width, height) = (image.width as usize, image.height as usize);
        if image.rgba.len() != width * height * 4 {
            return Err(format!("condition image {width}x{height}: {} bytes, not RGBA", image.rgba.len()));
        }
        let size = smart_resize(image.height, image.width, self.min_pixels, self.max_pixels)
            .map_err(|e| format!("condition image {width}x{height}: smart_resize refuses it ({e:?})"))?;
        let mut rgb = on_white(image.rgba);
        if (size.width, size.height) != (image.width, image.height) {
            rgb = resize_bicubic(&rgb, width, height, size.width as usize, size.height as usize);
        }
        Ok(self.patches(&rgb, size.width as usize, size.height as usize))
    }

    /// Normalize and patchify an RGB image whose sides are multiples of
    /// `PATCH · MERGE`.
    fn patches(&self, rgb: &[u8], width: usize, height: usize) -> VisionPatches {
        let (gh, gw) = (height / PATCH, width / PATCH);
        let lut: [[f32; 256]; 3] = [0, 1, 2].map(|c| std::array::from_fn(|v| (v as f32 - self.mean[c]) / self.std[c]));
        let mut pixels = vec![0f32; gh * gw * PATCH_FEATURES];
        let mut rows = pixels.chunks_exact_mut(PATCH_FEATURES);
        for block_y in 0..gh / MERGE {
            for block_x in 0..gw / MERGE {
                for merge_y in 0..MERGE {
                    for merge_x in 0..MERGE {
                        let row = rows.next().unwrap();
                        let (grid_y, grid_x) = (block_y * MERGE + merge_y, block_x * MERGE + merge_x);
                        for (channel, lut) in lut.iter().enumerate() {
                            for y in 0..PATCH {
                                let source = ((grid_y * PATCH + y) * width + grid_x * PATCH) * 3 + channel;
                                for x in 0..PATCH {
                                    let value = lut[rgb[source + x * 3] as usize];
                                    for temporal in 0..TEMPORAL {
                                        row[((channel * TEMPORAL + temporal) * PATCH + y) * PATCH + x] = value;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        VisionPatches { grid: Grid { t: 1, h: gh as u32, w: gw as u32 }, pixels }
    }
}

/// The encoder's copy of a condition image: RGBA composited on white, as
/// the pipeline flattens it (Pillow's `paste` with the alpha as mask:
/// `BLEND` and `DIV255` of `Paste.c`, exact for every value and alpha).
pub fn on_white(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).flat_map(|px| [blend_on_white(px[0], px[3]), blend_on_white(px[1], px[3]), blend_on_white(px[2], px[3])]).collect()
}

/// One colour `value` at `alpha` over white, as [`on_white`] flattens it:
/// `(value · alpha + 255 · (255 − alpha)) / 255` rounded to the nearest byte
/// (`DIV255`). An image answered `background: opaque` is composited by it
/// too (`images::job::on_white`).
pub fn blend_on_white(value: u8, alpha: u8) -> u8 {
    let a = u32::from(alpha);
    let t = 255 * (255 - a) + u32::from(value) * a + 128;
    (((t >> 8) + t) >> 8) as u8
}

// Ported from PyTorch 2.13.0, aten/src/ATen/native/cpu/UpSampleKernel.cpp
// (BSD-style licence, quoted in NOTICE), not copied verbatim: `resize_bicubic`,
// `convolve`, `cubic` and `int16_weights` restate its antialiased uint8 path
// in Rust.

/// torchvision's `resize(BICUBIC, antialias=True)` of a uint8 RGB image:
/// ATen's separable antialiased filter (Keys cubic, a = −0.5) in integer
/// arithmetic, a horizontal pass into a uint8 intermediate, then a vertical
/// one, each axis's weights scaled to int16 (`UpSampleKernel.cpp`,
/// `_compute_index_ranges_int16_weights`). An axis at its size is not
/// filtered.
pub fn resize_bicubic(rgb: &[u8], width: usize, height: usize, new_width: usize, new_height: usize) -> Vec<u8> {
    let mut rgb = rgb.to_vec();
    if new_width != width {
        let (taps, precision) = int16_weights(width, new_width);
        let mut out = vec![0u8; height * new_width * 3];
        for y in 0..height {
            for (x, (first, weights)) in taps.iter().enumerate() {
                for c in 0..3 {
                    out[(y * new_width + x) * 3 + c] = convolve(*first, weights, precision, |i| rgb[(y * width + i) * 3 + c]);
                }
            }
        }
        rgb = out;
    }
    if new_height != height {
        let (taps, precision) = int16_weights(height, new_height);
        let mut out = vec![0u8; new_height * new_width * 3];
        for (y, (first, weights)) in taps.iter().enumerate() {
            for x in 0..new_width {
                for c in 0..3 {
                    out[(y * new_width + x) * 3 + c] = convolve(*first, weights, precision, |i| rgb[(i * new_width + x) * 3 + c]);
                }
            }
        }
        rgb = out;
    }
    rgb
}

/// One output sample: `sample(i)` is input sample `i` along the axis.
fn convolve(first: usize, weights: &[i32], precision: u32, sample: impl Fn(usize) -> u8) -> u8 {
    let sum = (first..).zip(weights).fold(1i64 << (precision - 1), |sum, (i, &w)| sum + sample(i) as i64 * w as i64);
    (sum >> precision).clamp(0, 255) as u8
}

/// Keys cubic convolution, a = −0.5 (`HelperInterpCubic::aa_filter`).
fn cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// ATen's antialiased bicubic weights from `input` to `output` samples:
/// per output sample its first input sample and its int16 weights, and the
/// precision (fraction bits) they are scaled to. f64 throughout, in ATen's
/// order (`_compute_indices_min_size_weights_aa`).
fn int16_weights(input: usize, output: usize) -> (Vec<(usize, Vec<i32>)>, u32) {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 { 2.0 * scale } else { 2.0 };
    let max_taps = support.ceil() as i64 * 2 + 1;
    let invscale = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut largest = 0.0f64;
    let taps: Vec<(usize, Vec<f64>)> = (0..output)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            // `static_cast<int64_t>` truncates toward zero, as `as` does
            let first = ((center - support + 0.5) as i64).max(0);
            let count = (((center + support + 0.5) as i64).min(input as i64) - first).clamp(0, max_taps);
            let mut weights: Vec<f64> = (first..first + count).map(|j| cubic((j as f64 - center + 0.5) * invscale)).collect();
            let total = weights.iter().fold(0.0, |total, w| total + w);
            if total != 0.0 {
                for w in &mut weights {
                    *w /= total;
                    largest = largest.max(*w);
                }
            }
            (first as usize, weights)
        })
        .collect();
    let mut precision = 0;
    while precision < 22 && ((0.5 + largest * (1 << (precision + 1)) as f64) as i32) < 1 << 15 {
        precision += 1;
    }
    let to_int = |w: f64| {
        let v = w * (1 << precision) as f64;
        (if v < 0.0 { -0.5 + v } else { 0.5 + v }) as i32
    };
    (taps.into_iter().map(|(first, weights)| (first, weights.into_iter().map(to_int).collect())).collect(), precision)
}

/// The Qwen3-VL mrope positions of `input_ids`, axis-major `[3, L]`: each
/// run of `image_pad` takes the next grid, the 27B's rule
/// ([`assign_positions`]), which is what transformers' `get_rope_index`
/// computes for images. A run that is not its grid's size is refused.
pub fn mrope_positions(input_ids: &[u32], image_pad: u32, grids: &[Grid]) -> Result<Vec<i32>, String> {
    let types: Vec<u8> = input_ids.iter().map(|&id| if id == image_pad { IMAGE } else { TEXT }).collect();
    assign_positions(&types, grids).map(|(positions, _, _)| positions).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processor() -> ImageProcessor {
        ImageProcessor::from_json(&serde_json::json!({
            "do_normalize": true, "do_rescale": true, "do_resize": true, "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16, "resample": 3,
            "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "temporal_patch_size": 2,
        }))
        .unwrap()
    }

    #[test]
    fn the_checkpoints_config_normalizes_by_127_5() {
        let p = processor();
        assert_eq!((p.mean, p.std), ([127.5; 3], [127.5; 3]));
        assert_eq!((p.min_pixels, p.max_pixels), (65_536, 16_777_216));
    }

    #[test]
    fn a_config_this_processor_does_not_implement_is_refused() {
        let base = serde_json::json!({
            "do_normalize": true, "do_rescale": true, "do_resize": true, "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5], "merge_size": 2, "patch_size": 16, "resample": 3,
            "rescale_factor": 0.00392156862745098, "size": {"longest_edge": 16777216, "shortest_edge": 65536},
            "temporal_patch_size": 2,
        });
        for (key, value) in [("resample", serde_json::json!(1)), ("patch_size", 14.into()), ("do_rescale", false.into()),
                             ("image_std", serde_json::json!([0.5, 0.0, 0.5])), ("image_mean", serde_json::json!([0.5])),
                             ("size", serde_json::json!({"longest_edge": 16777216}))] {
            let mut config = base.clone();
            config[key] = value;
            assert!(ImageProcessor::from_json(&config).unwrap_err().contains(key), "{key}");
        }
    }

    #[test]
    fn patches_are_merge_blocks_of_channel_temporal_rows() {
        // a 64 x 32 image (2 x 4 patches, one row of two merge blocks), each pixel's red its x,
        // green its y, blue 0, all opaque
        let (width, height) = (64u32, 32u32);
        let rgba: Vec<u8> = (0..height).flat_map(|y| (0..width).flat_map(move |x| [x as u8, y as u8, 0, 255])).collect();
        let p = ImageProcessor { min_pixels: 1, ..processor() };
        let out = p.prepare(ConditionImage { width, height, rgba: &rgba }).unwrap();
        assert_eq!(out.grid, Grid { t: 1, h: 2, w: 4 });
        let value = |v: u8| (v as f32 - 127.5) / 127.5;
        let at = |row: usize, channel: usize, temporal: usize, y: usize, x: usize| {
            out.pixels[row * PATCH_FEATURES + ((channel * TEMPORAL + temporal) * PATCH + y) * PATCH + x]
        };
        // row 3 is block (0, 0)'s lower right patch: grid (1, 1); row 5 is block (0, 1)'s upper right: grid (0, 3)
        assert_eq!(at(3, 0, 0, 2, 5), value(16 + 5));
        assert_eq!(at(3, 1, 1, 2, 5), value(16 + 2));
        assert_eq!(at(5, 0, 1, 7, 0), value(48));
        assert_eq!(at(5, 1, 0, 7, 0), value(7));
        assert_eq!(at(5, 2, 0, 0, 0), value(0));
    }

    #[test]
    fn smart_resize_resamples_only_off_the_grid() {
        let p = processor();
        let flat = |w: u32, h: u32| [10u8, 200, 30, 255].repeat((w * h) as usize);
        let image = flat(2, 64);
        let out = p.prepare(ConditionImage { width: 2, height: 64, rgba: &image }).unwrap();
        // up to the minimum area: 64 x 1472, and a flat image stays flat
        assert_eq!(out.grid, Grid { t: 1, h: 92, w: 4 });
        let row: Vec<f32> = out.pixels[..PATCH_FEATURES].to_vec();
        assert!(row[..512].iter().all(|&v| v == (10.0 - 127.5) / 127.5) && row[1024..].iter().all(|&v| v == (30.0 - 127.5) / 127.5));
        let image = flat(2, 401);
        assert!(p.prepare(ConditionImage { width: 2, height: 401, rgba: &image }).unwrap_err().contains("AspectRatio"));
        assert!(p.prepare(ConditionImage { width: 2, height: 2, rgba: &[0; 15] }).unwrap_err().contains("not RGBA"));
    }

    #[test]
    fn int16_weights_sum_to_one_at_their_precision() {
        for (input, output) in [(2, 64), (64, 1472), (450, 448), (250, 256), (300, 7)] {
            let (taps, precision) = int16_weights(input, output);
            for (first, weights) in &taps {
                let sum: i32 = weights.iter().sum();
                assert!((sum - (1 << precision)).abs() <= weights.len() as i32, "{input}->{output} at {first}: {sum}");
                assert!(first + weights.len() <= input);
            }
        }
    }

    /// The count the server fits condition images by (`ImageRunner::condition_tokens`) is the one
    /// the leaf refuses by: the input this front end builds, laid out as the leaf lays it out
    /// (`PrefixLayout`), for text-to-image, empty and literal prompts, and one to three images.
    #[test]
    fn the_count_is_the_condition_prefix_the_leaf_lays_out() {
        use ignis_core::image_leaf::PrefixLayout;
        let frontend = EncoderFrontend { prompt: crate::image_prompt::tests::prompt(), processor: ImageProcessor { min_pixels: 1, ..processor() } };
        let cases: [(&str, Vec<ImageSize>); 5] = [
            ("a red fox in the snow", vec![]),
            ("", vec![]),
            ("make it {} blue <image1>", vec![ImageSize::new(64, 64)]),
            ("the two together", vec![ImageSize::new(96, 64), ImageSize::new(64, 128)]),
            ("x", vec![ImageSize::new(32, 32), ImageSize::new(128, 32), ImageSize::new(64, 96)]),
        ];
        for (prompt, sizes) in cases {
            let pixels: Vec<Vec<u8>> = sizes.iter().map(|s| vec![200; (s.width * s.height * 4) as usize]).collect();
            let images: Vec<ConditionImage> =
                sizes.iter().zip(&pixels).map(|(s, rgba)| ConditionImage { width: s.width, height: s.height, rgba }).collect();
            let input = frontend.input(prompt, &images).unwrap();
            let shapes: Vec<(u32, u32)> = sizes.iter().map(|s| (s.height / 16, s.width / 16)).collect();
            let layout = PrefixLayout::new(input.input_ids.len(), input.drop_idx, &input.image_pad_rows, &shapes).unwrap();
            let count = frontend.count(prompt, &sizes).unwrap();
            assert_eq!(
                count,
                ConditionCount {
                    encoder_rows: input.input_ids.len() as u64,
                    pads: input.image_pad_rows.len() as u64,
                    prefix: u64::from(layout.tokens),
                },
                "{prompt:?} with {sizes:?}"
            );
        }
    }

    /// A condition size the processor would not keep is not counted: its vision tokens would not
    /// line up with its latents, and the leaf refuses it. The budget's fit scales a panorama until
    /// its rounded sides pass 200:1 (13088x64), which `smart_resize` refuses; a size off the
    /// 32-pixel grid or below the pixel floor is resized.
    #[test]
    fn the_count_refuses_a_size_the_processor_would_not_keep() {
        let frontend = EncoderFrontend { prompt: crate::image_prompt::tests::prompt(), processor: processor() };
        assert!(frontend.count("x", &[ImageSize::new(13120, 96)]).is_ok(), "136:1 within the bounds");
        for (size, says) in [
            (ImageSize::new(13088, 64), "refuses"),
            (ImageSize::new(224, 192), "resizes it to 288x256"),
            (ImageSize::new(1024, 1008), "resizes it to 1024x1024"),
        ] {
            let refused = frontend.count("x", &[ImageSize::new(1024, 1024), size]).unwrap_err();
            assert!(refused.contains(&format!("condition image 1 is {size}")) && refused.contains(says), "{size}: {refused}");
        }
    }

    #[test]
    fn mrope_continues_the_text_after_each_image() {
        // text 2, a 4 x 6 grid (2 x 3 merged), text 1
        let pad = 9;
        let ids = [1, 1, pad, pad, pad, pad, pad, pad, 1];
        let positions = mrope_positions(&ids, pad, &[Grid { t: 1, h: 4, w: 6 }]).unwrap();
        assert_eq!(positions[..9], [0, 1, 2, 2, 2, 2, 2, 2, 5]);
        assert_eq!(positions[9..18], [0, 1, 2, 2, 2, 3, 3, 3, 5]);
        assert_eq!(positions[18..], [0, 1, 2, 3, 4, 2, 3, 4, 5]);
        assert!(mrope_positions(&ids, pad, &[]).is_err());
        assert!(mrope_positions(&ids, pad, &[Grid { t: 1, h: 4, w: 4 }]).is_err());
    }
}
