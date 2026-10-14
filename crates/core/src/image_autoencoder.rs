//! The image model's VAE (spec image-model/04, `vae_encode` and `vae_decode`; GitHub #321): the
//! encoder, an RGBA image to the DiT's condition latents, and the decoder, the denoised latents to
//! an RGBA image. The encoder is diffusers 0.41.0's `AutoencoderKLQwenImage21.encode` for
//! one frame (`QwenImage21Encoder3d` with `is_residual`, then `quant_conv` and the posterior's mode,
//! its first `z_dim` channels), driven op by op over spec 02's VAE ops ([`crate::image_vae`],
//! `kernel/include/ignis_image_vae.h`). The pipeline's normalization `(z − latents_mean) /
//! latents_std` is the leaf's ([`crate::image_leaf::LatentNorm`]).
//!
//! The encoder for one frame, every causal convolution a 2D one (the first chunk has no temporal
//! context, so no `time_conv` runs):
//! - `conv_in`, 3×3, RGBA → `base`;
//! - one down block per level, `out = base · dim_mult[i]`: `res_blocks` resnets, then, in every
//!   block but the last, the downsampler (3×3 stride 2 after the (0, 1, 0, 1) pad); plus the block
//!   input's AvgDown shortcut (factor_t 2 where the level downsamples in time, factor_s 2; the last
//!   block's is the identity);
//! - a resnet: `conv2(silu(norm2(conv1(silu(norm1(x)))))) + (conv_shortcut(x) or x)`;
//! - the mid block: a resnet, the single-head attention `x + proj(attn(to_qkv(norm(x))))`, a resnet;
//! - `norm_out`, SiLU, `conv_out` (3×3, → 2 `z_dim`), `quant_conv` (1×1): only its first `z_dim`
//!   output channels, the posterior's mean, are computed.
//!
//! Arithmetic is the ops' (spec 02 "As built"): BF16 NHWC activations, fp32 accumulation, the bias
//! and the residual added in the convolution's epilogue with one rounding, each norm rounded once.
//! The downsampler adds the block's shortcut as its residual: one rounding where diffusers' BF16
//! path rounds the convolution's output first. The latents come out fp32. The input is the
//! pipeline's `2 (u / 255) − 1` of the bytes, rounded to BF16 ([`input_bf16`]).
//!
//! The 3×3 weights are repacked to the convolution's layout per call, into a scratch region of the
//! workspace (the whole encoder's are ~0.2 GB of BF16 to read and write, a fraction of a
//! millisecond); a 1×1 weight's layout is already the packed one. The activations live in four
//! slots of the largest activation's size, taken and given back by liveness: a resnet holds three
//! at most, a down block one more for its shortcut, and the mid block's attention uses the fourth
//! as its workspace.
//!
//! The decoder is `AutoencoderKLQwenImage21.decode` for one frame (`post_quant_conv`, then
//! `QwenImage21Decoder3d` with `is_residual`), the pipeline's `z · std + mean` before it
//! ([`crate::image_leaf::LatentNorm`]) and its clamp and bytes after ([`rgba8`]):
//! - `post_quant_conv` (1×1), `conv_in` (3×3, `z_dim` → 1152), the mid block (as the encoder's);
//! - one up block per level, `decoder_base` times `[dim_mult[-1]]` + `dim_mult` reversed ([1152,
//!   1152, 1152, 576, 288, 144]): `res_blocks + 1` resnets, then, in every block but the last, the
//!   upsampler (nearest 2x, 3×3) and the block input's DupUp shortcut (factor_t 2 where the
//!   encoder's mirror level downsamples in time); no `time_conv` runs for one frame;
//! - `norm_out`, SiLU, `conv_out` (3×3, → RGBA) in fp32.
//!
//! The mid block runs on the whole latent grid (its attention is global) and its output stays.
//! Everything after it runs in horizontal bands of image rows ([`band_rows`]): each band computes
//! every stage over its rows plus the halo the next stage reads, from the mid block's output, so the
//! widest activations (2.4 GB at 2048²) never exist whole and the image is the same bit for bit in
//! any bands. The upsampler adds the DupUp shortcut as its residual, one rounding where diffusers'
//! BF16 path rounds the convolution's output first. The 3×3 weights are packed per call, so per band.
//!
//! Our own implementation (ADR 0051, tier 1).

use ignis_artifact::qwen_image::VaeGeometry;

use crate::image_vision::to_bf16;

/// The VAE config's settings the encoder and decoder depend on (`frontend/vae_config.json`, the checkpoint's
/// `vae/config.json` verbatim), checked against the artifact's inventory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VaeConfig;

impl VaeConfig {
    /// Refuses a config whose ladder is not `geometry`'s (the inventory's), or that the encoder does
    /// not run: `is_residual` off, a patch size above 1, attention at a down level.
    pub fn parse(bytes: &[u8], geometry: &VaeGeometry) -> Result<Self, String> {
        let json: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| format!("the VAE config is not JSON: {e}"))?;
        let uint = |key: &str| json.get(key).and_then(|v| v.as_u64()).ok_or(format!("the VAE config's {key} is missing"));
        for (key, want) in [
            ("in_channels", geometry.in_channels),
            ("out_channels", geometry.out_channels),
            ("z_dim", geometry.z_dim),
            ("base_dim", geometry.base),
            ("num_res_blocks", geometry.res_blocks as u64),
        ] {
            let got = uint(key)?;
            if got != want {
                return Err(format!("the VAE config's {key} is {got}, the leaf runs {want}"));
            }
        }
        // diffusers 0.41.0: decoder_base_dim defaults to 144, and None means base_dim.
        let decoder_base = match json.get("decoder_base_dim") {
            None => 144,
            Some(v) if v.is_null() => geometry.base,
            Some(v) => v.as_u64().ok_or("the VAE config's decoder_base_dim is not a number")?,
        };
        if decoder_base != geometry.decoder_base {
            return Err(format!("the VAE config's decoder_base_dim is {decoder_base}, the leaf runs {}", geometry.decoder_base));
        }
        let list = |key: &str| json.get(key).and_then(|v| v.as_array()).cloned().unwrap_or_default();
        let dim_mult: Vec<u64> = list("dim_mult").iter().filter_map(|v| v.as_u64()).collect();
        if dim_mult != geometry.dim_mult {
            return Err(format!("the VAE config's dim_mult is {dim_mult:?}, the leaf runs {:?}", geometry.dim_mult));
        }
        let temporal: Vec<bool> = list("temperal_downsample").iter().filter_map(|v| v.as_bool()).collect();
        if temporal != geometry.temporal_downsample {
            return Err(format!("the VAE config's temperal_downsample is {temporal:?}, the leaf runs {:?}", geometry.temporal_downsample));
        }
        // diffusers' defaults: is_residual true, patch_size None, no attention scales.
        if json.get("is_residual").and_then(|v| v.as_bool()) == Some(false) {
            return Err("the VAE config's is_residual is false: the leaf runs the residual down and up blocks".into());
        }
        if json.get("patch_size").and_then(|v| v.as_u64()).is_some_and(|p| p > 1) {
            return Err("the VAE config's patch_size is above 1: the leaf runs the unpatched VAE".into());
        }
        if !list("attn_scales").is_empty() {
            return Err("the VAE config's attn_scales is not empty: the leaf's down blocks have no attention".into());
        }
        Ok(Self)
    }
}

/// The encoder's channel ladder: `base` then `base · dim_mult[i]` per level.
fn ladder(g: &VaeGeometry) -> Vec<u32> {
    std::iter::once(1).chain(g.dim_mult.iter().copied()).map(|u| (g.base * u) as u32).collect()
}

fn align(n: u64) -> u64 {
    (n + 255) & !255
}

/// One activation slot's bytes at `height` x `width`: the largest BF16 activation of the encoder
/// (a down block's widest at its level, the mid block's `to_qkv` output).
fn slot_bytes(g: &VaeGeometry, height: u32, width: u32) -> u64 {
    let dims = ladder(g);
    let levels = g.dim_mult.len();
    let mut elems = 0u64;
    for i in 0..levels {
        let pixels = (height >> i) as u64 * (width >> i) as u64;
        elems = elems.max(dims[i].max(dims[i + 1]) as u64 * pixels);
    }
    let last = (height >> (levels - 1)) as u64 * (width >> (levels - 1)) as u64;
    elems = elems.max(3 * dims[levels] as u64 * last);
    align(elems * 2)
}

/// The largest 3×3 weight, BF16, the pack scratch holds.
fn scratch_bytes(g: &VaeGeometry) -> u64 {
    let dims = ladder(g);
    let levels = g.dim_mult.len();
    let mut elems = dims[0] as u64 * g.in_channels;
    for i in 0..levels {
        elems = elems.max(dims[i + 1] as u64 * dims[i].max(dims[i + 1]) as u64);
    }
    elems = elems.max(2 * g.z_dim * dims[levels] as u64);
    align(elems * 9 * 2)
}

/// The VAE's refusal of a size: sides positive multiples of the total downsampling.
pub fn check_size(g: &VaeGeometry, height: u32, width: u32) -> Result<(), String> {
    let factor = 1u32 << (g.dim_mult.len() - 1);
    if height == 0 || width == 0 || height % factor != 0 || width % factor != 0 || height as u64 * width as u64 > 1 << 24 {
        return Err(format!(
            "a {width} x {height} image: the VAE takes sides that are positive multiples of {factor}, at most 2^24 pixels"
        ));
    }
    Ok(())
}

/// The workspace [`encode`] needs at `height` x `width` (sides multiples of 16): four activation
/// slots and the pack scratch.
pub fn encode_workspace_bytes(g: &VaeGeometry, height: u32, width: u32) -> u64 {
    4 * slot_bytes(g, height, width) + scratch_bytes(g)
}

/// The encoder's input from RGBA8 bytes (`[height][width][4]`, row-major): the pipeline's
/// `2 (u / 255) − 1` in f32 (`VaeImageProcessor`'s `/ 255` then `normalize`), rounded to BF16
/// (nearest even), the convolution's NHWC input.
pub fn input_bf16(rgba: &[u8]) -> Vec<u16> {
    let table: Vec<u16> = (0..=255u32).map(|u| to_bf16(2.0f32 * (u as f32 / 255.0f32) - 1.0f32)).collect();
    rgba.iter().map(|&u| table[u as usize]).collect()
}

/// The decoder's input from the denormalized latents (packed fp32 `[tokens][z_dim]`): each value
/// rounded to BF16 (nearest even), post_quant_conv's input.
pub fn latents_bf16(latents: &[f32]) -> Vec<u16> {
    latents.iter().map(|&v| to_bf16(v)).collect()
}

/// The decode's tail on the host: conv_out's fp32 output (`[height][width][4]`) to RGBA8 bytes as
/// the pipeline makes them: the decode's clamp to [−1, 1], `VaeImageProcessor`'s `x · 0.5 + 0.5`
/// clamped to [0, 1] (two roundings), then numpy_to_pil's `(x · 255).round()` (half to even) as
/// bytes, every step in f32.
pub fn rgba8(image: &[f32]) -> Vec<u8> {
    image
        .iter()
        .map(|&v| {
            let x = v.clamp(-1.0, 1.0);
            let x = (x * 0.5 + 0.5).clamp(0.0, 1.0);
            (x * 255.0).round_ties_even() as u8
        })
        .collect()
}

/// Rows `[start, end)` of one of the decoder's stages.
pub type Rows = (u32, u32);

/// The decoder's channel ladder: `decoder_base` times `[dim_mult[-1]]` then `dim_mult` reversed:
/// conv_in's and the mid block's width, then each up block's output ([1152, 1152, 1152, 576, 288,
/// 144]).
fn decoder_ladder(g: &VaeGeometry) -> Vec<u32> {
    std::iter::once(*g.dim_mult.last().expect("a VAE has levels"))
        .chain(g.dim_mult.iter().rev().copied())
        .map(|u| (g.decoder_base * u) as u32)
        .collect()
}

/// The rows a 3×3 stride-1 convolution reads for its output rows `out` in a stage of `height` rows
/// (the rows outside `[0, height)` are its zero padding).
fn conv3_rows(out: Rows, height: u32) -> Rows {
    (out.0.saturating_sub(1), (out.1 + 1).min(height))
}

/// The rows of the `height`-row stage before the nearest 2x upsample that the 3×3 convolution after
/// it reads for its output rows `out` (upsampled row `r` is row `r >> 1`).
fn upsample_rows(out: Rows, height: u32) -> Rows {
    (out.0.saturating_sub(1) >> 1, ((out.1 >> 1) + 1).min(height))
}

/// One up block's rows in a band, each in its own stage's rows: the rows of its input its first
/// resnet reads, each resnet's output rows, and its output rows (at twice the resolution after an
/// upsampler, rounded out to even rows so that the DupUp shortcut maps whole input rows onto them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockRows {
    pub input: Rows,
    pub resnets: Vec<Rows>,
    pub output: Rows,
}

/// A band of the decode (spec image-model/04, the VAE decode at 2K): the image rows it writes and
/// each up block's rows. Every stage after the mid block computes its output rows plus the halo the
/// next stage reads, and a convolution pads by the global row, so a band's rows are the whole
/// image's bit for bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BandRows {
    pub out: Rows,
    pub blocks: Vec<BlockRows>,
}

/// The rows band `out` of a decode to `height` rows (a multiple of 16) computes at each stage:
/// from conv_out's halo back through the up blocks to the mid block's latent rows.
pub fn band_rows(g: &VaeGeometry, height: u32, out: Rows) -> BandRows {
    let levels = g.dim_mult.len();
    let latent = height >> (levels - 1);
    let mut blocks = Vec::with_capacity(levels);
    // The rows of each block's output the next stage reads, the last block's for conv_out first.
    let mut want = conv3_rows(out, height);
    for i in (0..levels).rev() {
        let h = latent << i;
        let up = i + 1 < levels;
        let output = if up { (want.0 & !1, ((want.1 + 1) & !1).min(2 * h)) } else { want };
        let mut rows = if up { upsample_rows(output, h) } else { output };
        let mut resnets = vec![(0, 0); g.res_blocks + 1];
        for r in resnets.iter_mut().rev() {
            *r = rows;
            rows = conv3_rows(conv3_rows(rows, h), h);
        }
        blocks.push(BlockRows { input: rows, resnets, output });
        want = rows;
    }
    blocks.reverse();
    BandRows { out, blocks }
}

/// `height` rows in bands of `band_rows` (> 0), the last one shorter when they do not divide.
pub fn bands(height: u32, band_rows: u32) -> Vec<Rows> {
    (0..height).step_by(band_rows as usize).map(|y| (y, (y + band_rows).min(height))).collect()
}

/// The largest BF16 activation band `out` holds at `width` pixels: each resnet's input, conv1's
/// output and its output (or shortcut), each upsampler's output, and norm_out's.
fn band_slot_bytes(g: &VaeGeometry, height: u32, width: u32, out: Rows) -> u64 {
    let dims = decoder_ladder(g);
    let levels = g.dim_mult.len();
    let plan = band_rows(g, height, out);
    let bytes = |rows: Rows, w: u32, c: u32| (rows.1 - rows.0) as u64 * w as u64 * c as u64 * 2;
    let mut most = 0;
    for (i, b) in plan.blocks.iter().enumerate() {
        let (h, w) = ((height >> (levels - 1)) << i, (width >> (levels - 1)) << i);
        let (input, out) = (dims[i], dims[i + 1]);
        for (k, &r) in b.resnets.iter().enumerate() {
            let mid = conv3_rows(r, h);
            most = most.max(bytes(conv3_rows(mid, h), w, if k == 0 { input } else { out })).max(bytes(mid, w, out)).max(bytes(r, w, out));
        }
        if i + 1 < levels {
            most = most.max(bytes(b.output, 2 * w, out));
        }
    }
    most = most.max(bytes(plan.blocks[levels - 1].output, width, dims[levels]));
    align(most)
}

/// The largest 3×3 weight of the decoder, BF16, the pack scratch holds.
fn decoder_scratch_bytes(g: &VaeGeometry) -> u64 {
    let dims = decoder_ladder(g);
    let levels = g.dim_mult.len();
    let mut elems = (dims[0] as u64 * g.z_dim).max(dims[0] as u64 * dims[0] as u64);
    for i in 0..levels {
        elems = elems.max(dims[i + 1] as u64 * dims[i].max(dims[i + 1]) as u64);
    }
    elems = elems.max(g.out_channels * dims[levels] as u64);
    align(elems * 9 * 2)
}

/// The workspace [`decode`] needs for a `height` x `width` image (multiples of 16) in bands of
/// `band_rows` rows: the mid block's output, kept whole while the bands run; four activation slots
/// of the widest of any band's activations and the mid block's attention input (to_qkv's output);
/// the pack scratch.
pub fn decode_workspace_bytes(g: &VaeGeometry, height: u32, width: u32, band_rows: u32) -> u64 {
    mid_bytes(g, height, width) + 4 * decode_slot_bytes(g, height, width, band_rows) + decoder_scratch_bytes(g)
}

/// The mid block's output, BF16, kept whole while the bands run.
fn mid_bytes(g: &VaeGeometry, height: u32, width: u32) -> u64 {
    let levels = g.dim_mult.len();
    align((height >> (levels - 1)) as u64 * (width >> (levels - 1)) as u64 * decoder_ladder(g)[0] as u64 * 2)
}

/// One activation slot of the decode: the widest of any band's activations and the mid block's
/// attention input.
fn decode_slot_bytes(g: &VaeGeometry, height: u32, width: u32, band_rows: u32) -> u64 {
    let levels = g.dim_mult.len();
    let latents = (height >> (levels - 1)) as u64 * (width >> (levels - 1)) as u64;
    let top = decoder_ladder(g)[0] as u64;
    bands(height, band_rows).into_iter().map(|b| band_slot_bytes(g, height, width, b)).fold(align(3 * top * latents * 2), u64::max)
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::os::raw::c_void;

    use ignis_artifact::qwen_image::VaeGeometry;
    use ignis_artifact::{MaterializedArtifact, ObjectHandle};

    use super::{
        bands, check_size, conv3_rows, decode_slot_bytes, decoder_ladder, decoder_scratch_bytes, ladder, mid_bytes, scratch_bytes, slot_bytes, BlockRows, Rows,
    };
    use crate::image_vae::{self as vae, Conv, ConvKind, ShortcutKind};

    /// A convolution's BF16 weight `[out][in][k][k]` as the checkpoint stores it, and its bias.
    #[derive(Debug, Clone, Copy)]
    struct ConvW {
        weight: *const c_void,
        bias: *const c_void,
        out: u32,
        input: u32,
        kernel: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct Resnet {
        norm1: *const c_void,
        conv1: ConvW,
        norm2: *const c_void,
        conv2: ConvW,
        shortcut: Option<ConvW>,
    }

    #[derive(Debug, Clone)]
    struct DownBlock {
        resnets: Vec<Resnet>,
        downsampler: Option<ConvW>,
        temporal: bool,
    }

    /// The encoder's weights where the materializer put them (`vae/encoder.*` and `vae/quant_conv`).
    #[derive(Debug)]
    pub struct VaeEncoderWeights {
        geometry: VaeGeometry,
        conv_in: ConvW,
        blocks: Vec<DownBlock>,
        mid: [Resnet; 2],
        attn_norm: *const c_void,
        to_qkv: ConvW,
        proj: ConvW,
        norm_out: *const c_void,
        conv_out: ConvW,
        quant_conv: ConvW,
    }

    impl VaeEncoderWeights {
        /// `handles` names the bound objects (`qwen_image::bind`).
        pub fn new(artifact: &MaterializedArtifact, handles: &BTreeMap<String, ObjectHandle>, geometry: &VaeGeometry) -> Result<Self, String> {
            let ptr = |name: &str| -> Result<*const c_void, String> {
                let handle = handles.get(name).ok_or(format!("{name} is not bound"))?;
                Ok(artifact.device_view(*handle).map_err(|e| format!("{name}: {e}"))?.base as *const c_void)
            };
            let conv = |prefix: &str, out: u32, input: u32, kernel: u32| -> Result<ConvW, String> {
                Ok(ConvW { weight: ptr(&format!("{prefix}.weight"))?, bias: ptr(&format!("{prefix}.bias"))?, out, input, kernel })
            };
            let resnet = |prefix: &str, input: u32, out: u32| -> Result<Resnet, String> {
                Ok(Resnet {
                    norm1: ptr(&format!("{prefix}.norm1.gamma"))?,
                    conv1: conv(&format!("{prefix}.conv1"), out, input, 3)?,
                    norm2: ptr(&format!("{prefix}.norm2.gamma"))?,
                    conv2: conv(&format!("{prefix}.conv2"), out, out, 3)?,
                    shortcut: if input != out { Some(conv(&format!("{prefix}.conv_shortcut"), out, input, 1)?) } else { None },
                })
            };
            let dims = ladder(geometry);
            let levels = geometry.dim_mult.len();
            let blocks = (0..levels)
                .map(|i| {
                    let (input, out) = (dims[i], dims[i + 1]);
                    Ok(DownBlock {
                        resnets: (0..geometry.res_blocks)
                            .map(|r| resnet(&format!("vae/encoder.down_blocks.{i}.resnets.{r}"), if r == 0 { input } else { out }, out))
                            .collect::<Result<_, String>>()?,
                        downsampler: if i + 1 < levels { Some(conv(&format!("vae/encoder.down_blocks.{i}.downsampler.resample.1"), out, out, 3)?) } else { None },
                        temporal: i + 1 < levels && geometry.temporal_downsample[i],
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let top = dims[levels];
            let z = geometry.z_dim as u32;
            Ok(Self {
                geometry: geometry.clone(),
                conv_in: conv("vae/encoder.conv_in", dims[0], geometry.in_channels as u32, 3)?,
                blocks,
                mid: [resnet("vae/encoder.mid_block.resnets.0", top, top)?, resnet("vae/encoder.mid_block.resnets.1", top, top)?],
                attn_norm: ptr("vae/encoder.mid_block.attentions.0.norm.gamma")?,
                to_qkv: conv("vae/encoder.mid_block.attentions.0.to_qkv", 3 * top, top, 1)?,
                proj: conv("vae/encoder.mid_block.attentions.0.proj", top, top, 1)?,
                norm_out: ptr("vae/encoder.norm_out.gamma")?,
                conv_out: conv("vae/encoder.conv_out", 2 * z, top, 3)?,
                // The posterior's mean: quant_conv's first z_dim output rows (weight and bias).
                quant_conv: conv("vae/quant_conv", z, 2 * z, 1)?,
            })
        }

        pub fn geometry(&self) -> &VaeGeometry {
            &self.geometry
        }
    }

    /// One of the decoder's up blocks: `res_blocks + 1` resnets, the upsampler (nearest 2x, then a
    /// 3×3) in every block but the last, and its DupUp shortcut's temporal factor (2 where the
    /// encoder's mirror level downsamples in time).
    #[derive(Debug, Clone)]
    struct UpBlock {
        resnets: Vec<Resnet>,
        upsampler: Option<ConvW>,
        factor_t: u32,
    }

    /// The decoder's weights where the materializer put them (`vae/post_quant_conv` and
    /// `vae/decoder.*`). The upsamplers' `time_conv`s never run for one frame (diffusers caches
    /// "Rep" on the first chunk instead).
    #[derive(Debug)]
    pub struct VaeDecoderWeights {
        geometry: VaeGeometry,
        post_quant_conv: ConvW,
        conv_in: ConvW,
        mid: [Resnet; 2],
        attn_norm: *const c_void,
        to_qkv: ConvW,
        proj: ConvW,
        blocks: Vec<UpBlock>,
        norm_out: *const c_void,
        conv_out: ConvW,
    }

    impl VaeDecoderWeights {
        /// `handles` names the bound objects (`qwen_image::bind`).
        pub fn new(artifact: &MaterializedArtifact, handles: &BTreeMap<String, ObjectHandle>, geometry: &VaeGeometry) -> Result<Self, String> {
            let ptr = |name: &str| -> Result<*const c_void, String> {
                let handle = handles.get(name).ok_or(format!("{name} is not bound"))?;
                Ok(artifact.device_view(*handle).map_err(|e| format!("{name}: {e}"))?.base as *const c_void)
            };
            let conv = |prefix: &str, out: u32, input: u32, kernel: u32| -> Result<ConvW, String> {
                Ok(ConvW { weight: ptr(&format!("{prefix}.weight"))?, bias: ptr(&format!("{prefix}.bias"))?, out, input, kernel })
            };
            let resnet = |prefix: &str, input: u32, out: u32| -> Result<Resnet, String> {
                Ok(Resnet {
                    norm1: ptr(&format!("{prefix}.norm1.gamma"))?,
                    conv1: conv(&format!("{prefix}.conv1"), out, input, 3)?,
                    norm2: ptr(&format!("{prefix}.norm2.gamma"))?,
                    conv2: conv(&format!("{prefix}.conv2"), out, out, 3)?,
                    shortcut: if input != out { Some(conv(&format!("{prefix}.conv_shortcut"), out, input, 1)?) } else { None },
                })
            };
            let dims = decoder_ladder(geometry);
            let levels = geometry.dim_mult.len();
            let blocks = (0..levels)
                .map(|i| {
                    let (input, out) = (dims[i], dims[i + 1]);
                    let up = i + 1 < levels;
                    Ok(UpBlock {
                        resnets: (0..=geometry.res_blocks)
                            .map(|r| resnet(&format!("vae/decoder.up_blocks.{i}.resnets.{r}"), if r == 0 { input } else { out }, out))
                            .collect::<Result<_, String>>()?,
                        upsampler: if up { Some(conv(&format!("vae/decoder.up_blocks.{i}.upsampler.resample.1"), out, out, 3)?) } else { None },
                        // temperal_upsample is temperal_downsample reversed.
                        factor_t: if up && geometry.temporal_downsample[levels - 2 - i] { 2 } else { 1 },
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let top = dims[0];
            let z = geometry.z_dim as u32;
            Ok(Self {
                geometry: geometry.clone(),
                post_quant_conv: conv("vae/post_quant_conv", z, z, 1)?,
                conv_in: conv("vae/decoder.conv_in", top, z, 3)?,
                mid: [resnet("vae/decoder.mid_block.resnets.0", top, top)?, resnet("vae/decoder.mid_block.resnets.1", top, top)?],
                attn_norm: ptr("vae/decoder.mid_block.attentions.0.norm.gamma")?,
                to_qkv: conv("vae/decoder.mid_block.attentions.0.to_qkv", 3 * top, top, 1)?,
                proj: conv("vae/decoder.mid_block.attentions.0.proj", top, top, 1)?,
                blocks,
                norm_out: ptr("vae/decoder.norm_out.gamma")?,
                conv_out: conv("vae/decoder.conv_out", geometry.out_channels as u32, dims[levels], 3)?,
            })
        }

        pub fn geometry(&self) -> &VaeGeometry {
            &self.geometry
        }
    }

    /// The four activation slots and the pack scratch, carved from the workspace.
    struct Run {
        slots: [*mut c_void; 4],
        free: [bool; 4],
        scratch: *mut c_void,
        slot_bytes: u64,
    }

    impl Run {
        fn take(&mut self) -> usize {
            let i = self.free.iter().position(|&f| f).expect("four slots cover the encoder's liveness");
            self.free[i] = false;
            i
        }

        fn give(&mut self, i: usize) {
            debug_assert!(!self.free[i]);
            self.free[i] = true;
        }

        /// `y = conv(x) + bias (+ residual)`, `x` of `height` x `width`; a 3×3 weight is packed into
        /// the scratch first.
        #[allow(clippy::too_many_arguments)]
        unsafe fn conv(
            &self,
            w: &ConvW,
            kind: ConvKind,
            x: *const c_void,
            height: u32,
            width: u32,
            residual: *const c_void,
            y: *mut c_void,
            y_f32: bool,
        ) -> Result<(), String> {
            let y_rows = if kind == ConvKind::K3x3S2 { height / 2 } else { height };
            unsafe { self.conv_rows(w, kind, false, x, (0, height), height, width, residual, y, (0, y_rows), y_f32) }
        }

        /// `y = conv(x) + bias (+ residual)` over row windows: `x` holds rows `x_rows` of a `height` x
        /// `width` stage (before the upsample, with `upsample`), the call writes output rows `y_rows`
        /// into `y`, and `residual` is laid out as `y`; a 3×3 weight is packed into the scratch first.
        #[allow(clippy::too_many_arguments)]
        unsafe fn conv_rows(
            &self,
            w: &ConvW,
            kind: ConvKind,
            upsample: bool,
            x: *const c_void,
            x_rows: Rows,
            height: u32,
            width: u32,
            residual: *const c_void,
            y: *mut c_void,
            y_rows: Rows,
            y_f32: bool,
        ) -> Result<(), String> {
            let weight = if w.kernel == 3 {
                unsafe { vae::pack_conv_weight(w.weight, w.out, w.input, 3, self.scratch, std::ptr::null_mut()) }?;
                self.scratch as *const c_void
            } else {
                w.weight
            };
            unsafe {
                vae::conv(
                    &Conv {
                        kind,
                        upsample,
                        x,
                        height,
                        width,
                        in_channels: w.input,
                        x_row0: x_rows.0,
                        x_rows: x_rows.1 - x_rows.0,
                        weight,
                        bias: w.bias,
                        out_channels: w.out,
                        residual,
                        y,
                        y_row0: y_rows.0,
                        y_rows: y_rows.1 - y_rows.0,
                        y_f32,
                    },
                    std::ptr::null_mut(),
                )
            }
        }

        unsafe fn norm(&self, x: usize, y: usize, pixels: u32, channels: u32, gamma: *const c_void, silu: bool) -> Result<(), String> {
            unsafe { vae::rms_norm(self.slots[x], pixels, channels, gamma, silu, self.slots[y], false, std::ptr::null_mut()) }
        }

        /// A resnet on slot `x` (`h` x `w`); returns the slot holding its output. `x` is given
        /// back unless `keep_x`, in which case it still holds the input afterwards.
        unsafe fn resnet(&mut self, r: &Resnet, x: usize, h: u32, w: u32, keep_x: bool) -> Result<usize, String> {
            let pixels = h * w;
            let (input, out) = (r.conv1.input, r.conv1.out);
            let shortcut = match &r.shortcut {
                Some(s) => {
                    let d = self.take();
                    unsafe { self.conv(s, ConvKind::K1x1, self.slots[x], h, w, std::ptr::null(), self.slots[d], false) }?;
                    Some(d)
                }
                None => None,
            };
            let b = self.take();
            unsafe { self.norm(x, b, pixels, input, r.norm1, true) }?;
            if shortcut.is_some() && !keep_x {
                self.give(x);
            }
            let c = self.take();
            unsafe { self.conv(&r.conv1, ConvKind::K3x3, self.slots[b], h, w, std::ptr::null(), self.slots[c], false) }?;
            self.give(b);
            unsafe { self.norm(c, c, pixels, out, r.norm2, true) }?;
            // The residual is added in place (the convolution allows residual == y for BF16).
            let y = match shortcut {
                Some(d) => d,
                None if !keep_x => x,
                None => self.take(),
            };
            let residual = self.slots[shortcut.unwrap_or(x)];
            unsafe { self.conv(&r.conv2, ConvKind::K3x3, self.slots[c], h, w, residual, self.slots[y], false) }?;
            self.give(c);
            Ok(y)
        }

        fn take_act(&mut self, rows: Rows, channels: u32) -> Act {
            let slot = self.take();
            Act { ptr: self.slots[slot], slot: Some(slot), rows, channels }
        }

        fn give_act(&mut self, a: Act) {
            if let Some(slot) = a.slot {
                self.give(slot);
            }
        }

        /// A resnet over output rows `out` of an `h` x `w` stage, `x` holding at least the rows its
        /// two 3×3 convolutions read. The output goes to `into` when given; otherwise in place (into
        /// the shortcut, or into x's own rows when x's slot is consumed) or into a new slot. With
        /// `consume`, x's slot is given back or holds the output; an `x` outside the slots is never
        /// written.
        #[allow(clippy::too_many_arguments)]
        unsafe fn resnet_rows(&mut self, r: &Resnet, x: Act, h: u32, w: u32, out: Rows, into: Option<Act>, consume: bool) -> Result<Act, String> {
            let null = std::ptr::null_mut();
            let mid = conv3_rows(out, h);
            let input = conv3_rows(mid, h);
            let (cin, cout) = (r.conv1.input, r.conv1.out);
            let shortcut = match &r.shortcut {
                Some(s) => {
                    let d = into.unwrap_or_else(|| self.take_act(out, cout));
                    unsafe { self.conv_rows(s, ConvKind::K1x1, false, x.ptr, x.rows, h, w, std::ptr::null(), d.ptr, out, false) }?;
                    Some(d)
                }
                None => None,
            };
            let b = self.take_act(input, cin);
            unsafe { vae::rms_norm(x.row(input.0, w), (input.1 - input.0) * w, cin, r.norm1, true, b.ptr, false, null) }?;
            if shortcut.is_some() && consume {
                self.give_act(x);
            }
            let c = self.take_act(mid, cout);
            unsafe { self.conv_rows(&r.conv1, ConvKind::K3x3, false, b.ptr, input, h, w, std::ptr::null(), c.ptr, mid, false) }?;
            self.give_act(b);
            unsafe { vae::rms_norm(c.ptr, (mid.1 - mid.0) * w, cout, r.norm2, true, c.ptr, false, null) }?;
            // The residual is added in place where it can be (residual == y is allowed for BF16).
            let y = match (shortcut, into) {
                (Some(d), _) => d,
                (None, Some(into)) => into,
                (None, None) if consume && x.slot.is_some() => Act { ptr: x.row(out.0, w), slot: x.slot, rows: out, channels: cout },
                (None, None) => self.take_act(out, cout),
            };
            let residual = shortcut.map_or(x.row(out.0, w), |d| d.ptr);
            unsafe { self.conv_rows(&r.conv2, ConvKind::K3x3, false, c.ptr, mid, h, w, residual, y.ptr, out, false) }?;
            self.give_act(c);
            if shortcut.is_none() && consume && y.slot != x.slot {
                self.give_act(x);
            }
            Ok(y)
        }

        /// Up block `block` over a band: `x` holds at least rows `rows.input` of its `h` x `w` input;
        /// returns its output rows `rows.output`. With an upsampler, the DupUp shortcut of the input
        /// goes into the output's slot first, and the upsampler's convolution adds it as its
        /// residual, in place: one rounding where diffusers' BF16 path rounds the convolution's
        /// output first.
        unsafe fn up_block(&mut self, block: &UpBlock, x: Act, h: u32, w: u32, rows: &BlockRows) -> Result<Act, String> {
            let (cin, cout) = (block.resnets[0].conv1.input, block.resnets[0].conv1.out);
            let shortcut = match block.upsampler {
                Some(_) => {
                    let s = self.take_act(rows.output, cout);
                    let (y0, y1) = rows.output;
                    unsafe {
                        vae::shortcut(ShortcutKind::DupUp, x.row(y0 / 2, w), (y1 - y0) / 2, w, cin, cout, block.factor_t, 2, s.ptr, false, false, std::ptr::null_mut())
                    }?;
                    Some(s)
                }
                None => None,
            };
            let mut y = x;
            for (r, &out) in block.resnets.iter().zip(&rows.resnets) {
                y = unsafe { self.resnet_rows(r, y, h, w, out, None, true) }?;
            }
            match (shortcut, &block.upsampler) {
                (Some(s), Some(up)) => {
                    unsafe { self.conv_rows(up, ConvKind::K3x3, true, y.ptr, y.rows, h, w, s.ptr, s.ptr, s.rows, false) }?;
                    self.give_act(y);
                    Ok(s)
                }
                _ => Ok(y),
            }
        }
    }

    /// A band's BF16 NHWC activation: rows `rows` of a stage at `ptr`, in the run's slot `slot`
    /// (None: outside the slots, the mid block's output).
    #[derive(Debug, Clone, Copy)]
    struct Act {
        ptr: *mut c_void,
        slot: Option<usize>,
        rows: Rows,
        channels: u32,
    }

    impl Act {
        /// Where row `r` starts, at `width` pixels a row.
        fn row(&self, r: u32, width: u32) -> *mut c_void {
            debug_assert!(self.rows.0 <= r && r <= self.rows.1, "row {r} of an activation holding {:?}", self.rows);
            unsafe { (self.ptr as *mut u8).add((r - self.rows.0) as usize * width as usize * self.channels as usize * 2) as *mut c_void }
        }
    }

    /// The encoder on `input` (BF16 NHWC `[height][width][4]`, [`super::input_bf16`]): the
    /// posterior's mean into `latents`, fp32 `[height / 16 · width / 16][z_dim]` (the packed order,
    /// token `y · w + x`), not yet normalized. Everything runs on the legacy stream.
    ///
    /// # Safety
    /// `weights` points at live device weights; `input`, `latents` and `workspace` (`workspace_bytes`,
    /// at least [`super::encode_workspace_bytes`]) are device pointers of the sizes the shapes imply,
    /// `input` and `latents` outside the workspace.
    pub unsafe fn encode(
        weights: &VaeEncoderWeights,
        input: *const c_void,
        height: u32,
        width: u32,
        latents: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
    ) -> Result<(), String> {
        let g = &weights.geometry;
        check_size(g, height, width)?;
        let slot = slot_bytes(g, height, width);
        let need = 4 * slot + scratch_bytes(g);
        if workspace.is_null() || (workspace as usize) % 256 != 0 || workspace_bytes < need {
            return Err(format!("the VAE encoder's workspace holds {workspace_bytes} bytes (256-byte aligned), it needs {need}"));
        }
        let at = |i: u64| unsafe { (workspace as *mut u8).add((i * slot) as usize) as *mut c_void };
        let mut run = Run { slots: [at(0), at(1), at(2), at(3)], free: [true; 4], scratch: at(4), slot_bytes: slot };

        let (mut h, mut w) = (height, width);
        let mut x = run.take();
        unsafe { run.conv(&weights.conv_in, ConvKind::K3x3, input, h, w, std::ptr::null(), run.slots[x], false) }?;
        for block in &weights.blocks {
            let (input, out) = (block.resnets[0].conv1.input, block.resnets[0].conv1.out);
            match &block.downsampler {
                Some(down) => {
                    // The shortcut first, into the downsampler's output: its residual.
                    let s = run.take();
                    let factor_t = if block.temporal { 2 } else { 1 };
                    unsafe { vae::shortcut(ShortcutKind::AvgDown, run.slots[x], h, w, input, out, factor_t, 2, run.slots[s], false, false, std::ptr::null_mut()) }?;
                    for r in &block.resnets {
                        x = unsafe { run.resnet(r, x, h, w, false) }?;
                    }
                    unsafe { run.conv(down, ConvKind::K3x3S2, run.slots[x], h, w, run.slots[s], run.slots[s], false) }?;
                    run.give(x);
                    x = s;
                    (h, w) = (h / 2, w / 2);
                }
                None => {
                    // The identity shortcut: the block's input, kept, added at the end.
                    let mut y = x;
                    for (i, r) in block.resnets.iter().enumerate() {
                        y = unsafe { run.resnet(r, y, h, w, i == 0) }?;
                    }
                    unsafe { vae::shortcut(ShortcutKind::AvgDown, run.slots[x], h, w, input, out, 1, 1, run.slots[y], true, false, std::ptr::null_mut()) }?;
                    run.give(x);
                    x = y;
                }
            }
        }

        // The mid block.
        let pixels = h * w;
        let top = weights.proj.out;
        x = unsafe { run.resnet(&weights.mid[0], x, h, w, false) }?;
        let (b, c, d) = (run.take(), run.take(), run.take());
        unsafe { run.norm(x, b, pixels, top, weights.attn_norm, false) }?;
        unsafe { run.conv(&weights.to_qkv, ConvKind::K1x1, run.slots[b], h, w, std::ptr::null(), run.slots[c], false) }?;
        let mut row_tile = pixels;
        while row_tile > 1 && vae::attn_workspace_bytes(pixels, top, row_tile) > run.slot_bytes {
            row_tile = row_tile.div_ceil(2);
        }
        let ws = vae::attn_workspace_bytes(pixels, top, row_tile);
        if ws == 0 || ws > run.slot_bytes {
            return Err(format!("the mid block's attention over {pixels} tokens does not fit a {}-byte slot", run.slot_bytes));
        }
        let scale = (1.0f64 / (top as f64).sqrt()) as f32;
        unsafe { vae::attn(run.slots[c], pixels, top, scale, row_tile, run.slots[d], ws, run.slots[b], false, std::ptr::null_mut()) }?;
        unsafe { run.conv(&weights.proj, ConvKind::K1x1, run.slots[b], h, w, run.slots[x], run.slots[x], false) }?;
        for i in [b, c, d] {
            run.give(i);
        }
        x = unsafe { run.resnet(&weights.mid[1], x, h, w, false) }?;

        // The head: norm_out, SiLU, conv_out, then quant_conv's mean rows in fp32.
        let b = run.take();
        unsafe { run.norm(x, b, pixels, top, weights.norm_out, true) }?;
        run.give(x);
        let c = run.take();
        unsafe { run.conv(&weights.conv_out, ConvKind::K3x3, run.slots[b], h, w, std::ptr::null(), run.slots[c], false) }?;
        unsafe { run.conv(&weights.quant_conv, ConvKind::K1x1, run.slots[c], h, w, std::ptr::null(), latents as *mut c_void, true) }?;
        run.give(b);
        run.give(c);
        Ok(())
    }

    /// The decoder on `latents` (BF16 `[height / 16 · width / 16][z_dim]` in the packed order, the
    /// denormalized latents, [`super::latents_bf16`]): conv_out's output into `image`, fp32
    /// `[height][width][out_channels]`, not yet clamped. post_quant_conv, conv_in and the mid block
    /// run on the whole latent grid (the attention is global); the up blocks, norm_out and conv_out
    /// run in bands of `band_rows` image rows ([`super::band_rows`]), each band from the mid block's
    /// output with its halos, so the image is the same bit for bit in any bands. Everything runs on
    /// the legacy stream.
    ///
    /// # Safety
    /// `weights` points at live device weights; `latents`, `image` and `workspace` (`workspace_bytes`,
    /// at least [`super::decode_workspace_bytes`]) are device pointers of the sizes the shapes imply,
    /// `latents` and `image` outside the workspace.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn decode(
        weights: &VaeDecoderWeights,
        latents: *const c_void,
        height: u32,
        width: u32,
        band_rows: u32,
        image: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
    ) -> Result<(), String> {
        let g = &weights.geometry;
        check_size(g, height, width)?;
        if band_rows == 0 {
            return Err("a decode in bands of 0 rows".into());
        }
        let mid_at = mid_bytes(g, height, width);
        let slot = decode_slot_bytes(g, height, width, band_rows);
        let need = mid_at + 4 * slot + decoder_scratch_bytes(g);
        if workspace.is_null() || (workspace as usize) % 256 != 0 || workspace_bytes < need {
            return Err(format!("the VAE decoder's workspace holds {workspace_bytes} bytes (256-byte aligned), it needs {need}"));
        }
        let at = |offset: u64| unsafe { (workspace as *mut u8).add(offset as usize) as *mut c_void };
        let mut run = Run {
            slots: [at(mid_at), at(mid_at + slot), at(mid_at + 2 * slot), at(mid_at + 3 * slot)],
            free: [true; 4],
            scratch: at(mid_at + 4 * slot),
            slot_bytes: slot,
        };
        let null = std::ptr::null_mut();
        let levels = g.dim_mult.len();
        let (h, w) = (height >> (levels - 1), width >> (levels - 1));
        let all = (0, h);
        let top = weights.proj.out;

        // post_quant_conv, conv_in, the mid block; its output stays at the workspace's start.
        let a = run.take_act(all, g.z_dim as u32);
        unsafe { run.conv_rows(&weights.post_quant_conv, ConvKind::K1x1, false, latents, all, h, w, std::ptr::null(), a.ptr, all, false) }?;
        let x = run.take_act(all, top);
        unsafe { run.conv_rows(&weights.conv_in, ConvKind::K3x3, false, a.ptr, all, h, w, std::ptr::null(), x.ptr, all, false) }?;
        run.give_act(a);
        let x = unsafe { run.resnet_rows(&weights.mid[0], x, h, w, all, None, true) }?;
        let pixels = h * w;
        let (b, c, d) = (run.take_act(all, top), run.take_act(all, 3 * top), run.take_act(all, top));
        unsafe { vae::rms_norm(x.ptr, pixels, top, weights.attn_norm, false, b.ptr, false, null) }?;
        unsafe { run.conv_rows(&weights.to_qkv, ConvKind::K1x1, false, b.ptr, all, h, w, std::ptr::null(), c.ptr, all, false) }?;
        let mut row_tile = pixels;
        while row_tile > 1 && vae::attn_workspace_bytes(pixels, top, row_tile) > run.slot_bytes {
            row_tile = row_tile.div_ceil(2);
        }
        let ws = vae::attn_workspace_bytes(pixels, top, row_tile);
        if ws == 0 || ws > run.slot_bytes {
            return Err(format!("the mid block's attention over {pixels} tokens does not fit a {}-byte slot", run.slot_bytes));
        }
        let scale = (1.0f64 / (top as f64).sqrt()) as f32;
        unsafe { vae::attn(c.ptr, pixels, top, scale, row_tile, d.ptr, ws, b.ptr, false, null) }?;
        unsafe { run.conv_rows(&weights.proj, ConvKind::K1x1, false, b.ptr, all, h, w, x.ptr, x.ptr, all, false) }?;
        for a in [b, c, d] {
            run.give_act(a);
        }
        let mid = Act { ptr: at(0), slot: None, rows: all, channels: top };
        unsafe { run.resnet_rows(&weights.mid[1], x, h, w, all, Some(mid), true) }?;

        // The bands: the up blocks from the mid block's output, then norm_out, SiLU and conv_out
        // into the band's rows of the image.
        let last = *decoder_ladder(g).last().expect("a VAE has levels");
        let out_channels = weights.conv_out.out as usize;
        for band in bands(height, band_rows) {
            let plan = super::band_rows(g, height, band);
            let mut x = mid;
            for (i, block) in weights.blocks.iter().enumerate() {
                x = unsafe { run.up_block(block, x, h << i, w << i, &plan.blocks[i]) }?;
            }
            let rows = plan.blocks[levels - 1].output;
            let b = run.take_act(rows, last);
            unsafe { vae::rms_norm(x.row(rows.0, width), (rows.1 - rows.0) * width, last, weights.norm_out, true, b.ptr, false, null) }?;
            run.give_act(x);
            let y = unsafe { image.add(band.0 as usize * width as usize * out_channels) } as *mut c_void;
            unsafe { run.conv_rows(&weights.conv_out, ConvKind::K3x3, false, b.ptr, rows, height, width, std::ptr::null(), y, band, true) }?;
            run.give_act(b);
            debug_assert!(run.free.iter().all(|&f| f), "a band gives back every slot");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::qwen_image::QwenImageGeometry;

    fn config(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        // The checkpoint's vae/config.json fields the parse reads (Qwen/Qwen-Image-2.1-Turbo at fc13ba19).
        let mut v = serde_json::json!({
            "_class_name": "AutoencoderKLQwenImage21", "attn_scales": [], "base_dim": 96, "decoder_base_dim": 144,
            "dim_mult": [1, 2, 4, 8, 8], "in_channels": 4, "is_residual": true, "num_res_blocks": 2, "out_channels": 4,
            "patch_size": null, "temperal_downsample": [false, true, true, true], "z_dim": 64
        });
        edit(&mut v);
        serde_json::to_vec(&v).unwrap()
    }

    #[test]
    fn the_vae_config_must_be_the_inventorys_residual_unpatched_vae() {
        let g = QwenImageGeometry::qwen_image_21_turbo().vae;
        assert_eq!(VaeConfig::parse(&config(|_| {}), &g), Ok(VaeConfig));
        // diffusers' defaults stand for absent keys.
        assert!(VaeConfig::parse(&config(|v| { v.as_object_mut().unwrap().remove("is_residual"); }), &g).is_ok());
        let refused = |edit: fn(&mut serde_json::Value)| VaeConfig::parse(&config(edit), &g).unwrap_err();
        assert!(refused(|v| v["is_residual"] = false.into()).contains("is_residual"));
        assert!(refused(|v| v["patch_size"] = 2.into()).contains("patch_size"));
        assert!(refused(|v| v["attn_scales"] = serde_json::json!([0.5])).contains("attn_scales"));
        assert!(refused(|v| v["z_dim"] = 16.into()).contains("z_dim is 16"));
        assert!(refused(|v| v["dim_mult"] = serde_json::json!([1, 2, 4, 4])).contains("dim_mult"));
        assert!(refused(|v| v["temperal_downsample"] = serde_json::json!([false, true, true, false])).contains("temperal_downsample"));
        // The decoder's width: absent is diffusers' default (144), null is base_dim's (96).
        assert!(VaeConfig::parse(&config(|v| { v.as_object_mut().unwrap().remove("decoder_base_dim"); }), &g).is_ok());
        assert!(refused(|v| v["decoder_base_dim"] = serde_json::Value::Null).contains("decoder_base_dim is 96"));
        assert!(refused(|v| v["decoder_base_dim"] = 128.into()).contains("decoder_base_dim is 128"));
    }

    #[test]
    fn the_workspace_is_four_slots_of_the_widest_activation_and_the_largest_3x3_weight() {
        let g = QwenImageGeometry::qwen_image_21_turbo().vae;
        // 1024²: level 0's 96 channels at full size are the widest (96 · 2^20 BF16 = 192 MiB); the
        // largest 3×3 weight is 768 x 768 x 9.
        assert_eq!(slot_bytes(&g, 1024, 1024), 96 << 21);
        assert_eq!(scratch_bytes(&g), 768 * 768 * 9 * 2);
        assert_eq!(encode_workspace_bytes(&g, 1024, 1024), 4 * (96 << 21) + 768 * 768 * 9 * 2);
        assert!(check_size(&g, 1024, 1008).is_ok());
        assert!(check_size(&g, 1024, 1000).unwrap_err().contains("multiples of 16"));
        assert!(check_size(&g, 0, 16).is_err());
    }

    #[test]
    fn a_decode_bands_rows_follow_each_stages_halo_back_to_the_latents() {
        let g = QwenImageGeometry::qwen_image_21_turbo().vae;
        // An interior band of a 2048² decode, rows [224, 448), worked by hand from the ops' halos
        // (3×3: one row each side; the upsample's 3×3: [(y0 - 1) >> 1, (y1 >> 1) + 1) before it).
        let band = band_rows(&g, 2048, (224, 448));
        assert_eq!(band.out, (224, 448));
        // conv_out reads [223, 449) of block 4's output (2048 rows, no upsampler): its three
        // resnets, two 3×3 each.
        let b4 = &band.blocks[4];
        assert_eq!((b4.output, b4.resnets.clone(), b4.input), ((223, 449), vec![(219, 453), (221, 451), (223, 449)], (217, 455)));
        // Block 3 (1024 rows, upsampled to 2048): its output rounded out to even rows, then the
        // upsample's rows, then the resnets'.
        let b3 = &band.blocks[3];
        assert_eq!((b3.output, b3.resnets.clone(), b3.input), ((216, 456), vec![(103, 233), (105, 231), (107, 229)], (101, 235)));
        let b2 = &band.blocks[2];
        assert_eq!((b2.output, b2.resnets.clone(), b2.input), ((100, 236), vec![(45, 123), (47, 121), (49, 119)], (43, 125)));
        let b1 = &band.blocks[1];
        assert_eq!((b1.output, b1.resnets.clone(), b1.input), ((42, 126), vec![(16, 68), (18, 66), (20, 64)], (14, 70)));
        // Block 0 reads the mid block's 128 latent rows: 42 of them for this band's 14.
        let b0 = &band.blocks[0];
        assert_eq!((b0.output, b0.resnets.clone(), b0.input), ((14, 70), vec![(2, 40), (4, 38), (6, 36)], (0, 42)));

        // The first band clamps at row 0; the whole image is every stage's whole height.
        let first = band_rows(&g, 2048, (0, 224));
        assert_eq!((first.blocks[4].output, first.blocks[3].output, first.blocks[0].input), ((0, 225), (0, 232), (0, 28)));
        let whole = band_rows(&g, 1024, (0, 1024));
        for (i, b) in whole.blocks.iter().enumerate() {
            let h = 64 << i;
            assert_eq!(b.input, (0, h), "block {i}'s input");
            assert!(b.resnets.iter().all(|&r| r == (0, h)), "block {i}'s resnets");
            assert_eq!(b.output, (0, if i < 4 { 2 * h } else { h }), "block {i}'s output");
        }
    }

    #[test]
    fn the_bands_tile_the_image_and_each_block_reads_rows_its_predecessor_wrote() {
        let g = QwenImageGeometry::qwen_image_21_turbo().vae;
        assert_eq!(bands(1024, 400), vec![(0, 400), (400, 800), (800, 1024)]);
        for (height, rows) in [(1024, 512), (1024, 1024), (2048, 224), (1792, 160), (2400, 400), (512, 48)] {
            let bands = bands(height, rows);
            assert_eq!(bands.first().unwrap().0, 0);
            assert_eq!(bands.last().unwrap().1, height);
            assert!(bands.windows(2).all(|w| w[0].1 == w[1].0), "{height}/{rows}: the bands tile the image");
            for &b in &bands {
                let plan = band_rows(&g, height, b);
                assert!(plan.blocks[0].input.1 <= height / 16, "{height}/{rows} {b:?}: the latents' rows");
                for i in 1..plan.blocks.len() {
                    let (have, need) = (plan.blocks[i - 1].output, plan.blocks[i].input);
                    assert!(have.0 <= need.0 && need.1 <= have.1, "{height}/{rows} {b:?}: block {i} reads {need:?}, block {} wrote {have:?}", i - 1);
                }
                for b in &plan.blocks[..4] {
                    assert!(b.output.0 % 2 == 0 && b.output.1 % 2 == 0, "an upsampled output starts and ends on even rows");
                }
            }
        }
    }

    #[test]
    fn the_decode_workspace_holds_the_mid_block_and_four_slots_of_the_bands_widest_activation() {
        let g = QwenImageGeometry::qwen_image_21_turbo().vae;
        let scratch = 1152 * 1152 * 9 * 2;
        // Unbanded at 1024²: the widest activation is block 3's output (block 4's input), 288
        // channels at 1024², 604 MB; the mid block's output stays whole (1152 channels at 64²);
        // the pack scratch holds a 1152 x 1152 3×3 weight.
        assert_eq!(decode_workspace_bytes(&g, 1024, 1024, 1024), 64 * 64 * 1152 * 2 + 4 * (1024 * 1024 * 288 * 2) + scratch);
        // In bands of 224 rows at 2048², an interior band's block 3 output is 240 rows of 2048.
        assert_eq!(decode_workspace_bytes(&g, 2048, 2048, 224), 128 * 128 * 1152 * 2 + 4 * (240 * 2048 * 288 * 2) + scratch);
        // Narrow bands of a tall image: the mid block's attention input (to_qkv's 3 x 1152
        // channels over the 128 x 16 latents) is the widest, and the slots are its size.
        assert_eq!(decode_workspace_bytes(&g, 2048, 256, 16), 128 * 16 * 1152 * 2 + 4 * (128 * 16 * 3 * 1152 * 2) + scratch);
    }

    #[test]
    fn the_decoded_image_is_the_pipelines_clamped_and_rounded_bytes() {
        // The expected bytes are diffusers 0.41.0's tail run on these values in the venv (the decode's
        // clamp to [-1, 1], VaeImageProcessor.postprocess, numpy_to_pil's (x * 255).round()).
        // fl(fl(tie * 0.5 + 0.5) * 255) is 126.5 exactly, and numpy rounds half to even.
        let tie = -65793.0f32 / 8388608.0;
        assert_eq!(rgba8(&[-2.0, -1.0, -0.5, 0.0, 0.3, 0.5, 1.0, 1.5, tie]), vec![0, 0, 64, 128, 166, 191, 255, 255, 126]);
    }

    #[test]
    fn the_decoders_latents_round_to_bf16_nearest_even() {
        // 1 + 2^-8 is halfway between BF16's 1 and 1 + 2^-7: the even one, 1. 1 + 3 x 2^-8 is
        // halfway between 1 + 2^-7 and 1 + 2^-6: the even one, 1 + 2^-6.
        assert_eq!(latents_bf16(&[1.0 + 1.0 / 256.0, 1.0 + 3.0 / 256.0, -2.5]), vec![0x3F80, 0x3F82, 0xC020]);
        // A NaN stays a NaN, whatever its bits: torch's quiet NaN, never a zero or an overflow
        // (bits at or above 0x7FFF8000 would carry into the sign, or past u32 for a negative one).
        let nans = [f32::NAN, f32::from_bits(0x7FFF_FFFF), f32::from_bits(0xFFFF_FFFF), f32::from_bits(0xFF80_0001)];
        assert_eq!(latents_bf16(&nans), vec![0x7FC0; 4]);
        assert_eq!(latents_bf16(&[f32::INFINITY, f32::NEG_INFINITY, f32::MAX]), vec![0x7F80, 0xFF80, 0x7F80]);
    }

    #[test]
    fn the_input_is_the_pipelines_normalized_bytes_in_bf16() {
        let got = input_bf16(&[0, 255, 128, 1]);
        // -1 and 1 exactly; 2 (128 / 255) - 1 = 2^-8 x 1.0039216 rounds to 2^-8 x (1 + 1/128), BF16 0x3B81.
        assert_eq!((got[0], got[1]), (0xBF80, 0x3F80));
        assert_eq!(got[2], 0x3B81);
        let v = 2.0f32 * (1.0f32 / 255.0) - 1.0;
        assert_eq!(got[3], to_bf16(v));
    }
}
