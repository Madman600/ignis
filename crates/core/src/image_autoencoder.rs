//! The image model's VAE encoder (spec image-model/04, `vae_encode`; GitHub #321): an RGBA image
//! to the DiT's condition latents. It is diffusers 0.41.0's `AutoencoderKLQwenImage21.encode` for
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
//! Our own implementation (ADR 0051, tier 1).

use ignis_artifact::qwen_image::VaeGeometry;

/// The VAE config's settings the encoder depends on (`frontend/vae_config.json`, the checkpoint's
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

/// The encoder's refusal of a size: sides positive multiples of the total downsampling.
pub fn check_size(g: &VaeGeometry, height: u32, width: u32) -> Result<(), String> {
    let factor = 1u32 << (g.dim_mult.len() - 1);
    if height == 0 || width == 0 || height % factor != 0 || width % factor != 0 || height as u64 * width as u64 > 1 << 24 {
        return Err(format!(
            "a {width} x {height} condition image: the VAE takes sides that are positive multiples of {factor}, at most 2^24 pixels"
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
    let table: Vec<u16> = (0..=255u32)
        .map(|u| {
            let v = 2.0f32 * (u as f32 / 255.0f32) - 1.0f32;
            let b = v.to_bits();
            ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
        })
        .collect();
    rgba.iter().map(|&u| table[u as usize]).collect()
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::os::raw::c_void;

    use ignis_artifact::qwen_image::VaeGeometry;
    use ignis_artifact::{MaterializedArtifact, ObjectHandle};

    use super::{check_size, ladder, scratch_bytes, slot_bytes};
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
            let weight = if w.kernel == 3 {
                unsafe { vae::pack_conv_weight(w.weight, w.out, w.input, 3, self.scratch, std::ptr::null_mut()) }?;
                self.scratch as *const c_void
            } else {
                w.weight
            };
            let y_rows = if kind == ConvKind::K3x3S2 { height / 2 } else { height };
            unsafe {
                vae::conv(
                    &Conv {
                        kind,
                        upsample: false,
                        x,
                        height,
                        width,
                        in_channels: w.input,
                        x_row0: 0,
                        x_rows: height,
                        weight,
                        bias: w.bias,
                        out_channels: w.out,
                        residual,
                        y,
                        y_row0: 0,
                        y_rows,
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
    fn the_input_is_the_pipelines_normalized_bytes_in_bf16() {
        let got = input_bf16(&[0, 255, 128, 1]);
        // -1 and 1 exactly; 2 (128 / 255) - 1 = 2^-8 x 1.0039216 rounds to 2^-8 x (1 + 1/128), BF16 0x3B81.
        assert_eq!((got[0], got[1]), (0xBF80, 0x3F80));
        assert_eq!(got[2], 0x3B81);
        let v = 2.0f32 * (1.0f32 / 255.0) - 1.0;
        assert_eq!(f32::from_bits((got[3] as u32) << 16), {
            let b = v.to_bits();
            f32::from_bits(((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16)
        });
    }
}
