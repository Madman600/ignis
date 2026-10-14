//! Qwen-Image-2.1-Turbo's artifact family (spec image-model/01, ADR 0050):
//! its object inventory and the binder plan that consumes every object.
//!
//! The container is an ordinary `.ninfer` v2 file, written by the packer
//! ([`crate::packer::Family::QwenImage21`]) from the converter's work tree
//! (`tools/image-converter`). What makes it this family is its identity
//! (`model_id` [`MODEL_ID`]) and its inventory, generated here from a
//! [`QwenImageGeometry`] — the checkpoint's or a synthetic fixture's:
//! - the text encoder's decoder layers under `text_encoder/` (the checkpoint
//!   name without `model.language_model.`): linears FP8 with a per-row
//!   scale, norms BF16, the token embedding BF16 and read on the host (only a
//!   prompt's rows go to the device);
//! - the vision tower, its merger and the DeepStack mergers under the 27B
//!   artifact's own names (`vision/...`,
//!   [`crate::inventory::vision_scope_27b`]), every matrix W8 (the patch
//!   embedding, the blocks' and the mergers' linears; the 27B's blocks are
//!   Q4/Q5/Q6), BF16 biases, norms and positions;
//! - the DiT under `transformer/` and the VAE under `vae/`, checkpoint names:
//!   the DiT's linears FP8, its modulation, timestep embedder and norms BF16,
//!   the VAE BF16 at the checkpoint's own shapes;
//! - the frontend resources ([`FRONTEND_FILES`]).
//!
//! `lm_head` and the text model's final norm are not stored: the encoder's
//! output is taken before that norm. [`bind`] requires every inventory
//! object with its exact format, layout and shape, refuses a missing one by
//! name, and fails on any object the inventory does not name (ADR 0002).

use std::collections::BTreeMap;

use crate::binder::{Binder, MaterializationPlan, ObjectHandle};
use crate::{fail, NumericFormat, Reader, ResourceEncoding, Result, StorageLayout};

/// The container identity's `model_id` for every artifact of the family:
/// its published name, which is also the id it is served under (ADR 0050).
pub const MODEL_ID: &str = "ignis-image-2.1-turbo";

/// The converter's work-tree units after `frontend/`, in container order:
/// each a `tensors.json` whose names all start with `<unit>/`.
pub const WORK_UNITS: [&str; 4] = ["text_encoder", "vision", "transformer", "vae"];

/// The frontend resources (`frontend/<file>`): the processor's tokenizer and
/// chat template, the checkpoint's configs, and the converter's
/// `image_model.json` (sampler sigmas, scheduler, image processor, latent
/// normalization, prompt templates, source revision).
pub const FRONTEND_FILES: [&str; 11] = [
    "chat_template.jinja",
    "image_model.json",
    "model_index.json",
    "processor_config.json",
    "scheduler_config.json",
    "text_encoder_config.json",
    "text_encoder_generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "transformer_config.json",
    "vae_config.json",
];

/// The Qwen3-VL text model the encoder runs (its decoder layers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextEncoderGeometry {
    pub layers: usize,
    pub hidden: u64,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub intermediate: u64,
    pub vocab: u64,
}

/// The Qwen3-VL vision tower with its merger and DeepStack mergers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionGeometry {
    pub depth: usize,
    pub hidden: u64,
    pub intermediate: u64,
    /// `in_channels · temporal_patch · patch²`: the patch embedding's columns.
    pub patch_dim: u64,
    pub positions: u64,
    pub merge_size: u64,
    /// The merger's output width: the text model's hidden.
    pub out_hidden: u64,
    pub deepstack_mergers: usize,
}

impl VisionGeometry {
    /// A merged token's width: `merge_size²` patches side by side.
    pub fn merged(&self) -> u64 {
        self.hidden * self.merge_size * self.merge_size
    }
}

/// The diffusion transformer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DitGeometry {
    pub blocks: usize,
    pub heads: u64,
    pub head_dim: u64,
    pub mlp: u64,
    /// Latent channels in and out (the VAE's `z_dim`, patch size 1).
    pub in_channels: u64,
    pub out_channels: u64,
    /// The text encoder's hidden, which `txt_in` reads.
    pub context: u64,
    /// The sinusoidal timestep projection's width.
    pub timestep: u64,
}

impl DitGeometry {
    pub fn hidden(&self) -> u64 {
        self.heads * self.head_dim
    }
}

/// The VAE (diffusers `AutoencoderKLQwenImage21`, `is_residual`): the
/// encoder's and decoder's channel ladders and where they resample.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VaeGeometry {
    pub in_channels: u64,
    pub out_channels: u64,
    pub z_dim: u64,
    pub base: u64,
    pub decoder_base: u64,
    pub dim_mult: Vec<u64>,
    pub res_blocks: usize,
    /// Per encoder level but the last: whether its downsampler carries a
    /// temporal (1×1 `time_conv`) branch; the decoder mirrors it.
    pub temporal_downsample: Vec<bool>,
}

/// Every count and width the inventory is generated from.
///
/// The checkpoint's numbers ([`QwenImageGeometry::qwen_image_21_turbo`]) are
/// its configs' at the pinned revision; `crates/artifact/tests/qwen_image.rs`
/// checks the DiT, VAE and text inventories against the converter's pinned
/// tensor list (`tools/image-converter/pin.json`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenImageGeometry {
    pub text: TextEncoderGeometry,
    pub vision: VisionGeometry,
    pub dit: DitGeometry,
    pub vae: VaeGeometry,
}

impl QwenImageGeometry {
    /// `Qwen/Qwen-Image-2.1-Turbo` at revision `fc13ba19`.
    pub fn qwen_image_21_turbo() -> Self {
        Self {
            text: TextEncoderGeometry {
                layers: 36,
                hidden: 4096,
                heads: 32,
                kv_heads: 8,
                head_dim: 128,
                intermediate: 12288,
                vocab: 151_936,
            },
            vision: VisionGeometry {
                depth: 27,
                hidden: 1152,
                intermediate: 4304,
                patch_dim: 3 * 2 * 16 * 16,
                positions: 2304,
                merge_size: 2,
                out_hidden: 4096,
                deepstack_mergers: 3,
            },
            dit: DitGeometry {
                blocks: 32,
                heads: 32,
                head_dim: 128,
                mlp: 12288,
                in_channels: 64,
                out_channels: 64,
                context: 4096,
                timestep: 256,
            },
            vae: VaeGeometry {
                in_channels: 4,
                out_channels: 4,
                z_dim: 64,
                base: 96,
                decoder_base: 144,
                dim_mult: vec![1, 2, 4, 8, 8],
                res_blocks: 2,
                temporal_downsample: vec![false, true, true, true],
            },
        }
    }

    /// The converter's synthetic checkpoint (`tools/image-converter/
    /// synthetic.py`): 2 text layers, a 3-block vision tower with 3 DeepStack
    /// mergers, 2 DiT blocks, a VAE with the real one's five levels.
    pub fn fixture() -> Self {
        Self {
            text: TextEncoderGeometry {
                layers: 2,
                hidden: 64,
                heads: 4,
                kv_heads: 2,
                head_dim: 16,
                intermediate: 128,
                vocab: 256,
            },
            vision: VisionGeometry {
                depth: 3,
                hidden: 32,
                intermediate: 64,
                patch_dim: 3 * 2 * 4 * 4,
                positions: 16,
                merge_size: 2,
                out_hidden: 64,
                deepstack_mergers: 3,
            },
            dit: DitGeometry {
                blocks: 2,
                heads: 4,
                head_dim: 16,
                mlp: 192,
                in_channels: 16,
                out_channels: 16,
                context: 64,
                timestep: 256,
            },
            vae: VaeGeometry {
                in_channels: 4,
                out_channels: 4,
                z_dim: 16,
                base: 2,
                decoder_base: 3,
                dim_mult: vec![1, 2, 4, 8, 8],
                res_blocks: 2,
                temporal_downsample: vec![false, true, true, true],
            },
        }
    }
}

// ---------------------------------------------------------------------------
// The inventory
// ---------------------------------------------------------------------------

/// Where a bound tensor goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// The device arena (the materializer uploads it).
    Device,
    /// Host memory: the token embedding, whose prompt rows are uploaded.
    Host,
}

/// One tensor of the inventory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub format: NumericFormat,
    pub layout: StorageLayout,
    pub shape: Vec<u64>,
    pub role: Role,
}

fn entry(name: String, (format, layout): (NumericFormat, StorageLayout), shape: &[u64]) -> Entry {
    Entry { name, format, layout, shape: shape.to_vec(), role: Role::Device }
}

const FP8: (NumericFormat, StorageLayout) = (NumericFormat::Fp8E4M3FnRowBf16S, StorageLayout::RowScaleV1);
const BF16: (NumericFormat, StorageLayout) = (NumericFormat::Bf16, StorageLayout::ContiguousLeV1);
const W8: (NumericFormat, StorageLayout) = (NumericFormat::W8G32F16S, StorageLayout::RowSplitK128V1);

/// The text encoder's objects: the embedding (host), then each layer's.
pub fn text_entries(g: &TextEncoderGeometry) -> Vec<Entry> {
    let mut entries = vec![Entry {
        role: Role::Host,
        ..entry("text_encoder/embed_tokens.weight".into(), BF16, &[g.vocab, g.hidden])
    }];
    let (q, kv) = (g.heads * g.head_dim, g.kv_heads * g.head_dim);
    for l in 0..g.layers {
        let n = |suffix: &str| format!("text_encoder/layers.{l}.{suffix}.weight");
        entries.push(entry(n("input_layernorm"), BF16, &[g.hidden]));
        entries.push(entry(n("post_attention_layernorm"), BF16, &[g.hidden]));
        entries.push(entry(n("self_attn.q_norm"), BF16, &[g.head_dim]));
        entries.push(entry(n("self_attn.k_norm"), BF16, &[g.head_dim]));
        entries.push(entry(n("self_attn.q_proj"), FP8, &[q, g.hidden]));
        entries.push(entry(n("self_attn.k_proj"), FP8, &[kv, g.hidden]));
        entries.push(entry(n("self_attn.v_proj"), FP8, &[kv, g.hidden]));
        entries.push(entry(n("self_attn.o_proj"), FP8, &[g.hidden, q]));
        entries.push(entry(n("mlp.gate_proj"), FP8, &[g.intermediate, g.hidden]));
        entries.push(entry(n("mlp.up_proj"), FP8, &[g.intermediate, g.hidden]));
        entries.push(entry(n("mlp.down_proj"), FP8, &[g.hidden, g.intermediate]));
    }
    entries
}

/// A merger's six objects under `prefix` (the merger, or one DeepStack
/// merger, whose norm runs after the shuffle: `merged` wide, not `hidden`).
fn merger_entries(g: &VisionGeometry, prefix: &str, norm: u64) -> Vec<Entry> {
    let m = g.merged();
    vec![
        entry(format!("{prefix}/fc1"), W8, &[m, m]),
        entry(format!("{prefix}/fc1_bias"), BF16, &[m]),
        entry(format!("{prefix}/fc2"), W8, &[g.out_hidden, m]),
        entry(format!("{prefix}/fc2_bias"), BF16, &[g.out_hidden]),
        entry(format!("{prefix}/norm/weight"), BF16, &[norm]),
        entry(format!("{prefix}/norm/bias"), BF16, &[norm]),
    ]
}

/// The vision tower's objects, in the 27B artifact's names with every matrix
/// W8, then the DeepStack mergers.
pub fn vision_entries(g: &VisionGeometry) -> Vec<Entry> {
    let h = g.hidden;
    let mut entries = vec![
        entry("vision/patch_embedding".into(), W8, &[h, g.patch_dim]),
        entry("vision/patch_embedding_bias".into(), BF16, &[h]),
        entry("vision/position_embedding".into(), BF16, &[g.positions, h]),
    ];
    for b in 0..g.depth {
        let n = |suffix: &str| format!("vision/layers/{b}/{suffix}");
        entries.push(entry(n("attention/qkv"), W8, &[3 * h, h]));
        entries.push(entry(n("attention/qkv_bias"), BF16, &[3 * h]));
        entries.push(entry(n("attention/output"), W8, &[h, h]));
        entries.push(entry(n("attention/output_bias"), BF16, &[h]));
        entries.push(entry(n("mlp/fc1"), W8, &[g.intermediate, h]));
        entries.push(entry(n("mlp/fc1_bias"), BF16, &[g.intermediate]));
        entries.push(entry(n("mlp/fc2"), W8, &[h, g.intermediate]));
        entries.push(entry(n("mlp/fc2_bias"), BF16, &[h]));
        for norm in ["norm1", "norm2"] {
            entries.push(entry(n(&format!("{norm}/weight")), BF16, &[h]));
            entries.push(entry(n(&format!("{norm}/bias")), BF16, &[h]));
        }
    }
    entries.extend(merger_entries(g, "vision/merger", h));
    for i in 0..g.deepstack_mergers {
        entries.extend(merger_entries(g, &format!("vision/deepstack/{i}"), g.merged()));
    }
    entries
}

/// The DiT's objects (`transformer/` + the checkpoint name).
pub fn dit_entries(g: &DitGeometry) -> Vec<Entry> {
    let h = g.hidden();
    let n = |suffix: &str| format!("transformer/{suffix}.weight");
    let mut entries = vec![
        entry(n("img_in"), FP8, &[h, g.in_channels]),
        entry(n("txt_in.text_norm"), BF16, &[g.context]),
        entry(n("txt_in.in_layer"), FP8, &[h, g.context]),
        entry(n("txt_in.out_layer"), FP8, &[h, h]),
        entry(n("time_text_embed.timestep_embedder.linear_1"), BF16, &[h, g.timestep]),
        entry(n("time_text_embed.timestep_embedder.linear_2"), BF16, &[h, h]),
        // the one modulation all blocks share: four vectors of `hidden`
        entry(n("modulation.1"), BF16, &[4 * h, h]),
    ];
    for b in 0..g.blocks {
        let block = |suffix: &str| n(&format!("transformer_blocks.{b}.{suffix}"));
        for linear in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0"] {
            entries.push(entry(block(linear), FP8, &[h, h]));
        }
        entries.push(entry(block("attn.norm_q"), BF16, &[g.head_dim]));
        entries.push(entry(block("attn.norm_k"), BF16, &[g.head_dim]));
        entries.push(entry(block("img_mlp.proj"), FP8, &[g.mlp, h]));
        entries.push(entry(block("img_mlp.gate_layer"), FP8, &[g.mlp, h]));
        entries.push(entry(block("img_mlp.out"), FP8, &[h, g.mlp]));
    }
    // norm_out's linear is its modulation (scale from the timestep embedding)
    entries.push(entry(n("norm_out.linear"), BF16, &[h, h]));
    entries.push(entry(n("proj_out"), FP8, &[g.out_channels, h]));
    entries
}

fn conv(entries: &mut Vec<Entry>, prefix: &str, out: u64, input: u64, kernel: u64) {
    entries.push(entry(format!("{prefix}.weight"), BF16, &[out, input, kernel, kernel]));
    entries.push(entry(format!("{prefix}.bias"), BF16, &[out]));
}

/// A residual block: two RMS-normed 3×3 convolutions and a 1×1 shortcut
/// when the width changes.
fn resnet(entries: &mut Vec<Entry>, prefix: &str, input: u64, out: u64) {
    entries.push(entry(format!("{prefix}.norm1.gamma"), BF16, &[input, 1, 1, 1]));
    conv(entries, &format!("{prefix}.conv1"), out, input, 3);
    entries.push(entry(format!("{prefix}.norm2.gamma"), BF16, &[out, 1, 1, 1]));
    conv(entries, &format!("{prefix}.conv2"), out, out, 3);
    if input != out {
        conv(entries, &format!("{prefix}.conv_shortcut"), out, input, 1);
    }
}

/// A mid block: a residual block, one single-head attention, a residual block.
fn mid_block(entries: &mut Vec<Entry>, prefix: &str, c: u64) {
    resnet(entries, &format!("{prefix}.resnets.0"), c, c);
    entries.push(entry(format!("{prefix}.attentions.0.norm.gamma"), BF16, &[c, 1, 1]));
    conv(entries, &format!("{prefix}.attentions.0.to_qkv"), 3 * c, c, 1);
    conv(entries, &format!("{prefix}.attentions.0.proj"), c, c, 1);
    resnet(entries, &format!("{prefix}.resnets.1"), c, c);
}

/// The VAE's objects (`vae/` + the checkpoint name), all BF16.
pub fn vae_entries(g: &VaeGeometry) -> Vec<Entry> {
    let mut e = Vec::new();
    let levels = g.dim_mult.len();
    // encoder: `base` times [1] + dim_mult, one down block per step
    let dims: Vec<u64> = std::iter::once(1).chain(g.dim_mult.iter().copied()).map(|u| g.base * u).collect();
    conv(&mut e, "vae/encoder.conv_in", dims[0], g.in_channels, 3);
    for (i, pair) in dims.windows(2).enumerate() {
        let (input, out) = (pair[0], pair[1]);
        for r in 0..g.res_blocks {
            resnet(&mut e, &format!("vae/encoder.down_blocks.{i}.resnets.{r}"), if r == 0 { input } else { out }, out);
        }
        if i != levels - 1 {
            conv(&mut e, &format!("vae/encoder.down_blocks.{i}.downsampler.resample.1"), out, out, 3);
            if g.temporal_downsample[i] {
                conv(&mut e, &format!("vae/encoder.down_blocks.{i}.downsampler.time_conv"), out, out, 1);
            }
        }
    }
    let top = dims[levels];
    mid_block(&mut e, "vae/encoder.mid_block", top);
    e.push(entry("vae/encoder.norm_out.gamma".into(), BF16, &[top, 1, 1, 1]));
    conv(&mut e, "vae/encoder.conv_out", 2 * g.z_dim, top, 3);
    conv(&mut e, "vae/quant_conv", 2 * g.z_dim, 2 * g.z_dim, 1);
    conv(&mut e, "vae/post_quant_conv", g.z_dim, g.z_dim, 1);
    // decoder: `decoder_base` times [dim_mult[-1]] + dim_mult reversed, one
    // up block per step; the temporal flags reversed
    let dims: Vec<u64> = std::iter::once(g.dim_mult[levels - 1])
        .chain(g.dim_mult.iter().rev().copied())
        .map(|u| g.decoder_base * u)
        .collect();
    let temporal_upsample: Vec<bool> = g.temporal_downsample.iter().rev().copied().collect();
    conv(&mut e, "vae/decoder.conv_in", dims[0], g.z_dim, 3);
    mid_block(&mut e, "vae/decoder.mid_block", dims[0]);
    for (i, pair) in dims.windows(2).enumerate() {
        let (input, out) = (pair[0], pair[1]);
        for r in 0..=g.res_blocks {
            resnet(&mut e, &format!("vae/decoder.up_blocks.{i}.resnets.{r}"), if r == 0 { input } else { out }, out);
        }
        if i != levels - 1 {
            conv(&mut e, &format!("vae/decoder.up_blocks.{i}.upsampler.resample.1"), out, out, 3);
            if temporal_upsample[i] {
                conv(&mut e, &format!("vae/decoder.up_blocks.{i}.upsampler.time_conv"), 2 * out, out, 1);
            }
        }
    }
    let last = dims[levels];
    e.push(entry("vae/decoder.norm_out.gamma".into(), BF16, &[last, 1, 1, 1]));
    conv(&mut e, "vae/decoder.conv_out", g.out_channels, last, 3);
    e
}

/// Every tensor of the family, in inventory order (binding is by name; the
/// container's own order is the converter's).
pub fn entries(g: &QwenImageGeometry) -> Vec<Entry> {
    let mut all = text_entries(&g.text);
    all.extend(vision_entries(&g.vision));
    all.extend(dit_entries(&g.dit));
    all.extend(vae_entries(&g.vae));
    all
}

// ---------------------------------------------------------------------------
// The binder plan
// ---------------------------------------------------------------------------

/// A Qwen-Image artifact bound: the plan (every object placed) and every
/// object's handle by name.
#[derive(Debug, Clone)]
pub struct QwenImagePlan {
    pub plan: MaterializationPlan,
    pub handles: BTreeMap<String, ObjectHandle>,
}

/// Refuses an artifact whose identity is not this family's.
fn require_family(reader: &Reader) -> Result<()> {
    if reader.identity().model_id != MODEL_ID {
        return Err(fail(format!(
            "artifact model_id {} is not a Qwen-Image-2.1 artifact ({MODEL_ID})",
            reader.identity().model_id
        )));
    }
    Ok(())
}

/// Requires `entry`'s tensor with its exact format, layout and shape and
/// places it by its role.
fn require_entry(binder: &mut Binder, entry: Entry, handles: &mut BTreeMap<String, ObjectHandle>) -> Result<()> {
    let handle = binder.require_tensor(&entry.name, entry.format, entry.layout, &entry.shape)?;
    match entry.role {
        Role::Device => binder.materialize_on_device(handle)?,
        Role::Host => binder.retain_tensor_on_host(handle)?,
    }
    handles.insert(entry.name, handle);
    Ok(())
}

/// Bind a Qwen-Image artifact: every frontend resource and every inventory
/// tensor is required with its exact format, layout and shape (a missing one
/// is refused by name), placed by its role, and any object the inventory does
/// not name fails the bind (ADR 0002).
pub fn bind(reader: &Reader, geometry: &QwenImageGeometry) -> Result<QwenImagePlan> {
    require_family(reader)?;
    let mut binder = Binder::new(reader);
    let mut handles = BTreeMap::new();
    for file in FRONTEND_FILES {
        let name = format!("frontend/{file}");
        let handle = binder.require_resource(&name, ResourceEncoding::RawBytesV1)?;
        binder.retain_on_host(handle)?;
        handles.insert(name, handle);
    }
    for entry in entries(geometry) {
        require_entry(&mut binder, entry, &mut handles)?;
    }
    Ok(QwenImagePlan { plan: binder.finish()?, handles })
}

/// The text encoder's config resource: the checkpoint's
/// `text_encoder/config.json` verbatim (eps, rope theta, mrope section).
pub const TEXT_ENCODER_CONFIG: &str = "frontend/text_encoder_config.json";

/// The DiT's config resource: the checkpoint's `transformer/config.json`
/// verbatim (eps, the RoPE axes, `causal_condition`).
pub const TRANSFORMER_CONFIG: &str = "frontend/transformer_config.json";

/// The converter's `image_model.json`: the sampler's sigmas and scheduler
/// config, the latent normalization, the processor, the prompt templates.
pub const IMAGE_MODEL: &str = "frontend/image_model.json";

/// The VAE's config resource: the checkpoint's `vae/config.json` verbatim
/// (the residual blocks, the patch size, the ladder).
pub const VAE_CONFIG: &str = "frontend/vae_config.json";

/// Bind the text encoder alone (spec image-model/03): its config resource
/// and every `text_encoder/` tensor, each with its exact format, layout and
/// shape and placed by its role. A scoped bind ([`Binder::plan`]): the
/// vision tower, the DiT, the VAE and the other resources are left alone,
/// for a test or tool that runs the encoder without the rest.
pub fn bind_text_encoder(reader: &Reader, geometry: &TextEncoderGeometry) -> Result<QwenImagePlan> {
    require_family(reader)?;
    let mut binder = Binder::new(reader);
    let mut handles = BTreeMap::new();
    let config = binder.require_resource(TEXT_ENCODER_CONFIG, ResourceEncoding::RawBytesV1)?;
    binder.retain_on_host(config)?;
    handles.insert(TEXT_ENCODER_CONFIG.to_string(), config);
    for entry in text_entries(geometry) {
        require_entry(&mut binder, entry, &mut handles)?;
    }
    Ok(QwenImagePlan { plan: binder.plan(), handles })
}

/// Bind the vision tower alone (spec image-model/03, GitHub #320): the
/// config resource (its `vision_config` names the DeepStack taps) and every
/// `vision/` tensor, the merger's and the DeepStack mergers' among them, as
/// [`bind_text_encoder`] binds the text model's. A scoped bind.
pub fn bind_vision(reader: &Reader, geometry: &VisionGeometry) -> Result<QwenImagePlan> {
    require_family(reader)?;
    let mut binder = Binder::new(reader);
    let mut handles = BTreeMap::new();
    let config = binder.require_resource(TEXT_ENCODER_CONFIG, ResourceEncoding::RawBytesV1)?;
    binder.retain_on_host(config)?;
    handles.insert(TEXT_ENCODER_CONFIG.to_string(), config);
    for entry in vision_entries(geometry) {
        require_entry(&mut binder, entry, &mut handles)?;
    }
    Ok(QwenImagePlan { plan: binder.plan(), handles })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::vision_scope_27b;

    /// The checkpoint's tensor counts (its safetensors headers at
    /// `fc13ba19`): 750 text-encoder tensors less `lm_head` and the final
    /// norm, 297 DiT tensors, 238 VAE tensors.
    #[test]
    fn the_real_inventory_has_the_checkpoints_counts() {
        let g = QwenImageGeometry::qwen_image_21_turbo();
        assert_eq!(text_entries(&g.text).len() + vision_entries(&g.vision).len(), 750 - 2);
        assert_eq!(dit_entries(&g.dit).len(), 297);
        assert_eq!(vae_entries(&g.vae).len(), 238);
        let all = entries(&g);
        let names: std::collections::BTreeSet<_> = all.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names.len(), all.len(), "names are unique");
        let fp8 = all.iter().filter(|e| e.format == NumericFormat::Fp8E4M3FnRowBf16S).count();
        assert_eq!(fp8, 36 * 7 + 32 * 7 + 4);
        let host: Vec<_> = all.iter().filter(|e| e.role == Role::Host).map(|e| e.name.as_str()).collect();
        assert_eq!(host, ["text_encoder/embed_tokens.weight"]);
    }

    /// The vision tower is the 27B's by name and shape, except where the
    /// image model differs: the merger writes 4096 wide (the 27B 5120), the
    /// three DeepStack mergers are new, and every matrix is W8 (the 27B's
    /// blocks are Q4/Q5/Q6), the rest the 27B's BF16.
    #[test]
    fn the_vision_tower_is_the_27bs_names_in_w8() {
        let ours = vision_entries(&QwenImageGeometry::qwen_image_21_turbo().vision);
        let theirs = vision_scope_27b();
        assert_eq!(ours.len(), theirs.len() + 3 * 6);
        for (a, b) in ours.iter().zip(&theirs) {
            assert_eq!(a.name, b.name);
            if !a.name.starts_with("vision/merger/fc2") {
                assert_eq!(a.shape, b.shape, "{}", a.name);
            }
            let want = if b.format == NumericFormat::Bf16 { BF16 } else { W8 };
            assert_eq!((a.format, a.layout), want, "{}", a.name);
        }
        let w8 = ours.iter().filter(|e| (e.format, e.layout) == W8).count();
        assert_eq!(w8, 1 + 27 * 4 + 2 + 3 * 2);
        assert!(ours.iter().all(|e| [W8, BF16].contains(&(e.format, e.layout))));
        assert_eq!(ours[theirs.len()].name, "vision/deepstack/0/fc1");
        assert_eq!(ours.last().unwrap().shape, [4608], "a DeepStack norm is post-shuffle");
    }
}
