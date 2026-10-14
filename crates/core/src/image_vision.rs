//! The image model's vision tower, Qwen3-VL-8B's (spec image-model/03, GitHub #320): what it reads
//! from the artifact's config (the DeepStack taps), the host side of one call (the condition
//! images' BF16 patch rows and the encoder control over all of them), and, with the `cuda`
//! feature, the flat C ABI of `kernel/include/ignis_image_vision.h`, 1:1, a thin wrapper that turns
//! its return code into a `Result` carrying the leaf's own message, and the weights' descriptor
//! over a materialized artifact.
//!
//! The tower is the 27B's program (`kernel/src/vision_encode.cu`) with the image model's
//! parameters: the merger 4096 wide, up to [`MAX_SEGMENTS`] condition images as attention segments
//! in one call, and the three DeepStack mergers. Its outputs feed the text encoder's hooks
//! (`image_encoder::ffi::ignis_image_encoder_vision`): the merger's rows over the image-pad rows,
//! DeepStack merger i's after text layer i.
//!
//! The device call is asynchronous on the stream it is given (null: the legacy default stream);
//! a caller that moves data through `CudaDevice`'s own stream orders the copies itself
//! (`moe::stream_sync(null)` after the call).
//!
//! Our own implementation (ADR 0051, tier 1); the program it runs is the 27B tower's (GitHub #178).

use ignis_artifact::qwen_image::VisionGeometry;
use ignis_artifact::vision::PATCH_FEATURES;

use crate::vision::{vision_item_control, Grid};

/// The condition images (one attention segment each) one call takes:
/// `IGNIS_IMAGE_VISION_MAX_SEGMENTS`.
pub const MAX_SEGMENTS: u32 = 10;

/// The vision config's numbers the tower reads (`vision_config` of
/// `frontend/text_encoder_config.json`, the checkpoint's `text_encoder/config.json` verbatim).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisionTowerConfig {
    pub num_heads: u32,
    /// The blocks after which the DeepStack mergers tap the residual (transformers'
    /// `deepstack_visual_indexes`), 0-based, one per DeepStack merger.
    pub deepstack_layers: Vec<u32>,
}

impl VisionTowerConfig {
    /// Reads the config's `vision_config`. Refuses one whose dimensions are not `geometry`'s (the
    /// artifact's inventory), whose block activation is not GELU-tanh, or whose DeepStack taps are
    /// not one per DeepStack merger, ascending, inside the tower.
    pub fn parse(bytes: &[u8], geometry: &VisionGeometry) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("the text encoder config is not JSON: {e}"))?;
        let vision = json.get("vision_config").ok_or("the text encoder config has no vision_config")?;
        let uint = |key: &str| vision.get(key).and_then(|v| v.as_u64()).ok_or(format!("vision_config.{key} is missing"));
        for (key, want) in [
            ("depth", geometry.depth as u64),
            ("hidden_size", geometry.hidden),
            ("intermediate_size", geometry.intermediate),
            ("out_hidden_size", geometry.out_hidden),
            ("spatial_merge_size", geometry.merge_size),
            ("num_position_embeddings", geometry.positions),
        ] {
            let got = uint(key)?;
            if got != want {
                return Err(format!("vision_config.{key} is {got}, the artifact's inventory has {want}"));
            }
        }
        let patch = uint("in_channels")? * uint("temporal_patch_size")? * uint("patch_size")?.pow(2);
        if patch != geometry.patch_dim {
            return Err(format!(
                "vision_config's patch (in_channels x temporal_patch_size x patch_size^2) is {patch}, the artifact's inventory has {}",
                geometry.patch_dim
            ));
        }
        if vision.get("hidden_act").and_then(|v| v.as_str()) != Some("gelu_pytorch_tanh") {
            return Err("vision_config.hidden_act is not gelu_pytorch_tanh".into());
        }
        let num_heads = uint("num_heads")? as u32;
        let taps: Vec<u64> = vision
            .get("deepstack_visual_indexes")
            .and_then(|v| v.as_array())
            .ok_or("vision_config.deepstack_visual_indexes is missing")?
            .iter()
            .map(|v| v.as_u64().ok_or("vision_config.deepstack_visual_indexes holds a non-integer"))
            .collect::<Result<_, _>>()?;
        if taps.len() != geometry.deepstack_mergers
            || taps.windows(2).any(|w| w[0] >= w[1])
            || taps.iter().any(|&t| t >= geometry.depth as u64)
        {
            return Err(format!(
                "vision_config.deepstack_visual_indexes {taps:?} is not {} ascending blocks below {}",
                geometry.deepstack_mergers, geometry.depth
            ));
        }
        Ok(Self { num_heads, deepstack_layers: taps.into_iter().map(|t| t as u32).collect() })
    }
}

/// One tower call's host inputs: the condition images one after the other, each image's patch
/// rows and encoder control as the 27B's media encode takes them
/// ([`crate::vision::vision_item_control`]).
#[derive(Debug, Clone, PartialEq)]
pub struct VisionCall {
    /// BF16 `[patches][1536]`, the f32 `pixel_values` rounded to nearest even.
    pub patches: Vec<u16>,
    /// Axis-major `[2][patches]`: every patch's row, then every patch's column, each image's
    /// from 0.
    pub position_ids: Vec<i32>,
    /// `[segments + 1]` segment bounds over the patches, one segment per frame.
    pub cu_seqlens: Vec<i32>,
    /// `[patches][4]` bilinear position-table corners and their weights.
    pub position_table_indices: Vec<i32>,
    pub position_table_weights: Vec<f32>,
}

impl VisionCall {
    pub fn patch_count(&self) -> u32 {
        (self.patches.len() / PATCH_FEATURES) as u32
    }

    /// The merged tokens the call emits: one per image-pad row, in order.
    pub fn tokens(&self) -> u32 {
        self.patch_count() / 4
    }

    pub fn segments(&self) -> u32 {
        (self.cu_seqlens.len() - 1) as u32
    }
}

/// `v` rounded to BF16 bits, nearest even, as torch's `.to(torch.bfloat16)`: a NaN is torch's
/// quiet NaN, 0x7FC0. The image leaf's one host rounding (the tower's patches, the VAE's input and
/// the decoder's latents).
pub(crate) fn to_bf16(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7FC0;
    }
    let b = v.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

/// The call over `images`, each its grid and its f32 `pixel_values` (`[raw patches][1536]`, the
/// processor's, as `image_encoder_input::VisionPatches` holds them). Each image keeps its own
/// positions and table coordinates (transformers' `rot_pos_emb` and `fast_pos_embed_interpolate`
/// run per image); the position planes and the segment bounds run over all of them.
pub fn vision_call(images: &[(Grid, &[f32])]) -> Result<VisionCall, String> {
    let segments: u64 = images.iter().map(|(g, _)| g.t as u64).sum();
    if images.is_empty() || segments > MAX_SEGMENTS as u64 {
        return Err(format!("a tower call takes 1 to {MAX_SEGMENTS} frames, not {segments}"));
    }
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut call = VisionCall {
        patches: Vec::new(),
        position_ids: Vec::new(),
        cu_seqlens: vec![0],
        position_table_indices: Vec::new(),
        position_table_weights: Vec::new(),
    };
    for (i, (grid, pixels)) in images.iter().enumerate() {
        let raw = grid.raw_patches() as usize;
        if pixels.len() != raw * PATCH_FEATURES || grid.h % 2 != 0 || grid.w % 2 != 0 {
            return Err(format!(
                "image {i}: {} pixel values for a {}x{}x{} grid (merge-aligned, {PATCH_FEATURES} per patch)",
                pixels.len(),
                grid.t,
                grid.h,
                grid.w
            ));
        }
        let control = vision_item_control(*grid);
        rows.extend_from_slice(&control.position_ids[..raw]);
        cols.extend_from_slice(&control.position_ids[raw..]);
        let offset = *call.cu_seqlens.last().unwrap();
        call.cu_seqlens.extend(control.cu_seqlens[1..].iter().map(|b| offset + b));
        call.position_table_indices.extend_from_slice(&control.position_table_indices);
        call.position_table_weights.extend_from_slice(&control.position_table_weights);
        call.patches.extend(pixels.iter().map(|&v| to_bf16(v)));
    }
    call.position_ids = rows;
    call.position_ids.extend(cols);
    Ok(call)
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::ffi::CStr;
    use std::os::raw::c_void;

    use ignis_artifact::qwen_image::VisionGeometry;
    use ignis_artifact::{MaterializedArtifact, ObjectHandle};

    use super::{VisionCall, VisionTowerConfig};

    /// 1:1 with `kernel/include/ignis_image_vision.h`.
    pub mod ffi {
        use std::os::raw::{c_char, c_void};

        pub const BLOCKS: u32 = 27;
        pub const HIDDEN: u32 = 1152;
        pub const INTERMEDIATE: u32 = 4304;
        pub const PATCH_DIM: u32 = 1536;
        pub const MERGE_UNIT: u32 = 4;
        pub const OUT_HIDDEN: u32 = 4096;
        pub const DEEPSTACK: usize = 3;
        pub const MAX_SEGMENTS: u32 = super::super::MAX_SEGMENTS;
        /// The leaf's attention heads (fixed in `vision_encode.cu`, not in the header).
        pub const HEADS: u32 = 16;
        /// The position table's rows (48 x 48).
        pub const POSITIONS: u32 = 2304;

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_vision_w8 {
            pub codes: *const c_void,
            pub scales: *const c_void,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_vision_block {
            pub qkv: ignis_image_vision_w8,
            pub qkv_bias: *const c_void,
            pub output: ignis_image_vision_w8,
            pub output_bias: *const c_void,
            pub fc1: ignis_image_vision_w8,
            pub fc1_bias: *const c_void,
            pub fc2: ignis_image_vision_w8,
            pub fc2_bias: *const c_void,
            pub norm1_weight: *const c_void,
            pub norm1_bias: *const c_void,
            pub norm2_weight: *const c_void,
            pub norm2_bias: *const c_void,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_vision_merger {
            pub fc1: ignis_image_vision_w8,
            pub fc1_bias: *const c_void,
            pub fc2: ignis_image_vision_w8,
            pub fc2_bias: *const c_void,
            pub norm_weight: *const c_void,
            pub norm_bias: *const c_void,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_vision_weights {
            pub patch_embedding: ignis_image_vision_w8,
            pub patch_embedding_bias: *const c_void,
            pub position_embedding: *const c_void,
            pub blocks: *const ignis_image_vision_block,
            pub block_count: u32,
            pub merger: ignis_image_vision_merger,
            pub deepstack: [ignis_image_vision_merger; DEEPSTACK],
            pub deepstack_layers: [u32; DEEPSTACK],
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_vision_input {
            pub patches: u32,
            pub segments: u32,
            pub patch_rows: *const u16,
            pub position_ids: *const i32,
            pub cu_seqlens: *const i32,
            pub position_table_indices: *const i32,
            pub position_table_weights: *const f32,
        }

        unsafe extern "C" {
            pub fn ignis_image_vision_workspace_bytes(tokens: u32, segments: u32) -> u64;
            pub fn ignis_image_vision_forward(
                weights: *const ignis_image_vision_weights,
                input: *const ignis_image_vision_input,
                embeds: *mut c_void,
                deepstack: *const *mut c_void,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_vision_last_error() -> *const c_char;
        }
    }

    /// The leaf's message for the most recent failed tower call on this thread.
    pub fn last_error() -> String {
        unsafe { CStr::from_ptr(ffi::ignis_image_vision_last_error()) }.to_string_lossy().into_owned()
    }

    /// The workspace a call of `tokens` merged tokens over `segments` segments needs, in bytes.
    pub fn workspace_bytes(tokens: u32, segments: u32) -> u64 {
        unsafe { ffi::ignis_image_vision_workspace_bytes(tokens, segments) }
    }

    /// The tower's descriptor over a materialized artifact: every `vision/` tensor where the
    /// materializer put it (each W8 payload's codes at its base, its scales at its scale plane,
    /// as the 27B's load reads a row-split tensor), with the config's DeepStack taps.
    #[derive(Debug)]
    pub struct VisionWeights {
        blocks: Vec<ffi::ignis_image_vision_block>,
        weights: ffi::ignis_image_vision_weights,
    }

    impl VisionWeights {
        /// `handles` names the bound objects (`qwen_image::bind` or `bind_vision`). Refuses a
        /// geometry or a config the leaf does not run.
        pub fn new(
            artifact: &MaterializedArtifact,
            handles: &BTreeMap<String, ObjectHandle>,
            geometry: &VisionGeometry,
            config: &VisionTowerConfig,
        ) -> Result<Self, String> {
            let leaf = (
                ffi::BLOCKS as usize,
                ffi::HIDDEN as u64,
                ffi::INTERMEDIATE as u64,
                ffi::PATCH_DIM as u64,
                ffi::POSITIONS as u64,
                2u64,
                ffi::OUT_HIDDEN as u64,
                ffi::DEEPSTACK,
            );
            let ours = (
                geometry.depth,
                geometry.hidden,
                geometry.intermediate,
                geometry.patch_dim,
                geometry.positions,
                geometry.merge_size,
                geometry.out_hidden,
                geometry.deepstack_mergers,
            );
            if ours != leaf {
                return Err(format!("the vision geometry {geometry:?} is not the one the leaf runs"));
            }
            if config.num_heads != ffi::HEADS || config.deepstack_layers.len() != ffi::DEEPSTACK {
                return Err(format!("the vision config {config:?} is not the one the leaf runs ({} heads)", ffi::HEADS));
            }
            let view = |name: &str| {
                let handle = handles.get(name).ok_or(format!("{name} is not bound"))?;
                artifact.device_view(*handle).map_err(|e| format!("{name}: {e}"))
            };
            let bf16 = |name: &str| -> Result<*const c_void, String> { Ok(view(name)?.base as *const c_void) };
            let w8 = |name: &str| -> Result<ffi::ignis_image_vision_w8, String> {
                let v = view(name)?;
                let scales = v.scale_plane().ok_or(format!("{name} has no scale plane"))?;
                Ok(ffi::ignis_image_vision_w8 { codes: v.base as *const c_void, scales: scales as *const c_void })
            };
            let merger = |prefix: &str| -> Result<ffi::ignis_image_vision_merger, String> {
                Ok(ffi::ignis_image_vision_merger {
                    fc1: w8(&format!("{prefix}/fc1"))?,
                    fc1_bias: bf16(&format!("{prefix}/fc1_bias"))?,
                    fc2: w8(&format!("{prefix}/fc2"))?,
                    fc2_bias: bf16(&format!("{prefix}/fc2_bias"))?,
                    norm_weight: bf16(&format!("{prefix}/norm/weight"))?,
                    norm_bias: bf16(&format!("{prefix}/norm/bias"))?,
                })
            };
            let blocks = (0..geometry.depth)
                .map(|b| {
                    let n = |suffix: &str| format!("vision/layers/{b}/{suffix}");
                    Ok(ffi::ignis_image_vision_block {
                        qkv: w8(&n("attention/qkv"))?,
                        qkv_bias: bf16(&n("attention/qkv_bias"))?,
                        output: w8(&n("attention/output"))?,
                        output_bias: bf16(&n("attention/output_bias"))?,
                        fc1: w8(&n("mlp/fc1"))?,
                        fc1_bias: bf16(&n("mlp/fc1_bias"))?,
                        fc2: w8(&n("mlp/fc2"))?,
                        fc2_bias: bf16(&n("mlp/fc2_bias"))?,
                        norm1_weight: bf16(&n("norm1/weight"))?,
                        norm1_bias: bf16(&n("norm1/bias"))?,
                        norm2_weight: bf16(&n("norm2/weight"))?,
                        norm2_bias: bf16(&n("norm2/bias"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let weights = ffi::ignis_image_vision_weights {
                patch_embedding: w8("vision/patch_embedding")?,
                patch_embedding_bias: bf16("vision/patch_embedding_bias")?,
                position_embedding: bf16("vision/position_embedding")?,
                blocks: std::ptr::null(),
                block_count: blocks.len() as u32,
                merger: merger("vision/merger")?,
                deepstack: [merger("vision/deepstack/0")?, merger("vision/deepstack/1")?, merger("vision/deepstack/2")?],
                deepstack_layers: [config.deepstack_layers[0], config.deepstack_layers[1], config.deepstack_layers[2]],
            };
            Ok(Self { blocks, weights })
        }

        /// The C descriptor; it points into `self`, so it lives no longer than `self`.
        pub fn descriptor(&self) -> ffi::ignis_image_vision_weights {
            ffi::ignis_image_vision_weights { blocks: self.blocks.as_ptr(), ..self.weights }
        }
    }

    /// The tower on `call`: the merger's output into `embeds`, BF16 `[tokens][4096]`, and DeepStack
    /// merger i's into `deepstack[i]` (null: that merger does not run; `None`: none runs).
    ///
    /// # Safety
    /// `weights` points at live device weights; `embeds`, each non-null `deepstack[i]` (`tokens x
    /// 4096` BF16 each) and `workspace` (`workspace_bytes`, at least
    /// [`workspace_bytes`]`(tokens, segments)`) are device pointers.
    pub unsafe fn forward(
        weights: &ffi::ignis_image_vision_weights,
        call: &VisionCall,
        embeds: *mut c_void,
        deepstack: Option<&[*mut c_void; ffi::DEEPSTACK]>,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let input = ffi::ignis_image_vision_input {
            patches: call.patch_count(),
            segments: call.segments(),
            patch_rows: call.patches.as_ptr(),
            position_ids: call.position_ids.as_ptr(),
            cu_seqlens: call.cu_seqlens.as_ptr(),
            position_table_indices: call.position_table_indices.as_ptr(),
            position_table_weights: call.position_table_weights.as_ptr(),
        };
        let deepstack = deepstack.map_or(std::ptr::null(), |d| d.as_ptr());
        let rc = unsafe {
            ffi::ignis_image_vision_forward(weights, &input, embeds, deepstack, workspace, workspace_bytes, stream)
        };
        if rc == 0 { Ok(()) } else { Err(last_error()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::qwen_image::QwenImageGeometry;

    /// The checkpoint's `vision_config` (Qwen/Qwen-Image-2.1-Turbo@fc13ba19, its
    /// `text_encoder/config.json`), edited by `edit`.
    fn checkpoint_config(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut v = serde_json::json!({
            "model_type": "qwen3_vl",
            "vision_config": {
                "deepstack_visual_indexes": [8, 16, 24], "depth": 27, "hidden_act": "gelu_pytorch_tanh",
                "hidden_size": 1152, "in_channels": 3, "intermediate_size": 4304, "num_heads": 16,
                "num_position_embeddings": 2304, "out_hidden_size": 4096, "patch_size": 16,
                "spatial_merge_size": 2, "temporal_patch_size": 2
            }
        });
        edit(&mut v["vision_config"]);
        serde_json::to_vec(&v).unwrap()
    }

    fn geometry() -> VisionGeometry {
        QwenImageGeometry::qwen_image_21_turbo().vision
    }

    #[test]
    fn the_checkpoints_vision_config_gives_the_deepstack_taps() {
        let c = VisionTowerConfig::parse(&checkpoint_config(|_| {}), &geometry()).unwrap();
        assert_eq!(c, VisionTowerConfig { num_heads: 16, deepstack_layers: vec![8, 16, 24] });
    }

    #[test]
    fn a_vision_config_the_inventory_does_not_match_is_refused_by_name() {
        let refused = |edit: fn(&mut serde_json::Value)| VisionTowerConfig::parse(&checkpoint_config(edit), &geometry()).unwrap_err();
        assert!(refused(|v| v["depth"] = 24.into()).contains("vision_config.depth is 24"));
        assert!(refused(|v| v["out_hidden_size"] = 5120.into()).contains("out_hidden_size is 5120"));
        assert!(refused(|v| v["patch_size"] = 14.into()).contains("patch (in_channels x temporal_patch_size x patch_size^2) is 1176"));
        assert!(refused(|v| v["hidden_act"] = "gelu".into()).contains("hidden_act"));
        assert!(refused(|v| v["deepstack_visual_indexes"] = serde_json::json!([8, 16])).contains("not 3 ascending blocks"));
        assert!(refused(|v| v["deepstack_visual_indexes"] = serde_json::json!([16, 8, 24])).contains("not 3 ascending blocks"));
        assert!(refused(|v| v["deepstack_visual_indexes"] = serde_json::json!([8, 16, 27])).contains("below 27"));
        assert!(refused(|v| { v.as_object_mut().unwrap().remove("num_heads"); }).contains("num_heads is missing"));
        assert!(VisionTowerConfig::parse(b"{}", &geometry()).unwrap_err().contains("no vision_config"));
    }

    /// Two images of different grids: each keeps its own positions and table coordinates (its
    /// 27B control), the position planes run over both (every row, then every column), the
    /// segment bounds continue, and the pixels round to BF16 to nearest even.
    #[test]
    fn a_call_over_two_images_concatenates_their_controls() {
        let (a, b) = (Grid { t: 1, h: 4, w: 6 }, Grid { t: 1, h: 2, w: 2 });
        let pa: Vec<f32> = (0..24 * PATCH_FEATURES).map(|i| i as f32 * 0.001).collect();
        let pb: Vec<f32> = (0..4 * PATCH_FEATURES).map(|i| -(i as f32) * 0.01).collect();
        let call = vision_call(&[(a, &pa), (b, &pb)]).unwrap();
        let (ca, cb) = (vision_item_control(a), vision_item_control(b));
        assert_eq!((call.patch_count(), call.tokens(), call.segments()), (28, 7, 2));
        assert_eq!(call.cu_seqlens, [0, 24, 28]);
        let rows: Vec<i32> = ca.position_ids[..24].iter().chain(&cb.position_ids[..4]).copied().collect();
        let cols: Vec<i32> = ca.position_ids[24..].iter().chain(&cb.position_ids[4..]).copied().collect();
        assert_eq!(call.position_ids[..28], rows[..]);
        assert_eq!(call.position_ids[28..], cols[..]);
        assert_eq!(call.position_table_indices, [ca.position_table_indices.clone(), cb.position_table_indices.clone()].concat());
        assert_eq!(call.position_table_weights, [ca.position_table_weights.clone(), cb.position_table_weights.clone()].concat());
        // 1 + 2^-8 is a tie between two BF16 neighbours (1 and 1 + 2^-7): it rounds to the even one, 1;
        // 1 + 3 * 2^-8 rounds up to 1 + 2^-6.
        let ties = vec![1.0f32 + 2f32.powi(-8), 1.0 + 3.0 * 2f32.powi(-8)];
        let mut px = vec![0.0f32; 4 * PATCH_FEATURES];
        px[..2].copy_from_slice(&ties);
        let one = vision_call(&[(b, &px)]).unwrap();
        assert_eq!(one.patches[..2], [0x3F80, 0x3F82]);
        assert_eq!(call.patches[24 * PATCH_FEATURES + 1], (pb[1].to_bits() >> 16) as u16 + 1, "-0.01 rounds up in magnitude");
    }

    #[test]
    fn a_call_is_refused_without_images_past_ten_or_with_the_wrong_pixel_count() {
        let g = Grid { t: 1, h: 2, w: 2 };
        let px = vec![0.0f32; 4 * PATCH_FEATURES];
        assert!(vision_call(&[]).unwrap_err().contains("1 to 10 frames, not 0"));
        let eleven: Vec<(Grid, &[f32])> = (0..11).map(|_| (g, &px[..])).collect();
        assert!(vision_call(&eleven).unwrap_err().contains("not 11"));
        assert_eq!(vision_call(&eleven[..10]).unwrap().segments(), 10);
        assert!(vision_call(&[(g, &px[1..])]).unwrap_err().contains("image 0"));
        assert!(vision_call(&[(Grid { t: 1, h: 3, w: 2 }, &px[..])]).unwrap_err().contains("merge-aligned"));
    }
}
