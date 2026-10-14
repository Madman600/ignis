//! The image model's diffusion transformer, Qwen-Image-2.1's DiT (spec image-model/04, GitHub
//! #321): the numbers it reads from the artifact's transformer config, and, with the `cuda`
//! feature, the flat C ABI of `kernel/include/ignis_image_transformer.h`, 1:1, thin wrappers that
//! turn its return codes into `Result`s carrying the leaf's own message, and the weights'
//! descriptors over a materialized artifact.
//!
//! The program is the image leaf's ([`crate::image_leaf`]), which drives it one call at a time.
//! Every device pointer argument is one the caller vouches for, so the device wrappers are
//! `unsafe`; the shapes and alignments they take are checked by the leaf.
//!
//! The device calls are asynchronous on the stream they are given (null: the legacy default
//! stream) and carry no synchronization: a caller that moves data through `CudaDevice`'s own
//! stream orders the copies itself (`moe::stream_sync(null)` after a call).
//!
//! Our own implementation (ADR 0051, tier 1).

use ignis_artifact::qwen_image::DitGeometry;

/// The transformer config's numbers the program reads (`frontend/transformer_config.json`, the
/// checkpoint's `transformer/config.json` verbatim).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformerConfig {
    /// Every norm's epsilon.
    pub eps: f32,
}

impl TransformerConfig {
    /// Reads the config. Refuses one whose dimensions are not `geometry`'s (the artifact's
    /// inventory), or that the program does not run: a patch size other than 1, RoPE axes other
    /// than (16, 56, 56) (the leaf's table), or `causal_condition` off (the condition prefix is
    /// only reusable when it is modulated at t = 0).
    pub fn parse(bytes: &[u8], geometry: &DitGeometry) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("the transformer config is not JSON: {e}"))?;
        let uint = |key: &str| json.get(key).and_then(|v| v.as_u64()).ok_or(format!("the transformer config's {key} is missing"));
        for (key, want) in [
            ("num_layers", geometry.blocks as u64),
            ("num_attention_heads", geometry.heads),
            ("attention_head_dim", geometry.head_dim),
            ("in_channels", geometry.in_channels),
            ("out_channels", geometry.out_channels),
            ("context_in_dim", geometry.context),
            ("mlp_ratio", geometry.mlp / geometry.hidden()),
            ("patch_size", 1),
        ] {
            let got = uint(key)?;
            if got != want {
                return Err(format!("the transformer config's {key} is {got}, the leaf runs {want}"));
            }
        }
        let axes: Vec<u64> = json
            .get("axes_dims_rope")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        if axes != [16, 56, 56] {
            return Err(format!("the transformer config's axes_dims_rope is {axes:?}, the leaf's RoPE table is [16, 56, 56]"));
        }
        if json.get("causal_condition").and_then(|v| v.as_bool()) != Some(true) {
            return Err("the transformer config's causal_condition is not true: the condition prefix would not be reusable".into());
        }
        let eps = json.get("eps").and_then(|v| v.as_f64()).ok_or("the transformer config's eps is missing")? as f32;
        if !(eps >= 0.0 && eps.is_finite()) {
            return Err(format!("the transformer config's eps {eps} is not a finite non-negative number"));
        }
        Ok(Self { eps })
    }
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::ffi::CStr;
    use std::os::raw::c_void;

    use ignis_artifact::qwen_image::DitGeometry;
    use ignis_artifact::{MaterializedArtifact, ObjectHandle};

    use super::TransformerConfig;

    /// 1:1 with `kernel/include/ignis_image_transformer.h`.
    pub mod ffi {
        use std::os::raw::{c_char, c_void};

        pub const HIDDEN: u32 = 4096;
        pub const HEADS: u32 = 32;
        pub const HEAD_DIM: u32 = 128;
        pub const MLP: u32 = 12288;
        pub const LATENT: u32 = 64;
        pub const TIMESTEP: u32 = 256;
        pub const MODULATION: u32 = 4 * HIDDEN;

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_transformer_block {
            pub to_q: *const c_void,
            pub to_k: *const c_void,
            pub to_v: *const c_void,
            pub to_out: *const c_void,
            pub norm_q: *const c_void,
            pub norm_k: *const c_void,
            pub mlp_gate: *const c_void,
            pub mlp_up: *const c_void,
            pub mlp_down: *const c_void,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_transformer_weights {
            pub img_in: *const c_void,
            pub txt_norm: *const c_void,
            pub txt_in: *const c_void,
            pub txt_out: *const c_void,
            pub time_linear_1: *const c_void,
            pub time_linear_2: *const c_void,
            pub modulation: *const c_void,
            pub norm_out: *const c_void,
            pub proj_out: *const c_void,
            pub blocks: *const ignis_image_transformer_block,
            pub block_count: u32,
            pub eps: f32,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct ignis_image_transformer_segment {
            pub start: u32,
            pub end: u32,
            pub text: u32,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_transformer_prefix {
            pub tokens: u32,
            pub runs: *const *mut c_void,
            pub segments: *const ignis_image_transformer_segment,
            pub segment_count: u32,
        }

        unsafe extern "C" {
            pub fn ignis_image_transformer_workspace_bytes(tokens: u32) -> u64;
            pub fn ignis_image_transformer_modulation(
                weights: *const ignis_image_transformer_weights,
                sigma: f32,
                modulation: *mut f32,
                out_scale: *mut f32,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_txt_in(
                weights: *const ignis_image_transformer_weights,
                x: *const f32,
                rows: u32,
                y: *mut f32,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_img_in(
                weights: *const ignis_image_transformer_weights,
                latents: *const f32,
                tokens: u32,
                y: *mut f32,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_blocks(
                weights: *const ignis_image_transformer_weights,
                first_block: u32,
                end_block: u32,
                x: *mut f32,
                tokens: u32,
                target: u32,
                rope: *const f32,
                modulation: *const f32,
                prefix: *const ignis_image_transformer_prefix,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_prefix_kv_bytes(tokens: u32) -> u64;
            pub fn ignis_image_transformer_prefix_blocks(
                weights: *const ignis_image_transformer_weights,
                first_block: u32,
                end_block: u32,
                x: *mut f32,
                rope: *const f32,
                modulation: *const f32,
                prefix: *const ignis_image_transformer_prefix,
                kv: *mut c_void,
                chunk: u32,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_head(
                weights: *const ignis_image_transformer_weights,
                x: *const f32,
                tokens: u32,
                out_scale: *const f32,
                velocity: *mut f32,
                latents: *mut f32,
                sigma: f32,
                sigma_next: f32,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_transformer_last_error() -> *const c_char;
        }
    }

    /// The leaf's message for the most recent failed transformer call on this thread.
    pub fn last_error() -> String {
        unsafe { CStr::from_ptr(ffi::ignis_image_transformer_last_error()) }.to_string_lossy().into_owned()
    }

    fn check(rc: i32) -> Result<(), String> {
        if rc == 0 { Ok(()) } else { Err(last_error()) }
    }

    /// The workspace a call over `tokens` needs, in bytes (0 for 0 or more than 2^20 tokens).
    pub fn workspace_bytes(tokens: u32) -> u64 {
        unsafe { ffi::ignis_image_transformer_workspace_bytes(tokens) }
    }

    /// The chunked prefix pass's BF16 K and V of `tokens` prefix tokens, in bytes (0 for 0 tokens).
    pub fn prefix_kv_bytes(tokens: u32) -> u64 {
        unsafe { ffi::ignis_image_transformer_prefix_kv_bytes(tokens) }
    }

    /// The DiT's descriptors over a materialized artifact: every `transformer/` tensor where the
    /// materializer put it (the FP8 payloads as stored), with the config's eps.
    #[derive(Debug)]
    pub struct TransformerWeights {
        blocks: Vec<ffi::ignis_image_transformer_block>,
        model: ffi::ignis_image_transformer_weights,
        pub config: TransformerConfig,
    }

    impl TransformerWeights {
        /// `handles` names the bound objects (`qwen_image::bind`).
        pub fn new(
            artifact: &MaterializedArtifact,
            handles: &BTreeMap<String, ObjectHandle>,
            geometry: &DitGeometry,
            config: TransformerConfig,
        ) -> Result<Self, String> {
            let at = |suffix: &str| -> Result<*const c_void, String> {
                let name = format!("transformer/{suffix}.weight");
                let handle = handles.get(&name).ok_or(format!("{name} is not bound"))?;
                Ok(artifact.device_view(*handle).map_err(|e| format!("{name}: {e}"))?.base as *const c_void)
            };
            let blocks = (0..geometry.blocks)
                .map(|b| {
                    let n = |s: &str| format!("transformer_blocks.{b}.{s}");
                    Ok(ffi::ignis_image_transformer_block {
                        to_q: at(&n("attn.to_q"))?,
                        to_k: at(&n("attn.to_k"))?,
                        to_v: at(&n("attn.to_v"))?,
                        to_out: at(&n("attn.to_out.0"))?,
                        norm_q: at(&n("attn.norm_q"))?,
                        norm_k: at(&n("attn.norm_k"))?,
                        mlp_gate: at(&n("img_mlp.gate_layer"))?,
                        mlp_up: at(&n("img_mlp.proj"))?,
                        mlp_down: at(&n("img_mlp.out"))?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let model = ffi::ignis_image_transformer_weights {
                img_in: at("img_in")?,
                txt_norm: at("txt_in.text_norm")?,
                txt_in: at("txt_in.in_layer")?,
                txt_out: at("txt_in.out_layer")?,
                time_linear_1: at("time_text_embed.timestep_embedder.linear_1")?,
                time_linear_2: at("time_text_embed.timestep_embedder.linear_2")?,
                modulation: at("modulation.1")?,
                norm_out: at("norm_out.linear")?,
                proj_out: at("proj_out")?,
                blocks: std::ptr::null(),
                block_count: 0,
                eps: config.eps,
            };
            Ok(Self { blocks, model, config })
        }

        /// The C descriptor; it points into `self`, so it lives no longer than `self`.
        pub fn descriptor(&self) -> ffi::ignis_image_transformer_weights {
            ffi::ignis_image_transformer_weights { blocks: self.blocks.as_ptr(), block_count: self.blocks.len() as u32, ..self.model }
        }
    }

    /// The timestep embedding at `sigma`: the shared modulation row (fp32 `[16384]`) and
    /// norm_out's scale (fp32 `[4096]`).
    ///
    /// # Safety
    /// `weights` points at live device weights; every pointer is a device pointer of the size
    /// the shapes imply; `workspace` holds `workspace_bytes`.
    pub unsafe fn modulation(
        weights: &ffi::ignis_image_transformer_weights,
        sigma: f32,
        modulation: *mut f32,
        out_scale: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe { ffi::ignis_image_transformer_modulation(weights, sigma, modulation, out_scale, workspace, workspace_bytes, stream) })
    }

    /// txt_in on the encoder's rows: fp32 `x [rows][4096]` to fp32 `y [rows][4096]`.
    ///
    /// # Safety
    /// As [`modulation`].
    pub unsafe fn txt_in(
        weights: &ffi::ignis_image_transformer_weights,
        x: *const f32,
        rows: u32,
        y: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe { ffi::ignis_image_transformer_txt_in(weights, x, rows, y, workspace, workspace_bytes, stream) })
    }

    /// img_in: fp32 latents `[tokens][64]` to fp32 `y [tokens][4096]`.
    ///
    /// # Safety
    /// As [`modulation`].
    pub unsafe fn img_in(
        weights: &ffi::ignis_image_transformer_weights,
        latents: *const f32,
        tokens: u32,
        y: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe { ffi::ignis_image_transformer_img_in(weights, latents, tokens, y, workspace, workspace_bytes, stream) })
    }

    /// Blocks `[first, end)` in place on `x`, fp32 `[tokens][4096]`: the prefix pass (`target`
    /// false; `tokens` is the prefix's, whose runs it writes) or a target pass over the prefix's
    /// runs. `rope` is these tokens' table, `modulation` the pass's row.
    ///
    /// # Safety
    /// As [`modulation`]; `prefix` points at live runs of `prefix.tokens` keys.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn blocks(
        weights: &ffi::ignis_image_transformer_weights,
        first: u32,
        end: u32,
        x: *mut f32,
        tokens: u32,
        target: bool,
        rope: *const f32,
        modulation: *const f32,
        prefix: &ffi::ignis_image_transformer_prefix,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_image_transformer_blocks(
                weights,
                first,
                end,
                x,
                tokens,
                target as u32,
                rope,
                modulation,
                prefix,
                workspace,
                workspace_bytes,
                stream,
            )
        })
    }

    /// The prefix pass in chunks of at most `chunk` tokens (a multiple of 128): blocks `[first,
    /// end)` on the prefix's `x`, bit for bit what [`blocks`] computes, through a workspace for
    /// `chunk` tokens and `kv` ([`prefix_kv_bytes`] of the prefix's tokens).
    ///
    /// # Safety
    /// As [`blocks`]; `kv` is a 256-byte aligned device buffer of [`prefix_kv_bytes`]`(prefix.tokens)`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prefix_blocks(
        weights: &ffi::ignis_image_transformer_weights,
        first: u32,
        end: u32,
        x: *mut f32,
        rope: *const f32,
        modulation: *const f32,
        prefix: &ffi::ignis_image_transformer_prefix,
        kv: *mut c_void,
        chunk: u32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_image_transformer_prefix_blocks(weights, first, end, x, rope, modulation, prefix, kv, chunk, workspace, workspace_bytes, stream)
        })
    }

    /// The head: `velocity [tokens][64] = proj_out(LayerNorm(x) * (1 + out_scale))`, then, with
    /// `latents`, the Euler update from `sigma` to `sigma_next`.
    ///
    /// # Safety
    /// As [`modulation`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn head(
        weights: &ffi::ignis_image_transformer_weights,
        x: *const f32,
        tokens: u32,
        out_scale: *const f32,
        velocity: *mut f32,
        latents: Option<*mut f32>,
        sigma: f32,
        sigma_next: f32,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let latents = latents.unwrap_or(std::ptr::null_mut());
        check(unsafe {
            ffi::ignis_image_transformer_head(weights, x, tokens, out_scale, velocity, latents, sigma, sigma_next, workspace, workspace_bytes, stream)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::qwen_image::QwenImageGeometry;

    /// The checkpoint's `transformer/config.json` (`fc13ba19`), edited by `edit`.
    fn checkpoint_config(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut v = serde_json::json!({
            "_class_name": "QwenImage21Transformer2DModel", "_diffusers_version": "0.41.0.dev0",
            "attention_head_dim": 128, "axes_dims_rope": [16, 56, 56], "causal_condition": true,
            "context_in_dim": 4096, "eps": 1e-06, "in_channels": 64, "mlp_ratio": 3, "num_attention_heads": 32,
            "num_layers": 32, "out_channels": 64, "patch_size": 1
        });
        edit(&mut v);
        serde_json::to_vec(&v).unwrap()
    }

    #[test]
    fn the_checkpoints_transformer_config_gives_eps() {
        let g = QwenImageGeometry::qwen_image_21_turbo().dit;
        assert_eq!(TransformerConfig::parse(&checkpoint_config(|_| {}), &g).unwrap(), TransformerConfig { eps: 1e-6 });
    }

    #[test]
    fn a_config_the_program_does_not_run_is_refused_by_name() {
        let g = QwenImageGeometry::qwen_image_21_turbo().dit;
        let refused = |edit: fn(&mut serde_json::Value)| TransformerConfig::parse(&checkpoint_config(edit), &g).unwrap_err();
        assert!(refused(|v| v["num_layers"] = 60.into()).contains("num_layers is 60"));
        assert!(refused(|v| v["mlp_ratio"] = 4.into()).contains("mlp_ratio is 4"));
        assert!(refused(|v| v["patch_size"] = 2.into()).contains("patch_size is 2"));
        assert!(refused(|v| v["axes_dims_rope"] = serde_json::json!([16, 56, 57])).contains("axes_dims_rope"));
        assert!(refused(|v| v["causal_condition"] = false.into()).contains("causal_condition"));
        assert!(refused(|v| { v.as_object_mut().unwrap().remove("eps"); }).contains("eps is missing"));
        assert!(TransformerConfig::parse(b"[", &g).unwrap_err().contains("not JSON"));
    }
}
