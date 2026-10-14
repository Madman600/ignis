//! The image model's text encoder, Qwen3-VL-8B's text model (spec image-model/03, GitHub #320):
//! the numbers it reads from the artifact's text config, the host side of its input (the token
//! embedding's rows), and, with the `cuda` feature, the flat C ABI of
//! `kernel/include/ignis_image_encoder.h`, 1:1, thin wrappers that turn its return codes into
//! `Result`s carrying the leaf's own message, and the weights' descriptors over a materialized
//! artifact.
//!
//! The forward is the image leaf's (spec image-model/04), which calls it inside the kernel leaf;
//! these bindings exist for Rust-side tests and tools that drive it directly. Every device
//! pointer argument is one the caller vouches for, so the device wrappers are `unsafe`; the shapes
//! and alignments they take are checked by the leaf.
//!
//! The device calls are asynchronous on the stream they are given (null: the legacy default
//! stream) and carry no synchronization: a caller that moves data through `CudaDevice`'s own
//! stream orders the copies itself (`moe::stream_sync(null)` after a call, as
//! `tests/image_encoder_gpu.rs` does).
//!
//! Our own implementation (ADR 0051, tier 1).

use ignis_artifact::qwen_image::TextEncoderGeometry;

/// The text config's numbers the forward reads (`frontend/text_encoder_config.json`, the
/// checkpoint's `text_encoder/config.json` verbatim).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextEncoderConfig {
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    /// The interleaved mrope's (t, h, w) frequency slots.
    pub mrope_section: [u32; 3],
}

impl TextEncoderConfig {
    /// Reads the config's `text_config`. Refuses one whose dimensions are not `geometry`'s (the
    /// artifact's inventory) or whose rope is not the interleaved mrope the leaf runs. The rope
    /// is read from `rope_parameters` (transformers 5's form) or else from `rope_theta` and
    /// `rope_scaling` (the checkpoint's, written by transformers 5.3).
    pub fn parse(bytes: &[u8], geometry: &TextEncoderGeometry) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("the text encoder config is not JSON: {e}"))?;
        let text = json.get("text_config").ok_or("the text encoder config has no text_config")?;
        let uint = |key: &str| text.get(key).and_then(|v| v.as_u64()).ok_or(format!("text_config.{key} is missing"));
        for (key, want) in [
            ("num_hidden_layers", geometry.layers as u64),
            ("hidden_size", geometry.hidden),
            ("num_attention_heads", geometry.heads),
            ("num_key_value_heads", geometry.kv_heads),
            ("head_dim", geometry.head_dim),
            ("intermediate_size", geometry.intermediate),
            ("vocab_size", geometry.vocab),
        ] {
            let got = uint(key)?;
            if got != want {
                return Err(format!("text_config.{key} is {got}, the artifact's inventory has {want}"));
            }
        }
        if text.get("hidden_act").and_then(|v| v.as_str()) != Some("silu") {
            return Err("text_config.hidden_act is not silu".into());
        }
        // The artifact binds no projection biases (the bind plan skips objects it does not name).
        if text.get("attention_bias").and_then(|v| v.as_bool()) == Some(true) {
            return Err("text_config.attention_bias is true: the leaf runs no projection biases".into());
        }
        let rms_norm_eps =
            text.get("rms_norm_eps").and_then(|v| v.as_f64()).ok_or("text_config.rms_norm_eps is missing")? as f32;
        let (rope, theta) = match text.get("rope_parameters") {
            Some(p) => (p, p.get("rope_theta")),
            None => (text.get("rope_scaling").ok_or("text_config has neither rope_parameters nor rope_scaling")?, text.get("rope_theta")),
        };
        let rope_theta = theta.and_then(|v| v.as_f64()).ok_or("the text config has no rope_theta")? as f32;
        if rope.get("rope_type").and_then(|v| v.as_str()) != Some("default") {
            return Err("the text config's rope_type is not default".into());
        }
        if rope.get("mrope_interleaved").and_then(|v| v.as_bool()) != Some(true) {
            return Err("the text config's mrope is not interleaved".into());
        }
        let section: Vec<u32> = rope
            .get("mrope_section")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|s| s.as_u64().map(|s| s as u32)).collect())
            .unwrap_or_default();
        let mrope_section: [u32; 3] =
            section.as_slice().try_into().map_err(|_| "the text config's mrope_section is not three integers")?;
        if mrope_section.iter().sum::<u32>() as u64 * 2 != geometry.head_dim {
            return Err(format!("the mrope_section {mrope_section:?} does not cover head_dim {}", geometry.head_dim));
        }
        Ok(Self { rms_norm_eps, rope_theta, mrope_section })
    }
}

/// The token embedding's rows for `ids`, BF16 `table [vocab][hidden]` (little-endian, the
/// artifact's host-retained `text_encoder/embed_tokens.weight`) widened exactly to fp32: the
/// forward's input x, `[ids.len()][hidden]`.
pub fn embed_rows(table: &[u8], hidden: usize, ids: &[u32]) -> Result<Vec<f32>, String> {
    let row_bytes = hidden * 2;
    if hidden == 0 || table.len() % row_bytes != 0 {
        return Err(format!("the embedding table's {} bytes are not rows of {hidden} BF16", table.len()));
    }
    let vocab = table.len() / row_bytes;
    let mut x = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        let id = id as usize;
        if id >= vocab {
            return Err(format!("token id {id} is outside the vocabulary of {vocab}"));
        }
        let row = &table[id * row_bytes..(id + 1) * row_bytes];
        x.extend(row.chunks_exact(2).map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16)));
    }
    Ok(x)
}

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::collections::BTreeMap;
    use std::ffi::CStr;
    use std::os::raw::c_void;

    use ignis_artifact::qwen_image::TextEncoderGeometry;
    use ignis_artifact::{MaterializedArtifact, ObjectHandle};

    use super::TextEncoderConfig;

    /// 1:1 with `kernel/include/ignis_image_encoder.h`.
    pub mod ffi {
        use std::os::raw::{c_char, c_void};

        pub const HIDDEN: u32 = 4096;
        pub const HEADS: u32 = 32;
        pub const KV_HEADS: u32 = 8;
        pub const HEAD_DIM: u32 = 128;
        pub const INTERMEDIATE: u32 = 12288;
        pub const DEEPSTACK: usize = 3;

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_encoder_layer {
            pub input_norm: *const c_void,
            pub q_proj: *const c_void,
            pub k_proj: *const c_void,
            pub v_proj: *const c_void,
            pub q_norm: *const c_void,
            pub k_norm: *const c_void,
            pub o_proj: *const c_void,
            pub post_norm: *const c_void,
            pub gate_proj: *const c_void,
            pub up_proj: *const c_void,
            pub down_proj: *const c_void,
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_encoder_weights {
            pub layers: *const ignis_image_encoder_layer,
            pub layer_count: u32,
            pub rms_norm_eps: f32,
            pub rope_theta: f32,
            pub mrope_section: [u32; 3],
        }

        #[repr(C)]
        #[derive(Debug, Clone, Copy)]
        pub struct ignis_image_encoder_vision {
            pub rows: u32,
            pub row_index: *const i32,
            pub embeds: *const c_void,
            pub deepstack: [*const c_void; DEEPSTACK],
        }

        unsafe extern "C" {
            pub fn ignis_image_encoder_workspace_bytes(tokens: u32) -> u64;
            pub fn ignis_image_encoder_forward(
                weights: *const ignis_image_encoder_weights,
                first_layer: u32,
                end_layer: u32,
                x: *mut f32,
                tokens: u32,
                positions: *const i32,
                vision: *const ignis_image_encoder_vision,
                workspace: *mut c_void,
                workspace_bytes: u64,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_encoder_residual_norm(
                x: *mut f32,
                delta: *const f32,
                tokens: u32,
                cols: u32,
                weight: *const c_void,
                eps: f32,
                y: *mut c_void,
                y_scales: *mut f32,
                y_f32: u32,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_encoder_qk_norm_mrope(
                q: *const f32,
                q_stride: u32,
                k: *const f32,
                k_stride: u32,
                tokens: u32,
                q_weight: *const c_void,
                k_weight: *const c_void,
                eps: f32,
                positions: *const i32,
                rope_theta: f32,
                mrope_section: *const u32,
                q_out: *mut c_void,
                k_out: *mut c_void,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_encoder_rows(
                x: *mut f32,
                tokens: u32,
                cols: u32,
                row_index: *const i32,
                rows: u32,
                src: *const c_void,
                add: u32,
                stream: *mut c_void,
            ) -> i32;
            pub fn ignis_image_encoder_last_error() -> *const c_char;
        }
    }

    /// The leaf's message for the most recent failed encoder call on this thread.
    pub fn last_error() -> String {
        unsafe { CStr::from_ptr(ffi::ignis_image_encoder_last_error()) }.to_string_lossy().into_owned()
    }

    fn check(rc: i32) -> Result<(), String> {
        if rc == 0 { Ok(()) } else { Err(last_error()) }
    }

    /// The forward's workspace for `tokens`, in bytes.
    pub fn workspace_bytes(tokens: u32) -> u64 {
        unsafe { ffi::ignis_image_encoder_workspace_bytes(tokens) }
    }

    /// The decoder layers' descriptors over a materialized artifact: every `text_encoder/` tensor
    /// where the materializer put it (the FP8 payloads as stored), with the config's eps and rope.
    #[derive(Debug)]
    pub struct TextEncoderWeights {
        layers: Vec<ffi::ignis_image_encoder_layer>,
        pub config: TextEncoderConfig,
    }

    impl TextEncoderWeights {
        /// `handles` names the bound objects (`qwen_image::bind` or `bind_text_encoder`).
        pub fn new(
            artifact: &MaterializedArtifact,
            handles: &BTreeMap<String, ObjectHandle>,
            geometry: &TextEncoderGeometry,
            config: TextEncoderConfig,
        ) -> Result<Self, String> {
            let at = |l: usize, suffix: &str| -> Result<*const c_void, String> {
                let name = format!("text_encoder/layers.{l}.{suffix}.weight");
                let handle = handles.get(&name).ok_or(format!("{name} is not bound"))?;
                Ok(artifact.device_view(*handle).map_err(|e| format!("{name}: {e}"))?.base as *const c_void)
            };
            let layers = (0..geometry.layers)
                .map(|l| {
                    Ok(ffi::ignis_image_encoder_layer {
                        input_norm: at(l, "input_layernorm")?,
                        q_proj: at(l, "self_attn.q_proj")?,
                        k_proj: at(l, "self_attn.k_proj")?,
                        v_proj: at(l, "self_attn.v_proj")?,
                        q_norm: at(l, "self_attn.q_norm")?,
                        k_norm: at(l, "self_attn.k_norm")?,
                        o_proj: at(l, "self_attn.o_proj")?,
                        post_norm: at(l, "post_attention_layernorm")?,
                        gate_proj: at(l, "mlp.gate_proj")?,
                        up_proj: at(l, "mlp.up_proj")?,
                        down_proj: at(l, "mlp.down_proj")?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(Self { layers, config })
        }

        /// The C descriptor; it points into `self`, so it lives no longer than `self`.
        pub fn descriptor(&self) -> ffi::ignis_image_encoder_weights {
            ffi::ignis_image_encoder_weights {
                layers: self.layers.as_ptr(),
                layer_count: self.layers.len() as u32,
                rms_norm_eps: self.config.rms_norm_eps,
                rope_theta: self.config.rope_theta,
                mrope_section: self.config.mrope_section,
            }
        }
    }

    /// Decoder layers `[first_layer, end_layer)` in place on `x`, fp32 `[tokens][4096]`, with
    /// mrope `positions` `[3][tokens]` (rows t, h, w) and the vision hooks, if any.
    ///
    /// # Safety
    /// `weights` points at live device weights; every pointer is a device pointer of the size the
    /// shapes imply; `workspace` holds `workspace_bytes` (at least [`workspace_bytes`]`(tokens)`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(
        weights: &ffi::ignis_image_encoder_weights,
        first_layer: u32,
        end_layer: u32,
        x: *mut f32,
        tokens: u32,
        positions: *const i32,
        vision: Option<&ffi::ignis_image_encoder_vision>,
        workspace: *mut c_void,
        workspace_bytes: u64,
        stream: *mut c_void,
    ) -> Result<(), String> {
        let vision = vision.map_or(std::ptr::null(), |v| v as *const _);
        check(unsafe {
            ffi::ignis_image_encoder_forward(
                weights,
                first_layer,
                end_layer,
                x,
                tokens,
                positions,
                vision,
                workspace,
                workspace_bytes,
                stream,
            )
        })
    }

    /// `x += delta` (when `delta` is not null), then, when `weight` is not null, the RMSNorm's
    /// output: E4M3 codes in `y` and their scales in `y_scales`, or fp32 in `y` when `y_f32`.
    ///
    /// # Safety
    /// Every pointer is a device pointer of the size the shapes imply.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn residual_norm(
        x: *mut f32,
        delta: *const f32,
        tokens: u32,
        cols: u32,
        weight: *const c_void,
        eps: f32,
        y: *mut c_void,
        y_scales: *mut f32,
        y_f32: bool,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_image_encoder_residual_norm(x, delta, tokens, cols, weight, eps, y, y_scales, y_f32 as u32, stream)
        })
    }

    /// The QK RMSNorm with the interleaved mrope, fp32 q (32 heads) and k (8 heads) in, BF16
    /// `q_out [tokens][4096]` and `k_out [tokens][1024]` out.
    ///
    /// # Safety
    /// Every pointer is a device pointer of the size the shapes imply.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn qk_norm_mrope(
        q: *const f32,
        q_stride: u32,
        k: *const f32,
        k_stride: u32,
        tokens: u32,
        q_weight: *const c_void,
        k_weight: *const c_void,
        eps: f32,
        positions: *const i32,
        rope_theta: f32,
        mrope_section: [u32; 3],
        q_out: *mut c_void,
        k_out: *mut c_void,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe {
            ffi::ignis_image_encoder_qk_norm_mrope(
                q,
                q_stride,
                k,
                k_stride,
                tokens,
                q_weight,
                k_weight,
                eps,
                positions,
                rope_theta,
                mrope_section.as_ptr(),
                q_out,
                k_out,
                stream,
            )
        })
    }

    /// `x[row_index[j]] = src[j]`, or `+=` when `add`, for `j < rows`; BF16 `src [rows][cols]`.
    ///
    /// # Safety
    /// Every pointer is a device pointer of the size the shapes imply.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rows(
        x: *mut f32,
        tokens: u32,
        cols: u32,
        row_index: *const i32,
        rows: u32,
        src: *const c_void,
        add: bool,
        stream: *mut c_void,
    ) -> Result<(), String> {
        check(unsafe { ffi::ignis_image_encoder_rows(x, tokens, cols, row_index, rows, src, add as u32, stream) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_artifact::qwen_image::QwenImageGeometry;

    /// The checkpoint's text config as its `text_encoder/config.json` writes it (transformers
    /// 5.3: `rope_theta` and `rope_scaling`), with `text_config` edited by `edit`.
    fn checkpoint_config(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut v = serde_json::json!({
            "model_type": "qwen3_vl",
            "text_config": {
                "head_dim": 128, "hidden_act": "silu", "hidden_size": 4096, "intermediate_size": 12288,
                "num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8,
                "rms_norm_eps": 1e-06, "vocab_size": 151936, "rope_theta": 5000000,
                "rope_scaling": {"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"}
            }
        });
        edit(&mut v["text_config"]);
        serde_json::to_vec(&v).unwrap()
    }

    fn geometry() -> TextEncoderGeometry {
        QwenImageGeometry::qwen_image_21_turbo().text
    }

    #[test]
    fn the_checkpoints_text_config_gives_eps_theta_and_section() {
        let c = TextEncoderConfig::parse(&checkpoint_config(|_| {}), &geometry()).unwrap();
        assert_eq!(c, TextEncoderConfig { rms_norm_eps: 1e-6, rope_theta: 5e6, mrope_section: [24, 20, 20] });
    }

    #[test]
    fn transformers_5s_rope_parameters_form_is_read_too() {
        let bytes = checkpoint_config(|t| {
            let o = t.as_object_mut().unwrap();
            o.remove("rope_theta");
            o.remove("rope_scaling");
            o.insert(
                "rope_parameters".into(),
                serde_json::json!({"mrope_interleaved": true, "mrope_section": [24, 20, 20], "rope_theta": 1e6, "rope_type": "default"}),
            );
        });
        assert_eq!(TextEncoderConfig::parse(&bytes, &geometry()).unwrap().rope_theta, 1e6);
    }

    #[test]
    fn a_config_the_leaf_does_not_run_is_refused_by_name() {
        let refused = |edit: fn(&mut serde_json::Value)| TextEncoderConfig::parse(&checkpoint_config(edit), &geometry()).unwrap_err();
        assert!(refused(|t| t["hidden_size"] = 2048.into()).contains("text_config.hidden_size is 2048"));
        assert!(refused(|t| t["num_hidden_layers"] = 28.into()).contains("num_hidden_layers is 28"));
        assert!(refused(|t| t["hidden_act"] = "gelu".into()).contains("hidden_act"));
        assert!(refused(|t| t["attention_bias"] = true.into()).contains("attention_bias"));
        assert!(refused(|t| t["rope_scaling"]["mrope_interleaved"] = false.into()).contains("not interleaved"));
        assert!(refused(|t| t["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20])).contains("three integers"));
        assert!(refused(|t| t["rope_scaling"]["mrope_section"] = serde_json::json!([24, 20, 24])).contains("does not cover"));
        assert!(refused(|t| t["rope_scaling"]["rope_type"] = "yarn".into()).contains("rope_type"));
        assert!(refused(|t| { t.as_object_mut().unwrap().remove("rope_theta"); }).contains("rope_theta"));
        assert!(TextEncoderConfig::parse(b"{", &geometry()).unwrap_err().contains("not JSON"));
    }

    #[test]
    fn embedding_rows_widen_bf16_exactly() {
        // vocab 3 x hidden 2: row r holds (r + 0.5, -(r + 1)) in BF16.
        let table: Vec<u8> = (0..3u32)
            .flat_map(|r| [r as f32 + 0.5, -(r as f32 + 1.0)])
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(embed_rows(&table, 2, &[2, 0, 2]).unwrap(), vec![2.5, -3.0, 0.5, -1.0, 2.5, -3.0]);
        assert!(embed_rows(&table, 2, &[3]).unwrap_err().contains("token id 3 is outside the vocabulary of 3"));
        assert!(embed_rows(&table[..5], 2, &[0]).unwrap_err().contains("not rows"));
    }
}
