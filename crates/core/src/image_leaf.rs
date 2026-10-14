//! The image leaf (spec image-model/04, GitHub #321): the family `Qwen-Image-2.1` driven from Rust
//! one call at a time (ADR 0009). This module holds what the leaf decides on the host:
//!
//! - [`Sampler`]: the denoising schedule, from the artifact's `frontend/image_model.json`: the
//!   `sample_sigmas` followed by a terminal 0, the DiT's timestep σ, and the Euler update
//!   `x += (σ_next − σ) · v` ([`euler_step`], the host's form of the device update). A scheduler
//!   config that would change the schedule is refused.
//! - [`ImagePlan`]: the load's device-memory plan: the weights, one sequential arena, and the
//!   **condition budget** (whatever the budget leaves, counted in condition-prefix tokens). A load
//!   whose weights and arena do not fit is refused, naming the shortfall.
//! - [`LatentNorm`]: the latents' per-channel normalization the DiT reads a condition image in.
//! - [`PrefixLayout`]: where the encoder's text rows and each condition image's latents go in the
//!   condition prefix, and its attention segments.
//!
//! With the `cuda` feature, [`ImageLeaf`] loads the family and runs its calls:
//! `encode_condition` (the condition images' VAE encodes, the vision tower and the text encoder,
//! `txt_in` and `img_in`, and the DiT over the prefix tokens at t = 0, storing the condition
//! prefix's K/V in 8 bits), `denoise_step` (the DiT over the target tokens, attending to the stored
//! prefix, then the Euler update on the device), `vae_decode` (the image's latents to RGBA bytes,
//! in the arena, in bands) and `vae_encode`. The DiT's program is the kernel leaf's
//! (`kernel/include/ignis_image_transformer.h`, [`crate::image_transformer`]); the VAE is
//! [`crate::image_autoencoder`]'s.

use crate::vram::VramMode;

/// `1 GiB`, for the plan line and the refusals.
const GIB: f64 = (1u64 << 30) as f64;

/// The denoising schedule (spec image-model/04, the sampler): diffusers'
/// `FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=sample_sigmas)` for a scheduler config that
/// neither shifts nor reshapes the sigmas.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampler {
    /// The sigmas, each the f32 of the config's number (numpy's `astype(float32)` of the Python
    /// float), then the terminal 0: one more than the steps.
    pub sigmas: Vec<f32>,
    /// `num_train_timesteps`: a timestep is σ times it, and the DiT reads timestep / it, σ.
    pub train_timesteps: u32,
}

impl Sampler {
    /// Reads `image_model.json`'s `sample_sigmas` and `scheduler`. Refuses a scheduler that is not
    /// `FlowMatchEulerDiscreteScheduler`, or whose config would move the sigmas away from the
    /// `sample_sigmas` (dynamic shifting, a shift other than 1, a terminal stretch, inverted,
    /// Karras, exponential or beta sigmas) or the update away from Euler (stochastic sampling).
    pub fn parse(image_model_json: &[u8]) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_slice(image_model_json).map_err(|e| format!("image_model.json is not JSON: {e}"))?;
        let scheduler = json.get("scheduler").ok_or("image_model.json has no scheduler")?;
        let class = scheduler.get("_class_name").and_then(|v| v.as_str());
        if class != Some("FlowMatchEulerDiscreteScheduler") {
            return Err(format!("the scheduler is {class:?}, the leaf runs FlowMatchEulerDiscreteScheduler"));
        }
        let flag = |key: &str| scheduler.get(key).and_then(|v| v.as_bool()).unwrap_or(false);
        for key in [
            "use_dynamic_shifting",
            "invert_sigmas",
            "use_karras_sigmas",
            "use_exponential_sigmas",
            "use_beta_sigmas",
            "stochastic_sampling",
        ] {
            if flag(key) {
                return Err(format!("scheduler.{key} is true: the leaf samples the sample_sigmas as they are, with Euler"));
            }
        }
        let shift = scheduler.get("shift").and_then(|v| v.as_f64()).ok_or("scheduler.shift is missing")?;
        if shift != 1.0 {
            return Err(format!("scheduler.shift is {shift}: the leaf samples the sample_sigmas unshifted (1.0)"));
        }
        if scheduler.get("shift_terminal").is_some_and(|v| !v.is_null()) {
            return Err("scheduler.shift_terminal is set: the leaf samples the sample_sigmas unstretched".into());
        }
        let train_timesteps = scheduler
            .get("num_train_timesteps")
            .and_then(|v| v.as_u64())
            .filter(|&t| t > 0 && t <= u32::MAX as u64)
            .ok_or("scheduler.num_train_timesteps is missing")? as u32;
        let sigmas: Vec<f32> = json
            .get("sample_sigmas")
            .and_then(|v| v.as_array())
            .ok_or("image_model.json has no sample_sigmas")?
            .iter()
            .map(|s| s.as_f64().map(|s| s as f32).ok_or("a sample_sigmas entry is not a number"))
            .collect::<Result<_, _>>()?;
        if sigmas.is_empty() {
            return Err("image_model.json's sample_sigmas is empty".into());
        }
        if !sigmas.windows(2).all(|w| w[0] > w[1]) || !sigmas.iter().all(|&s| s > 0.0 && s <= 1.0) {
            return Err(format!("the sample_sigmas {sigmas:?} are not decreasing within (0, 1]"));
        }
        Ok(Self { sigmas: sigmas.into_iter().chain(std::iter::once(0.0)).collect(), train_timesteps })
    }

    /// The denoising steps.
    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// Step `i`'s (σ, σ_next).
    pub fn sigma(&self, step: usize) -> (f32, f32) {
        (self.sigmas[step], self.sigmas[step + 1])
    }

    /// The scheduler's timesteps, σ · `num_train_timesteps` in f32 (one per step).
    pub fn timesteps(&self) -> Vec<f32> {
        self.sigmas[..self.steps()].iter().map(|&s| s * self.train_timesteps as f32).collect()
    }
}

/// The Euler update `latents += (σ_next − σ) · velocity`, as diffusers' scheduler computes it in
/// fp32: dt = σ_next − σ rounded, then dt · v rounded, then the sum rounded (two roundings, no
/// fused multiply-add). The device update is the same sequence.
pub fn euler_step(latents: &mut [f32], velocity: &[f32], sigma: f32, sigma_next: f32) {
    let dt = sigma_next - sigma;
    for (x, v) in latents.iter_mut().zip(velocity) {
        let step = dt * v;
        *x += step;
    }
}

/// The pipeline's latent normalization (`image_model.json`'s `latents_mean` and `latents_std`, the
/// VAE config's): the DiT reads a condition image's latents as `(z − mean) / std`, per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct LatentNorm {
    /// Per channel, each the f32 of the config's number (`torch.tensor` of the Python list).
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl LatentNorm {
    /// Reads the two lists, `channels` finite numbers each, every std positive.
    pub fn parse(image_model_json: &[u8], channels: usize) -> Result<Self, String> {
        let json: serde_json::Value =
            serde_json::from_slice(image_model_json).map_err(|e| format!("image_model.json is not JSON: {e}"))?;
        let list = |key: &str| -> Result<Vec<f32>, String> {
            json.get(key)
                .and_then(|v| v.as_array())
                .ok_or(format!("image_model.json has no {key}"))?
                .iter()
                .map(|v| v.as_f64().map(|x| x as f32).filter(|x| x.is_finite()).ok_or(format!("a {key} entry is not a finite number")))
                .collect()
        };
        let (mean, std) = (list("latents_mean")?, list("latents_std")?);
        if mean.len() != channels || std.len() != channels {
            return Err(format!("image_model.json has {} latents_mean and {} latents_std, the VAE has {channels} channels", mean.len(), std.len()));
        }
        if !std.iter().all(|&s| s > 0.0) {
            return Err("a latents_std entry is not positive".into());
        }
        Ok(Self { mean, std })
    }

    /// Packed latents `[tokens][channels]`, in place: `(z − mean) / std`, the difference rounded,
    /// then the quotient, as the pipeline's fp32 path computes them.
    pub fn normalize(&self, latents: &mut [f32]) {
        let channels = self.mean.len();
        for (i, z) in latents.iter_mut().enumerate() {
            let c = i % channels;
            *z = (*z - self.mean[c]) / self.std[c];
        }
    }

    /// Packed latents `[tokens][channels]`, in place: `z · std + mean`, the decode's input, the
    /// product rounded, then the sum, as the pipeline's fp32 path computes them.
    pub fn denormalize(&self, latents: &mut [f32]) {
        let channels = self.mean.len();
        for (i, z) in latents.iter_mut().enumerate() {
            let c = i % channels;
            *z = *z * self.std[c] + self.mean[c];
        }
    }
}

/// Where each part of a condition prefix goes (spec image-model/04, the joint sequence; diffusers'
/// `QwenImage21Transformer2DModel.forward`): the encoder's rows from `drop_idx` on, in order, a text
/// row as one prefix token (through `txt_in`), each condition image's image-pad rows as its block of
/// latent tokens (through `img_in`, 4 a pad: a pad stands for a 2 x 2 group). The image blocks follow
/// the images' token counts, so two images with no text between them stay two blocks
/// (`build_token_metadata`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixLayout {
    pub tokens: u32,
    /// Each run of text rows: (its first encoder row, its rows, its first prefix token).
    pub text_runs: Vec<(u32, u32, u32)>,
    /// Each condition image's block: (its first prefix token, (height, width) in latent tokens).
    pub blocks: Vec<(u32, (u32, u32))>,
    /// The attention's segments, `_qwenimage21_prefix_segments`: (start, end, text), a text run
    /// causal, an image block bidirectional.
    pub segments: Vec<(u32, u32, bool)>,
}

impl PrefixLayout {
    /// The layout of an encoder sequence of `rows` rows whose image-pad rows are `pad_rows`
    /// (ascending) for condition images of latent shapes `shapes` (in order). Refuses pads that do
    /// not add up to the images' (height · width / 4 each), an image whose pads are not consecutive,
    /// or a pad before `drop_idx`.
    pub fn new(rows: usize, drop_idx: usize, pad_rows: &[u32], shapes: &[(u32, u32)]) -> Result<Self, String> {
        if drop_idx >= rows {
            return Err(format!("drop_idx {drop_idx} leaves none of the {rows} tokens"));
        }
        let mut pads = Vec::with_capacity(shapes.len());
        for (k, &(h, w)) in shapes.iter().enumerate() {
            let tokens = h as u64 * w as u64;
            if tokens == 0 || tokens % 4 != 0 || tokens > 1 << 24 {
                return Err(format!("condition image {k}'s {h} x {w} latent tokens are not 1 .. 2^24 in groups of 4"));
            }
            pads.push((tokens / 4) as usize);
        }
        if pads.iter().sum::<usize>() != pad_rows.len() {
            return Err(format!("{} image-pad rows for condition images of {} pads", pad_rows.len(), pads.iter().sum::<usize>()));
        }
        let mut layout = Self { tokens: 0, text_runs: Vec::new(), blocks: Vec::new(), segments: Vec::new() };
        let mut token = 0u64;
        let text = |layout: &mut Self, token: &mut u64, row: usize, len: usize| {
            if len > 0 {
                layout.text_runs.push((row as u32, len as u32, *token as u32));
                layout.segments.push((*token as u32, (*token + len as u64) as u32, true));
                *token += len as u64;
            }
        };
        let (mut row, mut at) = (drop_idx, 0);
        for (k, (&n, &(h, w))) in pads.iter().zip(shapes).enumerate() {
            let first = pad_rows[at] as usize;
            if first < row || first + n > rows {
                return Err(format!("condition image {k}'s pads start at row {first}, not within rows {row} .. {rows}"));
            }
            if !pad_rows[at..at + n].iter().enumerate().all(|(i, &r)| r as usize == first + i) {
                return Err(format!("condition image {k}'s {n} pad rows from row {first} are not consecutive"));
            }
            text(&mut layout, &mut token, row, first - row);
            layout.blocks.push((token as u32, (h, w)));
            layout.segments.push((token as u32, (token + h as u64 * w as u64) as u32, false));
            token += h as u64 * w as u64;
            (row, at) = (first + n, at + n);
        }
        text(&mut layout, &mut token, row, rows - row);
        if token > 1 << 30 {
            return Err(format!("a condition prefix of {token} tokens"));
        }
        layout.tokens = token as u32;
        Ok(layout)
    }
}

/// What the image leaf's load knows before it allocates anything but its weights.
pub struct ImagePlanRequest<'a> {
    pub mode: VramMode,
    /// Device memory free before the first reservation.
    pub free_at_start_bytes: u64,
    /// The artifact's device arena (every bound tensor).
    pub weights_bytes: u64,
    /// The one sequential arena the calls share.
    pub arena_bytes: u64,
    /// The device bytes of a condition prefix of `n` tokens: every block's K/V run, as the leaf
    /// lays it out.
    pub prefix_bytes: &'a dyn Fn(u64) -> u64,
    /// `image.condition_budget_tokens` (spec image-model/05), when named: it can only lower the
    /// budget.
    pub condition_budget_cap: Option<u64>,
}

/// The plan the load carries out (spec image-model/04, load and plan).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImagePlan {
    pub mode: VramMode,
    pub free_at_start_bytes: u64,
    pub budget_bytes: u64,
    pub weights_bytes: u64,
    pub arena_bytes: u64,
    /// The largest condition prefix, in tokens, `encode_condition` accepts.
    pub condition_budget_tokens: u64,
    /// Its device bytes.
    pub condition_budget_bytes: u64,
}

/// A load the plan refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImagePlanError {
    /// An explicit budget above the memory free at start, without oversubscription allowed.
    BudgetAboveFree { budget_bytes: u64, free_bytes: u64 },
    /// The weights and the arena do not fit the budget.
    DoesNotFit { mode: VramMode, budget_bytes: u64, weights_bytes: u64, arena_bytes: u64 },
}

impl std::fmt::Display for ImagePlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::BudgetAboveFree { budget_bytes, free_bytes } => write!(
                f,
                "the image model's VRAM budget of {budget_bytes} bytes ({:.2} GiB) is more than the {free_bytes} bytes \
                 ({:.2} GiB) free at start; lower it, or allow oversubscription",
                budget_bytes as f64 / GIB,
                free_bytes as f64 / GIB
            ),
            Self::DoesNotFit { mode, budget_bytes, weights_bytes, arena_bytes } => {
                let needed = weights_bytes.saturating_add(arena_bytes);
                let short = needed.saturating_sub(budget_bytes);
                write!(
                    f,
                    "the image model does not fit: its weights ({weights_bytes} bytes, {:.2} GiB) and arena ({arena_bytes} \
                     bytes, {:.2} GiB) need {needed} bytes ({:.2} GiB), the {budget_bytes}-byte VRAM budget ({:.2} GiB) is \
                     short by {short} bytes ({:.2} GiB); ",
                    weights_bytes as f64 / GIB,
                    arena_bytes as f64 / GIB,
                    needed as f64 / GIB,
                    budget_bytes as f64 / GIB,
                    short as f64 / GIB
                )?;
                match mode {
                    VramMode::Derived { .. } => f.write_str("free device memory, or lower the VRAM headroom"),
                    VramMode::Explicit { .. } => f.write_str("raise the VRAM budget (or allow oversubscription)"),
                }
            }
        }
    }
}

impl std::error::Error for ImagePlanError {}

impl ImagePlan {
    /// Lays the plan out: the budget from `mode`, the weights and the arena first, then the
    /// condition budget, the largest prefix whose bytes fit in what they leave (and within the
    /// cap). Refuses a load whose weights and arena do not fit.
    pub fn new(request: &ImagePlanRequest<'_>) -> Result<Self, ImagePlanError> {
        let free = request.free_at_start_bytes;
        let budget_bytes = match request.mode {
            VramMode::Derived { headroom_bytes } => free.saturating_sub(headroom_bytes),
            VramMode::Explicit { budget_bytes, allow_oversubscription } => {
                if budget_bytes > free && !allow_oversubscription {
                    return Err(ImagePlanError::BudgetAboveFree { budget_bytes, free_bytes: free });
                }
                budget_bytes
            }
        };
        let fixed = request.weights_bytes.saturating_add(request.arena_bytes);
        if fixed > budget_bytes {
            return Err(ImagePlanError::DoesNotFit {
                mode: request.mode,
                budget_bytes,
                weights_bytes: request.weights_bytes,
                arena_bytes: request.arena_bytes,
            });
        }
        let left = budget_bytes - fixed;
        // The largest n with prefix_bytes(n) <= left: prefix_bytes grows with n.
        let fits = |n: u64| (request.prefix_bytes)(n) <= left;
        let (mut lo, mut hi) = (0u64, 1u64);
        while fits(hi) && hi < (1 << 40) {
            lo = hi;
            hi *= 2;
        }
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if fits(mid) { lo = mid } else { hi = mid }
        }
        let tokens = request.condition_budget_cap.map_or(lo, |cap| cap.min(lo));
        Ok(Self {
            mode: request.mode,
            free_at_start_bytes: free,
            budget_bytes,
            weights_bytes: request.weights_bytes,
            arena_bytes: request.arena_bytes,
            condition_budget_tokens: tokens,
            condition_budget_bytes: if tokens == 0 { 0 } else { (request.prefix_bytes)(tokens) },
        })
    }

    /// The plan line the load prints: weights, arena and condition budget.
    pub fn line(&self) -> String {
        format!(
            "image plan: weights {:.2} GiB, arena {:.2} GiB, condition budget {} tokens ({:.2} GiB); budget {:.2} GiB of \
             {:.2} GiB free ({})",
            self.weights_bytes as f64 / GIB,
            self.arena_bytes as f64 / GIB,
            self.condition_budget_tokens,
            self.condition_budget_bytes as f64 / GIB,
            self.budget_bytes as f64 / GIB,
            self.free_at_start_bytes as f64 / GIB,
            self.mode.as_str()
        )
    }
}

/// The largest target the arena is sized for, in latent tokens: 2400 x 1792 pixels (150 x 112),
/// the largest preset by cells (spec image-model/04's 2752 x 1536 is 16,512).
pub const MAX_TARGET_TOKENS: u32 = 16_800;

#[cfg(feature = "cuda")]
pub use device::*;

#[cfg(feature = "cuda")]
mod device {
    use std::os::raw::c_void;
    use std::time::Instant;

    use ignis_artifact::qwen_image::{self, QwenImageGeometry};
    use ignis_artifact::{materialize, CudaDevice, Device, DeviceBuffer, MaterializedArtifact, ObjectHandle, Reader};

    use super::{ImagePlan, ImagePlanError, ImagePlanRequest, LatentNorm, PrefixLayout, Sampler, MAX_TARGET_TOKENS};
    use crate::image_autoencoder::{self as autoencoder, VaeConfig, VaeDecoderWeights, VaeEncoderWeights};
    use crate::image_encoder::{self, embed_rows, TextEncoderConfig, TextEncoderWeights};
    use crate::image_transformer::{self as transformer, ffi as tffi, TransformerConfig, TransformerWeights};
    use crate::image_vision::{self, vision_call, VisionCall, VisionTowerConfig, VisionWeights};
    use crate::vision::Grid;
    use crate::vram::VramMode;
    use crate::{image_attn, image_dit, image_gemm, moe};

    const HIDDEN: usize = 4096;
    /// One fp32 residual row.
    const ROW: u64 = HIDDEN as u64 * 4;
    /// One token's RoPE table row: 64 (cos, sin) fp32 pairs.
    const ROPE_ROW: u64 = 64 * 2 * 4;
    const LATENT: usize = 64;

    fn align(n: u64) -> u64 {
        (n + 255) & !255
    }

    /// Why a call did not run.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ImageLeafError {
        /// The request is one the leaf refuses (a RoPE position outside the table, a prefix above
        /// the condition budget, a size above the arena, a token outside the vocabulary): the
        /// server answers it as a bad request, never a server error.
        Refused(String),
        /// The device or the leaf failed.
        Failed(String),
        /// The load does not fit the card.
        Plan(ImagePlanError),
    }

    impl std::fmt::Display for ImageLeafError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::Refused(why) => write!(f, "refused: {why}"),
                Self::Failed(why) => write!(f, "{why}"),
                Self::Plan(e) => write!(f, "{e}"),
            }
        }
    }

    impl std::error::Error for ImageLeafError {}

    fn failed(what: &str) -> impl Fn(String) -> ImageLeafError + '_ {
        move |e| ImageLeafError::Failed(format!("{what}: {e}"))
    }

    /// The load's knobs.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ImageLeafOptions {
        /// How the VRAM budget is chosen (the server's `vram` config).
        pub vram: VramMode,
        /// `image.condition_budget_tokens` (spec image-model/05), when named.
        pub condition_budget_cap: Option<u64>,
        /// The largest target, in latent tokens, the arena is sized for.
        pub max_target_tokens: u32,
    }

    /// A condition's text: the text encoder's input (the server's `EncoderFrontend` builds it).
    #[derive(Debug, Clone, Copy)]
    pub struct ConditionText<'a> {
        /// The prompt's token ids, the system message included, each image's pads expanded.
        pub input_ids: &'a [u32],
        /// The mrope positions, int32 `[3][tokens]` (rows t, h, w).
        pub positions: &'a [i32],
        /// The system message's length: the DiT reads the encoder's rows from here on.
        pub drop_idx: usize,
        /// The rows of `input_ids` that are image pads, ascending (none for text-to-image): where
        /// the tower's features go in the encoder, and each condition image's latents in the prefix.
        pub image_pad_rows: &'a [u32],
    }

    /// A condition image (spec image-model/04): what the vision tower reads and what the VAE reads,
    /// of one image (the server's `EncoderFrontend` and its media step build both).
    #[derive(Debug, Clone, Copy)]
    pub struct ConditionImage<'a> {
        /// The processor's grid in 16-pixel patches (`image_grid_thw`, one frame).
        pub grid: Grid,
        /// Its f32 `pixel_values`, `[raw patches][1536]`.
        pub pixels: &'a [f32],
        /// The VAE's input: RGBA8 `[height][width][4]`, row-major, the grid's size in pixels.
        pub width: u32,
        pub height: u32,
        pub rgba: &'a [u8],
    }

    /// A condition prefix (spec image-model/04): every block's K/V run of the prefix's tokens,
    /// stored once by `encode_condition` and read by every denoising step of every image of the
    /// job. Free it with [`ImageLeaf::release_condition`].
    #[derive(Debug)]
    pub struct Condition {
        runs: DeviceBuffer,
        run_ptrs: Vec<*mut c_void>,
        tokens: u32,
        segments: Vec<tffi::ignis_image_transformer_segment>,
        /// The prefix's image blocks (start, (height, width)): the joint sequence's RoPE
        /// positions depend on them.
        blocks: Vec<(u32, (u32, u32))>,
        /// The chunk the prefix pass ran in, when it did not run whole.
        chunk: Option<u32>,
    }

    impl Condition {
        /// The prefix's tokens.
        pub fn tokens(&self) -> u32 {
            self.tokens
        }

        /// The prefix's attention segments: (start, end, text).
        pub fn segments(&self) -> Vec<(u32, u32, bool)> {
            self.segments.iter().map(|g| (g.start, g.end, g.text != 0)).collect()
        }

        /// Its condition images' blocks: (start, (height, width) in latent tokens).
        pub fn blocks(&self) -> &[(u32, (u32, u32))] {
            &self.blocks
        }

        /// The chunk, in tokens, the prefix pass ran in when the whole pass did not fit the arena.
        pub fn prefix_chunk(&self) -> Option<u32> {
            self.chunk
        }

        /// Its device bytes.
        pub fn bytes(&self) -> u64 {
            self.runs.len()
        }

        fn descriptor(&self) -> tffi::ignis_image_transformer_prefix {
            tffi::ignis_image_transformer_prefix {
                tokens: self.tokens,
                runs: self.run_ptrs.as_ptr(),
                segments: self.segments.as_ptr(),
                segment_count: self.segments.len() as u32,
            }
        }
    }

    /// The arena: the image slot (one image's state across its steps), then the sequential region
    /// every call carves by phase. Byte offsets from the arena's start.
    #[derive(Debug, Clone, Copy)]
    struct Layout {
        max_target: u32,
        latents: u64,
        velocity: u64,
        positions: u64,
        rope: u64,
        modulation: u64,
        out_scale: u64,
        seq: u64,
        seq_bytes: u64,
    }

    impl Layout {
        fn new(max_target: u32) -> Self {
            let n = max_target as u64;
            let latents = 0;
            let velocity = latents + align(n * LATENT as u64 * 4);
            let positions = velocity + align(n * LATENT as u64 * 4);
            let rope = positions + align(n * 12);
            let modulation = rope + align(n * ROPE_ROW);
            let out_scale = modulation + align(tffi::MODULATION as u64 * 4);
            let seq = out_scale + align(ROW);
            Self { max_target, latents, velocity, positions, rope, modulation, out_scale, seq, seq_bytes: step_bytes(max_target) }
        }

        fn total(&self) -> u64 {
            self.seq + self.seq_bytes
        }
    }

    /// A denoising step's phase of the sequential region: the residual stream, then the DiT's
    /// workspace.
    fn step_bytes(tokens: u32) -> u64 {
        align(tokens as u64 * ROW) + transformer::workspace_bytes(tokens)
    }

    /// `encode_condition`'s phase of the sequential region. First the condition images' latents
    /// (fp32, normalized), which live until img_in; each image's VAE encode runs after them. Then the
    /// encoder's residual stream, its positions and pad rows, the prefix's RoPE positions, and a
    /// union: the tower's four outputs and the tower's or the encoder's workspace, or (once the
    /// encoder has run) the prefix's residual stream, its RoPE table, the modulation and the DiT's
    /// workspace. The prefix pass runs whole when its workspace fits what is left, else in the
    /// largest chunk (a multiple of 128) that fits ([`transformer::prefix_blocks`]).
    struct EncodePhase {
        latents: Vec<u64>,
        vae: u64,
        x: u64,
        positions: u64,
        pad_rows: u64,
        prefix_positions: u64,
        embeds: [u64; 4],
        encoder_workspace: u64,
        encoder_workspace_bytes: u64,
        prefix_x: u64,
        rope: u64,
        modulation: u64,
        out_scale: u64,
        workspace: u64,
        workspace_bytes: u64,
        chunk: Option<u32>,
    }

    impl EncodePhase {
        /// The phase of an encoder sequence of `tokens` rows with `pads` image pads (the tower's
        /// `call` over the condition images of `sizes`, (width, height) in pixels) and a prefix of
        /// `prefix` tokens, in a region of `region` bytes; Err names what does not fit.
        fn new(
            vae: &ignis_artifact::qwen_image::VaeGeometry,
            tokens: u32,
            pads: u32,
            call: Option<&VisionCall>,
            sizes: &[(u32, u32)],
            prefix: u32,
            region: u64,
        ) -> Result<Self, String> {
            let (l, p) = (tokens as u64, prefix as u64);
            let mut latents = Vec::with_capacity(sizes.len());
            let mut at = 0;
            for &(w, h) in sizes {
                latents.push(at);
                at += align((w / 16) as u64 * (h / 16) as u64 * LATENT as u64 * 4);
            }
            let vae_phase = sizes.iter().map(|&(w, h)| VaePhase::new(vae, w, h).bytes).max().unwrap_or(0);
            let vae_end = at + vae_phase;
            let x = at;
            let positions = x + align(l * ROW);
            let pad_rows = positions + align(l * 12);
            let prefix_positions = pad_rows + align(pads as u64 * 4);
            let union = prefix_positions + align(p * 12);
            let rows = align(pads as u64 * HIDDEN as u64 * 2);
            let embeds = [union, union + rows, union + 2 * rows, union + 3 * rows];
            let encoder_workspace = if call.is_some() { union + 4 * rows } else { union };
            let tower = call.map_or(0, |c| image_vision::workspace_bytes(c.tokens(), c.segments()));
            let encoder_workspace_bytes = image_encoder::workspace_bytes(tokens).max(tower);
            let encoder_end = encoder_workspace + encoder_workspace_bytes;
            let prefix_x = union;
            let rope = prefix_x + align(p * ROW);
            let modulation = rope + align(p * ROPE_ROW);
            let out_scale = modulation + align(tffi::MODULATION as u64 * 4);
            let workspace = out_scale + align(ROW);
            let need = |what: &str, end: u64| {
                if end > region {
                    Err(format!(
                        "the condition ({tokens} encoder tokens, {pads} image pads, {prefix} prefix tokens) needs {end} bytes of arena for \
                         its {what}, the arena holds {region}"
                    ))
                } else {
                    Ok(())
                }
            };
            need("VAE encode", vae_end)?;
            need("tower and encoder", encoder_end)?;
            need("prefix", workspace)?;
            let left = region - workspace;
            let whole = transformer::workspace_bytes(prefix);
            let chunk = if whole <= left {
                None
            } else {
                // The largest multiple of 128 whose workspace fits: the workspace grows with the chunk.
                let (mut lo, mut hi) = (0u32, prefix.div_ceil(128));
                while hi - lo > 1 {
                    let mid = lo + (hi - lo) / 2;
                    if transformer::workspace_bytes(mid * 128) <= left { lo = mid } else { hi = mid }
                }
                if lo == 0 {
                    need("prefix pass in chunks of 128", workspace + transformer::workspace_bytes(128))?;
                }
                Some(lo * 128)
            };
            Ok(Self {
                latents,
                vae: at,
                x,
                positions,
                pad_rows,
                prefix_positions,
                embeds,
                encoder_workspace,
                encoder_workspace_bytes,
                prefix_x,
                rope,
                modulation,
                out_scale,
                workspace,
                workspace_bytes: left,
                chunk,
            })
        }

        /// The rows txt_in, img_in and a pass take at once: the prefix's, or the chunk's.
        fn rows(&self, prefix: u32) -> u32 {
            self.chunk.unwrap_or(prefix)
        }
    }

    /// `vae_encode`'s phase of the sequential region, from where it starts: the input image (BF16
    /// NHWC RGBA), the latents (fp32), then the encoder's workspace.
    struct VaePhase {
        input: u64,
        latents: u64,
        workspace: u64,
        workspace_bytes: u64,
        bytes: u64,
    }

    impl VaePhase {
        fn new(geometry: &ignis_artifact::qwen_image::VaeGeometry, width: u32, height: u32) -> Self {
            let pixels = width as u64 * height as u64;
            let input = 0;
            let latents = input + align(pixels * 4 * 2);
            let workspace = latents + align(pixels / 256 * LATENT as u64 * 4);
            let workspace_bytes = autoencoder::encode_workspace_bytes(geometry, height, width);
            Self { input, latents, workspace, workspace_bytes, bytes: workspace + workspace_bytes }
        }
    }

    /// `vae_decode`'s phase of the sequential region, from where it starts: the decoder's input (the
    /// denormalized latents in BF16), its fp32 output image, then its workspace for bands of
    /// `band_rows` image rows.
    struct DecodePhase {
        input: u64,
        image: u64,
        workspace: u64,
        workspace_bytes: u64,
        bytes: u64,
        band_rows: u32,
    }

    impl DecodePhase {
        fn new(geometry: &ignis_artifact::qwen_image::VaeGeometry, width: u32, height: u32, band_rows: u32) -> Self {
            let pixels = width as u64 * height as u64;
            let input = 0;
            let image = input + align(pixels / 256 * LATENT as u64 * 2);
            let workspace = image + align(pixels * geometry.out_channels * 4);
            let workspace_bytes = autoencoder::decode_workspace_bytes(geometry, height, width, band_rows);
            Self { input, image, workspace, workspace_bytes, bytes: workspace + workspace_bytes, band_rows }
        }

        /// The phase in bands of the most image rows (a multiple of 16, at most the image's height)
        /// whose bytes fit `region`; Err when not even 16 rows do.
        fn fitting(geometry: &ignis_artifact::qwen_image::VaeGeometry, width: u32, height: u32, region: u64) -> Result<Self, String> {
            let mut rows = height;
            loop {
                let phase = Self::new(geometry, width, height, rows);
                if phase.bytes <= region {
                    return Ok(phase);
                }
                if rows <= 16 {
                    return Err(format!(
                        "the VAE decode of a {width} x {height} image needs {} bytes of arena in bands of 16 rows, the arena holds {region}",
                        phase.bytes
                    ));
                }
                rows -= 16;
            }
        }
    }

    fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn sync() -> Result<(), ImageLeafError> {
        moe::stream_sync(std::ptr::null_mut()).map_err(failed("the stream"))
    }

    /// The image leaf, loaded: the family's weights on the device (the text encoder, the vision
    /// tower, the DiT; the VAE's tensors materialized for the decode), its descriptors, the sampler,
    /// the plan and the arena.
    pub struct ImageLeaf {
        artifact: MaterializedArtifact,
        embed_tokens: ObjectHandle,
        text: TextEncoderWeights,
        vision: VisionWeights,
        transformer: TransformerWeights,
        vae: VaeEncoderWeights,
        decoder: VaeDecoderWeights,
        sampler: Sampler,
        latent_norm: LatentNorm,
        plan: ImagePlan,
        layout: Layout,
        arena: Option<DeviceBuffer>,
        /// The started image's shape, (height, width) in latent tokens.
        image: Option<(u32, u32)>,
        dev: CudaDevice,
    }

    impl ImageLeaf {
        /// Loads the family from the artifact: binds every object, plans the device memory (a
        /// load whose weights and arena do not fit is refused before anything is placed, naming
        /// the shortfall), materializes the weights and allocates the arena.
        pub fn load(reader: &Reader, mut dev: CudaDevice, options: &ImageLeafOptions) -> Result<Self, ImageLeafError> {
            if options.max_target_tokens == 0 || options.max_target_tokens > 1 << 20 {
                return Err(ImageLeafError::Refused(format!("max_target_tokens {} is not 1 .. 2^20", options.max_target_tokens)));
            }
            let geometry = QwenImageGeometry::qwen_image_21_turbo();
            let bound = qwen_image::bind(reader, &geometry).map_err(|e| failed("the artifact does not bind")(e.to_string()))?;
            let layout = Layout::new(options.max_target_tokens);
            let free = match CudaDevice::nvml_memory(0) {
                Ok((free, _)) => free,
                Err(_) => dev.free_bytes().ok_or_else(|| ImageLeafError::Failed("the device reports no free memory".into()))?,
            };
            let blocks = geometry.dit.blocks as u64;
            // A prefix's 32 runs, and the BF16 K and V a chunked prefix pass holds while it runs.
            let prefix_bytes = move |n: u64| match u32::try_from(n) {
                Ok(0) => 0,
                Ok(n) if n <= 1 << 20 => blocks * align(image_attn::kv8_bytes(n)) + transformer::prefix_kv_bytes(n),
                _ => u64::MAX,
            };
            let plan = ImagePlan::new(&ImagePlanRequest {
                mode: options.vram,
                free_at_start_bytes: free,
                weights_bytes: bound.plan.device_capacity_bytes,
                arena_bytes: layout.total(),
                prefix_bytes: &prefix_bytes,
                condition_budget_cap: options.condition_budget_cap,
            })
            .map_err(ImageLeafError::Plan)?;

            let start = Instant::now();
            let mut artifact = materialize(reader, &bound.plan, &mut dev, None).map_err(|e| failed("materialize")(e.to_string()))?;
            let built = (|| -> Result<_, ImageLeafError> {
                let resource = |name: &str| artifact.resource_bytes(bound.handles[name]).map_err(|e| failed(name)(e.to_string()));
                let text_json = resource(qwen_image::TEXT_ENCODER_CONFIG)?;
                let text_config = TextEncoderConfig::parse(text_json, &geometry.text).map_err(failed("the text config"))?;
                let tower_config = VisionTowerConfig::parse(text_json, &geometry.vision).map_err(failed("the vision config"))?;
                let transformer_config =
                    TransformerConfig::parse(resource(qwen_image::TRANSFORMER_CONFIG)?, &geometry.dit).map_err(failed("the transformer config"))?;
                let sampler = Sampler::parse(resource(qwen_image::IMAGE_MODEL)?).map_err(failed("the sampler"))?;
                let latent_norm =
                    LatentNorm::parse(resource(qwen_image::IMAGE_MODEL)?, geometry.vae.z_dim as usize).map_err(failed("the latent normalization"))?;
                VaeConfig::parse(resource(qwen_image::VAE_CONFIG)?, &geometry.vae).map_err(failed("the VAE config"))?;
                let text = TextEncoderWeights::new(&artifact, &bound.handles, &geometry.text, text_config).map_err(failed("the text encoder"))?;
                let vision = VisionWeights::new(&artifact, &bound.handles, &geometry.vision, &tower_config).map_err(failed("the vision tower"))?;
                let transformer = TransformerWeights::new(&artifact, &bound.handles, &geometry.dit, transformer_config).map_err(failed("the DiT"))?;
                let vae = VaeEncoderWeights::new(&artifact, &bound.handles, &geometry.vae).map_err(failed("the VAE encoder"))?;
                let decoder = VaeDecoderWeights::new(&artifact, &bound.handles, &geometry.vae).map_err(failed("the VAE decoder"))?;
                image_gemm::prepare().map_err(failed("image_gemm::prepare"))?;
                Ok((text, vision, transformer, vae, decoder, sampler, latent_norm))
            })();
            let (text, vision, transformer, vae, decoder, sampler, latent_norm) = match built {
                Ok(b) => b,
                Err(e) => {
                    let _ = artifact.release_arena(&mut dev);
                    return Err(e);
                }
            };
            let arena = match dev.allocate(layout.total()) {
                Ok(a) => a,
                Err(e) => {
                    let _ = artifact.release_arena(&mut dev);
                    return Err(failed("the arena")(e.to_string()));
                }
            };
            tracing::info!(load_s = start.elapsed().as_secs_f64(), "{}", plan.line());
            Ok(Self {
                artifact,
                embed_tokens: bound.handles["text_encoder/embed_tokens.weight"],
                text,
                vision,
                transformer,
                vae,
                decoder,
                sampler,
                latent_norm,
                plan,
                layout,
                arena: Some(arena),
                image: None,
                dev,
            })
        }

        /// The load's plan (its line: [`ImagePlan::line`]).
        pub fn plan(&self) -> &ImagePlan {
            &self.plan
        }

        /// The denoising schedule.
        pub fn sampler(&self) -> &Sampler {
            &self.sampler
        }

        /// The device's free memory (cudaMemGetInfo) once the legacy stream has drained: what a job's
        /// calls leave behind is measured against it (spec image-model/04 acceptance 8).
        pub fn device_free_bytes(&mut self) -> Result<u64, ImageLeafError> {
            sync()?;
            self.dev.free_bytes().ok_or_else(|| ImageLeafError::Failed("the device reports no free memory".into()))
        }

        /// The vision tower's descriptors (for `encode_condition`'s condition images).
        pub fn vision(&self) -> &VisionWeights {
            &self.vision
        }

        fn arena(&self) -> &DeviceBuffer {
            self.arena.as_ref().expect("the arena lives as long as the leaf")
        }

        fn at(&self, offset: u64) -> *mut c_void {
            unsafe { self.arena().base_ptr().add(offset as usize) as *mut c_void }
        }

        fn upload(&mut self, offset: u64, bytes: &[u8]) -> Result<(), ImageLeafError> {
            let arena = *self.arena();
            self.dev.copy_h2d(&arena, offset, bytes).map_err(|e| failed("an upload")(e.to_string()))
        }

        fn download_f32(&mut self, offset: u64, count: usize) -> Result<Vec<f32>, ImageLeafError> {
            let arena = *self.arena();
            self.download_f32_from(&arena, offset, count)
        }

        fn download_f32_from(&mut self, buffer: &DeviceBuffer, offset: u64, count: usize) -> Result<Vec<f32>, ImageLeafError> {
            sync()?;
            let mut bytes = vec![0u8; count * 4];
            self.dev.copy_d2h(buffer, offset, &mut bytes).map_err(|e| failed("a download")(e.to_string()))?;
            self.dev.synchronize().map_err(|e| failed("a download")(e.to_string()))?;
            Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        }

        /// The DiT's blocks in place on `x` (`tokens` rows at arena offset `x`), split after each
        /// block of `taps` (ascending, each below the block count) to copy the residual stream out:
        /// the split runs the same blocks, bit for bit (ignis_image_transformer.h). `chunked`, for
        /// the prefix pass only: its K/V buffer and chunk ([`transformer::prefix_blocks`]).
        #[allow(clippy::too_many_arguments)]
        fn blocks_tapped(
            &mut self,
            x: u64,
            tokens: u32,
            target: bool,
            rope: u64,
            modulation: u64,
            prefix: &tffi::ignis_image_transformer_prefix,
            chunked: Option<(*mut c_void, u32)>,
            workspace: u64,
            workspace_bytes: u64,
            taps: &[u32],
        ) -> Result<Vec<Vec<f32>>, ImageLeafError> {
            let desc = self.transformer.descriptor();
            let mut out = Vec::with_capacity(taps.len());
            let mut first = 0;
            for end in taps.iter().map(|&t| t + 1).chain(std::iter::once(desc.block_count)) {
                let (x_at, rope_at, modulation_at, ws) = (self.at(x) as *mut f32, self.at(rope) as *const f32, self.at(modulation) as *const f32, self.at(workspace));
                match chunked {
                    None => unsafe {
                        transformer::blocks(&desc, first, end, x_at, tokens, target, rope_at, modulation_at, prefix, ws, workspace_bytes, std::ptr::null_mut())
                    },
                    Some((kv, chunk)) => unsafe {
                        transformer::prefix_blocks(&desc, first, end, x_at, rope_at, modulation_at, prefix, kv, chunk, ws, workspace_bytes, std::ptr::null_mut())
                    },
                }
                .map_err(failed("the DiT's blocks"))?;
                if end < desc.block_count || taps.last() == Some(&(end - 1)) {
                    out.push(self.download_f32(x, tokens as usize * HIDDEN)?);
                }
                first = end;
                if first == desc.block_count {
                    break;
                }
            }
            Ok(out)
        }

        fn check_taps(&self, taps: &[u32]) -> Result<(), ImageLeafError> {
            let blocks = self.transformer.descriptor().block_count;
            if !taps.windows(2).all(|w| w[0] < w[1]) || taps.last().is_some_and(|&t| t >= blocks) {
                return Err(ImageLeafError::Refused(format!("the taps {taps:?} are not ascending blocks below {blocks}")));
            }
            Ok(())
        }

        /// `encode_condition` (spec image-model/04): the condition images' VAE encodes; the vision
        /// tower on them and the text encoder with its hooks (spec 03); txt_in on the encoder's text
        /// rows and img_in on each image's latents, laid out in the encoder's order
        /// ([`PrefixLayout`]); then the DiT over those prefix tokens at t = 0, storing each block's
        /// K/V run. No images: text-to-image.
        pub fn encode_condition(&mut self, text: &ConditionText<'_>, images: &[ConditionImage<'_>]) -> Result<Condition, ImageLeafError> {
            self.encode_condition_tapped(text, images, &[]).map(|(c, _)| c)
        }

        /// [`ImageLeaf::encode_condition`], with the prefix's residual stream copied out after
        /// each block of `taps` (spec image-model/04 acceptance 5's diagnosis).
        pub fn encode_condition_tapped(
            &mut self,
            text: &ConditionText<'_>,
            images: &[ConditionImage<'_>],
            taps: &[u32],
        ) -> Result<(Condition, Vec<Vec<f32>>), ImageLeafError> {
            self.check_taps(taps)?;
            let tokens = text.input_ids.len();
            if text.positions.len() != 3 * tokens {
                return Err(ImageLeafError::Refused(format!("{} mrope positions for {tokens} tokens", text.positions.len())));
            }
            for (k, image) in images.iter().enumerate() {
                if image.grid.t != 1 || image.grid.h as u64 * 16 != image.height as u64 || image.grid.w as u64 * 16 != image.width as u64 {
                    return Err(ImageLeafError::Refused(format!(
                        "condition image {k}: a {}x{}x{} grid is not its {} x {} pixels in 16-pixel patches, one frame",
                        image.grid.t, image.grid.h, image.grid.w, image.width, image.height
                    )));
                }
            }
            let shapes: Vec<(u32, u32)> = images.iter().map(|i| (i.height / 16, i.width / 16)).collect();
            let layout = PrefixLayout::new(tokens, text.drop_idx, text.image_pad_rows, &shapes).map_err(ImageLeafError::Refused)?;
            let prefix = layout.tokens;
            if prefix as u64 > self.plan.condition_budget_tokens {
                return Err(ImageLeafError::Refused(format!(
                    "the condition's {prefix} prefix tokens are above the condition budget of {} tokens",
                    self.plan.condition_budget_tokens
                )));
            }
            let starts: Vec<u32> = layout.blocks.iter().map(|b| b.0).collect();
            let prefix_positions = image_dit::rope_positions(prefix, &starts, &shapes).map_err(ImageLeafError::Refused)?;
            let call = if images.is_empty() {
                None
            } else {
                let call = vision_call(&images.iter().map(|i| (i.grid, i.pixels)).collect::<Vec<_>>()).map_err(ImageLeafError::Refused)?;
                if call.tokens() as usize != text.image_pad_rows.len() {
                    return Err(ImageLeafError::Refused(format!("the tower's {} tokens for {} image-pad rows", call.tokens(), text.image_pad_rows.len())));
                }
                Some(call)
            };
            let sizes: Vec<(u32, u32)> = images.iter().map(|i| (i.width, i.height)).collect();
            let phase = EncodePhase::new(
                self.vae.geometry(),
                tokens as u32,
                text.image_pad_rows.len() as u32,
                call.as_ref(),
                &sizes,
                prefix,
                self.layout.seq_bytes,
            )
            .map_err(ImageLeafError::Refused)?;
            let x0 = embed_rows(self.artifact.host_tensor_bytes(self.embed_tokens).map_err(|e| failed("the embedding")(e.to_string()))?, HIDDEN, text.input_ids)
                .map_err(ImageLeafError::Refused)?;

            let blocks = self.transformer.descriptor().block_count as usize;
            let run_bytes = align(image_attn::kv8_bytes(prefix));
            let runs = self.dev.allocate(run_bytes * blocks as u64).map_err(|e| failed("the condition prefix")(e.to_string()))?;
            let condition = Condition {
                run_ptrs: (0..blocks).map(|b| unsafe { runs.base_ptr().add(b * run_bytes as usize) as *mut c_void }).collect(),
                runs,
                tokens: prefix,
                segments: layout.segments.iter().map(|&(start, end, text)| tffi::ignis_image_transformer_segment { start, end, text: text as u32 }).collect(),
                blocks: layout.blocks.clone(),
                chunk: phase.chunk,
            };
            // The chunked pass's K/V, from the condition budget while the pass runs.
            let kv = match phase.chunk {
                None => None,
                Some(_) => match self.dev.allocate(transformer::prefix_kv_bytes(prefix)) {
                    Ok(kv) => Some(kv),
                    Err(e) => {
                        let _ = self.release_condition(condition);
                        return Err(failed("the prefix pass's K/V")(e.to_string()));
                    }
                },
            };
            let ran = self.run_prefix(text, images, call.as_ref(), &layout, &x0, &prefix_positions, &phase, &condition, kv.as_ref(), taps);
            if let Some(kv) = kv {
                let _ = sync();
                let _ = self.dev.deallocate(kv);
            }
            match ran {
                Ok(tapped) => Ok((condition, tapped)),
                Err(e) => {
                    let _ = self.release_condition(condition);
                    Err(e)
                }
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn run_prefix(
            &mut self,
            text: &ConditionText<'_>,
            images: &[ConditionImage<'_>],
            call: Option<&VisionCall>,
            layout: &PrefixLayout,
            x0: &[f32],
            prefix_positions: &[i32],
            phase: &EncodePhase,
            condition: &Condition,
            kv: Option<&DeviceBuffer>,
            taps: &[u32],
        ) -> Result<Vec<Vec<f32>>, ImageLeafError> {
            let seq = self.layout.seq;
            let tokens = text.input_ids.len() as u32;
            let prefix = condition.tokens;
            let null = std::ptr::null_mut();
            // The condition images' latents, each VAE encode in the region after them.
            for (k, image) in images.iter().enumerate() {
                let latents = self.vae_latents(image.width, image.height, image.rgba, seq + phase.vae)?;
                self.upload(seq + phase.latents[k], bytes_of(&latents))?;
                self.dev.synchronize().map_err(|e| failed("the latents' upload")(e.to_string()))?;
            }
            sync()?;
            let pad_rows: Vec<i32> = text.image_pad_rows.iter().map(|&r| r as i32).collect();
            self.upload(seq + phase.x, bytes_of(x0))?;
            self.upload(seq + phase.positions, bytes_of(text.positions))?;
            if !pad_rows.is_empty() {
                self.upload(seq + phase.pad_rows, bytes_of(&pad_rows))?;
            }
            self.upload(seq + phase.prefix_positions, bytes_of(prefix_positions))?;
            self.dev.synchronize().map_err(|e| failed("the uploads")(e.to_string()))?;

            // The tower, then the text encoder with its hooks (spec 03: the merger's rows over the
            // pads, the DeepStack mergers' added after the first text layers).
            let embeds = phase.embeds.map(|e| self.at(seq + e));
            let encoder_ws = self.at(seq + phase.encoder_workspace);
            if let Some(call) = call {
                let desc = self.vision.descriptor();
                unsafe { image_vision::forward(&desc, call, embeds[0], Some(&[embeds[1], embeds[2], embeds[3]]), encoder_ws, phase.encoder_workspace_bytes, null) }
                    .map_err(failed("the vision tower"))?;
            }
            let vision = call.map(|_| image_encoder::ffi::ignis_image_encoder_vision {
                rows: pad_rows.len() as u32,
                row_index: self.at(seq + phase.pad_rows) as *const i32,
                embeds: embeds[0],
                deepstack: [embeds[1] as *const c_void, embeds[2] as *const c_void, embeds[3] as *const c_void],
            });
            let text_desc = self.text.descriptor();
            unsafe {
                image_encoder::forward(
                    &text_desc,
                    0,
                    text_desc.layer_count,
                    self.at(seq + phase.x) as *mut f32,
                    tokens,
                    self.at(seq + phase.positions) as *const i32,
                    vision.as_ref(),
                    encoder_ws,
                    phase.encoder_workspace_bytes,
                    null,
                )
            }
            .map_err(failed("the text encoder"))?;

            // txt_in on each run of text rows, img_in on each image's latents, at their prefix rows;
            // both row by row, so in pieces of at most the pass's rows.
            let desc = self.transformer.descriptor();
            let workspace = seq + phase.workspace;
            let workspace_bytes = phase.workspace_bytes;
            let rows = phase.rows(prefix);
            for &(row, len, token) in &layout.text_runs {
                for at in (0..len).step_by(rows as usize) {
                    let n = rows.min(len - at);
                    unsafe {
                        transformer::txt_in(
                            &desc,
                            self.at(seq + phase.x + (row + at) as u64 * ROW) as *const f32,
                            n,
                            self.at(seq + phase.prefix_x + (token + at) as u64 * ROW) as *mut f32,
                            self.at(workspace),
                            workspace_bytes,
                            null,
                        )
                    }
                    .map_err(failed("txt_in"))?;
                }
            }
            for (k, &(token, (h, w))) in layout.blocks.iter().enumerate() {
                let len = h * w;
                for at in (0..len).step_by(rows as usize) {
                    let n = rows.min(len - at);
                    unsafe {
                        transformer::img_in(
                            &desc,
                            self.at(seq + phase.latents[k] + at as u64 * LATENT as u64 * 4) as *const f32,
                            n,
                            self.at(seq + phase.prefix_x + (token + at) as u64 * ROW) as *mut f32,
                            self.at(workspace),
                            workspace_bytes,
                            null,
                        )
                    }
                    .map_err(failed("img_in"))?;
                }
            }
            unsafe { image_dit::rope_table(self.at(seq + phase.prefix_positions) as *const i32, prefix, self.at(seq + phase.rope) as *mut f32, null) }
                .map_err(failed("the prefix's RoPE table"))?;
            unsafe {
                transformer::modulation(
                    &desc,
                    0.0,
                    self.at(seq + phase.modulation) as *mut f32,
                    self.at(seq + phase.out_scale) as *mut f32,
                    self.at(workspace),
                    workspace_bytes,
                    null,
                )
            }
            .map_err(failed("the t = 0 modulation"))?;
            let chunked = phase.chunk.zip(kv).map(|(chunk, kv)| (kv.base_ptr() as *mut c_void, chunk));
            let tapped = self.blocks_tapped(
                seq + phase.prefix_x,
                prefix,
                false,
                seq + phase.rope,
                seq + phase.modulation,
                &condition.descriptor(),
                chunked,
                workspace,
                workspace_bytes,
                taps,
            )?;
            sync()?;
            Ok(tapped)
        }

        /// `vae_encode` (spec image-model/04): an RGBA8 image (`[height][width][4]`, row-major, sides
        /// multiples of 16) to its condition latents as the DiT reads them, packed fp32 `[height / 16
        /// · width / 16][64]` (token `y · w + x`), normalized. It runs in the sequential region.
        pub fn vae_encode(&mut self, width: u32, height: u32, rgba: &[u8]) -> Result<Vec<f32>, ImageLeafError> {
            let seq = self.layout.seq;
            self.vae_latents(width, height, rgba, seq)
        }

        /// The VAE encode with its phase from arena offset `at` (within the sequential region):
        /// the normalized latents, on the host.
        fn vae_latents(&mut self, width: u32, height: u32, rgba: &[u8], at: u64) -> Result<Vec<f32>, ImageLeafError> {
            let geometry = self.vae.geometry().clone();
            autoencoder::check_size(&geometry, height, width).map_err(ImageLeafError::Refused)?;
            if rgba.len() as u64 != width as u64 * height as u64 * 4 {
                return Err(ImageLeafError::Refused(format!("{} bytes for a {width} x {height} RGBA image", rgba.len())));
            }
            let phase = VaePhase::new(&geometry, width, height);
            let end = self.layout.seq + self.layout.seq_bytes;
            if at + phase.bytes > end {
                return Err(ImageLeafError::Refused(format!(
                    "the VAE encode of a {width} x {height} image needs {} bytes of arena, {} are left",
                    phase.bytes,
                    end.saturating_sub(at)
                )));
            }
            sync()?;
            self.upload(at + phase.input, bytes_of(&autoencoder::input_bf16(rgba)))?;
            self.dev.synchronize().map_err(|e| failed("the image's upload")(e.to_string()))?;
            unsafe {
                autoencoder::encode(
                    &self.vae,
                    self.at(at + phase.input),
                    height,
                    width,
                    self.at(at + phase.latents) as *mut f32,
                    self.at(at + phase.workspace),
                    phase.workspace_bytes,
                )
            }
            .map_err(failed("the VAE encoder"))?;
            let tokens = (width / 16) as usize * (height / 16) as usize;
            let mut latents = self.download_f32(at + phase.latents, tokens * LATENT)?;
            self.latent_norm.normalize(&mut latents);
            Ok(latents)
        }

        /// Frees a condition prefix.
        pub fn release_condition(&mut self, condition: Condition) -> Result<(), ImageLeafError> {
            sync()?;
            self.dev.deallocate(condition.runs).map_err(|e| failed("release_condition")(e.to_string()))
        }

        /// Starts an image of `height` x `width` latent tokens over `condition`: its latents are
        /// `noise` (packed fp32 `[tokens][64]`, `packed[y * width + x][c] = noise[c][y][x]`), and its
        /// RoPE table the target block's rows of the joint sequence.
        pub fn start_image(&mut self, condition: &Condition, height: u32, width: u32, noise: &[f32]) -> Result<(), ImageLeafError> {
            let tokens = height as u64 * width as u64;
            if tokens == 0 || tokens > self.layout.max_target as u64 {
                return Err(ImageLeafError::Refused(format!(
                    "a {height} x {width} target is {tokens} latent tokens, the arena holds 1 .. {}",
                    self.layout.max_target
                )));
            }
            let tokens = tokens as u32;
            if noise.len() != tokens as usize * LATENT {
                return Err(ImageLeafError::Refused(format!("{} noise values for {tokens} tokens of 64", noise.len())));
            }
            let mut starts: Vec<u32> = condition.blocks.iter().map(|b| b.0).collect();
            let mut shapes: Vec<(u32, u32)> = condition.blocks.iter().map(|b| b.1).collect();
            starts.push(condition.tokens);
            shapes.push((height, width));
            let joint = condition.tokens.checked_add(tokens).ok_or_else(|| ImageLeafError::Refused("the joint sequence is too long".into()))?;
            let positions = image_dit::rope_positions(joint, &starts, &shapes).map_err(ImageLeafError::Refused)?;
            sync()?;
            self.upload(self.layout.latents, bytes_of(noise))?;
            self.upload(self.layout.positions, bytes_of(&positions[3 * condition.tokens as usize..]))?;
            self.dev.synchronize().map_err(|e| failed("the image's uploads")(e.to_string()))?;
            unsafe { image_dit::rope_table(self.at(self.layout.positions) as *const i32, tokens, self.at(self.layout.rope) as *mut f32, std::ptr::null_mut()) }
                .map_err(failed("the target's RoPE table"))?;
            sync()?;
            self.image = Some((height, width));
            Ok(())
        }

        /// Denoising step `step` of the started image: the DiT over the target tokens at the
        /// step's σ, attending to `condition`, then the Euler update of the latents on the device.
        pub fn denoise_step(&mut self, condition: &Condition, step: usize) -> Result<(), ImageLeafError> {
            self.denoise_step_tapped(condition, step, &[]).map(|_| ())
        }

        /// [`ImageLeaf::denoise_step`], with the target's residual stream copied out after each
        /// block of `taps`.
        pub fn denoise_step_tapped(&mut self, condition: &Condition, step: usize, taps: &[u32]) -> Result<Vec<Vec<f32>>, ImageLeafError> {
            self.check_taps(taps)?;
            let tokens = self.image.map(|(h, w)| h * w).ok_or_else(|| ImageLeafError::Refused("denoise_step before start_image".into()))?;
            if step >= self.sampler.steps() {
                return Err(ImageLeafError::Refused(format!("step {step} of a {}-step schedule", self.sampler.steps())));
            }
            let (sigma, sigma_next) = self.sampler.sigma(step);
            let seq = self.layout.seq;
            let x = seq;
            let workspace = seq + align(tokens as u64 * ROW);
            let workspace_bytes = self.layout.seq_bytes - align(tokens as u64 * ROW);
            let desc = self.transformer.descriptor();
            let null = std::ptr::null_mut();
            let l = self.layout;
            unsafe {
                transformer::modulation(&desc, sigma, self.at(l.modulation) as *mut f32, self.at(l.out_scale) as *mut f32, self.at(workspace), workspace_bytes, null)
            }
            .map_err(failed("the modulation"))?;
            unsafe { transformer::img_in(&desc, self.at(l.latents) as *const f32, tokens, self.at(x) as *mut f32, self.at(workspace), workspace_bytes, null) }
                .map_err(failed("img_in"))?;
            let tapped = self.blocks_tapped(x, tokens, true, l.rope, l.modulation, &condition.descriptor(), None, workspace, workspace_bytes, taps)?;
            unsafe {
                transformer::head(
                    &desc,
                    self.at(x) as *const f32,
                    tokens,
                    self.at(l.out_scale) as *const f32,
                    self.at(l.velocity) as *mut f32,
                    Some(self.at(l.latents) as *mut f32),
                    sigma,
                    sigma_next,
                    self.at(workspace),
                    workspace_bytes,
                    null,
                )
            }
            .map_err(failed("the head"))?;
            sync()?;
            Ok(tapped)
        }

        /// Overwrites the started image's latents (packed fp32 `[tokens][64]`): a teacher-forced
        /// step's input.
        pub fn set_latents(&mut self, latents: &[f32]) -> Result<(), ImageLeafError> {
            let tokens = self.image.map(|(h, w)| h * w).ok_or_else(|| ImageLeafError::Refused("set_latents before start_image".into()))?;
            if latents.len() != tokens as usize * LATENT {
                return Err(ImageLeafError::Refused(format!("{} latents for {tokens} tokens of 64", latents.len())));
            }
            sync()?;
            self.upload(self.layout.latents, bytes_of(latents))?;
            self.dev.synchronize().map_err(|e| failed("the latents' upload")(e.to_string()))
        }

        /// The started image's latents, packed fp32 `[tokens][64]`.
        pub fn latents(&mut self) -> Result<Vec<f32>, ImageLeafError> {
            let tokens = self.image.map(|(h, w)| h * w).ok_or_else(|| ImageLeafError::Refused("latents before start_image".into()))?;
            self.download_f32(self.layout.latents, tokens as usize * LATENT)
        }

        /// The last step's velocity, packed fp32 `[tokens][64]`.
        pub fn velocity(&mut self) -> Result<Vec<f32>, ImageLeafError> {
            let tokens = self.image.map(|(h, w)| h * w).ok_or_else(|| ImageLeafError::Refused("velocity before start_image".into()))?;
            self.download_f32(self.layout.velocity, tokens as usize * LATENT)
        }

        /// `vae_decode` (spec image-model/04): the started image's latents (the image slot), `z · std
        /// + mean`, the VAE decoder, the clamp: RGBA8 bytes `[16 h][16 w][4]` on the host. The decoder
        /// runs in the arena's sequential region, after the mid block in bands of the most image
        /// rows (a multiple of 16) that fit it; it allocates nothing.
        pub fn vae_decode(&mut self) -> Result<Vec<u8>, ImageLeafError> {
            let (h, w) = self.image.ok_or_else(|| ImageLeafError::Refused("vae_decode before start_image".into()))?;
            let phase = DecodePhase::fitting(self.decoder.geometry(), 16 * w, 16 * h, self.layout.seq_bytes).map_err(ImageLeafError::Refused)?;
            let arena = *self.arena();
            self.decode_in(&arena, self.layout.seq, &phase)
        }

        /// The image rows a band of [`ImageLeaf::vae_decode`] takes for the started image.
        pub fn vae_decode_band_rows(&self) -> Result<u32, ImageLeafError> {
            let (h, w) = self.image.ok_or_else(|| ImageLeafError::Refused("vae_decode before start_image".into()))?;
            Ok(DecodePhase::fitting(self.decoder.geometry(), 16 * w, 16 * h, self.layout.seq_bytes).map_err(ImageLeafError::Refused)?.band_rows)
        }

        /// [`ImageLeaf::vae_decode`] in bands of `band_rows` image rows (the image's height: unbanded).
        /// When the phase does not fit the arena's sequential region, its memory is allocated for the
        /// call, outside the plan: a test's or a diagnosis' reference (spec image-model/04 acceptance
        /// 6), never the server's path.
        pub fn vae_decode_in_bands(&mut self, band_rows: u32) -> Result<Vec<u8>, ImageLeafError> {
            let (h, w) = self.image.ok_or_else(|| ImageLeafError::Refused("vae_decode before start_image".into()))?;
            let geometry = self.decoder.geometry().clone();
            if band_rows == 0 || band_rows > 16 * h {
                return Err(ImageLeafError::Refused(format!("bands of {band_rows} rows for an image of {} rows", 16 * h)));
            }
            let phase = DecodePhase::new(&geometry, 16 * w, 16 * h, band_rows);
            if phase.bytes <= self.layout.seq_bytes {
                let arena = *self.arena();
                self.decode_in(&arena, self.layout.seq, &phase)
            } else {
                let buffer = self.dev.allocate(phase.bytes).map_err(|e| failed("the decode's workspace")(e.to_string()))?;
                let rgba = self.decode_in(&buffer, 0, &phase);
                let _ = sync();
                let _ = self.dev.deallocate(buffer);
                rgba
            }
        }

        /// The decode with its phase at `at` in `buffer`: the latents denormalized on the host (fp32,
        /// two roundings, the pipeline's) and rounded to BF16 once, the decoder, conv_out's fp32
        /// output to the host, then the pipeline's clamp and rounding to bytes.
        fn decode_in(&mut self, buffer: &DeviceBuffer, at: u64, phase: &DecodePhase) -> Result<Vec<u8>, ImageLeafError> {
            let (h, w) = self.image.ok_or_else(|| ImageLeafError::Refused("vae_decode before start_image".into()))?;
            let (height, width) = (16 * h, 16 * w);
            let mut latents = self.download_f32(self.layout.latents, (h * w) as usize * LATENT)?;
            self.latent_norm.denormalize(&mut latents);
            self.dev.copy_h2d(buffer, at + phase.input, bytes_of(&autoencoder::latents_bf16(&latents))).map_err(|e| failed("the latents' upload")(e.to_string()))?;
            self.dev.synchronize().map_err(|e| failed("the latents' upload")(e.to_string()))?;
            let base = buffer.base_ptr() as *mut u8;
            let ptr = |offset: u64| unsafe { base.add((at + offset) as usize) as *mut c_void };
            unsafe {
                autoencoder::decode(
                    &self.decoder,
                    ptr(phase.input),
                    height,
                    width,
                    phase.band_rows,
                    ptr(phase.image) as *mut f32,
                    ptr(phase.workspace),
                    phase.workspace_bytes,
                )
            }
            .map_err(failed("the VAE decoder"))?;
            let channels = self.decoder.geometry().out_channels as usize;
            let image = self.download_f32_from(buffer, at + phase.image, height as usize * width as usize * channels)?;
            Ok(autoencoder::rgba8(&image))
        }
    }

    impl Drop for ImageLeaf {
        fn drop(&mut self) {
            let _ = sync();
            if let Some(arena) = self.arena.take() {
                let _ = self.dev.deallocate(arena);
            }
            let _ = self.artifact.release_arena(&mut self.dev);
        }
    }

    impl ImageLeafOptions {
        /// The server's VRAM mode, no cap, the arena sized for [`MAX_TARGET_TOKENS`].
        pub fn new(vram: VramMode) -> Self {
            Self { vram, condition_budget_cap: None, max_target_tokens: MAX_TARGET_TOKENS }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// The checkpoint's `image_model.json` sampler and scheduler (`Qwen/Qwen-Image-2.1-Turbo` at
    /// `fc13ba19`: `model_index.json`'s `sample_sigmas`, `scheduler/scheduler_config.json`), with
    /// the scheduler edited by `edit`.
    fn image_model(edit: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
        let mut v = serde_json::json!({
            "schema": "ignis-image-model-v1",
            "sample_sigmas": [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568],
            "scheduler": {
                "_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5,
                "invert_sigmas": false, "max_image_seq_len": 8192, "max_shift": 0.9, "num_train_timesteps": 1000,
                "shift": 1.0, "shift_terminal": null, "stochastic_sampling": false, "time_shift_type": "exponential",
                "use_beta_sigmas": false, "use_dynamic_shifting": false, "use_exponential_sigmas": false,
                "use_karras_sigmas": false
            }
        });
        edit(&mut v["scheduler"]);
        serde_json::to_vec(&v).unwrap()
    }

    #[test]
    fn the_checkpoints_sampler_is_its_sigmas_then_zero() {
        let s = Sampler::parse(&image_model(|_| {})).unwrap();
        assert_eq!(s.steps(), 8);
        // The oracle's sigmas.npy (spec 01's t2i case.json), bit for bit: the f32 of each number.
        let recorded = [
            1.0f32, 0.9784529805183411, 0.9541800022125244, 0.9266260266304016, 0.8950799703598022, 0.8451480269432068,
            0.7045339941978455, 0.4145680069923401, 0.0,
        ];
        assert_eq!(s.sigmas.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), recorded.map(f32::to_bits));
        assert_eq!(s.sigma(7), (0.4145680069923401, 0.0));
        let t = s.timesteps();
        assert_eq!(t.len(), 8);
        assert_eq!(t[0], 1000.0);
        assert_eq!(t[1], 0.978453f32 * 1000.0);
    }

    #[test]
    fn a_scheduler_that_moves_the_schedule_is_refused_by_name() {
        let refused = |edit: fn(&mut serde_json::Value)| Sampler::parse(&image_model(edit)).unwrap_err();
        assert!(refused(|s| s["use_dynamic_shifting"] = true.into()).contains("use_dynamic_shifting"));
        assert!(refused(|s| s["shift"] = 3.0.into()).contains("scheduler.shift is 3"));
        assert!(refused(|s| s["shift_terminal"] = 0.02.into()).contains("shift_terminal"));
        assert!(refused(|s| s["invert_sigmas"] = true.into()).contains("invert_sigmas"));
        assert!(refused(|s| s["use_karras_sigmas"] = true.into()).contains("use_karras_sigmas"));
        assert!(refused(|s| s["use_exponential_sigmas"] = true.into()).contains("use_exponential_sigmas"));
        assert!(refused(|s| s["use_beta_sigmas"] = true.into()).contains("use_beta_sigmas"));
        assert!(refused(|s| s["stochastic_sampling"] = true.into()).contains("stochastic_sampling"));
        assert!(refused(|s| s["_class_name"] = "EulerDiscreteScheduler".into()).contains("FlowMatchEulerDiscreteScheduler"));
        assert!(refused(|s| { s.as_object_mut().unwrap().remove("shift"); }).contains("shift is missing"));
        assert!(Sampler::parse(b"{").unwrap_err().contains("not JSON"));
        let mut v: serde_json::Value = serde_json::from_slice(&image_model(|_| {})).unwrap();
        v["sample_sigmas"] = serde_json::json!([1.0, 1.0]);
        assert!(Sampler::parse(&serde_json::to_vec(&v).unwrap()).unwrap_err().contains("not decreasing"));
    }

    #[test]
    fn the_euler_update_rounds_the_product_then_the_sum() {
        // dt = 1 - 2^-24 and v = 1 + 2^-23: the exact product 1 + 2^-24 - 2^-47 rounds to 1, so
        // -1 + fl(dt v) is 0, where a fused multiply-add keeps 2^-24 - 2^-47.
        let (sigma, sigma_next) = (0.0f32, 1.0 - f32::EPSILON / 2.0);
        let v = 1.0 + f32::EPSILON;
        let mut x = [-1.0f32];
        euler_step(&mut x, &[v], sigma, sigma_next);
        let dt = sigma_next - sigma;
        assert_eq!(x[0].to_bits(), 0.0f32.to_bits());
        assert_ne!(dt.mul_add(v, -1.0), 0.0, "the values must tell the two apart");
    }

    #[test]
    fn the_latent_normalization_is_read_per_channel_and_rounds_twice() {
        let json = |mean: serde_json::Value, std: serde_json::Value| serde_json::to_vec(&serde_json::json!({"latents_mean": mean, "latents_std": std})).unwrap();
        let norm = LatentNorm::parse(&json(serde_json::json!([0.5126, -2.1025]), serde_json::json!([3.2001, 4.0379])), 2).unwrap();
        assert_eq!(norm.mean, vec![0.5126f32, -2.1025]);
        // Two tokens of two channels: each value (z - mean[c]) / std[c], two f32 roundings.
        let mut z = [1.0f32, 1.0, -3.0, 0.25];
        norm.normalize(&mut z);
        let want = [(1.0f32 - 0.5126) / 3.2001, (1.0f32 + 2.1025) / 4.0379, (-3.0f32 - 0.5126) / 3.2001, (0.25f32 + 2.1025) / 4.0379];
        assert_eq!(z.map(f32::to_bits), want.map(f32::to_bits));
        assert!(LatentNorm::parse(&json(serde_json::json!([0.0]), serde_json::json!([1.0, 1.0])), 2).unwrap_err().contains("1 latents_mean"));
        assert!(LatentNorm::parse(&json(serde_json::json!([0.0, 0.0]), serde_json::json!([1.0, 0.0])), 2).unwrap_err().contains("not positive"));
        assert!(LatentNorm::parse(b"{}", 2).unwrap_err().contains("no latents_mean"));
    }

    #[test]
    fn the_latents_are_denormalized_per_channel_with_two_roundings() {
        let norm = LatentNorm { mean: vec![0.25, -1.0], std: vec![3.5, 1.0 + f32::EPSILON] };
        // Channel 0: 2 x 3.5 + 0.25 = 7.25 exactly. Channel 1: (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46
        // rounds to 1 + 2^-22, minus 1 is 2^-22; a fused multiply-add would keep the 2^-46.
        let mut z = [2.0f32, 1.0 + f32::EPSILON, 1.0, 1.0];
        norm.denormalize(&mut z);
        assert_eq!(z, [7.25, 2f32.powi(-22), 3.75, f32::EPSILON]);
    }

    #[test]
    fn the_prefix_lays_text_rows_and_image_blocks_out_in_the_encoders_order() {
        // The oracle's edit case: 1064 rows, drop_idx 14, 1024 pads at rows 22 .. 1046 for a 64 x 64
        // condition image; case.json's segments are [0, 8) text, [8, 4104) image, [4104, 4122) text.
        let pads: Vec<u32> = (22..1046).collect();
        let edit = PrefixLayout::new(1064, 14, &pads, &[(64, 64)]).unwrap();
        assert_eq!(edit.tokens, 4122);
        assert_eq!(edit.segments, vec![(0, 8, true), (8, 4104, false), (4104, 4122, true)]);
        assert_eq!(edit.text_runs, vec![(14, 8, 0), (1046, 18, 4104)]);
        assert_eq!(edit.blocks, vec![(8, (64, 64))]);
        // Text to image: the rows from drop_idx, one causal segment.
        let t2i = PrefixLayout::new(45, 14, &[], &[]).unwrap();
        assert_eq!((t2i.tokens, t2i.segments.clone(), t2i.text_runs.clone()), (31, vec![(0, 31, true)], vec![(14, 31, 0)]));
        // Two images with no text between them stay two blocks: rows 2, 3 text, row 4 a 2 x 2
        // image's one pad, rows 5 .. 8 a 2 x 6 image's three, row 8 text.
        let adjacent = PrefixLayout::new(9, 2, &[4, 5, 6, 7], &[(2, 2), (2, 6)]).unwrap();
        assert_eq!(adjacent.segments, vec![(0, 2, true), (2, 6, false), (6, 18, false), (18, 19, true)]);
        assert_eq!(adjacent.blocks, vec![(2, (2, 2)), (6, (2, 6))]);
        assert_eq!(adjacent.text_runs, vec![(2, 2, 0), (8, 1, 18)]);
        // A sequence may end on an image.
        assert_eq!(PrefixLayout::new(6, 2, &[4, 5], &[(2, 4)]).unwrap().segments, vec![(0, 2, true), (2, 10, false)]);
        assert!(PrefixLayout::new(8, 2, &[4, 5, 6, 7], &[(2, 2), (2, 2)]).unwrap_err().contains("4 image-pad rows for condition images of 2 pads"));
        // Refusals: pads before drop_idx, an image's pads not consecutive, shapes not in 2 x 2 groups.
        assert!(PrefixLayout::new(10, 5, &[3, 4, 5, 6], &[(4, 4)]).unwrap_err().contains("not within rows 5"));
        assert!(PrefixLayout::new(10, 2, &[3, 4, 6, 7], &[(4, 4)]).unwrap_err().contains("not consecutive"));
        assert!(PrefixLayout::new(10, 2, &[3], &[(1, 3)]).unwrap_err().contains("groups of 4"));
        assert!(PrefixLayout::new(10, 10, &[], &[]).unwrap_err().contains("drop_idx"));
    }

    fn request(mode: VramMode, free: u64, cap: Option<u64>) -> ImagePlanRequest<'static> {
        ImagePlanRequest {
            mode,
            free_at_start_bytes: free,
            weights_bytes: 16 * GIB,
            arena_bytes: 2 * GIB,
            // 256 KiB a token plus 1 MiB of per-run constants.
            prefix_bytes: &|n| if n == 0 { 0 } else { (1 << 20) + n * (256 << 10) },
            condition_budget_cap: cap,
        }
    }

    #[test]
    fn the_condition_budget_is_what_the_weights_and_arena_leave() {
        let plan = ImagePlan::new(&request(VramMode::Derived { headroom_bytes: GIB }, 30 * GIB, None)).unwrap();
        assert_eq!(plan.budget_bytes, 29 * GIB);
        // 11 GiB left: (11 GiB - 1 MiB) / 256 KiB tokens.
        assert_eq!(plan.condition_budget_tokens, (11 * GIB - (1 << 20)) / (256 << 10));
        assert!(plan.condition_budget_bytes <= 11 * GIB);
        let line = plan.line();
        for part in ["weights 16.00 GiB", "arena 2.00 GiB", "condition budget 45052 tokens", "(derived)"] {
            assert!(line.contains(part), "{line}");
        }
        // A cap only lowers it.
        let capped = ImagePlan::new(&request(VramMode::Derived { headroom_bytes: GIB }, 30 * GIB, Some(1000))).unwrap();
        assert_eq!(capped.condition_budget_tokens, 1000);
        let above = ImagePlan::new(&request(VramMode::Derived { headroom_bytes: GIB }, 30 * GIB, Some(1 << 30))).unwrap();
        assert_eq!(above.condition_budget_tokens, plan.condition_budget_tokens);
    }

    #[test]
    fn a_load_that_does_not_fit_is_refused_naming_the_shortfall() {
        let err = ImagePlan::new(&request(VramMode::Derived { headroom_bytes: GIB }, 18 * GIB, None)).unwrap_err();
        assert_eq!(
            err,
            ImagePlanError::DoesNotFit {
                mode: VramMode::Derived { headroom_bytes: GIB },
                budget_bytes: 17 * GIB,
                weights_bytes: 16 * GIB,
                arena_bytes: 2 * GIB
            }
        );
        let message = err.to_string();
        assert!(message.contains(&format!("short by {} bytes (1.00 GiB)", GIB)), "{message}");
        assert!(message.contains("lower the VRAM headroom"), "{message}");
        let explicit = VramMode::Explicit { budget_bytes: 40 * GIB, allow_oversubscription: false };
        assert_eq!(
            ImagePlan::new(&request(explicit, 30 * GIB, None)).unwrap_err(),
            ImagePlanError::BudgetAboveFree { budget_bytes: 40 * GIB, free_bytes: 30 * GIB }
        );
        let allowed = VramMode::Explicit { budget_bytes: 40 * GIB, allow_oversubscription: true };
        assert_eq!(ImagePlan::new(&request(allowed, 30 * GIB, None)).unwrap().budget_bytes, 40 * GIB);
        // Exactly the weights and the arena: it loads, with no condition budget.
        let tight = ImagePlan::new(&request(VramMode::Derived { headroom_bytes: 0 }, 18 * GIB, None)).unwrap();
        assert_eq!((tight.condition_budget_tokens, tight.condition_budget_bytes), (0, 0));
    }
}
