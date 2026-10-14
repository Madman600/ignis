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
//!
//! With the `cuda` feature, [`ImageLeaf`] loads the family and runs its calls:
//! `encode_condition` (the text encoder, `txt_in` and the DiT over the prefix tokens at t = 0,
//! storing the condition prefix's K/V in 8 bits) and `denoise_step` (the DiT over the target
//! tokens, attending to the stored prefix, then the Euler update on the device). The DiT's program
//! is the kernel leaf's (`kernel/include/ignis_image_transformer.h`, [`crate::image_transformer`]).

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
