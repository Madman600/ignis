//! The image model's runner (spec image-model/05, ADR 0050): the calls a
//! server drives an **image job** with, one at a time — the image leaf's
//! Step ABI (spec image-model/04, ADR 0009) as Rust sees it — and
//! [`FakeImageRunner`], the deterministic CPU stand-in the server's tests load
//! in its place (ADR 0006), as [`crate::MockCompute`] stands in for the text
//! leaf.
//!
//! One image job, in order:
//! 1. [`ImageRunner::encode_condition`]: the prompt and the condition images
//!    become the job's **condition prefix**, once for every image it makes;
//! 2. for each target image, from the job's noise: one
//!    [`ImageRunner::denoise_step`] per sigma of [`ImagePlan::sample_sigmas`],
//!    each stepping to the next and the last to 0;
//! 3. [`ImageRunner::vae_decode`]: the final latents as RGBA;
//! 4. [`ImageRunner::release_condition`]: the prefix freed — also for a job
//!    cancelled between any two calls.
//!
//! The target's latents cross to the host between calls: the noise is drawn
//! on the host from the job's seed, and at most 4 MiB a step moves (2752 ×
//! 1536), against a denoising step of most of a second. So a cancelled job
//! leaves nothing on the device but its prefix. The leaf's `vae_encode` is
//! part of `encode_condition` here: no server drives it on its own.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// The latent channels of the VAE (`z_dim`): a target's latents are
/// `(1, 64, H / 16, W / 16)`.
pub const LATENT_CHANNELS: usize = 64;

/// The VAE's spatial downscale: one latent cell per 16 × 16 pixels.
pub const LATENT_SCALE: u32 = 16;

/// An image's size in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    /// `width` × `height`.
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Its latent cells, `(W / 16) · (H / 16)`: what a condition image costs
    /// the condition prefix in tokens, and a target in latents per channel.
    pub fn latent_cells(self) -> usize {
        (self.width / LATENT_SCALE) as usize * (self.height / LATENT_SCALE) as usize
    }

    /// `<width>x<height>`, as the API and the config spell a size.
    pub fn parse(text: &str) -> Result<Self, String> {
        let side = |side: &str| side.trim().parse::<u32>().ok().filter(|&n| n > 0);
        match text.trim().split_once(['x', 'X']) {
            Some((width, height)) => match (side(width), side(height)) {
                (Some(width), Some(height)) => Ok(Self { width, height }),
                _ => Err(format!("expects <width>x<height> in pixels, got `{text}`")),
            },
            None => Err(format!("expects <width>x<height> in pixels, got `{text}`")),
        }
    }
}

impl std::fmt::Display for ImageSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// An 8-bit RGBA image, rows top to bottom, four bytes a pixel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub size: ImageSize,
    pub pixels: Vec<u8>,
}

/// What an image job is conditioned on: its prompt, as the client sent it
/// (or as prompt enhancement revised it), and its condition images at the
/// size the condition budget left them — none for text-to-image. The runner
/// tokenizes and prepares them: the real one is the server's adapter over the
/// image leaf, which holds the artifact's tokenizer and templates and runs
/// the server's image preprocessing (spec image-model/03 §Seams, 05).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageCondition {
    pub prompt: String,
    pub images: Vec<RgbaImage>,
}

/// A target image's latents on the host: `(1, 64, H / 16, W / 16)` in f32,
/// channel by channel, each channel row by row.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetLatents {
    pub size: ImageSize,
    pub values: Vec<f32>,
}

impl TargetLatents {
    /// The latents of a `size` target, every value 0 (the job writes its
    /// noise over them).
    pub fn zeros(size: ImageSize) -> Self {
        Self { size, values: vec![0.0; LATENT_CHANNELS * size.latent_cells()] }
    }

    /// A `size` target's starting noise from `seed` alone (spec
    /// image-model/04 §Noise): standard normal values, ignis's own draw —
    /// splitmix64's stream from `seed`, two uniforms a pair of values through
    /// Box–Muller. The same seed gives the same latents on the same build and
    /// machine; it is not diffusers' noise (spec image-model/05 §Seams).
    pub fn noise(size: ImageSize, seed: u64) -> Self {
        let mut latents = Self::zeros(size);
        // The k-th output of splitmix64 seeded with `seed`, as a uniform in
        // [0, 1) on 53 bits.
        let uniform = |k: u64| {
            let mut z = seed.wrapping_add(k.wrapping_add(1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        for (pair, values) in latents.values.chunks_mut(2).enumerate() {
            let k = 2 * pair as u64;
            // 1 − u is in (0, 1]: its logarithm is finite.
            let radius = (-2.0 * (1.0 - uniform(k)).ln()).sqrt();
            let angle = std::f64::consts::TAU * uniform(k + 1);
            values[0] = (radius * angle.cos()) as f32;
            if let Some(second) = values.get_mut(1) {
                *second = (radius * angle.sin()) as f32;
            }
        }
        latents
    }
}

/// A condition prefix the runner holds, until
/// [`ImageRunner::release_condition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ConditionId(pub u64);

/// What a load planned, as the server reads it (spec image-model/04 §Load
/// and plan).
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePlan {
    /// How many condition-prefix tokens fit beside the weights and the
    /// arena: the **condition budget**. `image.condition_budget_tokens` can
    /// only lower it.
    pub condition_budget_tokens: u64,
    /// The artifact's sampler sigmas, one denoising step each; the last step
    /// goes to 0.
    pub sample_sigmas: Vec<f32>,
    /// The most latent cells a target may have ([`ImageSize::latent_cells`],
    /// one target token each): what the load sized its arena for. A size past
    /// it is not served, whatever `image.sizes` says.
    pub max_target_cells: usize,
}

impl ImagePlan {
    /// Whether the load can make a `size` target ([`ImagePlan::max_target_cells`]).
    pub fn takes(&self, size: ImageSize) -> bool {
        size.latent_cells() <= self.max_target_cells
    }
}

/// Why a runner call failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    /// The prefix would hold `tokens`, past the condition budget. The server
    /// scales condition images down to fit first, so this is a bug there.
    OverBudget { tokens: u64, budget: u64 },
    /// No prefix by that id: never encoded, or released already.
    NoSuchCondition(ConditionId),
    /// The runner refused the request as one the model cannot make, in its
    /// own words: a prompt that writes an image pad, a condition image the
    /// encoder's processor will not take, a RoPE position past the table
    /// (spec image-model/04). The request's fault, never the server's.
    Refused(String),
    /// The leaf failed the call, in its own words.
    Leaf(String),
}

impl std::fmt::Display for ImageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OverBudget { tokens, budget } => {
                write!(f, "the condition prefix holds {tokens} tokens, past the condition budget of {budget}")
            }
            Self::NoSuchCondition(id) => write!(f, "no condition prefix {}", id.0),
            Self::Refused(message) | Self::Leaf(message) => write!(f, "{message}"),
        }
    }
}

/// Why [`ImageRunner::condition_tokens`] gives a condition no count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConditionUnfit {
    /// Within the budget, but a limit beside it is short for condition
    /// images of these sizes — the leaf's arena, the RoPE positions the
    /// images take before the target — in the runner's words. Smaller images
    /// may fit: the server's fit scales them down for it, as for the budget.
    NoRoom(String),
    /// The runner refuses the request at these sizes, in its words: a
    /// condition size the encoder's processor would not keep (past 200:1), a
    /// prompt its template refuses, a prompt with no image too long for a
    /// limit beside the budget. No scaling makes it: `400 image_refused`.
    Refused(String),
}

/// The image model's compute seam: the only calls an image job makes on the
/// model, one at a time. The real runner is the server's adapter over the
/// image leaf (spec image-model/04, 05); [`FakeImageRunner`] stands in for it
/// on the CPU.
///
/// `&mut self`: one job runs at a time (spec image-model/05 §The queue), so
/// the runner has one caller, and its holder serializes the calls.
pub trait ImageRunner: Send {
    /// The load's plan: the condition budget, the sampler sigmas and the
    /// largest target.
    fn plan(&self) -> &ImagePlan;

    /// The tokens of the condition prefix of `prompt` with condition images
    /// of `images`' sizes: the prompt as the runner's template and tokenizer
    /// lay it out, plus [`ImageSize::latent_cells`] of each image (one
    /// image-pad slot becomes 2 × 2 latent tokens, spec image-model/04) —
    /// the count [`ImageRunner::encode_condition`] refuses past the budget.
    /// Host-side and cheap: the server, which holds no tokenizer for the
    /// image model, fits the condition images to the budget with it before it
    /// encodes them. A count past the budget is still the count. No count is
    /// a condition within the budget that the runner has no room for as it
    /// is ([`ConditionUnfit::NoRoom`]: the fit scales the images down for
    /// that limit too) or that it refuses at these sizes
    /// ([`ConditionUnfit::Refused`]).
    fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ConditionUnfit>;

    /// The encoder over the prompt and the condition images, the VAE encode
    /// of each image, and the DiT over the prefix tokens at t = 0: the
    /// job's condition prefix, kept until released. Refused past the
    /// condition budget.
    fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError>;

    /// One denoising step of `latents` against `condition` from `sigma` to
    /// `sigma_next`: the velocity, then `latents += (sigma_next − sigma) ·
    /// velocity`. With `velocity`, the velocity is also copied there (the
    /// tests' hook: spec image-model/04 AC 3 compares it with diffusers').
    fn denoise_step(
        &mut self,
        condition: ConditionId,
        latents: &mut TargetLatents,
        sigma: f32,
        sigma_next: f32,
        velocity: Option<&mut Vec<f32>>,
    ) -> Result<(), ImageError>;

    /// The VAE decode of final `latents`: the target image, RGBA.
    fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError>;

    /// Free `condition`'s prefix. An id it does not hold is ignored: a
    /// cancelled job releases whatever it got to.
    fn release_condition(&mut self, condition: ConditionId);

    /// The device bytes of the condition prefixes it holds now: what of the
    /// condition budget is in use (GitHub #324). Host-side and cheap: the job
    /// reads it after each encode and release, for the server to publish.
    fn condition_bytes(&self) -> u64;
}

/// The [`ImagePlan`] a [`FakeImageRunner`] has by default: room for eight
/// 1024² condition images (4,096 tokens each) and their prompts, eight
/// evenly spaced sigmas, and targets up to the checkpoint's largest preset
/// in latent cells, 2400 × 1792 (16,800).
pub fn fake_plan() -> ImagePlan {
    ImagePlan {
        condition_budget_tokens: 8 * 4096 + 512,
        sample_sigmas: (0..8).map(|step| 1.0 - step as f32 / 8.0).collect(),
        max_target_cells: ImageSize::new(2400, 1792).latent_cells(),
    }
}

/// The device bytes [`FakeImageRunner`] charges a condition-prefix token:
/// about the leaf's own, a K/V run per block.
pub const FAKE_PREFIX_TOKEN_BYTES: u64 = 256 << 10;

/// A deterministic image runner on the CPU (ADR 0006), for the server's tests
/// and a build without `cuda`: the same condition and noise give the same
/// bytes, and the noise — the seed — shows in them. Nothing it computes is
/// a picture of the prompt. Its prefix costs what the leaf's does in image
/// tokens (one per latent cell of each condition image), and one token per
/// word of the prompt; [`FAKE_PREFIX_TOKEN_BYTES`] a token.
#[derive(Debug)]
pub struct FakeImageRunner {
    plan: ImagePlan,
    /// Each held prefix's value, what its denoising steps pull towards, and
    /// its tokens.
    conditions: BTreeMap<ConditionId, (f32, u64)>,
    next: u64,
    held: Arc<AtomicUsize>,
}

impl Default for FakeImageRunner {
    /// A fake with [`fake_plan`].
    fn default() -> Self {
        Self::new(fake_plan())
    }
}

impl FakeImageRunner {
    /// A fake that plans `plan`.
    pub fn new(plan: ImagePlan) -> Self {
        Self { plan, conditions: BTreeMap::new(), next: 0, held: Arc::default() }
    }

    /// How many condition prefixes it holds, readable once the runner is
    /// boxed away: a finished or cancelled job leaves it where it was.
    pub fn held_conditions(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.held)
    }

    /// The prefix's tokens, as [`FakeImageRunner`] counts them: one token a
    /// word of the prompt, at least one, and the images' cells.
    fn count(prompt: &str, images: &[ImageSize]) -> u64 {
        let words = prompt.split_whitespace().count().max(1) as u64;
        words + images.iter().map(|size| size.latent_cells() as u64).sum::<u64>()
    }
}

/// FNV-1a over `bytes`, from `hash`.
fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3))
}

impl ImageRunner for FakeImageRunner {
    fn plan(&self) -> &ImagePlan {
        &self.plan
    }

    /// Its count, always: the fake has no limit beside the budget.
    fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ConditionUnfit> {
        Ok(Self::count(prompt, images))
    }

    fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
        let sizes: Vec<ImageSize> = condition.images.iter().map(|image| image.size).collect();
        let (tokens, budget) = (Self::count(&condition.prompt, &sizes), self.plan.condition_budget_tokens);
        if tokens > budget {
            return Err(ImageError::OverBudget { tokens, budget });
        }
        let hash = condition.images.iter().fold(fnv(0xcbf2_9ce4_8422_2325, condition.prompt.as_bytes()), |hash, image| {
            fnv(hash, &image.pixels)
        });
        let id = ConditionId(self.next);
        self.next += 1;
        // A value in [-1, 1) from the hash.
        self.conditions.insert(id, ((hash >> 40) as f32 / (1u64 << 23) as f32 - 1.0, tokens));
        self.held.store(self.conditions.len(), Ordering::SeqCst);
        Ok(id)
    }

    fn denoise_step(
        &mut self,
        condition: ConditionId,
        latents: &mut TargetLatents,
        sigma: f32,
        sigma_next: f32,
        velocity: Option<&mut Vec<f32>>,
    ) -> Result<(), ImageError> {
        let (value, _) = *self.conditions.get(&condition).ok_or(ImageError::NoSuchCondition(condition))?;
        let cells = latents.size.latent_cells();
        if latents.values.len() != LATENT_CHANNELS * cells {
            return Err(ImageError::Leaf(format!(
                "{} latents for a {} target, which takes {}",
                latents.values.len(),
                latents.size,
                LATENT_CHANNELS * cells
            )));
        }
        // Each channel pulled towards the condition's value, offset per
        // channel, with the noise kept in it.
        let v: Vec<f32> = latents
            .values
            .iter()
            .enumerate()
            .map(|(i, &x)| 0.5 * x - (value + (i / cells.max(1)) as f32 / LATENT_CHANNELS as f32))
            .collect();
        for (x, v) in latents.values.iter_mut().zip(&v) {
            *x += (sigma_next - sigma) * v;
        }
        if let Some(out) = velocity {
            *out = v;
        }
        Ok(())
    }

    fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
        let ImageSize { width, height } = latents.size;
        let (columns, cells) = ((width / LATENT_SCALE) as usize, latents.size.latent_cells());
        if latents.values.len() != LATENT_CHANNELS * cells {
            return Err(ImageError::Leaf(format!("{} latents for a {} target", latents.values.len(), latents.size)));
        }
        let byte = |value: f32| ((value.tanh() + 1.0) * 127.5).round() as u8;
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
        for y in 0..height as usize {
            let row = y / LATENT_SCALE as usize * columns;
            for x in 0..width as usize {
                let cell = row + x / LATENT_SCALE as usize;
                pixels.extend((0..4).map(|channel| byte(latents.values[channel * cells + cell])));
            }
        }
        Ok(RgbaImage { size: latents.size, pixels })
    }

    fn release_condition(&mut self, condition: ConditionId) {
        self.conditions.remove(&condition);
        self.held.store(self.conditions.len(), Ordering::SeqCst);
    }

    fn condition_bytes(&self) -> u64 {
        self.conditions.values().map(|&(_, tokens)| tokens * FAKE_PREFIX_TOKEN_BYTES).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Noise as a job would draw it: deterministic from `seed`.
    fn noise(size: ImageSize, seed: u64) -> TargetLatents {
        let mut latents = TargetLatents::zeros(size);
        let mut state = seed.wrapping_add(1);
        for value in &mut latents.values {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *value = (state >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
        }
        latents
    }

    /// One image the way a job makes it: the sigmas, then the decode.
    fn generate(runner: &mut dyn ImageRunner, condition: ConditionId, size: ImageSize, seed: u64) -> RgbaImage {
        let mut latents = noise(size, seed);
        let sigmas = runner.plan().sample_sigmas.clone();
        for (step, &sigma) in sigmas.iter().enumerate() {
            let next = sigmas.get(step + 1).copied().unwrap_or(0.0);
            runner.denoise_step(condition, &mut latents, sigma, next, None).expect("a held prefix");
        }
        runner.vae_decode(&latents).expect("decode")
    }

    fn prompt(text: &str) -> ImageCondition {
        ImageCondition { prompt: text.to_owned(), images: Vec::new() }
    }

    /// The fake is what the server's tests compare bytes on: the same seed
    /// and condition give the same image, another seed or prompt another,
    /// and the image is RGBA at the target's size.
    #[test]
    fn the_fake_is_deterministic_in_the_seed_and_the_condition() {
        let size = ImageSize::new(64, 32);
        let mut runner = FakeImageRunner::default();
        let cat = runner.encode_condition(&prompt("a cat")).unwrap();
        let dog = runner.encode_condition(&prompt("a dog")).unwrap();
        let first = generate(&mut runner, cat, size, 7);
        assert_eq!(first.size, size);
        assert_eq!(first.pixels.len(), 64 * 32 * 4);
        assert_eq!(generate(&mut runner, cat, size, 7), first, "same seed, same bytes");
        assert_ne!(generate(&mut runner, cat, size, 8), first, "the seed shows");
        assert_ne!(generate(&mut runner, dog, size, 7), first, "the prompt shows");
        let mut again = FakeImageRunner::default();
        let cat_again = again.encode_condition(&prompt("a cat")).unwrap();
        assert_eq!(generate(&mut again, cat_again, size, 7), first, "another runner, the same bytes");
    }

    /// Spec image-model/04: the step is the Euler update over the velocity,
    /// and the velocity can be copied out.
    #[test]
    fn a_denoising_step_is_the_euler_update_and_copies_the_velocity_out() {
        let size = ImageSize::new(32, 32);
        let mut runner = FakeImageRunner::default();
        let id = runner.encode_condition(&prompt("x")).unwrap();
        let before = noise(size, 3);
        let mut latents = before.clone();
        let mut velocity = Vec::new();
        runner.denoise_step(id, &mut latents, 1.0, 0.75, Some(&mut velocity)).unwrap();
        assert_eq!(velocity.len(), LATENT_CHANNELS * 4);
        for ((after, before), v) in latents.values.iter().zip(&before.values).zip(&velocity) {
            assert_eq!(*after, before + (0.75 - 1.0) * v);
        }
        let mut wrong = TargetLatents { size, values: vec![0.0; 3] };
        assert!(matches!(runner.denoise_step(id, &mut wrong, 1.0, 0.0, None), Err(ImageError::Leaf(_))));
    }

    /// A prefix is held from its encode to its release, a released one is
    /// gone, and the count is readable from outside the runner, its bytes
    /// from the runner (GitHub #324); a prefix past the budget is refused
    /// with both numbers and holds nothing.
    #[test]
    fn prefixes_are_held_until_released_and_refused_past_the_budget() {
        let mut runner = FakeImageRunner::new(ImagePlan { condition_budget_tokens: 4096 + 2, ..fake_plan() });
        let held = runner.held_conditions();
        let image = RgbaImage { size: ImageSize::new(1024, 1024), pixels: vec![0; 1024 * 1024 * 4] };
        let edit = ImageCondition { prompt: "make it blue".into(), images: vec![image.clone()] };
        assert_eq!(
            runner.encode_condition(&edit),
            Err(ImageError::OverBudget { tokens: 4096 + 3, budget: 4096 + 2 }),
            "three words and 64 x 64 latent cells"
        );
        assert_eq!(runner.condition_bytes(), 0);
        let fits = ImageCondition { prompt: "blue".into(), images: vec![image] };
        let id = runner.encode_condition(&fits).unwrap();
        let other = runner.encode_condition(&prompt("x")).unwrap();
        assert_eq!(held.load(Ordering::SeqCst), 2);
        assert_eq!(runner.condition_bytes(), (4096 + 1 + 1) * (256 << 10), "a word and 4,096 cells, and a word");
        runner.release_condition(id);
        runner.release_condition(id);
        assert_eq!(held.load(Ordering::SeqCst), 1, "a second release is ignored");
        assert_eq!(runner.condition_bytes(), 256 << 10);
        let mut latents = TargetLatents::zeros(ImageSize::new(32, 32));
        assert_eq!(runner.denoise_step(id, &mut latents, 1.0, 0.0, None), Err(ImageError::NoSuchCondition(id)));
        runner.release_condition(other);
        assert_eq!(held.load(Ordering::SeqCst), 0);
        assert_eq!(runner.condition_bytes(), 0);
    }

    /// The job's noise is a standard normal draw that only its seed decides.
    #[test]
    fn the_noise_is_standard_normal_and_the_seed_alone_decides_it() {
        let size = ImageSize::new(1024, 1024);
        let noise = TargetLatents::noise(size, 42);
        assert_eq!(noise.values.len(), LATENT_CHANNELS * 4096);
        let n = noise.values.len() as f64;
        let mean = noise.values.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let variance = noise.values.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / n;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!((variance - 1.0).abs() < 0.01, "variance {variance}");
        assert!(noise.values.iter().all(|v| v.is_finite()));
        assert_eq!(TargetLatents::noise(size, 42), noise, "the same seed, the same noise");
        assert_ne!(TargetLatents::noise(size, 43).values, noise.values);
    }

    /// The prefix the fake refuses is the one the server fits to
    /// (`condition_tokens`): the prompt's tokens plus each image's latent
    /// cells; the plan takes targets up to its largest.
    #[test]
    fn the_fake_counts_its_prefix_as_the_prompt_tokens_plus_the_latent_cells() {
        let mut runner = FakeImageRunner::new(ImagePlan { condition_budget_tokens: 3 + 4096, ..fake_plan() });
        let square = ImageSize::new(1024, 1024);
        assert_eq!(runner.condition_tokens("make it blue", &[square]), Ok(3 + 4096));
        assert_eq!(runner.condition_tokens("", &[]), Ok(1));
        assert_eq!(runner.condition_tokens("x", &[square, ImageSize::new(512, 256)]), Ok(1 + 4096 + 512));
        assert_eq!(runner.condition_tokens("x", &[square; 3]), Ok(1 + 3 * 4096), "past the budget, still the count");
        let image = RgbaImage { size: square, pixels: vec![0; 1024 * 1024 * 4] };
        let edit = ImageCondition { prompt: "make it blue".into(), images: vec![image] };
        assert!(runner.encode_condition(&edit).is_ok(), "exactly the budget fits");
        let plan = fake_plan();
        assert!(plan.takes(ImageSize::new(2400, 1792)) && plan.takes(ImageSize::new(1536, 2752)) && plan.takes(square));
        assert!(!plan.takes(ImageSize::new(2432, 1792)), "one column of cells past the largest preset");
    }

    #[test]
    fn a_size_is_width_x_height_in_pixels() {
        assert_eq!(ImageSize::parse("2400x1792"), Ok(ImageSize::new(2400, 1792)));
        assert_eq!(ImageSize::parse(" 1024X1024 "), Ok(ImageSize::new(1024, 1024)));
        for bad in ["1024", "0x1024", "x", "10x-2", "1024x1024x3", "big"] {
            assert!(ImageSize::parse(bad).unwrap_err().contains("<width>x<height>"), "{bad}");
        }
        assert_eq!(ImageSize::new(2752, 1536).to_string(), "2752x1536");
        assert_eq!(ImageSize::new(1024, 1024).latent_cells(), 4096);
        assert_eq!(fake_plan().sample_sigmas.len(), 8);
    }
}
