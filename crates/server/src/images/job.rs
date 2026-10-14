//! One image job (spec image-model/05 §The queue, §Streaming): what a request
//! asks the image model for once its route has checked it, and the job
//! running it — in its turn, one call on the runner at a time, cancelled
//! between any two calls.
//!
//! A job makes its `n` images one after another over one condition prefix:
//! the prompt and the condition images encoded once, then for each image
//! `i` its own noise (`seed + i`), every denoising step, and the decode, the
//! image going out as a PNG as soon as it is made. The request reads the
//! job's [`JobEvent`]s off a channel and, by dropping it, cancels the job:
//! while it waits for its turn, or at the next call once it runs.

use ignis_core::image::{ImageCondition, ImageError, ImageRunner, ImageSize, RgbaImage, TargetLatents};
use serde::Serialize;
use tokio::sync::mpsc;
use utoipa::ToSchema;

use super::{ImageModel, Waiting};

/// What the client's `background` asks of the images' alpha.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Background {
    /// The alpha channel as generated.
    Auto,
    /// The alpha channel as generated, as for `auto`.
    Transparent,
    /// Composited on white, alpha 255.
    Opaque,
}

impl Background {
    /// `auto`, `transparent` or `opaque`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "auto" => Some(Self::Auto),
            "transparent" => Some(Self::Transparent),
            "opaque" => Some(Self::Opaque),
            _ => None,
        }
    }

    /// As the request spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Transparent => "transparent",
            Self::Opaque => "opaque",
        }
    }
}

/// One job, as its route made it.
#[derive(Debug, Clone)]
pub struct ImageJob {
    /// The prompt, and the condition images at the size the condition budget
    /// left them (none for a generation).
    pub condition: ImageCondition,
    /// The images' size.
    pub size: ImageSize,
    /// How many images, one after another.
    pub n: u32,
    /// Image `i` starts from the noise of `seed + i`.
    pub seed: u64,
    pub background: Background,
}

/// The phase a progress event is in. The first three come before the image
/// model runs (prompt enhancement and the switches around it); the job
/// itself sends the last three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// A text model rewrites the prompt.
    Enhancing,
    /// The server switches to the model the job needs next.
    Switching,
    /// The server loads it.
    Loading,
    /// The prompt and the condition images become the job's condition
    /// prefix: once a job.
    Encoding,
    /// One denoising step of one image.
    Denoising,
    /// One image's latents through the VAE decoder.
    Decoding,
}

/// One unit of a job's work, announced as it starts: step `step` of `total`
/// of `phase`, for image `image` (0-based). Encoding is image 0, step 1 of
/// 1; each denoising step of image `i` is step `k` of the sigmas' count;
/// decoding image `i` is step 1 of 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ToSchema)]
pub struct Progress {
    pub phase: Phase,
    pub image: u32,
    pub step: u32,
    pub total: u32,
}

/// What a job tells its request, in order.
#[derive(Debug)]
pub enum JobEvent {
    Progress(Progress),
    /// Image `index`, an RGBA 8-bit PNG.
    Image { index: u32, png: Vec<u8> },
    /// The job ended before its last image, and why. A job that made all of
    /// them simply ends: the channel closes after its last image.
    Ended(JobEnd),
}

/// Why a job ended before its last image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobEnd {
    /// Its request went away (the channel has no reader): nobody is told.
    Cancelled,
    /// The model was stopped (a switch, a reload) before it ran or while it
    /// ran.
    Stopped,
    /// The runner refused a call.
    Failed(ImageError),
}

/// One call on the runner, holding it for that call alone; a model shut
/// down has none ([`JobEnd::Stopped`]).
fn call<T>(model: &ImageModel, calls: impl FnOnce(&mut dyn ImageRunner) -> Result<T, ImageError>) -> Result<T, JobEnd> {
    model.with_runner(calls).ok_or(JobEnd::Stopped)?.map_err(JobEnd::Failed)
}

/// A condition prefix the job holds, released when dropped — the job ended,
/// failed or was cancelled alike.
struct Held<'a> {
    model: &'a ImageModel,
    id: ignis_core::image::ConditionId,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        self.model.with_runner(|runner| runner.release_condition(self.id));
    }
}

/// Run `job` on `model` now: the condition prefix, then each image from its
/// noise through every denoising step and the decode, sent to `events` as a
/// PNG, every unit announced there as it starts ([`Progress`]). The runner
/// is held one call at a time, so a shutdown waits for one call, never for a
/// job; between any two calls the job stops when the model is closed
/// ([`JobEnd::Stopped`]) or when `events` has no reader, its request gone
/// ([`JobEnd::Cancelled`]). The prefix is released whichever way it ends.
/// Blocking: it runs on a blocking thread.
pub fn run<E: From<JobEvent>>(model: &ImageModel, job: &ImageJob, events: &mpsc::UnboundedSender<E>) -> Result<(), JobEnd> {
    let go_on = || {
        if model.jobs.is_closed() {
            Err(JobEnd::Stopped)
        } else if events.is_closed() {
            Err(JobEnd::Cancelled)
        } else {
            Ok(())
        }
    };
    let progress = |phase, image, step, total| {
        // A send fails only once the request is gone, which `go_on` sees.
        let _ = events.send(JobEvent::Progress(Progress { phase, image, step, total }).into());
    };
    go_on()?;
    progress(Phase::Encoding, 0, 1, 1);
    let held = Held { model, id: call(model, |runner| runner.encode_condition(&job.condition))? };
    let sigmas = &model.plan().sample_sigmas;
    let total = sigmas.len() as u32;
    for index in 0..job.n {
        let mut latents = TargetLatents::noise(job.size, job.seed.wrapping_add(u64::from(index)));
        for (step, &sigma) in sigmas.iter().enumerate() {
            go_on()?;
            progress(Phase::Denoising, index, step as u32 + 1, total);
            let next = sigmas.get(step + 1).copied().unwrap_or(0.0);
            call(model, |runner| runner.denoise_step(held.id, &mut latents, sigma, next, None))?;
        }
        go_on()?;
        progress(Phase::Decoding, index, 1, 1);
        let mut image = call(model, |runner| runner.vae_decode(&latents))?;
        if job.background == Background::Opaque {
            on_white(&mut image);
        }
        let _ = events.send(JobEvent::Image { index, png: encode_png(&image) }.into());
    }
    Ok(())
}

/// Queue `job`, admitted as `waiting`, on `model`: in its turn it runs
/// ([`run`]) on a blocking thread, its events sent to `events`, and a job
/// that ends early sends why, unless its request is gone. Dropping the
/// channel's receiver cancels the job: while it waits it leaves the queue at
/// once; once it runs, at its next call.
pub fn spawn<E: From<JobEvent> + Send + 'static>(model: ImageModel, job: ImageJob, waiting: Waiting, events: mpsc::UnboundedSender<E>) {
    tokio::spawn(async move {
        let running = tokio::select! {
            running = waiting.turn() => running,
            () = events.closed() => return,
        };
        let Some(running) = running else {
            let _ = events.send(JobEvent::Ended(JobEnd::Stopped).into());
            return;
        };
        let ran = tokio::task::spawn_blocking(move || {
            let ended = run(&model, &job, &events);
            // No longer counted before its request hears the end: a request
            // answered is never a job in flight.
            drop(running);
            match ended {
                Ok(()) | Err(JobEnd::Cancelled) => {}
                Err(end) => {
                    if let JobEnd::Failed(error) = &end {
                        tracing::error!(name: "ignis.image.job_failed", model = %model.model_id(), %error, "the image runner refused a call; the job ended");
                    }
                    let _ = events.send(JobEvent::Ended(end).into());
                }
            }
        })
        .await;
        if let Err(panicked) = ran {
            // The events' sender went with the thread: the request sees the
            // channel close before its last image.
            tracing::error!(name: "ignis.image.job_panicked", error = %panicked, "an image job panicked");
        }
    });
}

/// `image` composited on white, alpha 255: each colour `(c·a + 255·(255 −
/// a) + 127) / 255`, the blend rounded to the nearest byte.
pub fn on_white(image: &mut RgbaImage) {
    for px in image.pixels.chunks_exact_mut(4) {
        let alpha = u32::from(px[3]);
        for colour in &mut px[..3] {
            *colour = ((u32::from(*colour) * alpha + 255 * (255 - alpha) + 127) / 255) as u8;
        }
        px[3] = 255;
    }
}

/// `image` as an RGBA 8-bit PNG.
pub fn encode_png(image: &RgbaImage) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, image.size.width, image.size.height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // Into memory, from a buffer of the header's size: nothing to fail on.
    let mut writer = encoder.write_header().expect("a PNG header into memory");
    writer.write_image_data(&image.pixels).expect("an RGBA frame of the image's size");
    writer.finish().expect("a PNG into memory");
    out
}

/// `bytes` as standard, padded base64: the `b64_json` of a response.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | u32::from(b) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::image::{fake_plan, FakeImageRunner, ImagePlan};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    fn model(steps: usize) -> (ImageModel, Arc<std::sync::atomic::AtomicUsize>) {
        let plan = ImagePlan { condition_budget_tokens: 1 << 20, sample_sigmas: (0..steps).map(|k| 1.0 - k as f32 / steps as f32).collect(), ..fake_plan() };
        let runner = FakeImageRunner::new(plan);
        let held = runner.held_conditions();
        (ImageModel::new("ignis-image-2.1-turbo", Box::new(runner)), held)
    }

    fn job(n: u32, seed: u64, background: Background) -> ImageJob {
        ImageJob {
            condition: ImageCondition { prompt: "a red fox".into(), images: Vec::new() },
            size: ImageSize::new(64, 32),
            n,
            seed,
            background,
        }
    }

    /// Run `job` to its end, returning its events.
    fn events_of(model: &ImageModel, job: &ImageJob) -> (Result<(), JobEnd>, Vec<JobEvent>) {
        let (events, mut receiver) = mpsc::unbounded_channel();
        let result = run(model, job, &events);
        drop(events);
        let mut out = Vec::new();
        while let Ok(event) = receiver.try_recv() {
            out.push(event);
        }
        (result, out)
    }

    fn pngs(events: &[JobEvent]) -> Vec<Vec<u8>> {
        events.iter().filter_map(|event| if let JobEvent::Image { png, .. } = event { Some(png.clone()) } else { None }).collect()
    }

    fn decode(png: &[u8]) -> RgbaImage {
        crate::media::decode_condition_image(png, 64 << 20).expect("a PNG the job made")
    }

    /// AC 3 and AC 6: `n` RGBA PNGs from `seed + i`, the same seed the same
    /// bytes; the phases in order, one denoising event per step per image.
    #[test]
    fn a_job_makes_n_pngs_from_its_seed_announcing_every_unit() {
        let (model, held) = model(3);
        let (result, events) = events_of(&model, &job(2, 7, Background::Auto));
        assert_eq!(result, Ok(()));
        let images = pngs(&events);
        assert_eq!(images.len(), 2);
        assert_ne!(images[0], images[1], "each image its own noise");
        let (_, again) = events_of(&model, &job(2, 7, Background::Auto));
        assert_eq!(pngs(&again), images, "the same seed, the same bytes");
        let (_, shifted) = events_of(&model, &job(1, 8, Background::Auto));
        assert_eq!(pngs(&shifted)[0], images[1], "image 1 of seed 7 is image 0 of seed 8");
        let decoded = decode(&images[0]);
        assert_eq!(decoded.size, ImageSize::new(64, 32));
        assert!(decoded.pixels.chunks_exact(4).any(|px| px[3] != 255), "the alpha as generated");

        let units: Vec<(Phase, u32, u32, u32)> = events
            .iter()
            .filter_map(|event| if let JobEvent::Progress(p) = event { Some((p.phase, p.image, p.step, p.total)) } else { None })
            .collect();
        let mut expected = vec![(Phase::Encoding, 0, 1, 1)];
        for image in 0..2 {
            expected.extend((1..=3).map(|step| (Phase::Denoising, image, step, 3)));
            expected.push((Phase::Decoding, image, 1, 1));
        }
        assert_eq!(units, expected);
        assert!(matches!(events.last(), Some(JobEvent::Image { index: 1, .. })), "each image right after its decode");
        assert_eq!(held.load(Ordering::SeqCst), 0, "the prefix released");
    }

    /// AC 3: `background=opaque` is alpha 255 composited on white.
    #[test]
    fn an_opaque_background_is_composited_on_white() {
        let (model, _) = model(2);
        let auto = decode(&pngs(&events_of(&model, &job(1, 3, Background::Auto)).1)[0]);
        let opaque = decode(&pngs(&events_of(&model, &job(1, 3, Background::Opaque)).1)[0]);
        let mut expected = auto.clone();
        on_white(&mut expected);
        assert_eq!(opaque, expected);
        assert!(opaque.pixels.chunks_exact(4).all(|px| px[3] == 255));
        let mut pixel = RgbaImage { size: ImageSize::new(1, 1), pixels: vec![200, 0, 100, 128] };
        on_white(&mut pixel);
        // (200·128 + 255·127 + 127) / 255 = 227.9, (255·127 + 127) / 255 =
        // 127.5, (100·128 + 255·127 + 127) / 255 = 177.7, each truncated.
        assert_eq!(pixel.pixels, [227, 127, 177, 255]);
    }

    /// AC 7: a closed model stops the job between steps, a request gone
    /// cancels it, a runner that is gone stops it — and the prefix is
    /// released every time.
    #[test]
    fn a_job_stops_between_calls_and_releases_its_prefix() {
        let (model, held) = model(4);
        let (events, receiver) = mpsc::unbounded_channel::<JobEvent>();
        drop(receiver);
        assert_eq!(run(&model, &job(1, 1, Background::Auto), &events), Err(JobEnd::Cancelled));
        model.jobs.close();
        let (events, _receiver) = mpsc::unbounded_channel::<JobEvent>();
        assert_eq!(run(&model, &job(1, 1, Background::Auto), &events), Err(JobEnd::Stopped));
        assert_eq!(held.load(Ordering::SeqCst), 0);

        let (model, held) = model_with_budget(1);
        let (result, _) = events_of(&model, &job(1, 1, Background::Auto));
        assert!(matches!(result, Err(JobEnd::Failed(ImageError::OverBudget { .. }))), "{result:?}");
        assert_eq!(held.load(Ordering::SeqCst), 0);
        model.shut_down();
        assert_eq!(events_of(&model, &job(1, 1, Background::Auto)).0, Err(JobEnd::Stopped));
    }

    /// `FakeImageRunner`, running `at_step` once its `step`-th denoising
    /// step is done.
    struct AfterStep {
        inner: FakeImageRunner,
        step: usize,
        steps: usize,
        at_step: Option<Box<dyn FnOnce() + Send>>,
    }

    impl ImageRunner for AfterStep {
        fn plan(&self) -> &ImagePlan {
            self.inner.plan()
        }

        fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> u64 {
            self.inner.condition_tokens(prompt, images)
        }

        fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ignis_core::image::ConditionId, ImageError> {
            self.inner.encode_condition(condition)
        }

        fn denoise_step(
            &mut self,
            condition: ignis_core::image::ConditionId,
            latents: &mut TargetLatents,
            sigma: f32,
            sigma_next: f32,
            velocity: Option<&mut Vec<f32>>,
        ) -> Result<(), ImageError> {
            self.inner.denoise_step(condition, latents, sigma, sigma_next, velocity)?;
            self.steps += 1;
            if self.steps == self.step {
                if let Some(at_step) = self.at_step.take() {
                    at_step();
                }
            }
            Ok(())
        }

        fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
            self.inner.vae_decode(latents)
        }

        fn release_condition(&mut self, condition: ignis_core::image::ConditionId) {
            self.inner.release_condition(condition);
        }
    }

    /// Review-322b L6: a job whose request goes away, or whose model is
    /// closed, while its prefix is held — after step `k` of 4 — stops at its
    /// next call with nothing more made, and the prefix is released.
    #[test]
    fn a_job_stopped_mid_run_releases_the_prefix_it_holds() {
        for closes_queue in [false, true] {
            let inner = FakeImageRunner::new(ImagePlan { sample_sigmas: vec![1.0, 0.75, 0.5, 0.25], ..fake_plan() });
            let held = inner.held_conditions();
            let (events, receiver) = mpsc::unbounded_channel();
            let receiver = Arc::new(std::sync::Mutex::new(Some(receiver)));
            let queue: Arc<std::sync::OnceLock<Arc<super::super::ImageJobs>>> = Arc::default();
            let at_step: Box<dyn FnOnce() + Send> = if closes_queue {
                let queue = Arc::clone(&queue);
                Box::new(move || queue.get().expect("the model's queue").close())
            } else {
                let receiver = Arc::clone(&receiver);
                Box::new(move || drop(receiver.lock().unwrap().take()))
            };
            let runner = AfterStep { inner, step: 2, steps: 0, at_step: Some(at_step) };
            let model = ImageModel::new("ignis-image-2.1-turbo", Box::new(runner));
            queue.set(Arc::clone(&model.jobs)).unwrap();
            let ended = run(&model, &job(1, 1, Background::Auto), &events);
            assert_eq!(ended, Err(if closes_queue { JobEnd::Stopped } else { JobEnd::Cancelled }), "closes the queue: {closes_queue}");
            assert_eq!(held.load(Ordering::SeqCst), 0, "the prefix released (closes the queue: {closes_queue})");
            if let Some(mut receiver) = receiver.lock().unwrap().take() {
                let mut steps = 0;
                while let Ok(event) = receiver.try_recv() {
                    assert!(!matches!(event, JobEvent::Image { .. } | JobEvent::Progress(Progress { phase: Phase::Decoding, .. })), "{event:?}");
                    steps += usize::from(matches!(event, JobEvent::Progress(Progress { phase: Phase::Denoising, .. })));
                }
                assert_eq!(steps, 2, "no step announced past the close");
            }
        }
    }

    fn model_with_budget(budget: u64) -> (ImageModel, Arc<std::sync::atomic::AtomicUsize>) {
        let runner = FakeImageRunner::new(ImagePlan { condition_budget_tokens: budget, sample_sigmas: vec![1.0], ..fake_plan() });
        let held = runner.held_conditions();
        (ImageModel::new("ignis-image-2.1-turbo", Box::new(runner)), held)
    }

    #[test]
    fn base64_is_standard_and_padded() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(&[0xfb, 0xff, 0xbf]), "+/+/");
    }
}
