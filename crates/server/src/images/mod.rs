//! Serving the image model (spec image-model/05, ADR 0050): the loaded image
//! model as the server holds it — its runner ([`ignis_core::image`]) and the
//! count of the image jobs on it.
//!
//! The image routes ([`routes`]), the job and its queue ([`job`]), the
//! condition budget ([`budget`]) and the PNG build on this: the queue admits
//! each job through [`ImageJobs::admit`] and runs it in its turn
//! ([`Waiting::turn`]) through [`ImageModel::with_runner`], one at a time,
//! so the idle clock and a switch's drain read the jobs
//! ([`ImageJobs::in_flight`]) as they read a text model's lanes.

pub mod budget;
pub mod enhance;
pub mod job;
pub mod leaf;
mod prompts;
pub(crate) mod routes;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use ignis_core::image::{ImagePlan, ImageRunner, ImageSize};

use crate::active::InFlight;

/// The loaded image model: its served id, what its load planned, its runner
/// and its jobs. Cloning shares all of them, like a clone of an engine.
#[derive(Clone)]
pub struct ImageModel {
    model: String,
    plan: ImagePlan,
    /// The runner until the model is shut down ([`ImageModel::shut_down`]),
    /// which drops it — and every device buffer of its leaf — before a switch
    /// loads the next model: the image model's counterpart of joining a text
    /// model's thread. Behind a lock because its calls are serialized anyway
    /// (one job at a time); a call made after the shutdown finds nothing.
    runner: Arc<Mutex<Option<Box<dyn ImageRunner>>>>,
    /// The jobs waiting and running on it.
    pub jobs: Arc<ImageJobs>,
}

impl ImageModel {
    /// The image model served as `model`, computing with `runner`.
    pub fn new(model: impl Into<String>, runner: Box<dyn ImageRunner>) -> Self {
        Self {
            model: model.into(),
            plan: runner.plan().clone(),
            runner: Arc::new(Mutex::new(Some(runner))),
            jobs: Arc::default(),
        }
    }

    /// The id it is served under.
    pub fn model_id(&self) -> String {
        self.model.clone()
    }

    /// What its load planned: the condition budget, the sampler sigmas and
    /// the largest target.
    pub fn plan(&self) -> &ImagePlan {
        &self.plan
    }

    /// Whether `other` is this model (a clone of it), not another load.
    pub fn is(&self, other: &ImageModel) -> bool {
        Arc::ptr_eq(&self.jobs, &other.jobs)
    }

    /// Run `calls` on the runner, holding it: `None` once the model was shut
    /// down. A job holds it for as many calls as it makes in one go, and a
    /// shutdown waits for that. A call that panicked leaves the lock
    /// poisoned, not the runner unreachable: the job's prefix is still
    /// released, and the shutdown still drops it — which the job does at
    /// once (`job::spawn`).
    pub fn with_runner<T>(&self, calls: impl FnOnce(&mut dyn ImageRunner) -> T) -> Option<T> {
        let mut runner = self.runner.lock().unwrap_or_else(PoisonError::into_inner);
        runner.as_mut().map(|runner| calls(runner.as_mut()))
    }

    /// Drop the runner, once whatever holds it lets go. Blocking; calling it
    /// twice is harmless.
    pub fn shut_down(&self) {
        drop(self.runner.lock().unwrap_or_else(PoisonError::into_inner).take());
    }
}

/// Of `sizes` (`image.sizes`), the ones a load that planned `plan` makes
/// ([`ImagePlan::takes`]): a size past its largest target is not served,
/// whatever the config names — its load warns of it, and a request for it is
/// refused.
pub fn served_sizes(sizes: &[ImageSize], plan: &ImagePlan) -> Vec<ImageSize> {
    sizes.iter().copied().filter(|&size| plan.takes(size)).collect()
}

/// The image jobs waiting and running on the loaded image model: what tells
/// the idle clock it is in use and a switch's drain when it is done, and the
/// queue that runs them (spec image-model/05 §The queue): one at a time, in
/// the order they were admitted. The queue counts each job here, from its
/// admission until its last image, and stops admitting once it is closed.
#[derive(Debug, Default)]
pub struct ImageJobs {
    state: Mutex<Queue>,
    /// Told whenever the turn may have passed: a job ended, a waiting one
    /// left, the queue closed.
    changed: tokio::sync::Notify,
    closed: AtomicBool,
    /// Where the queue's depth, and the condition prefixes' bytes its jobs
    /// hold, are published ([`ImageJobs::report_to`]).
    gauge: std::sync::OnceLock<Arc<crate::metrics::Metrics>>,
}

/// The queue under its lock.
#[derive(Debug, Default)]
struct Queue {
    /// The tickets of the jobs waiting, in the order they were admitted: the
    /// order is fixed there, whenever each job then asks for its turn.
    waiting: VecDeque<u64>,
    running: u32,
    next: u64,
}

impl ImageJobs {
    fn queue(&self) -> std::sync::MutexGuard<'_, Queue> {
        self.state.lock().expect("image queue lock")
    }

    /// From now, every change of the queue sets `metrics`' image queue
    /// gauge, under the queue's lock: the last change to land is the last
    /// the gauge shows, whichever way a job leaves (GitHub #324).
    pub fn report_to(&self, metrics: &Arc<crate::metrics::Metrics>) {
        let _ = self.gauge.set(Arc::clone(metrics));
    }

    /// The bytes of the condition prefixes `runner` holds now, to the
    /// gauge (GitHub #324). Called with the runner held, right after an
    /// encode or a release: the last figure a runner gives is the last the
    /// gauge shows, and none lands once its model was shut down and the next
    /// load's reservations reset it.
    pub fn publish_conditions(&self, runner: &dyn ImageRunner) {
        if let Some(metrics) = self.gauge.get() {
            metrics.set_condition_budget_used(runner.condition_bytes());
        }
    }

    /// The queue as it stands, to the gauge. Called with the lock held.
    fn publish(&self, queue: &Queue) {
        if let Some(metrics) = self.gauge.get() {
            metrics.set_image_queue(queue.waiting.len() as u32, queue.running);
        }
    }

    /// One job counted as waiting, at the back of the queue, whatever the
    /// depth ([`ImageJobs::admit`] without its bound): it waits until
    /// [`Waiting::turn`], and leaves the count when the guard is dropped —
    /// run, refused or cancelled alike. The queue must be open.
    pub fn wait(self: &Arc<Self>) -> Waiting {
        self.admit(u32::MAX).expect("an open queue admits without a bound")
    }

    /// One job admitted to the back of the queue, counted as waiting, when
    /// at most `depth` (`image.queue_depth`) would then wait behind the one
    /// running; `None` when the queue is full or closed.
    pub fn admit(self: &Arc<Self>, depth: u32) -> Option<Waiting> {
        let mut queue = self.queue();
        if self.is_closed() || queue.waiting.len() as u64 + u64::from(queue.running) > u64::from(depth) {
            return None;
        }
        let ticket = queue.next;
        queue.next += 1;
        queue.waiting.push_back(ticket);
        self.publish(&queue);
        Some(Waiting { jobs: Some(Arc::clone(self)), ticket })
    }

    /// The jobs waiting and running now.
    pub fn in_flight(&self) -> InFlight {
        let queue = self.queue();
        InFlight { waiting: queue.waiting.len() as u32, running: queue.running }
    }

    /// The model is going away (a switch, an unload): the queue admits no job
    /// from now, the waiting ones never get their turn, and the running one
    /// is cancelled between denoising steps.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.changed.notify_waiters();
    }

    /// Whether [`ImageJobs::close`] was called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// A job counted as waiting ([`ImageJobs::wait`], [`ImageJobs::admit`]),
/// until dropped or its turn comes.
#[derive(Debug)]
pub struct Waiting {
    jobs: Option<Arc<ImageJobs>>,
    ticket: u64,
}

impl Waiting {
    /// Wait for the job's turn — no job running, and every job admitted
    /// before it run or gone — and start it: counted as running until the
    /// returned guard is dropped, and never as neither nor both in between.
    /// `None` once the queue is closed. Dropping the future leaves the queue.
    pub async fn turn(mut self) -> Option<Running> {
        let jobs = Arc::clone(self.jobs.as_ref().expect("a waiting job"));
        loop {
            // Asked before the state is read, so no change between the two
            // goes unseen.
            let changed = jobs.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            {
                let mut queue = jobs.queue();
                if jobs.is_closed() {
                    return None;
                }
                if queue.running == 0 && queue.waiting.front() == Some(&self.ticket) {
                    queue.waiting.pop_front();
                    queue.running += 1;
                    jobs.publish(&queue);
                    self.jobs = None;
                    drop(queue);
                    return Some(Running(Arc::clone(&jobs)));
                }
            }
            changed.await;
        }
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        if let Some(jobs) = &self.jobs {
            let mut queue = jobs.queue();
            queue.waiting.retain(|&ticket| ticket != self.ticket);
            jobs.publish(&queue);
            drop(queue);
            jobs.changed.notify_waiters();
        }
    }
}

/// A job counted as running ([`Waiting::turn`]), until dropped.
#[derive(Debug)]
pub struct Running(Arc<ImageJobs>);

impl Drop for Running {
    fn drop(&mut self) {
        let mut queue = self.0.queue();
        queue.running -= 1;
        self.0.publish(&queue);
        drop(queue);
        self.0.changed.notify_waiters();
    }
}

/// The edits decoding their condition images now: at most
/// `image.max_condition_decodes` at once (review-322b F2), since each may
/// decode `image.max_condition_images` frames of up to
/// `image.max_condition_mipx` before the queue sees it, and the queue behind
/// runs one job at a time anyway — decodes side by side buy memory, not
/// throughput.
#[derive(Debug, Default)]
pub struct Decodes {
    running: Mutex<u32>,
    /// Told whenever a decode ends.
    ended: tokio::sync::Notify,
}

impl Decodes {
    /// A decode's turn, once fewer than `limit` run — the limit as the caller
    /// reads it now, so a live change applies from the next decode that
    /// begins or ends. Counted until the guard is dropped: moved into the
    /// blocking task that decodes, a request that goes away mid-decode does
    /// not free it before the decode ends.
    pub async fn enter(self: &Arc<Self>, limit: u32) -> Decoding {
        loop {
            // Asked before the count is read, so no end between the two goes
            // unseen.
            let ended = self.ended.notified();
            tokio::pin!(ended);
            ended.as_mut().enable();
            {
                let mut running = self.running.lock().expect("decodes lock");
                if *running < limit.max(1) {
                    *running += 1;
                    return Decoding(Arc::clone(self));
                }
            }
            ended.await;
        }
    }

    /// The decodes running now.
    pub fn running(&self) -> u32 {
        *self.running.lock().expect("decodes lock")
    }
}

/// A decode counted as running ([`Decodes::enter`]), until dropped.
#[derive(Debug)]
pub struct Decoding(Arc<Decodes>);

impl Drop for Decoding {
    fn drop(&mut self) {
        *self.0.running.lock().expect("decodes lock") -= 1;
        self.0.ended.notify_waiters();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::image::{ConditionId, FakeImageRunner, ImageCondition, ImageError, RgbaImage, TargetLatents};

    /// A job is counted from its admission until its guard goes, waiting then
    /// running, and is never between the two.
    #[tokio::test]
    async fn a_job_counts_as_waiting_then_running_until_its_guard_goes() {
        let jobs = Arc::new(ImageJobs::default());
        let (first, second) = (jobs.wait(), jobs.wait());
        assert_eq!(jobs.in_flight(), InFlight { waiting: 2, running: 0 });
        let running = first.turn().await.expect("the first's turn at once");
        assert_eq!(jobs.in_flight(), InFlight { waiting: 1, running: 1 });
        drop(second);
        drop(running);
        assert!(jobs.in_flight().is_idle());
        assert!(!jobs.is_closed());
        jobs.close();
        assert!(jobs.is_closed());
    }

    /// Spec image-model/05 §The queue: `depth` jobs wait behind the running
    /// one and one more is refused; a closed queue admits none.
    #[tokio::test]
    async fn the_queue_admits_up_to_its_depth_behind_the_running_job() {
        let jobs = Arc::new(ImageJobs::default());
        let running = jobs.admit(2).expect("an empty queue admits").turn().await.expect("its turn at once");
        let waiting: Vec<Waiting> = (0..2).map(|_| jobs.admit(2).expect("room to wait")).collect();
        assert_eq!(jobs.in_flight(), InFlight { waiting: 2, running: 1 });
        assert!(jobs.admit(2).is_none(), "a third would wait past the depth");
        assert_eq!(jobs.in_flight(), InFlight { waiting: 2, running: 1 }, "a refusal counts nothing");
        drop(running);
        let again = jobs.admit(2);
        assert!(again.is_some(), "room again");
        drop((again, waiting));
        let one = jobs.admit(0).expect("depth 0 runs one").turn().await;
        assert!(one.is_some(), "depth 0 runs one");
        assert!(jobs.admit(0).is_none(), "with none waiting");
        drop(one);
        jobs.close();
        assert!(jobs.admit(4).is_none(), "a closed queue admits nothing");
        assert!(jobs.in_flight().is_idle());
    }

    /// The turn goes one at a time in the order of admission, whenever each
    /// job asks for it; a job that leaves the queue gives its place up; a
    /// closed queue gives no turn.
    #[tokio::test]
    async fn the_turn_goes_one_at_a_time_in_the_order_of_admission() {
        let jobs = Arc::new(ImageJobs::default());
        let first = jobs.admit(8).unwrap().turn().await.expect("an empty queue's turn at once");
        let tickets: Vec<Waiting> = (0..4).map(|_| jobs.admit(8).unwrap()).collect();
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waiters = Vec::new();
        // They ask for their turn in the reverse order, and the second leaves.
        for (job, waiting) in tickets.into_iter().enumerate().rev() {
            if job == 1 {
                drop(waiting);
                continue;
            }
            let order = Arc::clone(&order);
            waiters.push(tokio::spawn(async move {
                let running = waiting.turn().await.expect("an open queue");
                order.lock().unwrap().push(job);
                tokio::task::yield_now().await;
                drop(running);
            }));
            tokio::task::yield_now().await;
        }
        assert_eq!(jobs.in_flight(), InFlight { waiting: 3, running: 1 });
        drop(first);
        for waiter in waiters {
            waiter.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), [0, 2, 3]);
        assert!(jobs.in_flight().is_idle());

        let running = jobs.admit(8).unwrap().turn().await.expect("an empty queue's turn at once");
        let waiting = jobs.admit(8).unwrap();
        let closed = tokio::spawn(async move { waiting.turn().await.is_none() });
        tokio::task::yield_now().await;
        jobs.close();
        assert!(closed.await.unwrap(), "a waiting job gets no turn once closed");
        drop(running);
        assert!(jobs.in_flight().is_idle());
    }

    /// Review-322cf L1: at most `limit` decodes run, read as each one asks,
    /// and one waiting gets its turn when one ends.
    #[tokio::test]
    async fn decodes_run_up_to_the_limit_each_one_reads() {
        let decodes = Arc::new(Decodes::default());
        let (first, second) = (decodes.enter(2).await, decodes.enter(2).await);
        assert_eq!(decodes.running(), 2);
        let third = tokio::spawn({
            let decodes = Arc::clone(&decodes);
            async move { decodes.enter(2).await }
        });
        tokio::task::yield_now().await;
        assert!(!third.is_finished(), "a third waits at a limit of 2");
        drop(first);
        let third = third.await.unwrap();
        assert_eq!(decodes.running(), 2);
        let fourth = tokio::time::timeout(std::time::Duration::from_secs(5), decodes.enter(3)).await.expect("room at a limit of 3");
        drop((second, third, fourth));
        assert_eq!(decodes.running(), 0);
        let _one = decodes.enter(0).await;
        assert_eq!(decodes.running(), 1, "a limit of 0 still lets one run");
    }

    /// Spec image-model/05 §Sizes: the fake's plan makes every default size,
    /// the checkpoint's largest preset included; a smaller plan drops what is
    /// past it and keeps the order of the rest.
    #[test]
    fn the_sizes_served_are_the_configs_that_the_plan_takes() {
        let defaults = crate::config::DEFAULT_IMAGE_SIZES;
        assert_eq!(served_sizes(&defaults, &ignis_core::image::fake_plan()), defaults);
        let square = ImagePlan { max_target_cells: ImageSize::new(2048, 2048).latent_cells(), ..ignis_core::image::fake_plan() };
        assert_eq!(served_sizes(&defaults, &square), [ImageSize::new(2048, 2048), ImageSize::new(1024, 1024)]);
        let huge = [ImageSize::new(8192, 8192), ImageSize::new(1024, 1024)];
        assert_eq!(served_sizes(&huge, &ignis_core::image::fake_plan()), [ImageSize::new(1024, 1024)]);
    }

    /// `FakeImageRunner`, but every denoising step panics.
    struct PanicsOnStep(FakeImageRunner);

    impl ImageRunner for PanicsOnStep {
        fn plan(&self) -> &ImagePlan {
            self.0.plan()
        }

        fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ignis_core::image::ConditionUnfit> {
            self.0.condition_tokens(prompt, images)
        }

        fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
            self.0.encode_condition(condition)
        }

        fn denoise_step(&mut self, _: ConditionId, _: &mut TargetLatents, _: f32, _: f32, _: Option<&mut Vec<f32>>) -> Result<(), ImageError> {
            panic!("a leaf fault");
        }

        fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
            self.0.vae_decode(latents)
        }

        fn release_condition(&mut self, condition: ConditionId) {
            self.0.release_condition(condition);
        }

        fn condition_bytes(&self) -> u64 {
            self.0.condition_bytes()
        }
    }

    /// A runner call that panics leaves the lock poisoned, not the runner
    /// stuck: the job's prefix is still released through it, and the
    /// shutdown still drops it — and every buffer of its leaf with it.
    #[test]
    fn a_runner_call_that_panics_neither_keeps_its_prefix_nor_survives_the_shutdown() {
        let inner = FakeImageRunner::default();
        let held = inner.held_conditions();
        let alive = Arc::downgrade(&held);
        let model = ImageModel::new("ignis-image-2.1-turbo", Box::new(PanicsOnStep(inner)));
        let condition = ImageCondition { prompt: "a cat".into(), images: Vec::new() };
        let id = model.with_runner(|runner| runner.encode_condition(&condition)).unwrap().unwrap();
        let mut latents = TargetLatents::zeros(ImageSize::new(32, 32));
        let step = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.with_runner(|runner| runner.denoise_step(id, &mut latents, 1.0, 0.0, None))
        }));
        assert!(step.is_err(), "the step panicked");
        model.with_runner(|runner| runner.release_condition(id)).expect("the runner is still there");
        assert_eq!(held.load(std::sync::atomic::Ordering::SeqCst), 0, "the prefix released");
        drop(held);
        model.shut_down();
        assert_eq!(alive.strong_count(), 0, "the runner was dropped");
        assert!(model.with_runner(|_| ()).is_none());
    }

    /// The shutdown drops the runner, and a call after it finds none.
    #[test]
    fn a_shut_down_model_has_no_runner() {
        let model = ImageModel::new("ignis-image-2.1-turbo", Box::new(FakeImageRunner::default()));
        let condition = ImageCondition { prompt: "a cat".into(), images: Vec::new() };
        assert!(model.with_runner(|runner| runner.encode_condition(&condition)).is_some());
        assert_eq!(model.plan().sample_sigmas.len(), 8);
        model.clone().shut_down();
        assert!(model.with_runner(|runner| runner.encode_condition(&condition)).is_none());
        model.shut_down();
        assert_eq!(model.model_id(), "ignis-image-2.1-turbo");
    }
}
