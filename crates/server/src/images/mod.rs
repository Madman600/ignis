//! Serving the image model (spec image-model/05, ADR 0050): the loaded image
//! model as the server holds it — its runner ([`ignis_core::image`]) and the
//! count of the image jobs on it.
//!
//! The image routes, the job queue, the condition budget and the PNG build on
//! this: the queue admits each job through [`ImageJobs::wait`] and runs it
//! through [`ImageModel::with_runner`], one at a time, so the idle clock and a
//! switch's drain read the jobs ([`ImageJobs::in_flight`]) as they read a text
//! model's lanes.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use ignis_core::image::{ImagePlan, ImageRunner};

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

    /// What its load planned: the condition budget and the sampler sigmas.
    pub fn plan(&self) -> &ImagePlan {
        &self.plan
    }

    /// Run `calls` on the runner, holding it: `None` once the model was shut
    /// down. A job holds it for as many calls as it makes in one go, and a
    /// shutdown waits for that.
    pub fn with_runner<T>(&self, calls: impl FnOnce(&mut dyn ImageRunner) -> T) -> Option<T> {
        let mut runner = self.runner.lock().expect("image runner lock");
        runner.as_mut().map(|runner| calls(runner.as_mut()))
    }

    /// Drop the runner, once whatever holds it lets go. Blocking; calling it
    /// twice is harmless.
    pub fn shut_down(&self) {
        drop(self.runner.lock().expect("image runner lock").take());
    }
}

/// The image jobs waiting and running on the loaded image model: what tells
/// the idle clock it is in use and a switch's drain when it is done (spec
/// image-model/05 §The queue). The queue counts each job here, from its
/// admission until its last image, and stops admitting once it is closed.
#[derive(Debug, Default)]
pub struct ImageJobs {
    waiting: AtomicU32,
    running: AtomicU32,
    closed: AtomicBool,
}

impl ImageJobs {
    /// One job admitted: it waits until [`Waiting::start`], and leaves the
    /// count when the guard is dropped — run, refused or cancelled alike.
    pub fn wait(self: &Arc<Self>) -> Waiting {
        self.waiting.fetch_add(1, Ordering::SeqCst);
        Waiting(Arc::clone(self))
    }

    /// The jobs waiting and running now.
    pub fn in_flight(&self) -> InFlight {
        InFlight { waiting: self.waiting.load(Ordering::SeqCst), running: self.running.load(Ordering::SeqCst) }
    }

    /// The model is going away (a switch, an unload): the queue admits no job
    /// from now, and cancels the ones it has between denoising steps.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Whether [`ImageJobs::close`] was called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// A job counted as waiting ([`ImageJobs::wait`]), until dropped or started.
#[derive(Debug)]
pub struct Waiting(Arc<ImageJobs>);

impl Waiting {
    /// The job runs: counted as running until the returned guard is dropped,
    /// and never as neither in between.
    pub fn start(self) -> Running {
        self.0.running.fetch_add(1, Ordering::SeqCst);
        Running(Arc::clone(&self.0))
    }
}

impl Drop for Waiting {
    fn drop(&mut self) {
        self.0.waiting.fetch_sub(1, Ordering::SeqCst);
    }
}

/// A job counted as running ([`Waiting::start`]), until dropped.
#[derive(Debug)]
pub struct Running(Arc<ImageJobs>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.running.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::image::{FakeImageRunner, ImageCondition};

    /// A job is counted from its admission until its guard goes, waiting then
    /// running, and is never between the two.
    #[test]
    fn a_job_counts_as_waiting_then_running_until_its_guard_goes() {
        let jobs = Arc::new(ImageJobs::default());
        let (first, second) = (jobs.wait(), jobs.wait());
        assert_eq!(jobs.in_flight(), InFlight { waiting: 2, running: 0 });
        let running = first.start();
        assert_eq!(jobs.in_flight(), InFlight { waiting: 1, running: 1 });
        drop(second);
        drop(running);
        assert!(jobs.in_flight().is_idle());
        assert!(!jobs.is_closed());
        jobs.close();
        assert!(jobs.is_closed());
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
