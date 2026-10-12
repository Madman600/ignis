//! A [`ModelLoader`] over `MockCompute` for the model-switch tests (spec
//! model-switch/01), included via `#[path]` by `model_switch.rs` and
//! `model_switch_http.rs`.
//!
//! Every model it builds — the server's first one included — runs on an
//! engine started with `Engine::with_clock_and_driver`, its driver handle
//! kept on the [`ActiveModel`], so a switch's teardown really joins the
//! model thread. And it keeps a weak handle on every compute it built: a
//! compute is alive exactly as long as the scheduler that owns it, so the
//! count of live ones when a load begins is how many models were still
//! resident then — the mock's proof that the old model was gone first.

#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, Weak};

use ignis_core::compute::ModelFamily;
use ignis_core::mock::{GateController, GatedCompute, MockCompute};
use ignis_core::{Compute, ConcreteScheduler, SchedulerConfig};
use ignis_server::config_http::ConfigState;
use ignis_server::engine::Engine;
use ignis_server::model_switch::{load_options, ModelLoader, PreparedLoad};
use ignis_server::load::LoadedModel;
use ignis_server::telemetry::SystemClock;
use ignis_server::template::SimpleTemplateProvider;
use ignis_server::{ActiveModel, ModelSource};

/// The loader's shared state: what its prepared loads also reach.
#[derive(Default)]
struct State {
    /// Every compute built, weakly.
    built: Mutex<Vec<Weak<dyn Compute>>>,
    /// How many built models were still alive as each load began.
    resident_at_load: Mutex<Vec<usize>>,
    /// The ids loaded, in order.
    loads: Mutex<Vec<String>>,
    /// Ids [`ModelLoader::prepare`] refuses.
    refused: Mutex<HashSet<String>>,
    /// Ids whose load fails after prepare accepted them.
    broken: Mutex<HashSet<String>>,
    /// Each load takes the first of these and waits for a message on it
    /// before it builds: how a test holds a switch inside its load step (a
    /// second one holds the reload that follows a failed load).
    hold: Mutex<std::collections::VecDeque<Receiver<()>>>,
    /// The configuration every prepared load reports as the one it runs
    /// with ([`PreparedLoad::options`]), as the production loader does:
    /// what makes a switch re-apply the server-level values once a model
    /// serves.
    serving: Mutex<Option<ignis_server::config::Config>>,
    /// The options each prepare was handed (spec config-v2/02: a reload
    /// with a changed config hands its patched config; a switch none).
    options: Mutex<Vec<Option<ignis_server::config::Config>>>,
    /// The artifact each prepare was asked to load, in order.
    artifacts: Mutex<Vec<std::path::PathBuf>>,
    /// The server's configuration, and the family each id's artifact is
    /// ([`MockLoader::fit_from`]): when set, every prepare fits the options
    /// its load runs with from the configuration running at that moment, by
    /// the production loader's own function.
    fitting: Mutex<Option<(Arc<ConfigState>, Vec<(String, ModelFamily)>)>>,
    /// The options each fitted load runs with, in order.
    fitted: Mutex<Vec<ignis_server::config::Config>>,
}

/// The mock loader. Clone the `Arc` into a `Switcher`, keep one to steer it.
#[derive(Default)]
pub struct MockLoader {
    state: Arc<State>,
}

/// Where a mock model `id` "comes from": what a switch to it names.
pub fn source(id: &str) -> ModelSource {
    ModelSource { artifact: format!("{id}.ninfer").into(), model: id.to_owned() }
}

impl MockLoader {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// A model served as `id`, built as a switch would build it (driver
    /// kept, source recorded) — the server's first model.
    pub fn model(&self, id: &str) -> ActiveModel {
        build(&self.state, id, Arc::new(MockCompute::new()))
    }

    /// [`MockLoader::model`] over a gated compute (spec server/05's
    /// primitive): arm it, then drop the `Arc` unless the test needs to arm
    /// it again — a test holding it keeps the model "resident".
    pub fn gated_model(&self, id: &str) -> (ActiveModel, Arc<GatedCompute>, GateController) {
        let (gated, controller) = GatedCompute::new(Arc::new(MockCompute::new()));
        let model = build(&self.state, id, Arc::clone(&gated) as Arc<dyn Compute>);
        (model, gated, controller)
    }

    /// Refuse `id` at prepare, as a path with no file would be.
    pub fn refuse(&self, id: &str) {
        self.state.refused.lock().unwrap().insert(id.to_owned());
    }

    /// Fail `id`'s load after prepare accepted it, as a kernel load error
    /// would.
    pub fn break_load(&self, id: &str) {
        self.state.broken.lock().unwrap().insert(id.to_owned());
    }

    /// Let `id` load again ([`MockLoader::break_load`] undone): the way a
    /// test proves a later attempt is made, and can succeed.
    pub fn mend(&self, id: &str) {
        self.state.broken.lock().unwrap().remove(id);
    }

    /// Hold the next load until the returned sender sends (or is dropped).
    /// Called again before that load runs, it holds the one after.
    pub fn hold_next_load(&self) -> Sender<()> {
        let (release, hold) = channel();
        self.state.hold.lock().unwrap().push_back(hold);
        release
    }

    /// Have every prepared load report `config` as the one it runs with, as
    /// the production loader does.
    pub fn serve_with(&self, config: ignis_server::config::Config) {
        *self.state.serving.lock().unwrap() = Some(config);
    }

    /// Fit every load's options from `config`, the server's running
    /// configuration, as the production loader does
    /// ([`ignis_server::model_switch::load_options`]); an id `families` does
    /// not name is a 27B. What proves a value set for a family not loaded
    /// reaches that family's load, not only a fixed configuration
    /// ([`MockLoader::serve_with`]) handed back.
    pub fn fit_from(&self, config: Arc<ConfigState>, families: &[(&str, ModelFamily)]) {
        let families = families.iter().map(|(id, family)| (id.to_string(), *family)).collect();
        *self.state.fitting.lock().unwrap() = Some((config, families));
    }

    /// The options each fitted load ran with, in order.
    pub fn fitted_options(&self) -> Vec<ignis_server::config::Config> {
        self.state.fitted.lock().unwrap().clone()
    }

    /// How many built models were alive as each load began.
    pub fn resident_at_load(&self) -> Vec<usize> {
        self.state.resident_at_load.lock().unwrap().clone()
    }

    /// The ids loaded, in order.
    pub fn loads(&self) -> Vec<String> {
        self.state.loads.lock().unwrap().clone()
    }

    /// How many built models are alive now.
    pub fn resident(&self) -> usize {
        resident(&self.state)
    }

    /// The artifact each prepare was asked to load, in order.
    pub fn prepared_artifacts(&self) -> Vec<std::path::PathBuf> {
        self.state.artifacts.lock().unwrap().clone()
    }

    /// The options each prepare was handed, in order.
    pub fn prepared_options(&self) -> Vec<Option<ignis_server::config::Config>> {
        self.state.options.lock().unwrap().clone()
    }
}

fn resident(state: &State) -> usize {
    state.built.lock().unwrap().iter().filter(|built| built.strong_count() > 0).count()
}

fn build(state: &State, id: &str, compute: Arc<dyn Compute>) -> ActiveModel {
    state.built.lock().unwrap().push(Arc::downgrade(&compute));
    let scheduler =
        ConcreteScheduler::with_config(SchedulerConfig { model: id.into(), ..SchedulerConfig::default() }, compute);
    let (engine, driver) = Engine::with_clock_and_driver(Box::new(scheduler), Arc::new(SystemClock));
    ActiveModel::new(engine, Arc::new(SimpleTemplateProvider)).with_driver(driver).with_source(source(id))
}

impl ModelLoader for MockLoader {
    fn prepare(&self, target: &ModelSource, options: Option<&ignis_server::config::Config>) -> Result<Box<dyn PreparedLoad>, String> {
        self.state.options.lock().unwrap().push(options.cloned());
        self.state.artifacts.lock().unwrap().push(target.artifact.clone());
        if self.state.refused.lock().unwrap().contains(&target.model) {
            return Err(format!("{} refused at prepare (simulated)", target.model));
        }
        let fitted = match &*self.state.fitting.lock().unwrap() {
            Some((config, families)) => {
                let family = families.iter().find(|(id, _)| *id == target.model).map_or(ModelFamily::Qwen38_27b, |(_, family)| *family);
                let fitted = load_options(&config.current(), target, family, options)?;
                self.state.fitted.lock().unwrap().push(fitted.clone());
                Some(fitted)
            }
            None => None,
        };
        let options = fitted.or_else(|| self.state.serving.lock().unwrap().clone());
        Ok(Box::new(MockPrepared { state: Arc::clone(&self.state), id: target.model.clone(), options }))
    }
}

struct MockPrepared {
    state: Arc<State>,
    id: String,
    options: Option<ignis_server::config::Config>,
}

impl PreparedLoad for MockPrepared {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        let hold = self.state.hold.lock().unwrap().pop_front();
        if let Some(hold) = hold {
            let _ = hold.recv();
        }
        self.state.resident_at_load.lock().unwrap().push(resident(&self.state));
        self.state.loads.lock().unwrap().push(self.id.clone());
        if self.state.broken.lock().unwrap().contains(&self.id) {
            return Err(format!("{} failed to load (simulated kernel load error)", self.id));
        }
        Ok(LoadedModel { model: build(&self.state, &self.id, Arc::new(MockCompute::new())), reservations: None })
    }

    fn options(&self) -> Option<&ignis_server::config::Config> {
        self.options.as_ref()
    }
}
