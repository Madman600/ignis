//! The runtime model switch (spec model-switch/01, GitHub #305): one process
//! serves the 27B or Qwen3.8-Flash-Next, and `POST /v1/models/switch` moves
//! it from one to the other without a restart.
//!
//! A switch is a **full reload**: the old model's VRAM, pinned host memory,
//! KV-RAM and retained slots are all freed before the new model allocates
//! anything, so the two are never resident together and the switch itself
//! spends no VRAM. In order, on one spawned task ([`begin`]):
//!
//! 1. **Gate** — [`ModelStatus::Switching`] is stored, and the `/v1` routes
//!    answer `503 model_switching` (`crate::api`).
//! 2. **Prepare** — everything that can refuse the target without the GPU
//!    ([`ModelLoader::prepare`]: the file, its checksum, its model, the flags
//!    it takes). A refusal here touches nothing: the old model never stopped.
//! 3. **Drain** — the old engine's in-flight counts (the same wait-free
//!    snapshot `/metrics` reads) are polled until nothing is waiting or
//!    running, or `--switch-drain-timeout` passes.
//! 4. **Tear down** — the old model thread is told to shut down (whatever is
//!    still running ends with an error) and joined: only once the join
//!    returns has its scheduler — and every GPU buffer and pinned KV-RAM
//!    arena its leaf owns — been dropped.
//! 5. **Load** — the target is loaded on the freed card and warmed up, then
//!    published whole ([`crate::Server::active`]), and the API serves again.
//!
//! **Why tear down before loading.** The spec's first draft built the new
//! model before releasing the old one, so a failed load could fall back to a
//! model that never stopped. On the card that cannot work at all, in either
//! direction: every load sizes its VRAM plan from the memory NVML reports
//! free at its start (`crate::runtime`), which the old model still holds, so
//! the new plan would either refuse or be built around the old weights — and
//! the two models resident together is what the spec rules out. On top of
//! that, the 27B's KV-RAM arena is a process-wide singleton whose create
//! refuses while one exists ("destroy it before creating another",
//! `kernel/src/seq.cu`; Flash-Next's arena is its own instance's). So the
//! failure contract is kept the other way round: a target that can be refused
//! without the GPU is refused in step 2, before anything stops; a load that
//! fails on the GPU after the teardown reloads the **old** model from the
//! artifact it came from ([`crate::ModelSource`]), so a bad switch degrades
//! to a cold restart of what was serving, never to a stopped server. Only a
//! reload that fails too leaves [`ModelStatus::Failed`] standing, with the
//! process up and a further switch still accepted.
//!
//! **Who may begin one.** `POST /v1/models/switch`, naming the artifact; and
//! (spec §Implicit switch) a request whose own `model` names another model the
//! operator listed ([`implicit_switch`]), which waits for that same switch and
//! is then served on the model it named. Both run [`begin`]: one mechanism,
//! two callers.
//!
//! **Idle unload** (spec model-switch/02, ADR 0049). The idle clock
//! (`crate::idle`) tears a model nobody uses down through this module's own
//! slot and teardown (`claim_unload`, `unload`): the status is
//! [`ModelStatus::Unloaded`], the [`ActiveModel`] is kept — its id and its
//! source — and its engine is never called again. Leaving that state is a
//! load and nothing more, [`ModelStatus::Loading`]: whatever begins it — a
//! request that needs the model ([`ensure_loaded`]), `POST
//! /v1/models/switch`, a reloading `PATCH /v1/config` — runs `run` with no
//! drain and no teardown (nothing is resident), and a load that fails leaves
//! [`ModelStatus::Failed`], since there is no previous model to reload. An
//! unload and a switch, a reload or a load never overlap: they share
//! [`Switcher::running`].

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::engine::Engine;
use crate::load::LoadedModel;
use crate::{ActiveModel, ModelSource, ModelStatus, Server, StatusCell};

/// How often the drain reads the old engine's in-flight counts. A switch is
/// an operator's action, seconds long; 10 ms of latency on noticing the last
/// request left is nothing beside the load that follows.
const DRAIN_POLL: Duration = Duration::from_millis(10);

/// Loads models for a switch — the seam the tests replace (ADR 0006: the
/// orchestration is proven on `MockCompute`, the GPU load by its own
/// `#[ignore]`d test).
pub trait ModelLoader: Send + Sync {
    /// Everything that can refuse `target` without touching the GPU, done
    /// while the old model is still serving; the error says why. Blocking
    /// (it reads and verifies files): called on `spawn_blocking`.
    ///
    /// `options` is the configuration to load with when a reload changes it
    /// (`PATCH /v1/config`, spec config-v2/02: already patched and validated,
    /// fitted to the family it reloads); `None` is the running one, fitted
    /// to `target`'s family as a switch fits it.
    fn prepare(&self, target: &ModelSource, options: Option<&crate::config::Config>) -> Result<Box<dyn PreparedLoad>, String>;
}

/// A target [`ModelLoader::prepare`] accepted: what is left is the load
/// itself, on a card the old model has already left.
pub trait PreparedLoad: Send {
    /// Load the model and start its engine. Blocking for the length of a GPU
    /// load: called on `spawn_blocking`, inside the runtime (the engine
    /// spawns its telemetry consumer).
    fn load(self: Box<Self>) -> Result<LoadedModel, String>;

    /// The configuration the load runs with, fitted to its family: what
    /// `GET /v1/config` shows once it serves. `None` for a loader that keeps
    /// none (the tests' mock).
    fn options(&self) -> Option<&crate::config::Config> {
        None
    }
}

/// The production [`ModelLoader`]: the start-up load path
/// ([`crate::load::prepare_model`] / [`crate::load::load_model`]) on
/// the running configuration, as a restart with `--model-artifact` and
/// `--model-id` changed would run it — except that a value only the *other*
/// model takes is dropped rather than refused
/// ([`crate::config::fit_to_family`]), and said so.
///
/// The running configuration is the server's own
/// ([`crate::config_http::ConfigState`]), shared: a live `PATCH /v1/config`
/// is in it, so a later switch loads with the change, and a reload that
/// failed is not, so the model it falls back to loads as it did before.
pub struct ArtifactLoader {
    state: Arc<crate::config_http::ConfigState>,
}

impl ArtifactLoader {
    /// Load every switch target with `options`, kept by the loader alone (a
    /// test's loader; `main` shares the server's with
    /// [`ArtifactLoader::sharing`]).
    pub fn new(options: crate::config::Config) -> Self {
        Self::sharing(Arc::new(crate::config_http::ConfigState::new(options, Arc::new(crate::config::file::NoFiles))))
    }

    /// Load every switch target with the server's running configuration.
    pub fn sharing(state: Arc<crate::config_http::ConfigState>) -> Self {
        Self { state }
    }
}

impl ModelLoader for ArtifactLoader {
    fn prepare(&self, target: &ModelSource, options: Option<&crate::config::Config>) -> Result<Box<dyn PreparedLoad>, String> {
        if !target.artifact.exists() {
            return Err(format!("no such file: {}", target.artifact.display()));
        }
        // The family first: which flags this load can take depends on it.
        let family = crate::loader::artifact_family(&target.artifact)
            .map_err(|e| e.to_string())?
            .unwrap_or(ignis_core::compute::ModelFamily::Qwen38_27b);
        let options = load_options(&self.state.current(), target, family, options)?;
        let prepared = crate::load::prepare_model(&options, &target.artifact).map_err(|e| e.to_string())?;
        Ok(Box::new(prepared))
    }
}

/// The configuration a load of `target`, an artifact of `family`, runs
/// with: `given` as it is when it was fitted to that family (a reload with a
/// changed config, patched, validated and fitted by the caller), else
/// `given` or `running` with the target's artifact and id over it, fitted to
/// `family` as a switch fits it. What [`ArtifactLoader`] loads with, apart
/// from the load itself so a test's loader fits a switch the same way: a
/// value a live `PATCH /v1/config` set for a family not loaded reaches that
/// family's load through here (spec playground/05 AC 19).
pub fn load_options(
    running: &crate::config::Config,
    target: &ModelSource,
    family: ignis_core::compute::ModelFamily,
    given: Option<&crate::config::Config>,
) -> Result<crate::config::Config, String> {
    if let Some(options) = given.filter(|options| options.basis.family() == Some(family)) {
        return Ok(options.clone());
    }
    // The target's artifact and id over the running options, as a patch:
    // the options are resolved again from their sources, so a value set on
    // the struct would not survive the fit.
    let base = given.unwrap_or(running);
    let named = crate::config::target_patch(&target.artifact, &target.model)
        .and_then(|patch| base.general_with(&patch))
        .map_err(|e| e.to_string())?;
    let (options, dropped) = crate::config::fit_to_family(&named, family).map_err(|e| e.to_string())?;
    log_flags_dropped(&target.model, family, &dropped);
    Ok(options)
}

/// Say which of the running options a switch to `to`, a model of `family`,
/// left off because the family does not take them (spec config-v2/01 AC 8):
/// `ignis.model.switch_flags_dropped`, nothing when nothing was dropped.
fn log_flags_dropped(to: &str, family: ignis_core::compute::ModelFamily, dropped: &[String]) {
    if dropped.is_empty() {
        return;
    }
    tracing::info!(
        name: "ignis.model.switch_flags_dropped",
        to = %to,
        family = family.name(),
        flags = %dropped.join(","),
        "start flags this model does not take are off for this load"
    );
}

impl PreparedLoad for crate::load::PreparedModel {
    fn load(self: Box<Self>) -> Result<LoadedModel, String> {
        crate::load::load_model(*self).map_err(|e| e.to_string())
    }

    fn options(&self) -> Option<&crate::config::Config> {
        Some(self.config())
    }
}

/// What a server switches models with (`Server::with_switcher`): the loader,
/// the drain window, and the one switch that may run at a time.
pub struct Switcher {
    loader: Arc<dyn ModelLoader>,
    /// `--switch-drain-timeout`; a live `PATCH /v1/config` may change it.
    drain_timeout: Mutex<Duration>,
    /// The switch under way, if any. Switches are serialized, not queued:
    /// a second one is refused naming this one (spec model-switch/01).
    running: Mutex<Option<SwitchStarted>>,
    /// The models a request's own `model` may switch to, each with the
    /// artifact it loads from ([`implicit_switch`]). Empty unless
    /// [`Switcher::with_known_models`] named some — `main` names none under
    /// `--switch-allow-implicit false` — and then a request naming another
    /// model is refused by name, as it always was. A live `PATCH
    /// /v1/config` may change it.
    known: Mutex<BTreeMap<String, PathBuf>>,
    /// The catalog a request's `model` is looked up in when `known` does not
    /// name it ([`Switcher::with_catalog`]), and whether it is — off with
    /// `--switch-allow-implicit false`, which a live `PATCH /v1/config` may
    /// change. Kept apart from `known` on purpose: `known` is the operator's
    /// list, written back to the config file; the catalog's contribution is
    /// derived, never written anywhere.
    catalog: Option<CatalogModels>,
    catalog_on: AtomicBool,
    /// The models an explicit switch by id may name (`POST /v1/models/switch`
    /// with `model` alone, spec playground/05): the operator's list and the
    /// start model, kept whether or not `known` is — `known` is emptied by
    /// `--switch-allow-implicit false`, which gates the switch a request
    /// triggers by naming a model, not one an operator asked for. Read with
    /// [`Switcher::listed_models`].
    listed: Mutex<BTreeMap<String, PathBuf>>,
    /// How the last switch or reload ended, once there has been one
    /// ([`Switcher::last_switch`]).
    last: Mutex<Option<LastSwitch>>,
    /// Whether the switch under way has said how it ended
    /// ([`Switcher::record`]): a task that dies without having is recorded
    /// as failed by its guard.
    settled: AtomicBool,
    /// The model the server started on, kept for the server's life: a switch
    /// replaces the loaded model, but the way back to this one must stay
    /// listed ([`Switcher::tables`]). Set once, by the start or by the first
    /// switch, whichever sees it first.
    start: Mutex<Option<ModelSource>>,
    /// Whether nothing is resident because the idle clock unloaded the model
    /// (spec model-switch/02): set by the unload, cleared when a load
    /// publishes a model. What tells a failed load out of `unloaded` — which
    /// a later request may try again — from a failed switch with nothing to
    /// reload, which only a switch recovers from.
    unloaded: AtomicBool,
}

/// The two tables a server's models are named by, computed together from the
/// operator's list ([`Switcher::tables`]).
#[derive(Debug, Clone, Default)]
pub struct SwitchTables {
    /// The models an explicit switch by id may name: the operator's and the
    /// start model, whatever `switch.allow_implicit` says.
    pub listed: BTreeMap<String, PathBuf>,
    /// The models a request naming one switches to: `listed`, or none with
    /// implicit switching off.
    pub known: BTreeMap<String, PathBuf>,
}

/// How the last switch or reload since start ended, for `GET /v1/models`
/// (spec playground/05 §`last_switch`): a page polling the status sees a
/// refused or rolled-back switch for one tick only, and this is how it learns
/// why afterwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LastSwitch {
    /// The id served when it began.
    pub from: String,
    /// The id it loaded: equal to `from` for a reload from `PATCH /v1/config`.
    pub to: String,
    /// How it ended.
    pub outcome: SwitchOutcome,
}

/// Why an id named alone cannot be switched to (`POST /v1/models/switch`
/// with no `artifact`): `404 model_not_found` says which, and how to fix it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unswitchable {
    /// Neither a known model nor a catalog entry.
    Unknown,
    /// A known model whose artifact is not a file.
    KnownMissing {
        /// The path the known models name for it.
        artifact: PathBuf,
    },
    /// A catalog entry whose artifact is not under `download.path`: it has to
    /// be fetched first (`model download`), a switch never starts a download.
    NotDownloaded,
}

/// A catalog's entries as known models (spec model-download/02 §Known
/// models): an entry whose artifact is under `dir` **when a request names
/// it**, so a `model download` into a running server's `download.path` makes
/// it switchable with no restart — and one not there is refused as an
/// unknown model is, since a switch never starts a download.
#[derive(Debug, Clone)]
pub struct CatalogModels {
    /// The merged catalog the server started with.
    pub catalog: Arc<crate::download::Catalog>,
    /// `download.path`.
    pub dir: PathBuf,
}

impl CatalogModels {
    /// The artifact a request naming `model` would switch to: the entry
    /// whose id it is, exactly as a known model's is matched, once its
    /// artifact is on disk.
    fn artifact(&self, model: &str) -> Option<PathBuf> {
        let entry = self.catalog.entries().iter().find(|entry| entry.id == model)?;
        let path = entry.artifact_path(&self.dir);
        path.is_file().then_some(path)
    }
}

impl Switcher {
    /// Switch with `loader`, giving the old model's requests `drain_timeout`
    /// to finish (`--switch-drain-timeout`).
    pub fn new(loader: Arc<dyn ModelLoader>, drain_timeout: Duration) -> Self {
        Self {
            loader,
            drain_timeout: Mutex::new(drain_timeout),
            running: Mutex::new(None),
            known: Mutex::new(BTreeMap::new()),
            catalog: None,
            catalog_on: AtomicBool::new(false),
            listed: Mutex::new(BTreeMap::new()),
            last: Mutex::new(None),
            settled: AtomicBool::new(true),
            start: Mutex::new(None),
            unloaded: AtomicBool::new(false),
        }
    }

    /// Whether the idle clock unloaded the model and no load has published
    /// one since (spec model-switch/02): nothing is resident.
    pub fn is_unloaded(&self) -> bool {
        self.unloaded.load(Ordering::SeqCst)
    }

    /// What holds the switch's slot now — a switch, a reload, a load, or an
    /// unload (whose two ends are the same id) — if anything does.
    pub fn running(&self) -> Option<SwitchStarted> {
        self.running.lock().expect("switch lock").clone()
    }

    /// Remember `start`, the model the server started on, as one the server
    /// can always switch back to. The first one remembered stays: it is the
    /// start, not a model a switch brought.
    pub fn note_start(&self, start: Option<&ModelSource>) {
        let mut kept = self.start.lock().expect("start model lock");
        if kept.is_none() {
            *kept = start.cloned();
        }
    }

    /// The model the server started on, once it is known.
    pub fn start(&self) -> Option<ModelSource> {
        self.start.lock().expect("start model lock").clone()
    }

    /// The tables of models a switch may name, from the operator's list
    /// `named` and the start model: what `main` sets at start and
    /// [`Server::apply_config`] rebuilds after every switch and config
    /// change, so the two can never disagree. `allow_implicit` is
    /// `switch.allow_implicit`: it empties the table a request's own `model`
    /// reads and leaves the one an explicit switch by id reads.
    pub fn tables(&self, named: &BTreeMap<String, PathBuf>, allow_implicit: bool) -> SwitchTables {
        let listed = known_models(named, self.start().as_ref());
        let known = if allow_implicit { listed.clone() } else { BTreeMap::new() };
        SwitchTables { listed, known }
    }

    /// Let an explicit switch by id name `listed`'s models even with implicit
    /// switching off; `main` passes [`Switcher::tables`]' table, whatever
    /// `--switch-allow-implicit` says.
    pub fn with_listed_models(self, listed: BTreeMap<String, PathBuf>) -> Self {
        *self.listed.lock().expect("listed models lock") = listed;
        self
    }

    /// Change the models an explicit switch may name, on a running server
    /// (see [`Switcher::set_knobs`], which sets the implicit ones).
    pub fn set_listed_models(&self, listed: BTreeMap<String, PathBuf>) {
        *self.listed.lock().expect("listed models lock") = listed;
    }

    /// Let a request naming one of `known`'s ids switch to it
    /// ([`implicit_switch`]); `main` passes [`known_models`]' table.
    pub fn with_known_models(self, known: BTreeMap<String, PathBuf>) -> Self {
        *self.known.lock().expect("known models lock") = known;
        self
    }

    /// Let a request naming a catalog entry whose artifact is on disk switch
    /// to it too, when `on` (`--switch-allow-implicit`). An explicit
    /// known-models entry wins for its id.
    pub fn with_catalog(self, catalog: CatalogModels, on: bool) -> Self {
        self.catalog_on.store(on, Ordering::SeqCst);
        Self { catalog: Some(catalog), ..self }
    }

    /// Change the drain window, the known models and whether the catalog is
    /// consulted, on a running server (a live `PATCH /v1/config`, spec
    /// config-v2/02): the next switch reads them; one already under way
    /// keeps what it began with.
    pub fn set_knobs(&self, drain_timeout: Duration, known: BTreeMap<String, PathBuf>, catalog_on: bool) {
        *self.drain_timeout.lock().expect("drain timeout lock") = drain_timeout;
        *self.known.lock().expect("known models lock") = known;
        self.catalog_on.store(catalog_on, Ordering::SeqCst);
    }

    /// The drain window a switch beginning now gets.
    pub fn drain_timeout(&self) -> Duration {
        *self.drain_timeout.lock().expect("drain timeout lock")
    }

    /// Every model the operator's list and the start make switchable, with
    /// its artifact — the catalog's entries apart, since which of them are
    /// switchable is decided on disk at each request.
    pub fn known_models(&self) -> BTreeMap<String, PathBuf> {
        self.known.lock().expect("known models lock").clone()
    }

    /// The models an explicit switch by id may name: the table
    /// [`Switcher::set_listed_models`] keeps, with the implicit one
    /// ([`Switcher::known_models`]) over it, so a switcher built with only the
    /// latter resolves its models too.
    pub fn listed_models(&self) -> BTreeMap<String, PathBuf> {
        let mut listed = self.listed.lock().expect("listed models lock").clone();
        listed.extend(self.known_models());
        listed
    }

    /// The catalog a switch can go to, with the directory it is looked up in.
    pub fn catalog(&self) -> Option<&CatalogModels> {
        self.catalog.as_ref()
    }

    /// The artifact an explicit switch with `model` alone loads (spec
    /// playground/05 §Switch by id): the listed models' entry for it, which
    /// wins as for an implicit switch, else a catalog entry whose artifact is
    /// a file under `download.path`. `switch.allow_implicit` does not apply —
    /// it gates the switch a request triggers by naming a model, not an
    /// operator's. Looked up at each call: a download landing in the
    /// directory needs no restart.
    pub fn resolve(&self, model: &str) -> Result<PathBuf, Unswitchable> {
        if let Some(listed) = self.listed_models().remove(model) {
            return if listed.is_file() { Ok(listed) } else { Err(Unswitchable::KnownMissing { artifact: listed }) };
        }
        let catalog = self.catalog.as_ref().ok_or(Unswitchable::Unknown)?;
        let entry = catalog.catalog.entries().iter().find(|entry| entry.id == model).ok_or(Unswitchable::Unknown)?;
        let artifact = entry.artifact_path(&catalog.dir);
        if artifact.is_file() { Ok(artifact) } else { Err(Unswitchable::NotDownloaded) }
    }

    /// How the last switch or reload ended; `None` before the first.
    pub fn last_switch(&self) -> Option<LastSwitch> {
        self.last.lock().expect("last switch lock").clone()
    }

    /// Say how a switch ended. Called **before** the status leaves
    /// `switching` or `failed`, so a client that reads `serving` finds its
    /// reason already here.
    fn record(&self, from: &str, to: &str, outcome: &SwitchOutcome) {
        self.settled.store(true, Ordering::SeqCst);
        *self.last.lock().expect("last switch lock") =
            Some(LastSwitch { from: from.to_owned(), to: to.to_owned(), outcome: outcome.clone() });
    }

    /// The artifact a request naming `model` would switch to, if it is known:
    /// the known models' entry for it, else a catalog entry on disk.
    fn known(&self, model: &str) -> Option<PathBuf> {
        if let Some(listed) = self.known.lock().expect("known models lock").get(model) {
            return Some(listed.clone());
        }
        if !self.catalog_on.load(Ordering::SeqCst) {
            return None;
        }
        self.catalog.as_ref()?.artifact(model)
    }
}

/// The models a request may switch to by naming them: the operator's
/// (`--switch-known-models`, [`crate::config::Config::known_models`]) and the one the
/// server started on, `start` — its served id and the artifact it loaded,
/// which only its load knows. An operator's entry for the start's own id
/// gives way to the start's, which is the artifact actually loaded and so the
/// one a switch back must reload; `ignis.config.known_model_replaced` says so
/// when the two paths differ. A start built in process (the placeholder) has
/// no artifact and adds nothing.
pub fn known_models(named: &BTreeMap<String, PathBuf>, start: Option<&ModelSource>) -> BTreeMap<String, PathBuf> {
    let mut known = named.clone();
    if let Some(start) = start {
        if let Some(listed) = known.insert(start.model.clone(), start.artifact.clone()) {
            if listed != start.artifact {
                tracing::warn!(
                    name: "ignis.config.known_model_replaced",
                    model = %start.model,
                    listed = %listed.display(),
                    loaded = %start.artifact.display(),
                    "--switch-known-models names the start model at another path; a switch back reloads the one loaded"
                );
            }
        }
    }
    known
}

/// A switch that has begun: the ids it moves between, as `202` reports them
/// and as a second switch's `409` names the one under way.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwitchStarted {
    /// The id served when it began — the model that is drained and torn
    /// down, and reloaded if the target fails on the GPU.
    pub from: String,
    /// The id it is loading: the request's `model`, which the target
    /// artifact's own model must not contradict.
    pub to: String,
}

/// Why a switch did not begin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchRefusal {
    /// The server was built without a [`Switcher`].
    Unavailable,
    /// The first traversal of the loaded model has not run yet; switching
    /// under it would race its own warm-up.
    WarmingUp,
    /// Another switch is under way (`409`): this is it.
    InProgress(SwitchStarted),
    /// A reload with a changed config (`PATCH /v1/config`) was asked of a
    /// model not loaded from an artifact (the placeholder start): there is
    /// nothing to load it from again.
    NotReloadable,
    /// The load a request began by needing an unloaded model
    /// ([`ensure_loaded`]) found one loaded, or loading, already — another
    /// request's load or a switch got there first. Never answered: the
    /// request looks at the status again.
    NotUnloaded,
}

/// How a switch ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SwitchOutcome {
    /// The target is serving.
    Switched,
    /// The target was refused before anything was torn down: the old model
    /// never stopped serving.
    Refused {
        /// What refused it — a path with no file, a checksum that is not
        /// clean, flags the target cannot take — as the prepare step said
        /// it, and as `ignis.model.switch_failed` logs it.
        reason: String,
    },
    /// The target failed to load after the old model was torn down, and the
    /// old model was loaded again from its artifact: it is serving.
    Restored {
        /// Why the target failed on the GPU (or in its warm-up).
        reason: String,
    },
    /// The target failed, and the old model could not be loaded again
    /// either: nothing is serving until a switch succeeds.
    Failed {
        /// Why, both failures said: the target's, then the reload's.
        reason: String,
    },
}

/// Ends a switch's hold on the server when its task ends, however it ends:
/// [`Switcher::running`] is cleared so the next switch may begin, and a task
/// that ended without settling the status — a panic, or a runtime shutting
/// down under it — leaves [`ModelStatus::Failed`] rather than a gate stuck
/// at `Switching` (or `Loading`) with nobody left to open it. Freeing the
/// slot wakes the requests waiting on the status: one that found the slot
/// taken by an unload loads the model now (spec model-switch/02).
struct RunningGuard {
    switcher: Arc<Switcher>,
    status: Arc<StatusCell>,
    /// Whether how it ended goes to `last_switch`: a switch or a reload,
    /// not an unload nor the load a request began after one (spec
    /// model-switch/02 AC 14).
    records: bool,
}

impl Drop for RunningGuard {
    fn drop(&mut self) {
        // Mid-switch, or in the reload after a failed load (the status is
        // already `failed` then, but the switch has not said how it ended).
        let status = self.status.load_full();
        let reason = match &*status {
            ModelStatus::Switching { .. } | ModelStatus::Loading => {
                Some("the switch ended before it finished (see ignis.model.switch_panicked)".to_owned())
            }
            ModelStatus::Failed { reason } if self.records && !self.switcher.settled.load(Ordering::SeqCst) => Some(reason.clone()),
            _ => None,
        };
        if let Some(reason) = reason {
            if let Some(started) = self.switcher.running.lock().expect("switch lock").as_ref().filter(|_| self.records) {
                self.switcher.record(&started.from, &started.to, &SwitchOutcome::Failed { reason: reason.clone() });
            }
            tracing::error!(
                name: "ignis.model.switch_panicked",
                panicking = std::thread::panicking(),
                "the switch task ended mid-switch; a switch to a model that loads is the way back"
            );
            self.status.store(Arc::new(ModelStatus::Failed { reason }));
        }
        *self.switcher.running.lock().expect("switch lock") = None;
        self.status.wake();
    }
}

/// Begin switching `server` to `target`: the gate closes now
/// ([`ModelStatus::Switching`]), and the rest — prepare, drain, tear down,
/// load — runs on the returned task, so an HTTP caller answers `202` without
/// waiting on a model load. Must be called inside a tokio runtime.
pub fn begin(
    server: &Server,
    target: ModelSource,
) -> Result<(SwitchStarted, tokio::task::JoinHandle<SwitchOutcome>), SwitchRefusal> {
    begin_with(server, target, None, false)
}

/// A reload asked for by a config change (spec config-v2/02 §`PATCH`).
struct Reconfigure {
    /// The configuration to load with: the running one with `patch` on top,
    /// validated and fitted to the loaded model's family by the caller.
    config: crate::config::Config,
    /// The change itself, written to the config file once the reload has
    /// taken.
    patch: crate::config::source::Layer,
}

/// Reload the loaded model with `config` — the running configuration with
/// `patch` applied, already validated — through the very switch [`begin`]
/// runs (spec config-v2/02 §`PATCH` step 4): gate, drain, teardown, load,
/// and the old model reloaded as it was if the new load fails. Only once
/// the reloaded model serves do `config` become the running configuration,
/// its server-level values take effect, and `patch` reach the config file —
/// never part of it before, never any of it after a failure. The task runs
/// detached; `GET /v1/models` and `GET /v1/config` report it.
pub fn reconfigure(
    server: &Server,
    config: crate::config::Config,
    patch: crate::config::source::Layer,
) -> Result<SwitchStarted, SwitchRefusal> {
    if server.switcher.is_none() {
        return Err(SwitchRefusal::Unavailable);
    }
    let target = server.active().source.clone().ok_or(SwitchRefusal::NotReloadable)?;
    begin_with(server, target, Some(Reconfigure { config, patch }), false).map(|(started, _task)| started)
}

/// Where a run of the switch's slot begins, which decides its steps.
#[derive(Debug, Clone)]
struct Begun {
    /// What a target refused before anything stopped puts back: `serving`
    /// when the old model never stopped, `unloaded` when the unloaded model
    /// is still the one a request loads; nothing after a failure, which
    /// stays.
    back: Option<ModelStatus>,
    /// Nothing is resident — the idle clock unloaded the model (spec
    /// model-switch/02): no drain, no teardown, and no previous model to
    /// reload when the load fails.
    unloaded: bool,
    /// The load a request began by needing the unloaded model
    /// ([`ensure_loaded`]): not a switch, so it is logged
    /// `ignis.model.reloaded` and never written to `last_switch` (AC 14).
    on_demand: bool,
}

/// [`begin`], with the config change a reload carries, if any; `on_demand`
/// is the load [`ensure_loaded`] begins, refused
/// ([`SwitchRefusal::NotUnloaded`]) unless the model is still unloaded when
/// the slot is taken — one load, never a switch over a model another
/// request's load has just published.
fn begin_with(
    server: &Server,
    target: ModelSource,
    reconfigure: Option<Reconfigure>,
    on_demand: bool,
) -> Result<(SwitchStarted, tokio::task::JoinHandle<SwitchOutcome>), SwitchRefusal> {
    let switcher = server.switcher.clone().ok_or(SwitchRefusal::Unavailable)?;
    let (started, begun) = {
        let mut running = switcher.running.lock().expect("switch lock");
        if let Some(started) = running.as_ref() {
            return Err(SwitchRefusal::InProgress(started.clone()));
        }
        let before = server.status();
        if matches!(*before, ModelStatus::WarmingUp) {
            return Err(SwitchRefusal::WarmingUp);
        }
        // Unloaded, or a load out of `unloaded` that failed: nothing is
        // resident, so this run only loads.
        let unloaded = matches!(*before, ModelStatus::Unloaded)
            || (matches!(*before, ModelStatus::Failed { .. }) && switcher.is_unloaded());
        if on_demand && !unloaded {
            return Err(SwitchRefusal::NotUnloaded);
        }
        let back = match &*before {
            _ if on_demand => None,
            ModelStatus::Serving => Some(ModelStatus::Serving),
            ModelStatus::Unloaded => Some(ModelStatus::Unloaded),
            _ => None,
        };
        let started = SwitchStarted { from: server.active().engine.model_id(), to: target.model.clone() };
        *running = Some(started.clone());
        // The model being replaced is the start model if nothing else said so.
        switcher.note_start(server.active().source.as_ref());
        if !on_demand {
            switcher.settled.store(false, Ordering::SeqCst);
        }
        (started, Begun { back, unloaded, on_demand })
    };
    if begun.unloaded {
        // Nothing to drain: every request is let through, and the ones that
        // need the model wait for this load (spec model-switch/02).
        server.status.store(Arc::new(ModelStatus::Loading));
        tracing::info!(
            name: "ignis.model.load_started",
            from = %started.from,
            to = %started.to,
            artifact = %target.artifact.display(),
            on_demand,
            "loading a model while none is loaded: the requests that need it wait for it"
        );
    } else {
        server
            .status
            .store(Arc::new(ModelStatus::Switching { from: started.from.clone(), to: started.to.clone() }));
        tracing::info!(
            name: "ignis.model.switch_started",
            from = %started.from,
            to = %started.to,
            artifact = %target.artifact.display(),
            "switching models: /v1 answers 503 model_switching until it is done"
        );
    }
    let guard = RunningGuard { switcher, status: Arc::clone(&server.status), records: !begun.on_demand };
    let task = tokio::spawn(run(server.clone(), guard, target, started.clone(), begun, reconfigure));
    Ok((started, task))
}

/// Why a request whose `model` named another model was not served on it
/// ([`implicit_switch`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImplicitRefusal {
    /// A switch was already under way when the request named its model —
    /// another request's, or one `POST /v1/models/switch` began — whatever
    /// its target, this request's own included. Not joined, not queued
    /// behind, not retargeted (spec §Implicit switch, ordering 5): answered
    /// `503 model_switching`, exactly as the gate answers a request sent a
    /// moment later.
    Switching(SwitchStarted),
    /// The switch this request began did not leave `to` serving — refused
    /// before anything stopped, or failed on the GPU with the previous model
    /// reloaded or not. Never served on another model instead: the client
    /// named one, and gets it or the reason it did not.
    Failed {
        /// The model the request named.
        to: String,
        /// The switch's own reason, as `GET /v1/models` reports a failed
        /// switch's.
        reason: String,
    },
}

/// Switch to the model a request named, and wait until it serves (spec
/// model-switch/01 §Implicit switch): what lets an OpenAI client that names
/// the other model in `model` simply be answered by it, the first such
/// request taking as long as the switch takes.
///
/// `requested` is the request's `model` with its lane tag already stripped.
/// Nothing named, the model already loaded, or one the switcher does not know
/// (switching off, or a name `--switch-known-models` never listed) is not this
/// function's to refuse: it returns at once, and the request goes on to the
/// check every endpoint already makes, which refuses a model it does not load
/// by name. A known model other than the loaded one begins the very switch
/// `POST /v1/models/switch` would ([`begin`]) — the gate closes before this
/// returns control to anything else, the old model's requests drain, it is
/// torn down, the named one loads — and only once that switch has ended does
/// this return: `Ok` with the named model serving, so the caller, taking the
/// loaded model only now, runs exactly as if it had always been loaded.
///
/// Call it on the server every handler shares, before
/// [`crate::Server::pinned`]: a pinned copy holds a model of its own, and a
/// switch published into it would reach no one.
pub async fn implicit_switch(server: &Server, requested: Option<&str>) -> Result<(), ImplicitRefusal> {
    let Some(requested) = requested.filter(|model| !model.is_empty()) else {
        return Ok(());
    };
    let Some(artifact) = server.switcher.as_ref().and_then(|switcher| switcher.known(requested)) else {
        return Ok(());
    };
    if server.active().engine.model_id() == requested {
        return Ok(());
    }
    let target = ModelSource { artifact: artifact.clone(), model: requested.to_owned() };
    let task = match begin(server, target) {
        Ok((started, task)) => {
            tracing::info!(
                name: "ignis.model.switch_requested",
                from = %started.from,
                to = %started.to,
                "a request named a known model other than the loaded one; it is served once the switch lands"
            );
            task
        }
        Err(SwitchRefusal::InProgress(running)) => return Err(ImplicitRefusal::Switching(running)),
        // None reaches a handler: there is a switcher, the gate holds every
        // request until the warm-up has run, and only a reload refuses a
        // model with no artifact. Were one to, the request's own model check
        // refuses it by name.
        Err(
            SwitchRefusal::Unavailable | SwitchRefusal::WarmingUp | SwitchRefusal::NotReloadable | SwitchRefusal::NotUnloaded,
        ) => return Ok(()),
    };
    let failed = |reason: String| Err(ImplicitRefusal::Failed { to: requested.to_owned(), reason });
    match task.await {
        Ok(SwitchOutcome::Switched) => Ok(()),
        Ok(SwitchOutcome::Refused { reason } | SwitchOutcome::Restored { reason } | SwitchOutcome::Failed { reason }) => {
            failed(reason)
        }
        Err(ended) => failed(format!("the switch ended before it finished: {ended}")),
    }
}

/// Load the model a request needs when the idle clock unloaded it, and wait
/// until it serves (spec model-switch/02, ADR 0049): what lets the first
/// request after a quiet spell simply take as long as a cold load, rather
/// than be refused for a state the server leaves by itself.
///
/// From [`ModelStatus::Unloaded`] this loads the model `requested` names
/// when it is one a request may switch to — the implicit switch's target
/// resolved here, so leaving `unloaded` is one load, never a load and then
/// a switch — else the unloaded model again, from its source. A request
/// arriving while that load runs ([`ModelStatus::Loading`]) waits for it:
/// concurrent arrivals share one load. Woken by the status itself
/// ([`StatusCell::subscribe`]), never by polling it.
///
/// `Ok` once a model serves — or at once in any other state, which is the
/// gate's and the request's own business, as before. `Err` with the reason
/// when the load this request waited on failed (AC 7); a failed load the
/// request did not wait on is one it tries again.
///
/// Called where [`implicit_switch`] is, after the API key and before
/// [`crate::Server::pinned`]: an unauthenticated request never costs a load
/// (the serving gate, outside the key, lets every route through while
/// unloaded), and the pin then takes the loaded model.
pub async fn ensure_loaded(server: &Server, requested: Option<&str>) -> Result<(), String> {
    let Some(switcher) = server.switcher.clone() else {
        return Ok(());
    };
    // Whether a load this request can count as its own has run: a failure
    // seen after one is this request's answer.
    let mut waited = false;
    loop {
        let mut changed = server.status.subscribe();
        let status = server.status();
        let begin = match &*status {
            ModelStatus::Loading => {
                waited = true;
                false
            }
            ModelStatus::Failed { reason } if switcher.is_unloaded() => {
                if waited {
                    return Err(reason.clone());
                }
                true
            }
            ModelStatus::Unloaded => true,
            _ => return Ok(()),
        };
        if begin {
            let Some(target) = load_target(server, &switcher, requested) else {
                // Nothing names an artifact to load (never so after an
                // unload, which needs one): the torn-down engine answers.
                return Ok(());
            };
            match begin_with(server, target, None, true) {
                Ok(_) => waited = true,
                // The slot is an unload's still tearing down, or another
                // load or a switch got there first: its end wakes this.
                Err(SwitchRefusal::InProgress(_) | SwitchRefusal::NotUnloaded) => {}
                Err(SwitchRefusal::Unavailable | SwitchRefusal::WarmingUp | SwitchRefusal::NotReloadable) => return Ok(()),
            }
        }
        if changed.changed().await.is_err() {
            return Ok(());
        }
    }
}

/// What a request needing the unloaded model loads ([`ensure_loaded`]): the
/// model it names when that is one a request may switch to and not the
/// unloaded one, else the unloaded one from its source.
fn load_target(server: &Server, switcher: &Switcher, requested: Option<&str>) -> Option<ModelSource> {
    let unloaded = server.active().source.clone();
    let other = requested
        .filter(|model| !model.is_empty())
        .filter(|model| unloaded.as_ref().map_or(true, |source| source.model != *model));
    other
        .and_then(|model| switcher.known(model).map(|artifact| ModelSource { artifact, model: model.to_owned() }))
        .or(unloaded)
}

/// The switch's slot, held by an idle unload from the moment it closes the
/// model to new work until its teardown is done ([`claim_unload`]).
pub(crate) struct Unloading {
    _slot: RunningGuard,
}

/// Take the switch's slot for an idle unload (spec model-switch/02) and
/// close the model: the status is [`ModelStatus::Unloaded`] from now, so a
/// request admitted after this loads the model again rather than reaching an
/// engine about to go. `None` — the unload waits for a later tick — while a
/// switch, a reload or a load holds the slot (AC 12), when the model is not
/// serving, or when it was not loaded from an artifact and could not be
/// loaded again.
///
/// The idle clock calls it holding the in-flight count's lock
/// ([`crate::idle::Usage`]), which a request's admission takes too: the
/// count it saw at zero stays zero until the status says `unloaded`.
pub(crate) fn claim_unload(server: &Server) -> Option<Unloading> {
    let switcher = server.switcher.clone()?;
    let mut running = switcher.running.lock().expect("switch lock");
    if running.is_some() || !matches!(*server.status(), ModelStatus::Serving) || server.active().source.is_none() {
        return None;
    }
    let model = server.active().engine.model_id();
    *running = Some(SwitchStarted { from: model.clone(), to: model });
    switcher.unloaded.store(true, Ordering::SeqCst);
    server.status.store(Arc::new(ModelStatus::Unloaded));
    drop(running);
    Some(Unloading { _slot: RunningGuard { switcher, status: Arc::clone(&server.status), records: false } })
}

/// Tear the model down the way a switch does ([`ActiveModel::shut_down`]:
/// the model thread drops its scheduler, and with it every GPU buffer and
/// pinned arena), after [`claim_unload`] closed it, `idle` after its last
/// use. The [`ActiveModel`] stays published — its id and its source are
/// what `GET /v1/models` names and what the next load reads — and its
/// engine is never called again. Not a switch: `ignis.model.unloaded`, and
/// `last_switch` is left as it was (AC 14). The slot is freed, and the
/// requests that arrived meanwhile woken, when this returns.
pub(crate) async fn unload(server: &Server, unloading: Unloading, idle: Duration) {
    let model = server.active();
    let teardown_began = Instant::now();
    shut_down(&model).await;
    tracing::info!(
        name: "ignis.model.unloaded",
        model = %model.engine.model_id(),
        idle_secs = idle.as_secs(),
        teardown_ms = teardown_began.elapsed().as_millis() as u64,
        "nobody used the model for model.idle_unload seconds: it is unloaded, and the next request that needs it loads it again"
    );
    drop(unloading);
}

/// The switch itself, steps 2 to 5 of the module doc, from where `begun`
/// says it began: a switch may also begin from [`ModelStatus::Failed`], the
/// way out of a switch that left nothing serving, and a refusal then must
/// not reopen the gate over a dead engine; and from
/// [`ModelStatus::Unloaded`], where it only loads (spec model-switch/02).
async fn run(
    server: Server,
    running: RunningGuard,
    target: ModelSource,
    started: SwitchStarted,
    begun: Begun,
    reconfigure: Option<Reconfigure>,
) -> SwitchOutcome {
    let switcher = Arc::clone(&running.switcher);
    let began = Instant::now();
    let SwitchStarted { from, to } = started;

    let options = reconfigure.as_ref().map(|reconfigure| reconfigure.config.clone());
    let prepared = match prepare(&switcher, target.clone(), options).await {
        Ok(prepared) => prepared,
        Err(reason) if begun.on_demand => {
            // Spec model-switch/02 AC 7: the requests waiting on this load
            // are answered with its reason, and `failed` stays until a later
            // request or a switch tries again.
            tracing::error!(
                name: "ignis.model.reload_failed",
                model = %to,
                stage = "prepare",
                %reason,
                "loading the unloaded model failed; the next request tries again"
            );
            server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
            return SwitchOutcome::Failed { reason };
        }
        Err(reason) => {
            tracing::error!(
                name: "ignis.model.switch_failed",
                from = %from,
                to = %to,
                stage = "prepare",
                %reason,
                "the switch was refused before anything stopped; the old model is as it was"
            );
            // Failed for one tick (`GET /v1/models` may read it), then the
            // old model, which never stopped, serves again — or stays
            // unloaded, for a request to load — or, when the switch began
            // from a failed one, nothing still serves. The reason is kept
            // (`last_switch`) before either is stored.
            let outcome = SwitchOutcome::Refused { reason: reason.clone() };
            switcher.record(&from, &to, &outcome);
            server.status.store(Arc::new(ModelStatus::Failed { reason }));
            if let Some(back) = begun.back {
                server.status.store(Arc::new(back));
            }
            return outcome;
        }
    };

    // What the configuration will be once this load serves: the patched
    // one for a reload, the one the loader fitted to the target otherwise.
    let serving_config = match &reconfigure {
        Some(reconfigure) => Some(reconfigure.config.clone()),
        None => prepared.options().cloned(),
    };
    let old = server.active();
    // An unloaded model has nothing in flight and was already torn down.
    let (drain_ms, teardown_ms) = if begun.unloaded {
        (0, 0)
    } else {
        let drain_began = Instant::now();
        let drain_timeout = switcher.drain_timeout();
        if let Err(left) = drain(&old.engine, drain_timeout).await {
            // The window is over: what is still running is cut now, before
            // the line saying so — whoever reads it can count on the cut
            // being under way. The join below waits for it.
            old.engine.shutdown();
            tracing::warn!(
                name: "ignis.model.switch_drain_timed_out",
                from = %from,
                to = %to,
                waiting = left.waiting,
                running = left.running,
                drain_timeout_ms = drain_timeout.as_millis() as u64,
                "requests still on the old model past the drain window are cancelled"
            );
        }
        let drain_ms = drain_began.elapsed().as_millis() as u64;

        let teardown_began = Instant::now();
        shut_down(&old).await;
        (drain_ms, teardown_began.elapsed().as_millis() as u64)
    };

    let load_began = Instant::now();
    match load_and_warm(&server, prepared).await {
        Ok((new, warm_up)) => {
            publish(&server, new);
            // Spec config-v2/02: the configuration changes only now, with a
            // model serving under it — and is written down only now.
            if let Some(config) = serving_config {
                server.apply_config(&config);
                if let Some(state) = &server.config {
                    state.publish(config);
                    if let Some(reconfigure) = &reconfigure {
                        state.persist(&reconfigure.patch);
                    }
                }
            }
            let load_ms = (load_began.elapsed() - warm_up).as_millis() as u64;
            if begun.on_demand {
                // Not a switch: logged as the load after an unload, and
                // `last_switch` is left as it was (spec model-switch/02).
                tracing::info!(
                    name: "ignis.model.reloaded",
                    model = %to,
                    artifact = %target.artifact.display(),
                    load_ms,
                    warm_up_ms = warm_up.as_millis() as u64,
                    took_ms = began.elapsed().as_millis() as u64,
                    "the model is loaded again: the requests waiting on it are served"
                );
            } else {
                tracing::info!(
                    name: "ignis.model.switched",
                    from = %from,
                    to = %to,
                    artifact = %target.artifact.display(),
                    drain_ms,
                    teardown_ms,
                    load_ms,
                    warm_up_ms = warm_up.as_millis() as u64,
                    took_ms = began.elapsed().as_millis() as u64,
                    "model switched: the API admits requests"
                );
                switcher.record(&from, &to, &SwitchOutcome::Switched);
            }
            server.status.store(Arc::new(ModelStatus::Serving));
            SwitchOutcome::Switched
        }
        Err(reason) if begun.unloaded => {
            // Nothing was resident before this load, so there is nothing to
            // reload: `failed` stays, with the process up, until a request
            // (after an unload) or a switch loads a model.
            let outcome = SwitchOutcome::Failed { reason: reason.clone() };
            if begun.on_demand {
                tracing::error!(
                    name: "ignis.model.reload_failed",
                    model = %to,
                    stage = "load",
                    %reason,
                    "loading the unloaded model failed; the next request tries again"
                );
            } else {
                tracing::error!(
                    name: "ignis.model.switch_failed",
                    from = %from,
                    to = %to,
                    stage = "load",
                    %reason,
                    "the target failed to load and no model was loaded before it; nothing is serving"
                );
                switcher.record(&from, &to, &outcome);
            }
            server.status.store(Arc::new(ModelStatus::Failed { reason }));
            outcome
        }
        Err(reason) => {
            tracing::error!(
                name: "ignis.model.switch_failed",
                from = %from,
                to = %to,
                stage = "load",
                %reason,
                "the target failed to load after the old model was torn down; reloading the old model"
            );
            server.status.store(Arc::new(ModelStatus::Failed { reason: reason.clone() }));
            restore(&server, &switcher, &old, (&from, &to), reason).await
        }
    }
}

/// Step 2: [`ModelLoader::prepare`] off the async workers, with `options`
/// when a reload changes the configuration.
async fn prepare(
    switcher: &Arc<Switcher>,
    target: ModelSource,
    options: Option<crate::config::Config>,
) -> Result<Box<dyn PreparedLoad>, String> {
    let loader = Arc::clone(&switcher.loader);
    tokio::task::spawn_blocking(move || loader.prepare(&target, options.as_ref()))
        .await
        .unwrap_or_else(|panicked| Err(format!("preparing the load panicked: {panicked}")))
}

/// Step 3: wait until `engine` has nothing waiting or running, or `timeout`
/// passes — then what was still in flight.
async fn drain(engine: &Engine, timeout: Duration) -> Result<(), crate::telemetry::IntervalCounters> {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let counters = engine.interval_counters();
        if counters.waiting == 0 && counters.running == 0 {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(counters);
        }
        tokio::time::sleep(DRAIN_POLL).await;
    }
}

/// Step 4: stop `model` and wait until its thread has dropped everything it
/// held ([`ActiveModel::shut_down`]), off the async workers.
async fn shut_down(model: &Arc<ActiveModel>) {
    let model = Arc::clone(model);
    if let Err(panicked) = tokio::task::spawn_blocking(move || model.shut_down()).await {
        tracing::error!(name: "ignis.model.teardown_panicked", error = %panicked, "tearing the old model down panicked");
    }
}

/// Step 5 up to publishing: load `prepared`, run its first traversal, and
/// point `--server-metrics` at it. A model whose warm-up fails is torn down again
/// and reported as a failed load: better refused than silently cold. Returns
/// the model and how long its warm-up took.
async fn load_and_warm(server: &Server, prepared: Box<dyn PreparedLoad>) -> Result<(ActiveModel, Duration), String> {
    let loaded = tokio::task::spawn_blocking(move || prepared.load())
        .await
        .unwrap_or_else(|panicked| Err(format!("the load panicked: {panicked}")))?;
    let warm_up = match crate::warm_up_engine(&loaded.model.engine).await {
        Ok(took) => took,
        Err(error) => {
            shut_down(&Arc::new(loaded.model)).await;
            return Err(error);
        }
    };
    server.observe_load(&loaded.model, loaded.reservations);
    Ok((loaded.model, warm_up))
}

/// Make `model` the one every request reads, with a fresh fork history: the
/// old one's match keys name KV the new model never computed. A model is
/// resident again, and its idle clock starts now (spec model-switch/02):
/// a model loaded after a long quiet spell is not unloaded at the next tick.
fn publish(server: &Server, model: ActiveModel) {
    *server.fork_history.lock().expect("fork history lock") = crate::reuse::ForkHistory::default();
    server.active.store(Arc::new(model));
    if let Some(switcher) = &server.switcher {
        switcher.unloaded.store(false, Ordering::SeqCst);
    }
    server.usage.touch();
    crate::decide::log_load_heads(&server.active());
}

/// After a failed load: load `old` again from where it came from, so a bad
/// switch ends with the previous model serving. `reason` is why the target
/// failed; `(from, to)` are the switch's ends, for `last_switch`.
async fn restore(
    server: &Server,
    switcher: &Arc<Switcher>,
    old: &ActiveModel,
    (from, to): (&str, &str),
    reason: String,
) -> SwitchOutcome {
    let Some(source) = old.source.clone() else {
        let reason = format!("{reason}; the previous model was not loaded from an artifact, so there is nothing to reload");
        tracing::error!(name: "ignis.model.switch_restore_failed", %reason, "nothing is serving until a switch succeeds");
        let outcome = SwitchOutcome::Failed { reason: reason.clone() };
        switcher.record(from, to, &outcome);
        server.status.store(Arc::new(ModelStatus::Failed { reason }));
        return outcome;
    };
    // The running configuration, unchanged: a reload that failed never
    // published its own.
    let restored = match prepare(switcher, source.clone(), None).await {
        Ok(prepared) => load_and_warm(server, prepared).await.map(|(model, _)| model),
        Err(error) => Err(error),
    };
    match restored {
        Ok(model) => {
            publish(server, model);
            tracing::warn!(
                name: "ignis.model.switch_restored",
                model = %source.model,
                artifact = %source.artifact.display(),
                %reason,
                "the switch failed; the previous model was reloaded and serves again"
            );
            let outcome = SwitchOutcome::Restored { reason };
            switcher.record(from, to, &outcome);
            server.status.store(Arc::new(ModelStatus::Serving));
            outcome
        }
        Err(again) => {
            let reason = format!("{reason}; reloading {} failed too: {again}", source.model);
            tracing::error!(name: "ignis.model.switch_restore_failed", %reason, "nothing is serving until a switch succeeds");
            let outcome = SwitchOutcome::Failed { reason: reason.clone() };
            switcher.record(from, to, &outcome);
            server.status.store(Arc::new(ModelStatus::Failed { reason }));
            outcome
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ignis_core::compute::ModelFamily;

    fn logged(run: impl FnOnce()) -> Vec<serde_json::Value> {
        use tracing_subscriber::layer::SubscriberExt;
        let sink = Arc::new(ignis_logging::MemorySink::new());
        let subscriber = tracing_subscriber::registry().with(ignis_logging::JsonLayer::new(sink.clone()));
        tracing::subscriber::with_default(subscriber, run);
        sink.lines().iter().map(|line| serde_json::from_str(line).expect("json")).collect()
    }

    /// Spec config-v2/01 AC 8: a value the target family cannot take is
    /// dropped, and the switch says which, from the dropped list the fit
    /// returned — the list each field's own applicability produces.
    #[test]
    fn a_switch_names_what_it_dropped_and_says_nothing_when_nothing_was() {
        let started = match crate::config::resolve(&["--vision-enabled".to_owned()], |_| None).unwrap() {
            crate::config::ConfigOutcome::Config(config) => config,
            other => panic!("{other:?}"),
        };
        let (_, dropped) = crate::config::fit_to_family(&started, ModelFamily::FlashNext).unwrap();
        let records = logged(|| log_flags_dropped("qwen3.8-flash-next", ModelFamily::FlashNext, &dropped));
        assert_eq!(records.len(), 1, "{records:?}");
        let event = &records[0];
        assert_eq!(event["event_name"], "ignis.model.switch_flags_dropped", "{event}");
        let field = |name: &str| event.get(name).or_else(|| event["attributes"].get(name)).cloned();
        assert_eq!(field("flags"), Some("--vision-enabled".into()), "{event}");
        assert_eq!(field("to"), Some("qwen3.8-flash-next".into()), "{event}");
        assert!(logged(|| log_flags_dropped("qwen3.8-27b", ModelFamily::Qwen38_27b, &[])).is_empty());
    }
}
