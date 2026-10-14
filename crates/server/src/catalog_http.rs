//! `GET /v1/models/catalog` (spec playground/05): every model this server
//! can switch to or fetch, with its state on disk — what a page needs to
//! offer a switch by id (`POST /v1/models/switch` with `model` alone) without
//! knowing a path on the server's machine.
//!
//! One row per catalog entry (built-in, then the operator's), then one per
//! known model (`switch.known_models`, the model the server started on) that
//! is not a catalog entry. An entry's state is what `model list` computes
//! ([`crate::download::command::list_rows`]: sizes only, never a hash); a
//! known model is `complete` when its artifact is a file, else `absent`. No
//! row carries a path: the page switches by id.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::Json;
use serde::Serialize;
use utoipa::ToSchema;

use crate::api::ApiError;
use crate::config::field::scope;
use crate::download::catalog::CatalogLayer;
use crate::download::command::OnDisk;
use crate::{ModelStatus, Server};

/// Where a model is listed from.
#[derive(Serialize, ToSchema, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
enum CatalogSource {
    /// An entry of the catalog compiled into the binary.
    BuiltIn,
    /// An entry of the operator's `download.catalog`.
    Operator,
    /// Not a catalog entry: in `switch.known_models`, or the model the server
    /// started on.
    Known,
}

impl From<CatalogLayer> for CatalogSource {
    fn from(layer: CatalogLayer) -> Self {
        match layer {
            CatalogLayer::BuiltIn => Self::BuiltIn,
            CatalogLayer::Operator => Self::Operator,
        }
    }
}

/// One model the server can switch to or fetch.
#[derive(Serialize, ToSchema)]
pub(crate) struct CatalogRow {
    /// The id the model is served under, and the one a switch names.
    id: String,
    /// `built-in` or `operator` (a catalog entry), or `known` (in
    /// `switch.known_models`, or the model the server started on, and not a
    /// catalog entry).
    source: CatalogSource,
    /// `complete`, `partial` or `absent` under `download.path`; `complete`
    /// too when the file the server would load the model from (a listed
    /// model's artifact, the one it started on) is there. A known model is
    /// `complete` when its artifact is a file, else `absent`.
    on_disk: OnDisk,
    /// An entry's total size, or the artifact's for a known model on disk;
    /// absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    /// The scope name (`qwen38`, `qwen38flashnext`) the artifact's header
    /// names, when it is on disk; for the loaded model always the running
    /// family.
    #[serde(skip_serializing_if = "Option::is_none")]
    family: Option<&'static str>,
    /// This is the model serving now (the start model before any switch).
    /// Exactly one row is, except while nothing serves (a failed switch).
    loaded: bool,
    /// `POST /v1/models/switch` with this `model` alone would begin a switch
    /// now: the id resolves, its artifact is on disk, it is not the loaded
    /// one, and no switch is under way.
    switchable: bool,
}

/// The envelope of `GET /v1/models/catalog`, as `GET /v1/models`' is.
#[derive(Serialize, ToSchema)]
pub(crate) struct CatalogList {
    /// Always `list`.
    object: &'static str,
    /// The models, catalog entries first.
    data: Vec<CatalogRow>,
}

/// The rows, as of now. Reads the filesystem (sizes, and the header of an
/// artifact that is there), so a caller on the async runtime moves it off
/// the workers.
fn rows(server: &Server) -> Vec<CatalogRow> {
    let active = server.active();
    let loaded_id = active.model_id();
    let switcher = server.switcher.as_deref();
    // A switch beginning now: serving, failed (the way out of a failure
    // is a switch), or unloaded (a switch loads, spec model-switch/02). Not
    // during one or a load, not before the warm-up has run.
    let status = server.status();
    let free = matches!(*status, ModelStatus::Serving | ModelStatus::Failed { .. } | ModelStatus::Unloaded);
    // After a failed switch, or an unload, nothing is resident: the engine
    // `active` still holds is the torn-down one, so no row is loaded and its
    // model is a way out — "Load" on the unloaded model loads it.
    let serving = !status.no_model_loaded();
    let size_of = |path: &Path| std::fs::metadata(path).ok().filter(|meta| meta.is_file()).map(|meta| meta.len());
    let family_of = |path: &Path| crate::loader::artifact_family(path).ok().flatten();
    let row = |id: &str, source, on_disk: OnDisk, bytes, family| {
        let loaded = serving && id == loaded_id;
        CatalogRow {
            id: id.to_owned(),
            source,
            on_disk,
            bytes,
            family: if loaded { Some(scope(active.family)) } else { family },
            loaded,
            switchable: free && !loaded && switcher.is_some_and(|switcher| switcher.resolve(id).is_ok()),
        }
    };

    let mut rows = Vec::new();
    let mut listed = switcher.map(|switcher| switcher.listed_models()).unwrap_or_default();
    // The model loaded by an explicit artifact is the server's whether or not
    // anyone listed it; the one row that is `loaded` must always exist.
    if let Some(source) = &active.source {
        listed.entry(source.model.clone()).or_insert_with(|| source.artifact.clone());
    }
    let mut seen = BTreeSet::new();
    if let Some(catalog) = switcher.and_then(|switcher| switcher.catalog()) {
        for found in crate::download::command::list_rows(&catalog.catalog, &catalog.dir, &size_of, &family_of) {
            seen.insert(found.entry.id.clone());
            // A listed model of the same id wins over `download.path`, as it
            // does for a switch (`Switcher::resolve`): its file is where the
            // server would load it from, and gone it is absent, whatever
            // `download.path` holds.
            let (on_disk, bytes, family) = match listed.get(&found.entry.id) {
                Some(artifact) => match size_of(artifact) {
                    Some(bytes) => (OnDisk::Complete, bytes, family_of(artifact).map(scope)),
                    None => (OnDisk::Absent, found.entry.total_bytes(), None),
                },
                None => (found.on_disk, found.entry.total_bytes(), found.family.map(scope)),
            };
            rows.push(row(&found.entry.id, found.entry.layer.into(), on_disk, Some(bytes), family));
        }
    }
    for (id, artifact) in listed.iter().filter(|(id, _)| !seen.contains(*id)) {
        let bytes = size_of(artifact);
        let on_disk = if bytes.is_some() { OnDisk::Complete } else { OnDisk::Absent };
        let family = bytes.and_then(|_| family_of(artifact)).map(scope);
        rows.push(row(id, CatalogSource::Known, on_disk, bytes, family));
    }
    // A loaded model with no artifact at all (a start built in process).
    if serving && !rows.iter().any(|row| row.loaded) {
        rows.push(row(&loaded_id, CatalogSource::Known, OnDisk::Absent, None, None));
    }
    rows
}

/// `GET /v1/models/catalog` — the models this server can switch to or fetch.
#[utoipa::path(
    get,
    path = "/v1/models/catalog",
    tag = "models",
    operation_id = "list_model_catalog",
    summary = "The models this server can switch to or fetch",
    description = "One row per model the server knows of: every catalog entry (the built-in ones, then the operator's `download.catalog`), then every known model (`switch.known_models`, and the model the server started on) that is not a catalog entry. Each row says where the model is from (`source`), how much of it is there (`on_disk`: `complete`, `partial` or `absent` under `download.path`, or `complete` when the artifact a listed model or the start names is a file — file sizes only, never a hash, as `ignis-server model list` computes it; a known model is `complete` when its artifact is a file), its size, the model family its artifact's header names, whether it is the model serving now (`loaded`: exactly one row, none while a failed switch has left nothing serving) and whether `POST /v1/models/switch` with this `model` alone would begin a switch now (`switchable`: the id resolves, the artifact is on disk, it is not the loaded one, and no switch is under way). No row carries a path: switch by id. A model not on disk is fetched with `ignis-server model download <id>`; a switch never starts a download.\n\nAnswers during a model switch and while one has failed, like `GET /v1/models`.",
    responses(
        (status = 200, description = "The models, catalog entries first.", body = CatalogList),
        (status = 401, description = "The server was started with an API key and the request carried no matching bearer token.", body = ApiError),
        (status = 503, description = "The loaded model's first traversal has not run yet (`server_not_ready`).", body = ApiError),
    ),
)]
pub(crate) async fn get_models_catalog(State(server): State<Arc<Server>>) -> Json<CatalogList> {
    let server = Server::clone(&server);
    let rows = tokio::task::spawn_blocking(move || rows(&server)).await.expect("listing the catalog does not panic");
    Json(CatalogList { object: "list", data: rows })
}
