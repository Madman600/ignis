//! What an image request waits on before its job (spec image-model/05
//! §Streaming, §Prompt enhancement, GitHub #322): the load of an unloaded
//! model and the switch to the image model, streamed as `loading` and
//! `switching` phases ahead of the job's own. The models are
//! `support/switch.rs`'s mock loader's: the text models on `MockCompute`,
//! the image model on `FakeImageRunner`.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::compute::ModelKind;
use ignis_server::model_switch::{visit, LastSwitch, SwitchOutcome, SwitchRefusal, SwitchStarted, SwitchStep, Switcher};
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::{source, MockLoader};

/// The image model's served id: its family's own.
const IMAGE: &str = "ignis-image-2.1-turbo";

const PATIENT: Duration = Duration::from_secs(30);

/// The idle clock's tick and `model.idle_unload`, as `idle_unload.rs` runs
/// them.
const TICK: Duration = Duration::from_millis(50);
const IDLE: Duration = Duration::from_secs(1);

/// The known-models table naming each of `ids` at the artifact [`source`]
/// gives it.
fn known(ids: &[&str]) -> BTreeMap<String, PathBuf> {
    ids.iter().map(|id| (id.to_string(), source(id).artifact)).collect()
}

/// A server on `start` whose requests may switch to any of `ids`.
fn implicit_on(loader: &Arc<MockLoader>, start: &str, ids: &[&str]) -> Server {
    Server::from_active(loader.model(start))
        .with_switcher(Switcher::new(Arc::clone(loader) as _, PATIENT).with_known_models(known(ids)))
}

fn generation(body: Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/images/generations")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

/// A streamed answer's events, `(event, data)` in order.
async fn events_of(app: &axum::Router, body: Value) -> Vec<(String, Value)> {
    let response = app.clone().oneshot(generation(body)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()[header::CONTENT_TYPE], "text/event-stream");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    text.split("\n\n")
        .filter(|block| !block.trim().is_empty())
        .map(|block| {
            let field = |name: &str| block.lines().find_map(|line| line.strip_prefix(name)).map(str::trim).unwrap_or_default().to_owned();
            (field("event:"), serde_json::from_str(&field("data:")).expect("a JSON event"))
        })
        .collect()
}

/// Each event as `phase step/total`, or its type when it has no phase.
fn phases(events: &[(String, Value)]) -> Vec<String> {
    events
        .iter()
        .map(|(event, data)| match data["phase"].as_str() {
            Some(phase) => format!("{phase} {}/{}", data["step"], data["total"]),
            None => event.clone(),
        })
        .collect()
}

/// The phases of a job of one image after what comes before it.
fn then_the_job(before: &[&str]) -> Vec<String> {
    let mut expected: Vec<String> = before.iter().map(|phase| phase.to_string()).collect();
    expected.push("encoding 1/1".into());
    expected.extend((1..=8).map(|step| format!("denoising {step}/8")));
    expected.push("decoding 1/1".into());
    expected.push("image_generation.completed".into());
    expected
}

/// Wait (bounded) until the status reads `wanted`.
async fn until(server: &Server, wanted: &str, within: Duration) {
    let deadline = Instant::now() + within;
    while server.status().as_str() != wanted {
        assert!(Instant::now() < deadline, "the status never read {wanted}: {:?}", server.status());
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Spec image-model/05 §Streaming: a streamed request that switches from a
/// text model to the image model it names hears `switching` and `loading`
/// before its job's phases; one on an image model the idle clock unloaded
/// hears `loading`. A refusal that needs the image model loaded comes after
/// the stream began: it is the stream's `error` event.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stream_hears_the_switch_and_the_load_before_its_job() {
    let loader = MockLoader::new();
    let server = implicit_on(&loader, "mock-a", &["mock-a", IMAGE]).with_idle_unload(IDLE);
    let app = server.app();
    let body = json!({ "prompt": "a lighthouse", "model": IMAGE, "size": "1024x1024", "seed": 1, "stream": true });
    let events = events_of(&app, body.clone()).await;
    assert_eq!(phases(&events), then_the_job(&["switching 1/1", "loading 1/1"]));
    assert_eq!(events[0].0, "image_generation.progress");
    assert_eq!((server.active().kind(), loader.loads()), (ModelKind::Image, vec![IMAGE.to_owned()]));

    let _clock = server.spawn_idle_clock(TICK).expect("a clock");
    until(&server, "unloaded", IDLE * 5).await;
    let mut unnamed = body.clone();
    unnamed.as_object_mut().unwrap().remove("model");
    let events = events_of(&app, unnamed).await;
    assert_eq!(phases(&events), then_the_job(&["loading 1/1"]));
    assert_eq!(loader.loads(), [IMAGE, IMAGE]);

    // Back on the text model, a prompt over the condition budget is known
    // only once the image model has loaded: an error event after the switch.
    let (status, _) = send_chat(&app).await;
    assert_eq!(status, StatusCode::OK);
    server.update_live(|live| live.image.condition_budget_tokens = 1);
    let events = events_of(&app, json!({ "prompt": "far too long a prompt", "model": IMAGE, "stream": true })).await;
    assert_eq!(phases(&events), ["switching 1/1", "loading 1/1", "error"]);
    assert_eq!(events[2].1["error"]["code"], "condition_budget_exceeded", "{:?}", events[2]);
}

/// A chat request naming `mock-a`, which switches back to it.
async fn send_chat(app: &axum::Router) -> (StatusCode, Value) {
    let body = json!({ "model": "mock-a", "messages": [{ "role": "user", "content": "hello" }], "max_tokens": 2 });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

// ── the visit (model_switch::visit) ───────────────────────────────────────

/// What a visit's switches told, in order.
fn recording_steps() -> (ignis_server::model_switch::Steps, Arc<std::sync::Mutex<Vec<SwitchStep>>>) {
    let told = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = Arc::clone(&told);
    (Arc::new(move |step| sink.lock().unwrap().push(step)), told)
}

/// A server on the image model, its `last_switch` set by an explicit switch
/// to it from `mock-b`, so that "left as it was" is a real assertion.
async fn on_the_image_model(loader: &Arc<MockLoader>) -> Server {
    let server = implicit_on(loader, "mock-b", &["mock-a", "mock-b", IMAGE]);
    let (_, task) = ignis_server::model_switch::begin(&server, source(IMAGE)).expect("a switch begins");
    assert_eq!(task.await.unwrap(), SwitchOutcome::Switched);
    assert_eq!(server.active().model_id(), IMAGE);
    server
}

fn the_switch_to_the_image_model() -> Option<LastSwitch> {
    Some(LastSwitch { from: "mock-b".into(), to: IMAGE.into(), outcome: SwitchOutcome::Switched })
}

/// Spec image-model/05 §Prompt enhancement: a visit is two switches in one
/// hold — to the visited model, handed over with the gate still closed
/// (every other request `503 model_switching`, a switch `409`), then back —
/// each model off the card before the next loads, and `last_switch` left as
/// it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_goes_to_the_visited_model_and_back_with_the_gate_closed_throughout() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    let app = server.app();
    let (steps, told) = recording_steps();
    let mut visit = visit(&server, source("mock-a"), source(IMAGE), Some(steps)).expect("a visit begins");
    let visited = visit.arrived().await.expect("the visited model serves");
    assert_eq!((visited.active().model_id(), visited.active().kind()), ("mock-a".to_owned(), ModelKind::Text));
    assert_eq!(*server.status(), ModelStatus::Switching { from: "mock-a".into(), to: IMAGE.into() });
    let (status, refused) = send_chat(&app).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::SERVICE_UNAVAILABLE, Some("model_switching")), "{refused}");
    let response = app.clone().oneshot(generation(json!({ "prompt": "x", "size": "1024x1024" }))).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(response.headers()[header::RETRY_AFTER], "1");
    let explicit = ignis_server::model_switch::begin(&server, source("mock-b")).map(|(started, _)| started);
    assert_eq!(explicit, Err(SwitchRefusal::InProgress(SwitchStarted { from: "mock-a".into(), to: IMAGE.into() })));

    assert_eq!(visit.on().await, SwitchOutcome::Switched);
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));
    assert_eq!(loader.loads(), [IMAGE, "mock-a", IMAGE]);
    assert_eq!(loader.resident_at_load(), [0, 0, 0], "each model left the card before the next loaded");
    assert_eq!(*told.lock().unwrap(), [SwitchStep::Switching, SwitchStep::Loading, SwitchStep::Switching, SwitchStep::Loading]);
    assert_eq!(server.switcher.as_ref().unwrap().last_switch(), the_switch_to_the_image_model());
}

/// A visit whose way there fails is over: refused before anything stopped,
/// the image model never stopped serving; failed on the card, the image
/// model is loaded again. A visit whose way back fails leaves the visited
/// model serving. A visit its beginner leaves goes back on its own. None of
/// them writes `last_switch`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_visit_that_fails_either_way_ends_with_a_model_serving_and_writes_no_last_switch() {
    let loader = MockLoader::new();
    let server = on_the_image_model(&loader).await;
    let last = || server.switcher.as_ref().unwrap().last_switch();

    loader.refuse("mock-a");
    let mut refused = visit(&server, source("mock-a"), source(IMAGE), None).unwrap();
    assert!(refused.arrived().await.is_none());
    assert!(matches!(refused.on().await, SwitchOutcome::Refused { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id(), loader.loads().len()), ("serving", IMAGE.to_owned(), 1));
    assert_eq!(last(), the_switch_to_the_image_model());

    loader.break_load("mock-b");
    let mut broken = visit(&server, source("mock-b"), source(IMAGE), None).unwrap();
    assert!(broken.arrived().await.is_none());
    assert!(matches!(broken.on().await, SwitchOutcome::Restored { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", IMAGE.to_owned()));
    assert_eq!(loader.loads(), [IMAGE, "mock-b", IMAGE]);
    assert_eq!(last(), the_switch_to_the_image_model());

    loader.mend("mock-b");
    loader.break_load(IMAGE);
    let mut back_broken = visit(&server, source("mock-b"), source(IMAGE), None).unwrap();
    assert!(back_broken.arrived().await.is_some());
    assert!(matches!(back_broken.on().await, SwitchOutcome::Restored { .. }));
    assert_eq!((server.status().as_str(), server.active().model_id()), ("serving", "mock-b".to_owned()));
    assert_eq!(last(), the_switch_to_the_image_model());

    // From mock-b again: an explicit switch, which writes `last_switch`.
    loader.mend(IMAGE);
    let (_, task) = ignis_server::model_switch::begin(&server, source(IMAGE)).unwrap();
    assert_eq!(task.await.unwrap(), SwitchOutcome::Switched);
    let loads = loader.loads().len();
    drop(visit(&server, source("mock-b"), source(IMAGE), None).unwrap());
    until(&server, "serving", PATIENT).await;
    assert_eq!(server.active().model_id(), IMAGE, "a visit left by its beginner goes back");
    assert_eq!(loader.loads()[loads..], ["mock-b", IMAGE]);
    assert_eq!(last(), the_switch_to_the_image_model());
}
