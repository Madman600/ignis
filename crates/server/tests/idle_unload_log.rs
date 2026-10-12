//! The log lines of idle unload (spec model-switch/02 AC 2 and 6): the
//! unload is `ignis.model.unloaded`, once, and the load a request begins
//! after it `ignis.model.reloaded` — neither of them a switch. A binary of
//! its own with one test, which installs the process's subscriber: the
//! unload's line is written on the blocking thread that tears the model
//! down, which a subscriber scoped to the test's thread never sees.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::model_switch::Switcher;
use ignis_server::{ModelStatus, Server};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

#[path = "support/switch.rs"]
mod switch_support;
use switch_support::MockLoader;

/// Every line `log` holds named `name`.
fn named(log: &MemorySink, name: &str) -> Vec<Value> {
    log.lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == name)
        .collect()
}

/// An event's field, top level or under `attributes`.
fn field(event: &Value, name: &str) -> Value {
    event.get(name).or_else(|| event["attributes"].get(name)).cloned().unwrap_or(Value::Null)
}

#[tokio::test]
async fn the_unload_and_the_load_after_it_are_logged_and_neither_is_a_switch() {
    let log = Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone())))
        .expect("the binary's only subscriber");
    let loader = MockLoader::new();
    let server = Server::from_active(loader.model("mock-a"))
        .with_switcher(Switcher::new(Arc::clone(&loader) as _, Duration::from_secs(30)))
        .with_idle_unload(Duration::from_secs(1));
    let app = server.app();
    let _clock = server.spawn_idle_clock(Duration::from_millis(50)).expect("a clock");

    let deadline = Instant::now() + Duration::from_secs(10);
    while named(&log, "ignis.model.unloaded").is_empty() {
        assert!(Instant::now() < deadline, "never unloaded");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(*server.status(), ModelStatus::Unloaded);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let unloaded = named(&log, "ignis.model.unloaded");
    assert_eq!(unloaded.len(), 1, "logged once: {unloaded:?}");
    assert_eq!(field(&unloaded[0], "model"), json!("mock-a"));
    assert_eq!(field(&unloaded[0], "idle_secs"), json!(1));

    server.update_live(|live| live.idle_unload = Duration::ZERO);
    let body = json!({ "messages": [{ "role": "user", "content": "hello there" }], "max_tokens": 3 });
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Its body read or not, a response held is a request in use.
    drop(response);

    let reloaded = named(&log, "ignis.model.reloaded");
    assert_eq!(reloaded.len(), 1, "{reloaded:?}");
    assert_eq!(field(&reloaded[0], "model"), json!("mock-a"));
    for timing in ["took_ms", "load_ms", "warm_up_ms"] {
        assert!(field(&reloaded[0], timing).is_u64(), "{timing}: {}", reloaded[0]);
    }
    assert!(named(&log, "ignis.model.switched").is_empty(), "neither is a switch");

    // Unloaded again, and the next load refused before it reaches the GPU:
    // `reload_failed` at the prepare stage, still not a switch.
    server.update_live(|live| live.idle_unload = Duration::from_secs(1));
    let deadline = Instant::now() + Duration::from_secs(10);
    while named(&log, "ignis.model.unloaded").len() < 2 {
        assert!(Instant::now() < deadline, "never unloaded again");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    server.update_live(|live| live.idle_unload = Duration::ZERO);
    loader.refuse("mock-a");
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    let failed = named(&log, "ignis.model.reload_failed");
    assert_eq!(failed.len(), 1, "{failed:?}");
    assert_eq!((field(&failed[0], "model"), field(&failed[0], "stage")), (json!("mock-a"), json!("prepare")), "{}", failed[0]);
    assert!(named(&log, "ignis.model.switch_failed").is_empty(), "not a switch either");
}
