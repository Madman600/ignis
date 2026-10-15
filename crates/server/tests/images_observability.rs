//! What an image job leaves behind (GitHub #324 item 4): one `ignis.image.job`
//! log line per finished job and the `ignis_image_*` families on `/metrics`,
//! over the router on a fake image runner (ADR 0006).
//!
//! A binary of its own with one test, which installs the process's
//! subscriber: the job's line is written on the blocking thread it runs on,
//! which a subscriber scoped to the test's thread never sees, and a callsite
//! another test settled with no subscriber would never be captured.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_core::image::{
    fake_plan, ConditionId, ConditionUnfit, FakeImageRunner, ImageCondition, ImageError, ImagePlan, ImageRunner, ImageSize, RgbaImage, TargetLatents,
};
use ignis_logging::{JsonLayer, MemorySink};
use ignis_server::images::ImageModel;
use ignis_server::{ActiveModel, Server};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tower::ServiceExt;
use tracing_subscriber::layer::SubscriberExt;

const IMAGE: &str = "ignis-image-2.1-turbo";

/// The fake runner whose decode fails while `fail_decode` is set: a job that
/// ends in `failed`.
struct Flaky {
    inner: FakeImageRunner,
    fail_decode: Arc<AtomicBool>,
}

impl ImageRunner for Flaky {
    fn plan(&self) -> &ImagePlan {
        self.inner.plan()
    }

    fn condition_tokens(&self, prompt: &str, images: &[ImageSize]) -> Result<u64, ConditionUnfit> {
        self.inner.condition_tokens(prompt, images)
    }

    fn encode_condition(&mut self, condition: &ImageCondition) -> Result<ConditionId, ImageError> {
        self.inner.encode_condition(condition)
    }

    fn denoise_step(
        &mut self,
        condition: ConditionId,
        latents: &mut TargetLatents,
        sigma: f32,
        sigma_next: f32,
        velocity: Option<&mut Vec<f32>>,
    ) -> Result<(), ImageError> {
        self.inner.denoise_step(condition, latents, sigma, sigma_next, velocity)
    }

    fn vae_decode(&mut self, latents: &TargetLatents) -> Result<RgbaImage, ImageError> {
        match self.fail_decode.load(Ordering::SeqCst) {
            true => Err(ImageError::Leaf("the decode broke".into())),
            false => self.inner.vae_decode(latents),
        }
    }

    fn release_condition(&mut self, condition: ConditionId) {
        self.inner.release_condition(condition);
    }

    fn condition_bytes(&self) -> u64 {
        self.inner.condition_bytes()
    }
}

async fn post(app: &axum::Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&bytes).unwrap_or(Value::Null))
}

async fn scrape(metrics: &axum::Router) -> String {
    let request = Request::builder().method(Method::GET).uri("/metrics").body(Body::empty()).unwrap();
    let bytes = metrics.clone().oneshot(request).await.unwrap().into_body().collect().await.unwrap().to_bytes();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The value of the sample `series` (name and labels as printed).
fn sample(text: &str, series: &str) -> String {
    text.lines()
        .find_map(|line| line.strip_prefix(series).and_then(|rest| rest.strip_prefix(' ')))
        .unwrap_or_else(|| panic!("no `{series}` in:\n{text}"))
        .to_owned()
}

/// A job that makes every image is one log line saying what the request
/// was, and moves the families; a job that fails is one more line and one
/// more count. A request refused after it was admitted leaves no job in the
/// queue gauge (review-324serving F1).
#[tokio::test]
async fn a_finished_image_job_logs_one_line_and_moves_the_image_families() {
    let log = std::sync::Arc::new(MemorySink::new());
    tracing::subscriber::set_global_default(tracing_subscriber::registry().with(JsonLayer::new(log.clone()))).expect("the only subscriber of this binary");

    let fail_decode = Arc::new(AtomicBool::new(false));
    let runner = Flaky { inner: FakeImageRunner::new(fake_plan()), fail_decode: Arc::clone(&fail_decode) };
    let server = Server::from_active(ActiveModel::image(ImageModel::new(IMAGE, Box::new(runner)))).with_metrics();
    server.update_live(|live| live.image.sizes = vec![ImageSize::new(64, 64)]);
    let (app, metrics) = (server.app(), server.metrics_app().expect("metrics are on"));
    assert!(!scrape(&metrics).await.contains("ignis_image_"), "a server that has run no image job renders none");

    let (status, answer) = post(&app, "/v1/images/generations", &json!({ "prompt": "a red fox", "n": 2, "size": "64x64", "seed": 3 })).await;
    assert_eq!(status, StatusCode::OK, "{answer}");

    let lines: Vec<Value> = log
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.image.job")
        .collect();
    assert_eq!(lines.len(), 1, "one line per finished job: {lines:?}");
    let field = |name: &str| lines[0].get(name).or_else(|| lines[0]["attributes"].get(name)).cloned().unwrap_or(Value::Null);
    assert!(field("job_id").as_str().is_some_and(|id| id.starts_with("img-")), "{}", lines[0]);
    for (name, want) in [
        ("kind", json!("generation")),
        ("outcome", json!("completed")),
        ("model", json!(IMAGE)),
        ("n", json!(2)),
        ("images", json!(2)),
        ("size", json!("64x64")),
        ("condition_images", json!(0)),
        ("enhancement", json!("off")),
    ] {
        assert_eq!(field(name), want, "{name}: {}", lines[0]);
    }
    for phase in ["enhance_ms", "switch_ms", "queued_ms", "encode_ms", "denoise_ms", "decode_ms", "run_ms", "total_ms"] {
        assert!(field(phase).is_u64(), "{phase}: {}", lines[0]);
    }
    assert!(field("total_ms").as_u64() >= field("run_ms").as_u64(), "{}", lines[0]);

    let text = scrape(&metrics).await;
    assert_eq!(sample(&text, "ignis_image_jobs_total{outcome=\"completed\"}"), "1", "{text}");
    assert_eq!(sample(&text, "ignis_image_jobs_total{outcome=\"failed\"}"), "0", "{text}");
    assert_eq!(sample(&text, "ignis_image_images_total"), "2", "{text}");
    assert_eq!(sample(&text, "ignis_image_job_duration_seconds_count"), "1", "{text}");
    assert_eq!(sample(&text, "ignis_image_queue_jobs{state=\"waiting\"}"), "0", "{text}");
    assert_eq!(sample(&text, "ignis_image_queue_jobs{state=\"running\"}"), "0", "{text}");
    for phase in ["queued", "encoding", "denoising", "decoding"] {
        assert!(text.contains(&format!("ignis_image_job_phase_seconds_total{{phase=\"{phase}\"}}")), "{phase}: {text}");
    }

    // Past admission, the condition budget refuses the request: no job ran,
    // and none is left counted as waiting.
    let budget = server.live().image.condition_budget_tokens;
    server.update_live(|live| live.image.condition_budget_tokens = 1);
    let (status, refused) = post(&app, "/v1/images/generations", &json!({ "prompt": "far too long a prompt", "size": "64x64" })).await;
    assert_eq!((status, refused["error"]["code"].as_str()), (StatusCode::BAD_REQUEST, Some("condition_budget_exceeded")), "{refused}");
    server.update_live(|live| live.image.condition_budget_tokens = budget);
    let text = scrape(&metrics).await;
    assert_eq!(sample(&text, "ignis_image_queue_jobs{state=\"waiting\"}"), "0", "{text}");
    assert_eq!(sample(&text, "ignis_image_queue_jobs{state=\"running\"}"), "0", "{text}");

    // A job whose decode fails is one more line, `failed`, and one more
    // count; it is not a duration.
    fail_decode.store(true, Ordering::SeqCst);
    let (status, _) = post(&app, "/v1/images/generations", &json!({ "prompt": "a red fox", "size": "64x64", "seed": 3 })).await;
    assert!(!status.is_success(), "{status}");
    let outcomes: Vec<Value> = log
        .lines()
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("json"))
        .filter(|event| event["event_name"] == "ignis.image.job")
        .map(|event| event.get("outcome").or_else(|| event["attributes"].get("outcome")).cloned().unwrap_or(Value::Null))
        .collect();
    assert_eq!(outcomes, [json!("completed"), json!("failed")]);
    let text = scrape(&metrics).await;
    assert_eq!(sample(&text, "ignis_image_jobs_total{outcome=\"failed\"}"), "1", "{text}");
    assert_eq!(sample(&text, "ignis_image_job_duration_seconds_count"), "1", "{text}");
    assert_eq!(sample(&text, "ignis_image_queue_jobs{state=\"running\"}"), "0", "{text}");
}
