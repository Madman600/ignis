//! The image leaf's runner end to end (spec image-model/04 and 05, GitHub #321): the real
//! artifact loaded through the server's loader (`ArtifactLoader`, as a start, a switch and the idle
//! unload's reload load it), then the image routes through the router.
//!
//! - **The count.** The runner's `condition_tokens` for the oracle's two cases is the prefix
//!   diffusers laid out (`case.json`'s segments): the real tokenizer and templates.
//! - **A generation**, 1024², n = 1, a fixed seed: a 1024x1024 RGBA PNG, and the same bytes again.
//! - **An edit** with the oracle's condition image: a 1024x1024 RGBA PNG, the same bytes again.
//! - **A refusal**: a prompt whose RoPE positions run past the table is `400 image_refused`,
//!   never a 500.
//! - **A runner dropped holding a prefix** (a switch or an unload mid-job, when the job's own
//!   release finds no runner) frees it: the device's free memory after that drop is what it was
//!   after a clean one.
//! - **The arena beside the budget** (review-321ab I3): on a budget larger than this card's, ten
//!   condition images at the worst aspect ratio fit the budget but not the arena, and the count
//!   says so, so the fit scales them.
//! - **Every default size decodes** (review-321c L2) within the arena the load sizes for the
//!   largest of them: the leaf's own band plan, host-side.
//!
//! The PNGs are written beside the free runs' (`<oracle>/../freerun/http/`) for the eye. The
//! artifact is `$IGNIS_IMAGE_ARTIFACT`, else
//! `F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer`; the oracle
//! `$IGNIS_IMAGE_ORACLE`, else `F:/ai/models/ignis-image-2.1-turbo/oracle`. Explicit GPU profile
//! (ADR 0006): outside `IGNIS_GPU_PROFILE=1` a missing GPU or fixture is a skip, under it a hard
//! failure (`ignis_core::gpu_profile`). Run via `scripts/gpu-profile.ps1`.

#![cfg(feature = "cuda")]

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use http_body_util::BodyExt;
use ignis_artifact::CudaDevice;
use ignis_core::gpu_profile;
use ignis_core::image::{ConditionUnfit, ImageCondition, ImageSize, RgbaImage};
use ignis_server::model_switch::{ArtifactLoader, ModelLoader};
use ignis_server::{ActiveModel, ModelSource, Server};
use serde_json::{json, Value};
use tower::ServiceExt;

const IMAGE: &str = "ignis-image-2.1-turbo";

fn machine_local(var: &str, default: &str) -> PathBuf {
    PathBuf::from(std::env::var(var).ok().filter(|s| !s.is_empty()).unwrap_or_else(|| default.to_string()))
}

/// The artifact and the oracle, or None after skip_or_fail.
fn fixtures() -> Option<(PathBuf, PathBuf)> {
    let artifact = machine_local("IGNIS_IMAGE_ARTIFACT", "F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer");
    let oracle = machine_local("IGNIS_IMAGE_ORACLE", "F:/ai/models/ignis-image-2.1-turbo/oracle");
    if !artifact.is_file() {
        gpu_profile::skip_or_fail(&format!("the image runner: no artifact at {} (IGNIS_IMAGE_ARTIFACT)", artifact.display()));
        return None;
    }
    if !oracle.join("edit/condition.png").is_file() || !oracle.join("t2i/case.json").is_file() {
        gpu_profile::skip_or_fail(&format!("the image runner: no oracle at {} (IGNIS_IMAGE_ORACLE)", oracle.display()));
        return None;
    }
    if let Err(e) = CudaDevice::nvml_memory(0) {
        gpu_profile::skip_or_fail(&format!("the image runner: no GPU: {e}"));
        return None;
    }
    Some((artifact, oracle))
}

fn case(dir: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(dir.join("case.json")).unwrap()).unwrap()
}

/// The prefix diffusers laid out for a case: its last segment's end.
fn recorded_prefix(case: &Value) -> u64 {
    case["segments"].as_array().and_then(|segments| segments.last()).and_then(|last| last[1].as_u64()).expect("the case's segments")
}

/// The image model loaded through the server's loader, on the options `args` give over the
/// defaults, and how long the load took; the load's own refusal is a failure under the profile.
async fn load(artifact: &Path, args: &[&str]) -> Option<(ActiveModel, Duration)> {
    let args: Vec<String> = args.iter().map(|arg| (*arg).to_owned()).collect();
    let options = match ignis_server::config::resolve(&args, |_| None).expect("the options resolve") {
        ignis_server::config::ConfigOutcome::Config(config) => config,
        other => panic!("{other:?}"),
    };
    let target = ModelSource { artifact: artifact.to_path_buf(), model: IMAGE.to_owned() };
    let start = Instant::now();
    let loaded = tokio::task::spawn_blocking(move || ArtifactLoader::new(options).prepare(&target, None)?.load()).await.unwrap();
    match loaded {
        Ok(loaded) => Some((loaded.model, start.elapsed())),
        Err(e) => {
            gpu_profile::skip_or_fail(&format!("the image model's load: {e}"));
            None
        }
    }
}

/// The device's free memory as NVML reads it, once it has settled.
fn free_memory() -> u64 {
    std::thread::sleep(Duration::from_millis(500));
    CudaDevice::nvml_memory(0).expect("NVML").0
}

fn json_request(uri: &str, body: &Value) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

const BOUNDARY: &str = "ignis-image-runner-gpu";

/// An edit: `fields` as text parts, `image` as the one condition image.
fn edit_request(fields: &[(&str, &str)], image: &[u8]) -> Request<Body> {
    let mut body = Vec::new();
    for (name, text) in fields {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{text}\r\n").bytes());
    }
    body.extend(
        format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"image[]\"; filename=\"condition.png\"\r\nContent-Type: image/png\r\n\r\n")
            .bytes(),
    );
    body.extend_from_slice(image);
    body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
    Request::builder()
        .method(Method::POST)
        .uri("/v1/images/edits")
        .header(header::CONTENT_TYPE, format!("multipart/form-data; boundary={BOUNDARY}"))
        .body(Body::from(body))
        .unwrap()
}

/// The answer's status and body, and how long it took.
async fn send(app: &axum::Router, request: Request<Body>) -> (StatusCode, Value, Duration) {
    let start = Instant::now();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body = serde_json::from_slice(&bytes).unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
    (status, body, start.elapsed())
}

/// Standard base64, padded.
fn unbase64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let mut out = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        let digits: Vec<u8> = chunk.iter().filter(|&&c| c != b'=').map(|&c| value(c)).collect();
        let n = digits.iter().enumerate().fold(0u32, |n, (i, &d)| n | u32::from(d) << (18 - 6 * i));
        out.extend((0..digits.len() - 1).map(|i| (n >> (16 - 8 * i)) as u8));
    }
    out
}

/// The answer's images' PNG bytes, in order.
fn pngs_of(answer: &Value) -> Vec<Vec<u8>> {
    answer["data"].as_array().expect("data").iter().map(|image| unbase64(image["b64_json"].as_str().expect("b64_json"))).collect()
}

/// The answer's one image: its PNG bytes, decoded as RGBA8.
fn png_of(answer: &Value) -> (Vec<u8>, RgbaImage) {
    let pngs = pngs_of(answer);
    assert_eq!(pngs.len(), 1, "{answer}");
    let png = pngs[0].clone();
    let mut reader = png::Decoder::new(Cursor::new(png.clone())).read_info().expect("a PNG");
    let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
    let frame = reader.next_frame(&mut pixels).unwrap();
    assert_eq!((frame.color_type, frame.bit_depth), (png::ColorType::Rgba, png::BitDepth::Eight));
    (png, RgbaImage { size: ImageSize::new(frame.width, frame.height), pixels })
}

/// Each channel's standard deviation over the image: a picture, not a flat field.
fn spread(image: &RgbaImage) -> [f64; 4] {
    let n = (image.pixels.len() / 4) as f64;
    std::array::from_fn(|c| {
        let mean = image.pixels.chunks_exact(4).map(|px| f64::from(px[c])).sum::<f64>() / n;
        (image.pixels.chunks_exact(4).map(|px| (f64::from(px[c]) - mean).powi(2)).sum::<f64>() / n).sqrt()
    })
}

fn keep(out: Option<&Path>, name: &str, png: &[u8]) {
    if let Some(out) = out {
        std::fs::create_dir_all(out).unwrap();
        std::fs::write(out.join(name), png).unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
async fn the_image_model_serves_generations_and_edits_over_http_and_frees_what_it_holds() {
    let Some((artifact, oracle)) = fixtures() else { return };
    let (t2i, edit) = (case(&oracle.join("t2i")), case(&oracle.join("edit")));
    let condition = std::fs::read(oracle.join("edit/condition.png")).unwrap();
    let out = oracle.parent().map(|models| models.join("freerun")).filter(|freerun| freerun.is_dir()).map(|freerun| freerun.join("http"));
    let Some((model, took)) = load(&artifact, &[]).await else { return };
    let image = model.image_model().expect("the image model").clone();
    let plan = image.plan().clone();
    println!(
        "loaded through the server's loader in {:.1} s: condition budget {} tokens, {} sigmas, max target {} cells",
        took.as_secs_f64(),
        plan.condition_budget_tokens,
        plan.sample_sigmas.len(),
        plan.max_target_cells
    );
    assert_eq!(plan.max_target_cells, 16_800, "the arena for the largest default size, 2400x1792");

    // The count is the prefix diffusers laid out, with the artifact's own tokenizer and templates.
    let square = ImageSize::new(1024, 1024);
    let prompt = |case: &Value| case["case"]["prompt"].as_str().unwrap().to_owned();
    let counted = image.with_runner(|runner| (runner.condition_tokens(&prompt(&t2i), &[]), runner.condition_tokens(&prompt(&edit), &[square]))).unwrap();
    assert_eq!(counted, (Ok(recorded_prefix(&t2i)), Ok(recorded_prefix(&edit))), "the runner's count is diffusers' prefix");

    let server = Server::from_active(model);
    let app = server.app();

    // A generation, twice: the same seed, the same bytes.
    let body = json!({ "prompt": prompt(&t2i), "size": "1024x1024", "n": 1, "seed": 7 });
    let (status, first, took) = send(&app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (png, generated) = png_of(&first);
    println!("generation 1024², 8 steps: {:.2} s (the first job: lazy kernel loads)", took.as_secs_f64());
    let (status, again, took) = send(&app, json_request("/v1/images/generations", &body)).await;
    assert_eq!(status, StatusCode::OK, "{again}");
    println!("generation 1024² again: {:.2} s", took.as_secs_f64());
    assert_eq!(generated.size, square);
    assert_eq!(png_of(&again).0, png, "the same seed, the same bytes");
    let spread_t2i = spread(&generated);
    println!("generation: channel spread {spread_t2i:.1?}");
    assert!(spread_t2i[..3].iter().all(|&s| s > 10.0), "a picture, not a flat field: {spread_t2i:?}");
    keep(out.as_deref(), "t2i-seed7.png", &png);
    // n = 2 (review-321d Info): image i starts from seed + i, image 1 after image 0's decode in
    // the same job over the same prefix. Image 0 is seed 7's, image 1 seed 8's alone.
    let (status, two, took) = send(&app, json_request("/v1/images/generations", &json!({ "prompt": prompt(&t2i), "size": "1024x1024", "n": 2, "seed": 7 }))).await;
    assert_eq!(status, StatusCode::OK, "{two}");
    println!("generation 1024², n = 2: {:.2} s", took.as_secs_f64());
    let (status, eight, _) = send(&app, json_request("/v1/images/generations", &json!({ "prompt": prompt(&t2i), "size": "1024x1024", "n": 1, "seed": 8 }))).await;
    assert_eq!(status, StatusCode::OK, "{eight}");
    let two = pngs_of(&two);
    assert_eq!(two.len(), 2);
    assert_eq!(two[0], png, "image 0 of seed 7 is seed 7's");
    assert_eq!(two[1], png_of(&eight).0, "image 1 of seed 7 is image 0 of seed 8");
    assert_ne!(two[0], two[1]);

    // An edit of the oracle's condition image, twice.
    let fields = [("prompt", prompt(&edit)), ("size", "1024x1024".to_owned()), ("seed", "3".to_owned())];
    let fields: Vec<(&str, &str)> = fields.iter().map(|(name, value)| (*name, value.as_str())).collect();
    let (status, edited, took) = send(&app, edit_request(&fields, &condition)).await;
    assert_eq!(status, StatusCode::OK, "{edited}");
    println!("edit 1024² (one 1024² condition image): {:.2} s", took.as_secs_f64());
    let (png, made) = png_of(&edited);
    assert_eq!(made.size, square);
    assert!(edited.get("condition_sizes").is_none(), "a 1024² condition image fits the budget as it is: {edited}");
    let (status, again, _) = send(&app, edit_request(&fields, &condition)).await;
    assert_eq!(status, StatusCode::OK, "{again}");
    assert_eq!(png_of(&again).0, png, "the same seed, the same bytes");
    println!("edit: channel spread {:.1?}, alpha below 255 in {} pixels", spread(&made), made.pixels.chunks_exact(4).filter(|px| px[3] < 255).count());
    keep(out.as_deref(), "edit-seed3.png", &png);

    // A prompt past the RoPE table (8,192 positions) and within the budget: refused as the
    // request's, before the leaf allocates anything.
    let long = "fox ".repeat(9000);
    let (status, refused, _) = send(&app, json_request("/v1/images/generations", &json!({ "prompt": long, "size": "1024x1024", "seed": 1 }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{refused}");
    assert_eq!(refused["error"]["code"], "image_refused", "{refused}");
    println!("a 9,000-word prompt: 400, {}", refused["error"]["message"]);

    // A clean unload, then the free memory: the reference.
    let active = server.active();
    tokio::task::spawn_blocking(move || active.shut_down()).await.unwrap();
    drop((app, server, image));
    let clean = free_memory();

    // Loaded again, a prefix held, and the model shut down under it (a switch's or an unload's
    // drain mid-job): the runner's drop frees the prefix, ~1 GiB, with the rest of the model.
    let Some((model, _)) = load(&artifact, &[]).await else { return };
    let image = model.image_model().expect("the image model").clone();
    let mut pixels = png::Decoder::new(Cursor::new(&condition)).read_info().unwrap();
    let mut rgba = vec![0; pixels.output_buffer_size().unwrap()];
    let frame = pixels.next_frame(&mut rgba).unwrap();
    let held = ImageCondition { prompt: prompt(&edit), images: vec![RgbaImage { size: ImageSize::new(frame.width, frame.height), pixels: rgba }] };
    let loaded = free_memory();
    image.with_runner(|runner| runner.encode_condition(&held)).unwrap().expect("the edit's prefix encodes");
    let holding = free_memory();
    tokio::task::spawn_blocking(move || model.shut_down()).await.unwrap();
    drop(image);
    let dropped = free_memory();
    let mib = |bytes: u64| bytes as f64 / (1u64 << 20) as f64;
    println!(
        "free after a clean unload {clean} B; loaded again {loaded} B; holding the edit's prefix {holding} B ({:.1} MiB); after the drop that held it \
         {dropped} B ({:+.1} MiB from the clean unload)",
        mib(loaded.saturating_sub(holding)),
        (dropped as f64 - clean as f64) / (1u64 << 20) as f64
    );
    assert!(loaded.saturating_sub(holding) > 512 << 20, "the held prefix shows: {:.1} MiB", mib(loaded.saturating_sub(holding)));
    // NVML's free memory is the card's, the desktop's own use in it: two runs of 321f's gate read
    // +84.6 and -46.5 MiB here. A prefix left behind reads about -1,042 MiB.
    assert!(dropped.abs_diff(clean) < 256 << 20, "the runner's drop frees the prefix it held: {:.1} MiB apart", mib(dropped.abs_diff(clean)));
}

/// Review-321ab I3: the leaf's arena is a second limit on a condition beside the budget. With a
/// budget past what the card has (an explicit 40 GiB, oversubscribed: a larger card's), ten
/// condition images at the worst condition size (13120x96, 1,230 merged tokens each, spec
/// image-model/03: a 16380x100 source at 1024² of area) fit the budget, and the count says the
/// runner has no room for their tower and encoder; the fit then scales them to sizes the arena
/// takes, from the default 512 floor, whose own sizes (6560x32) the processor refuses (review-321d:
/// the fit no longer reads the floor first). A size the processor would not keep (13088x64, past
/// 200:1) is refused in its words. The RoPE table is a third limit (review-321d L1): five 12800x64
/// images take 4,000 of its 8,192 positions, and a prompt of ~4,300 words on top has no room for
/// the target, which scaling the images makes. Host-side counts only, after the load: nothing is
/// encoded.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "GPU: run under scripts/gpu-profile.ps1"]
async fn the_count_holds_the_arena_and_the_rope_table_beside_the_budget() {
    use ignis_server::images::budget;
    let Some((artifact, oracle)) = fixtures() else { return };
    let edit = case(&oracle.join("edit"));
    let prompt = edit["case"]["prompt"].as_str().unwrap();
    let Some((model, _)) = load(&artifact, &["--vram-budget-bytes", "40G", "--vram-allow-oversubscription"]).await else { return };
    let image = model.image_model().expect("the image model").clone();
    let budget = image.plan().condition_budget_tokens;
    let sources = vec![ImageSize::new(16380, 100); 10];
    let worst = vec![budget::condition_size(1024.0 * 1024.0, 163.8); 10];
    assert_eq!(worst[0], ImageSize::new(13120, 96), "the worst condition size");
    let squares = vec![ImageSize::new(1024, 1024); 10];
    let long = format!("{prompt} {}", "fox ".repeat(4300));
    let five = vec![ImageSize::new(12800, 64); 5];
    let (counted_worst, counted_squares, fitted, past_ratio) = image
        .with_runner(|runner| {
            let fitted = budget::fit(|sizes| runner.condition_tokens(prompt, sizes), &sources, budget, 512);
            let past_ratio = runner.condition_tokens(prompt, &[ImageSize::new(13088, 64); 10]);
            (runner.condition_tokens(prompt, &worst), runner.condition_tokens(prompt, &squares), fitted, past_ratio)
        })
        .unwrap();
    let cells = |sizes: &[ImageSize]| sizes.iter().map(|size| size.latent_cells() as u64).sum::<u64>();
    println!(
        "a {budget}-token budget: ten 13120x96 images ({} latent cells) count {counted_worst:?}; ten 1024² ({} cells) count          {counted_squares:?}; the fit: {:?}; ten 13088x64: {past_ratio:?}",
        cells(&worst),
        cells(&squares),
        fitted
    );
    assert!(cells(&worst) < budget, "the worst case fits the budget: {} cells", cells(&worst));
    assert!(matches!(counted_worst, Err(ConditionUnfit::NoRoom(_))), "past the arena's room: {counted_worst:?}");
    assert!(matches!(&past_ratio, Err(ConditionUnfit::Refused(why)) if why.contains("200:1")), "past 200:1: {past_ratio:?}");
    assert!(matches!(counted_squares, Ok(tokens) if tokens <= budget && tokens > cells(&squares)), "ten 1024² images fit both");
    let fitted = fitted.expect("the fit scales the worst case into the arena, from the default floor");
    assert!(fitted.scaled, "{fitted:?}");
    let counted = image.with_runner(|runner| runner.condition_tokens(prompt, &fitted.sizes)).unwrap();
    assert!(matches!(counted, Ok(tokens) if tokens <= budget), "the fitted sizes count within the budget, the arena and the processor: {counted:?}");

    // The RoPE table: five 12800x64 images take 5 x 800 positions before the target.
    let (short, long_five, long_fitted) = image
        .with_runner(|runner| {
            let fitted = budget::fit(|sizes| runner.condition_tokens(&long, sizes), &[ImageSize::new(16380, 100); 5], budget, 512);
            (runner.condition_tokens(prompt, &five), runner.condition_tokens(&long, &five), fitted)
        })
        .unwrap();
    println!("five 12800x64: {short:?} with the edit's prompt, {long_five:?} with ~4,300 more words; the fit then: {long_fitted:?}");
    assert!(matches!(short, Ok(tokens) if tokens <= budget), "{short:?}");
    assert!(matches!(&long_five, Err(ConditionUnfit::NoRoom(why)) if why.contains("8192")), "no RoPE position left for the target: {long_five:?}");
    let long_fitted = long_fitted.expect("smaller images leave the target a position");
    assert!(long_fitted.scaled && long_fitted.sizes[0].width < 12800, "{long_fitted:?}");
    let counted = image.with_runner(|runner| runner.condition_tokens(&long, &long_fitted.sizes)).unwrap();
    assert!(matches!(counted, Ok(tokens) if tokens <= budget), "{counted:?}");
    tokio::task::spawn_blocking(move || model.shut_down()).await.unwrap();
}

/// Review-321c L2: `vae_decode` refuses a size whose bands do not fit the arena, and the load
/// sizes the arena for the largest of `image.sizes` within `image.max_target_cells` (`max_target_cells`). Every default size must
/// then decode in it. Host-side, the kernel leaf's sizes: no GPU call.
#[test]
#[ignore = "the kernel leaf (cuda): run under scripts/gpu-profile.ps1"]
fn every_default_size_decodes_within_the_arena_the_load_sizes() {
    let sizes = ignis_server::config::DEFAULT_IMAGE_SIZES;
    let max_target = u32::try_from(ignis_server::images::leaf::max_target_cells(&sizes, ignis_core::image_leaf::MAX_TARGET_TOKENS)).unwrap();
    for size in sizes {
        match ignis_core::image_leaf::decode_band_rows(max_target, size.width, size.height) {
            Ok(rows) => println!("{size}: decodes in bands of {rows} rows in an arena for {max_target} cells"),
            Err(e) => panic!("{size}: {e}"),
        }
    }
}
