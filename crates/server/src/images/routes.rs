//! The image routes (spec image-model/05 §Image routes, ADR 0050):
//! `POST /v1/images/generations` (JSON) and `POST /v1/images/edits`
//! (multipart), OpenAI's Images API on the image model, with the job queue
//! behind them ([`super::job`]).
//!
//! A request is checked whole before it costs anything: its fields against
//! `image.*`, an edit's condition images decoded — then the kind check and
//! the implicit switch ([`crate::api::switch_to_named_model`]), the queue's
//! admission, the condition budget ([`super::budget`]) and the resize, and
//! the job. Its answer is one JSON body, or with `stream` its events as
//! Server-Sent Events, one per progress unit and one per finished image.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Multipart, State};
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_core::Stream;
use ignis_core::compute::ModelKind;
use ignis_core::image::{ImageCondition, ImageSize, RgbaImage};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use utoipa::ToSchema;

use super::budget::{self, Fitted};
use super::job::{self, Background, ImageJob, JobEnd, JobEvent, Progress};
use super::ImageModel;
use crate::api::{
    bad_request_param, check_served_kind, error_response, error_response_naming, split_model_lane,
    switch_to_named_model, ApiError, ENGINE_ERROR_MESSAGE,
};
use crate::config::schema::ImageGroup;
use crate::Server;

/// `POST /v1/images/generations`'s body, and the fields of an edit's form
/// beside its images.
#[derive(Debug, Default, Deserialize, ToSchema)]
pub(crate) struct GenerationRequest {
    /// Absent, or the loaded image model's id: served as it is. Another known
    /// image model is switched to first (`switch.allow_implicit`); a text
    /// model is `409 model_kind_mismatch`.
    model: Option<String>,
    /// What to make. Required; passed to the model as it is.
    #[schema(required = true, value_type = String)]
    prompt: Option<String>,
    /// How many images, 1 to `image.max_n` (default 1): made one after
    /// another from one condition prefix.
    #[schema(minimum = 1)]
    n: Option<u32>,
    /// One of `image.sizes` (`<width>x<height>`), or `auto` (the default).
    size: Option<String>,
    /// `b64_json` only: the images come back inline. `url` is refused.
    response_format: Option<String>,
    /// `png` only.
    output_format: Option<String>,
    /// `auto` (the default) or `transparent`: the alpha channel as generated.
    /// `opaque`: composited on white, alpha 255.
    background: Option<String>,
    /// ignis: image `i` starts from the noise of `seed + i`. Absent, one is
    /// drawn and returned. The same seed gives the same bytes on the same
    /// build and machine.
    seed: Option<u64>,
    /// ignis: rewrite the prompt with the text model `image.prompt_enhancer`
    /// first; absent, `image.prompt_enhancement` decides.
    prompt_enhancement: Option<bool>,
    /// `true`: Server-Sent Events, the progress and each image as it is made.
    #[serde(default)]
    stream: bool,
}

/// `POST /v1/images/edits`'s form (`multipart/form-data`): the condition
/// images and the generation's fields, each a text part.
#[derive(Debug, Default, ToSchema)]
pub(crate) struct EditForm {
    /// The condition images, 1 to `image.max_condition_images`, in order:
    /// PNG, JPEG or WebP, with or without alpha. One may be sent as `image`.
    #[serde(rename = "image[]")]
    #[schema(value_type = Vec<String>, format = Binary)]
    images: Vec<Vec<u8>>,
    /// Not supported: `400`.
    #[schema(value_type = Option<String>, format = Binary)]
    mask: Option<Vec<u8>>,
    #[serde(flatten)]
    fields: GenerationRequest,
}

/// The images, as one JSON body.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ImagesResponse {
    /// Unix seconds.
    created: u64,
    data: Vec<ImageData>,
    /// ignis: the job's seed, the request's or the one drawn for it; image
    /// `i` started from `seed + i`.
    seed: u64,
    /// The images' size, `auto` resolved.
    size: String,
    background: &'static str,
    output_format: &'static str,
    /// ignis: each condition image's size as encoded, in order — only when
    /// the condition budget scaled them below 1024² of area.
    #[serde(skip_serializing_if = "Option::is_none")]
    condition_sizes: Option<Vec<String>>,
}

/// One image.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ImageData {
    /// The PNG, RGBA 8-bit, base64.
    b64_json: String,
    /// The prompt as prompt enhancement rewrote it, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
}

/// One Server-Sent Event of a streamed image request: `event:` is its
/// `type`, `data:` the object. `image_generation.*` on `/v1/images/generations`,
/// `image_edit.*` on `/v1/images/edits` (OpenAI's names for each route's
/// events).
#[derive(Debug, Serialize, ToSchema)]
#[serde(untagged)]
pub(crate) enum ImageStreamEvent {
    Progress(ProgressEvent),
    Completed(CompletedEvent),
    Error(ErrorEvent),
}

/// `image_generation.progress` / `image_edit.progress`: a unit of the job's
/// work starting — `encoding` once, then per image each `denoising` step
/// and its `decoding`. `enhancing`, `switching` and `loading` come first
/// when the prompt is rewritten.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ProgressEvent {
    r#type: &'static str,
    #[serde(flatten)]
    progress: Progress,
}

/// `image_generation.completed` / `image_edit.completed`: one per image, as
/// soon as it is made. No partial images are sent.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct CompletedEvent {
    r#type: &'static str,
    /// Which image, 0-based.
    image: u32,
    b64_json: String,
    created_at: u64,
    size: String,
    background: &'static str,
    output_format: &'static str,
    /// The job's seed: this image started from `seed + image`.
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    revised_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    condition_sizes: Option<Vec<String>>,
}

/// `error`: the job ended before its last image, the status already sent.
#[derive(Debug, Serialize, ToSchema)]
pub(crate) struct ErrorEvent {
    r#type: &'static str,
    #[serde(flatten)]
    error: ApiError,
}

/// Which route a request came by: the names of its streamed events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Generations,
    Edits,
}

impl Route {
    fn progress(self) -> &'static str {
        match self {
            Self::Generations => "image_generation.progress",
            Self::Edits => "image_edit.progress",
        }
    }

    fn completed(self) -> &'static str {
        match self {
            Self::Generations => "image_generation.completed",
            Self::Edits => "image_edit.completed",
        }
    }
}

/// A request whose fields passed their rules.
#[derive(Debug)]
struct ImageRequest {
    model: Option<String>,
    prompt: String,
    n: u32,
    /// `None`: `auto`.
    size: Option<ImageSize>,
    background: Background,
    seed: Option<u64>,
    stream: bool,
}

/// The fields' rules (spec image-model/05 §Image routes) against `limits`.
fn checked(fields: GenerationRequest, limits: &ImageGroup) -> Result<ImageRequest, Response> {
    let Some(prompt) = fields.prompt else {
        return Err(bad_request_param("prompt is required", "prompt"));
    };
    let n = fields.n.unwrap_or(1);
    if n == 0 || n > limits.max_n {
        return Err(bad_request_param(&format!("n must be between 1 and {} (image.max_n)", limits.max_n), "n"));
    }
    let size = match fields.size.as_deref() {
        None | Some("auto") => None,
        Some(asked) => match ImageSize::parse(asked).ok().filter(|size| limits.sizes.contains(size)) {
            Some(size) => Some(size),
            None => {
                let sizes: Vec<String> = limits.sizes.iter().map(ToString::to_string).collect();
                return Err(bad_request_param(
                    &format!("size `{asked}` is not served: it must be auto or one of {} (image.sizes)", sizes.join(", ")),
                    "size",
                ));
            }
        },
    };
    match fields.response_format.as_deref() {
        None | Some("b64_json") => {}
        Some("url") => {
            return Err(bad_request_param(
                "response_format url is not served: the images come back inline as b64_json, never stored to be fetched",
                "response_format",
            ))
        }
        Some(other) => return Err(bad_request_param(&format!("response_format `{other}` is not b64_json"), "response_format")),
    }
    match fields.output_format.as_deref() {
        None | Some("png") => {}
        Some(other) => return Err(bad_request_param(&format!("output_format `{other}` is not served: png only"), "output_format")),
    }
    let background = match fields.background.as_deref() {
        None => Background::Auto,
        Some(asked) => Background::parse(asked).ok_or_else(|| {
            bad_request_param(&format!("background `{asked}` is not auto, transparent or opaque"), "background")
        })?,
    };
    Ok(ImageRequest { model: fields.model, prompt, n, size, background, seed: fields.seed, stream: fields.stream })
}

/// `auto` (spec image-model/05 §Image routes): among `sizes`, the one whose
/// aspect ratio is nearest `ratio` — 1 for a generation, the last condition
/// image's for an edit — the larger on a tie, so 2048x2048 for a generation
/// with the default sizes.
fn auto_size(sizes: &[ImageSize], ratio: f64) -> Option<ImageSize> {
    // On the log scale, where 2:1 and 1:2 are as far from 1:1.
    let distance = |size: &ImageSize| ((f64::from(size.width) / f64::from(size.height)).ln() - ratio.ln()).abs();
    sizes.iter().copied().min_by(|a, b| {
        distance(a).total_cmp(&distance(b)).then_with(|| b.latent_cells().cmp(&a.latent_cells()))
    })
}

/// `503` with `code` and `Retry-After: 1`.
fn unavailable(type_: &str, code: &str, message: impl Into<String>) -> Response {
    let mut response = error_response(StatusCode::SERVICE_UNAVAILABLE, type_, code, message);
    response.headers_mut().insert(axum::http::header::RETRY_AFTER, HeaderValue::from_static("1"));
    response
}

/// What a request whose job the image model's stop ended is told.
const MODEL_STOPPED_MESSAGE: &str =
    "the image model was stopped (a model switch or a reload) before the job finished; retry once GET /v1/models reports `serving`";

/// The image model was stopped (a switch, a reload) before the job could
/// finish: `503 model_stopped`, with `Retry-After`.
fn model_stopped() -> Response {
    unavailable("server_error", "model_stopped", MODEL_STOPPED_MESSAGE)
}

/// The runner refused a call: `500 engine_error`.
fn engine_error() -> Response {
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "engine_error", "engine_error", ENGINE_ERROR_MESSAGE)
}

/// What a request carries past its checks into the job.
struct Prepared {
    model: ImageModel,
    job: ImageJob,
    fitted: Fitted,
}

#[utoipa::path(
    post,
    path = "/v1/images/generations",
    tag = "images",
    operation_id = "create_image",
    summary = "Images from a prompt",
    description = "OpenAI's image generation on the image model (`ignis-image-2.1-turbo`, ADR 0050): `n` RGBA PNGs from `prompt`, as `b64_json`.

`size` is one of `image.sizes` (by default the checkpoint's presets 2048x2048, 2400x1792, 1792x2400, 2528x1696, 1696x2528, 2752x1536, 1536x2752, and 1024x1024) or `auto`, 2048x2048. `response_format` is `b64_json` only (`url` is a 400), `output_format` `png` only. `background: opaque` composites the image on white with alpha 255; `auto` and `transparent` keep the alpha as generated.

ignis adds `seed` (image `i` starts from `seed + i`; absent, one is drawn and returned in `seed`) and `prompt_enhancement`. Accepted and inert: `quality`, `style`, `moderation`, `output_compression`, `partial_images` (no partial image is ever sent), `user`; any other field is ignored.

One job runs at a time and up to `image.queue_depth` wait, in order; past that the answer is `503 engine_full` with `Retry-After`. A client that disconnects cancels its job, while it waits or between two denoising steps.

`stream: false` answers one JSON body. `stream: true` answers `text/event-stream`, each event an `event:` named by its `type` and a `data:` object: `image_generation.progress` (`{phase, image, step, total}`) as each unit of work starts -- `encoding` once, then per image one `denoising` per step and its `decoding` -- and one `image_generation.completed` per image with its `b64_json`, as soon as it is made. A job that ends early sends one `error` event (`{type, error}`) instead of the rest.

`model` absent, or the loaded image model's id, is served as it is; another known image model is switched to first when `switch.allow_implicit` is on. With a text model loaded and no `model` naming the image model, the answer is `409 model_kind_mismatch`.",
    request_body = GenerationRequest,
    responses(
        (status = 200, description = "The images. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ImagesResponse = "application/json"),
            (ImageStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "A field outside its rule: no `prompt`, `n` outside 1..`image.max_n`, a `size` not in `image.sizes` (the message lists them), `response_format: url`, an `output_format` other than `png`, an unknown `background`; or a prompt longer than the condition budget (`condition_budget_exceeded`).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 409, description = "The model that would serve the request is a text model (`model_kind_mismatch`, naming both kinds and the loaded id).", body = ApiError),
        (status = 500, description = "The image model refused a call (`engine_error`).", body = ApiError),
        (status = 503, description = "The queue is full (`engine_full`, with `Retry-After`); a model switch is under way (`model_switching`, with `Retry-After`) or did not land (`model_switch_failed`); the image model was stopped before the job finished (`model_stopped`, with `Retry-After`); or the model was unloaded while idle and loading it again failed (`model_load_failed`).", body = ApiError),
    ),
)]
pub(crate) async fn create_image(State(server): State<Arc<Server>>, Json(fields): Json<GenerationRequest>) -> Response {
    let request = match checked(fields, &server.live().image) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    serve(server, request, Vec::new(), Route::Generations).await
}

#[utoipa::path(
    post,
    path = "/v1/images/edits",
    tag = "images",
    operation_id = "create_image_edit",
    summary = "Images from a prompt and condition images",
    description = "OpenAI's image edit on the image model: `n` RGBA PNGs from `prompt` and 1 to `image.max_condition_images` condition images (`image[]`, or `image` for one; PNG, JPEG or WebP, with or without alpha), as a `multipart/form-data` form whose other parts are the generation's fields as text. `mask` is not supported (400).

Each condition image is resized to 1024x1024 pixels of area at its own aspect ratio, each side a multiple of 32 (diffusers' `calculate_dimensions`). When the prompt and the images exceed the condition budget -- the load's, lowered by `image.condition_budget_tokens`; a condition image costs one token per 16x16 pixels, 4,096 at 1024x1024 -- every image is scaled down by the same factor until they fit, and `condition_sizes` reports the sizes encoded; below `image.condition_floor_px` squared of area the request is a 400 naming the budget.

`size: auto` is the size in `image.sizes` whose aspect ratio is nearest the last condition image's. Streaming, the events are `image_edit.progress` and `image_edit.completed`, otherwise as for a generation. Every other field, the queue and the switch are as for `POST /v1/images/generations`.",
    request_body(content = EditForm, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "The images. `application/json` when `stream` is false or absent; `text/event-stream` when it is true.", content(
            (ImagesResponse = "application/json"),
            (ImageStreamEvent = "text/event-stream"),
        )),
        (status = 400, description = "As for a generation; or no condition image, more than `image.max_condition_images`, one that is not a PNG, JPEG or WebP, a `mask`, or condition images that do not fit the condition budget even at `image.condition_floor_px` (`condition_budget_exceeded`).", body = ApiError),
        (status = 401, description = "The server was started with `--server-api-key` and the request carried no matching bearer token.", body = ApiError),
        (status = 404, description = "The request named a model this server neither loads nor may switch to.", body = ApiError),
        (status = 409, description = "The model that would serve the request is a text model (`model_kind_mismatch`).", body = ApiError),
        (status = 500, description = "The image model refused a call (`engine_error`).", body = ApiError),
        (status = 503, description = "As for a generation: `engine_full`, `model_switching`, `model_switch_failed`, `model_stopped`, `model_load_failed`.", body = ApiError),
    ),
)]
pub(crate) async fn create_image_edit(State(server): State<Arc<Server>>, multipart: Multipart) -> Response {
    let form = match read_form(multipart).await {
        Ok(form) => form,
        Err(refused) => return refused,
    };
    let limits = server.live().image.clone();
    if form.mask.is_some() {
        return bad_request_param("mask is not supported: an edit is conditioned on whole images", "mask");
    }
    let request = match checked(form.fields, &limits) {
        Ok(request) => request,
        Err(refused) => return refused,
    };
    let count = form.images.len();
    if count == 0 || count > limits.max_condition_images as usize {
        return bad_request_param(
            &format!(
                "an edit takes 1 to {} condition images (image.max_condition_images), got {count}",
                limits.max_condition_images
            ),
            "image",
        );
    }
    let decoded = tokio::task::spawn_blocking(move || {
        form.images
            .iter()
            .enumerate()
            .map(|(i, bytes)| crate::media::decode_condition_image(bytes).map_err(|why| format!("condition image {i}: {why}")))
            .collect::<Result<Vec<_>, _>>()
    })
    .await
    .expect("decoding condition images does not panic");
    match decoded {
        Ok(images) => serve(server, request, images, Route::Edits).await,
        Err(why) => bad_request_param(&why, "image"),
    }
}

/// An edit's form: the image parts in order, the mask's presence, and the
/// text parts as the generation's fields. Parts it does not know are
/// skipped, OpenAI's inert ones among them.
async fn read_form(mut multipart: Multipart) -> Result<EditForm, Response> {
    let malformed = |error: axum::extract::multipart::MultipartError| {
        bad_request_param(&format!("the multipart form could not be read: {}", error.body_text()), "image")
    };
    let mut form = EditForm::default();
    while let Some(field) = multipart.next_field().await.map_err(malformed)? {
        let name = field.name().unwrap_or_default().to_owned();
        match name.as_str() {
            "image" | "image[]" => form.images.push(field.bytes().await.map_err(malformed)?.to_vec()),
            "mask" => form.mask = Some(field.bytes().await.map_err(malformed)?.to_vec()),
            _ => {
                let text = field.text().await.map_err(malformed)?;
                set_field(&mut form.fields, &name, text)?;
            }
        }
    }
    Ok(form)
}

/// One text part of an edit's form into `fields`, parsed as the JSON body
/// would be.
fn set_field(fields: &mut GenerationRequest, name: &str, text: String) -> Result<(), Response> {
    fn parse<T: std::str::FromStr>(name: &str, text: &str, what: &str) -> Result<T, Response> {
        text.trim().parse().map_err(|_| bad_request_param(&format!("{name} must be {what}, got `{text}`"), name))
    }
    match name {
        "model" => fields.model = Some(text),
        "prompt" => fields.prompt = Some(text),
        "n" => fields.n = Some(parse(name, &text, "an integer")?),
        "size" => fields.size = Some(text),
        "response_format" => fields.response_format = Some(text),
        "output_format" => fields.output_format = Some(text),
        "background" => fields.background = Some(text),
        "seed" => fields.seed = Some(parse(name, &text, "an unsigned integer")?),
        "prompt_enhancement" => fields.prompt_enhancement = Some(parse(name, &text, "true or false")?),
        "stream" => fields.stream = parse(name, &text, "true or false")?,
        _ => {}
    }
    Ok(())
}

/// Serve a checked request on the image model: the kind check and the
/// implicit switch, the queue's admission, the condition budget and the
/// resize, then the job — answered whole, or streamed.
async fn serve(server: Arc<Server>, request: ImageRequest, images: Vec<RgbaImage>, route: Route) -> Response {
    if let Err(refused) = switch_to_named_model(&server, ModelKind::Image, request.model.as_deref()).await {
        return refused;
    }
    // One model for the whole request (spec model-switch/01), the image
    // model (spec image-model/05).
    let server = Arc::new(server.pinned());
    if let Err(refused) = check_served_kind(&server, ModelKind::Image, split_model_lane(request.model.clone()).0.as_deref()) {
        return refused;
    }
    let model = server.active().image_model().expect("the kind check passed: the image model").clone();
    let limits = server.live().image.clone();
    let Some(waiting) = model.jobs.admit(limits.queue_depth) else {
        if model.jobs.is_closed() {
            return model_stopped();
        }
        return unavailable(
            "engine_full",
            "engine_full",
            format!("the image queue is full: one job runs and {} wait (image.queue_depth); retry", limits.queue_depth),
        );
    };
    let ratio = images.last().map_or(1.0, |image| f64::from(image.size.width) / f64::from(image.size.height));
    let Some(size) = request.size.or_else(|| auto_size(&limits.sizes, ratio)) else {
        return bad_request_param("size auto has no size to pick: image.sizes is empty", "size");
    };
    let seed = request.seed.or(server.seedless_seed).unwrap_or_else(crate::api::fresh_seed);
    let ImageRequest { prompt, n, background, stream, .. } = request;
    let prepared = {
        let model = model.clone();
        tokio::task::spawn_blocking(move || prepare(model, &limits, prompt, images, size, n, seed, background))
            .await
            .expect("fitting condition images does not panic")
    };
    let Prepared { model, job, fitted } = match prepared {
        Ok(prepared) => prepared,
        Err(refused) => return refused,
    };
    let created = crate::api::now();
    let answer = Answer {
        route,
        created,
        seed,
        size: size.to_string(),
        background: background.as_str(),
        condition_sizes: fitted.scaled.then(|| fitted.sizes.iter().map(ToString::to_string).collect()),
    };
    let n = job.n;
    let events = job::spawn(model, job, waiting);
    if stream {
        return Sse::new(ImageSse { events, answer, ended: false }).into_response();
    }
    collect(events, answer, n).await
}

/// The condition budget and the resize, then the job: blocking, since the
/// prompt's tokens are the runner's to count — it holds the image model's
/// tokenizer — and the resize is per pixel.
#[allow(clippy::too_many_arguments)]
fn prepare(
    model: ImageModel,
    limits: &ImageGroup,
    prompt: String,
    images: Vec<RgbaImage>,
    size: ImageSize,
    n: u32,
    seed: u64,
    background: Background,
) -> Result<Prepared, Response> {
    let Some(prompt_tokens) = model.with_runner(|runner| runner.prompt_tokens(&prompt, images.len())) else {
        return Err(model_stopped());
    };
    let budget = budget::budget(model.plan(), limits.condition_budget_tokens);
    let sources: Vec<ImageSize> = images.iter().map(|image| image.size).collect();
    let fitted = budget::fit(prompt_tokens, &sources, budget, limits.condition_floor_px).map_err(|over| {
        let param = if images.is_empty() { "prompt" } else { "image" };
        error_response_naming(StatusCode::BAD_REQUEST, "invalid_request_error", "condition_budget_exceeded", over.to_string(), Some(param))
    })?;
    let images = images.iter().zip(&fitted.sizes).map(|(image, &size)| crate::media::resize_rgba(image, size)).collect();
    let job = ImageJob { condition: ImageCondition { prompt, images }, size, n, seed, background };
    Ok(Prepared { model, job, fitted })
}

/// What every answer to one request says beside its images.
struct Answer {
    route: Route,
    created: u64,
    seed: u64,
    size: String,
    background: &'static str,
    condition_sizes: Option<Vec<String>>,
}

/// A request that did not stream: the job's `n` images, collected, or why
/// it ended without them.
async fn collect(mut events: mpsc::UnboundedReceiver<JobEvent>, answer: Answer, n: u32) -> Response {
    let mut data = Vec::with_capacity(n as usize);
    while let Some(event) = events.recv().await {
        match event {
            JobEvent::Progress(_) => {}
            JobEvent::Image { png, .. } => data.push(ImageData { b64_json: job::base64(&png), revised_prompt: None }),
            JobEvent::Ended(JobEnd::Stopped) => return model_stopped(),
            JobEvent::Ended(_) => return engine_error(),
        }
    }
    if data.len() != n as usize {
        // The job's thread went away without saying why: it panicked.
        return engine_error();
    }
    Json(ImagesResponse {
        created: answer.created,
        data,
        seed: answer.seed,
        size: answer.size,
        background: answer.background,
        output_format: "png",
        condition_sizes: answer.condition_sizes,
    })
    .into_response()
}

/// A streamed request: the job's events as Server-Sent Events. Dropped — the
/// client gone — it drops the channel, which cancels the job.
struct ImageSse {
    events: mpsc::UnboundedReceiver<JobEvent>,
    answer: Answer,
    /// The last event was sent.
    ended: bool,
}

impl ImageSse {
    fn event(event: &ImageStreamEvent) -> Event {
        let kind = match event {
            ImageStreamEvent::Progress(progress) => progress.r#type,
            ImageStreamEvent::Completed(completed) => completed.r#type,
            ImageStreamEvent::Error(error) => error.r#type,
        };
        Event::default().event(kind).data(serde_json::to_string(event).expect("an image event serializes"))
    }
}

impl Stream for ImageSse {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.ended {
            return Poll::Ready(None);
        }
        let event = match this.events.poll_recv(cx) {
            Poll::Pending => return Poll::Pending,
            Poll::Ready(Some(JobEvent::Progress(progress))) => {
                ImageStreamEvent::Progress(ProgressEvent { r#type: this.answer.route.progress(), progress })
            }
            Poll::Ready(Some(JobEvent::Image { index, png })) => ImageStreamEvent::Completed(CompletedEvent {
                r#type: this.answer.route.completed(),
                image: index,
                b64_json: job::base64(&png),
                created_at: this.answer.created,
                size: this.answer.size.clone(),
                background: this.answer.background,
                output_format: "png",
                seed: this.answer.seed,
                revised_prompt: None,
                condition_sizes: this.answer.condition_sizes.clone(),
            }),
            Poll::Ready(Some(JobEvent::Ended(end))) => {
                // What the request would have been answered had it not
                // streamed: the status is already 200.
                this.ended = true;
                let error = match end {
                    JobEnd::Stopped => ApiError::new("server_error", "model_stopped", MODEL_STOPPED_MESSAGE),
                    _ => ApiError::new("engine_error", "engine_error", ENGINE_ERROR_MESSAGE),
                };
                ImageStreamEvent::Error(ErrorEvent { r#type: "error", error })
            }
            // The job made its images, or its thread went away: the stream
            // ends either way, and a client counts the images it got.
            Poll::Ready(None) => return Poll::Ready(None),
        };
        Poll::Ready(Some(Ok(Self::event(&event))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DEFAULT_IMAGE_SIZES;

    fn limits() -> ImageGroup {
        ImageGroup::default()
    }

    fn request(fields: serde_json::Value) -> Result<ImageRequest, Response> {
        checked(serde_json::from_value(fields).unwrap(), &limits())
    }

    async fn refusal(result: Result<ImageRequest, Response>) -> (StatusCode, serde_json::Value) {
        let response = result.expect_err("refused");
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }

    /// AC 4: the field rules.
    #[tokio::test]
    async fn every_field_rule_refuses_with_a_400_naming_the_field() {
        let ok = request(serde_json::json!({ "prompt": "a fox" })).unwrap();
        assert_eq!((ok.n, ok.size, ok.background, ok.stream), (1, None, Background::Auto, false));
        let full = request(serde_json::json!({
            "prompt": "", "n": 10, "size": "1536x2752", "response_format": "b64_json", "output_format": "png",
            "background": "opaque", "seed": 9, "stream": true, "quality": "high", "user": "u"
        }))
        .unwrap();
        assert_eq!((full.n, full.size, full.background, full.seed), (10, Some(ImageSize::new(1536, 2752)), Background::Opaque, Some(9)));
        for (fields, param, words) in [
            (serde_json::json!({}), "prompt", "required"),
            (serde_json::json!({ "prompt": "x", "n": 0 }), "n", "between 1 and 10"),
            (serde_json::json!({ "prompt": "x", "n": 11 }), "n", "image.max_n"),
            (serde_json::json!({ "prompt": "x", "size": "512x512" }), "size", "2048x2048, 2400x1792"),
            (serde_json::json!({ "prompt": "x", "size": "big" }), "size", "1024x1024 (image.sizes)"),
            (serde_json::json!({ "prompt": "x", "response_format": "url" }), "response_format", "b64_json"),
            (serde_json::json!({ "prompt": "x", "response_format": "xml" }), "response_format", "xml"),
            (serde_json::json!({ "prompt": "x", "output_format": "webp" }), "output_format", "png only"),
            (serde_json::json!({ "prompt": "x", "background": "green" }), "background", "green"),
        ] {
            let (status, body) = refusal(request(fields.clone())).await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{fields}");
            assert_eq!(body["error"]["param"], param, "{fields}: {body}");
            assert!(body["error"]["message"].as_str().unwrap().contains(words), "{fields}: {body}");
        }
    }

    /// AC 4: `auto` is 2048x2048 for a generation and, for an edit, the
    /// preset nearest the last condition image's aspect ratio.
    #[test]
    fn auto_is_the_square_or_the_preset_nearest_the_last_images_ratio() {
        let sizes = DEFAULT_IMAGE_SIZES;
        assert_eq!(auto_size(&sizes, 1.0), Some(ImageSize::new(2048, 2048)), "the larger square on the tie");
        assert_eq!(auto_size(&sizes, 16.0 / 9.0), Some(ImageSize::new(2752, 1536)));
        assert_eq!(auto_size(&sizes, 9.0 / 16.0), Some(ImageSize::new(1536, 2752)));
        assert_eq!(auto_size(&sizes, 4.0 / 3.0), Some(ImageSize::new(2400, 1792)));
        assert_eq!(auto_size(&sizes, 3.0 / 2.0), Some(ImageSize::new(2528, 1696)));
        assert_eq!(auto_size(&sizes, 1.0 / 1.4), Some(ImageSize::new(1792, 2400)));
        assert_eq!(auto_size(&[ImageSize::new(1024, 1024)], 3.0), Some(ImageSize::new(1024, 1024)));
        assert_eq!(auto_size(&[], 1.0), None);
    }

    /// An edit's text parts parse as the JSON fields do, and a malformed
    /// number is a 400 naming its part.
    #[tokio::test]
    async fn an_edits_text_parts_parse_as_the_json_fields() {
        let mut fields = GenerationRequest::default();
        for (name, text) in [("prompt", "a cat"), ("n", "3"), ("seed", "42"), ("stream", "true"), ("prompt_enhancement", "false"), ("quality", "high")] {
            set_field(&mut fields, name, text.to_owned()).unwrap();
        }
        assert_eq!((fields.prompt.as_deref(), fields.n, fields.seed, fields.stream), (Some("a cat"), Some(3), Some(42), true));
        assert_eq!(fields.prompt_enhancement, Some(false));
        let refused = set_field(&mut fields, "n", "three".to_owned()).unwrap_err();
        let (status, body) = refusal(Err(refused)).await;
        assert_eq!((status, body["error"]["param"].as_str()), (StatusCode::BAD_REQUEST, Some("n")));
    }
}
