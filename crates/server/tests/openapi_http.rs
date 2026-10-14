//! GitHub #251: the `/v1` surface documents itself. The document is
//! generated from the handler annotations by the same call that registers
//! the routes, so these tests are what stops the two drifting — a route
//! added with `.route()` alone fails [`the_document_lists_exactly_the_v1_surface`],
//! and a monitoring path that leaks in fails
//! [`the_document_carries_no_monitoring_path`].
//!
//! The page, the document and their assets answer without a key even on a
//! keyed server (the API's shape is not its load); the handler routes still
//! do not.

use std::collections::BTreeSet;
use std::sync::Arc;

use axum::body::Body;
use axum::http::{Method, Request, StatusCode, header};
use http_body_util::BodyExt;
use ignis_core::{ConcreteScheduler, MockCompute, SchedulerConfig};
use ignis_server::Server;
use ignis_server::config::ApiKey;
use ignis_server::engine::Engine;
use ignis_server::template::SimpleTemplateProvider;
use tower::ServiceExt;

const MODEL: &str = "test-model";
const KEY: &str = "sk-test-key";

fn server() -> Server {
    let scheduler = ConcreteScheduler::with_config(
        SchedulerConfig { model: MODEL.into(), ..SchedulerConfig::default() },
        Arc::new(MockCompute::new()),
    );
    Server::new(Engine::new(Box::new(scheduler)), Box::new(SimpleTemplateProvider))
}

async fn get(app: &axum::Router, uri: &str) -> axum::response::Response {
    let request = Request::builder().uri(uri).body(Body::empty()).unwrap();
    app.clone().oneshot(request).await.unwrap()
}

async fn body_bytes(response: axum::response::Response) -> Vec<u8> {
    response.into_body().collect().await.unwrap().to_bytes().to_vec()
}

fn header_value(response: &axum::response::Response, name: header::HeaderName) -> String {
    response
        .headers()
        .get(name)
        .map(|v| v.to_str().unwrap().to_owned())
        .unwrap_or_default()
}

/// The document itself, as the JSON a client generator reads. Asserting
/// over the serialized form rather than over `utoipa`'s types keeps these
/// tests about the contract, not about the generator's internals.
fn document() -> serde_json::Value {
    serde_json::to_value(ignis_server::api::openapi()).unwrap()
}

/// The document's paths and the methods each one serves.
fn documented() -> BTreeSet<String> {
    document()["paths"]
        .as_object()
        .expect("paths")
        .iter()
        .flat_map(|(path, item)| {
            item.as_object()
                .expect("a path item")
                .keys()
                .map(move |method| format!("{method} {path}"))
                .collect::<Vec<_>>()
        })
        .collect()
}

// ── the document ─────────────────────────────────────────────────────────

#[test]
fn the_document_lists_exactly_the_v1_surface() {
    // Set equality, both ways: a new `/v1` route that ships without its
    // `#[utoipa::path]` fails here, and so does a path entry whose route
    // was removed.
    let expected: BTreeSet<String> = [
        "get /v1/models",
        // Spec model-switch/01: replace the loaded model.
        "post /v1/models/switch",
        // Spec playground/05: the models a switch can go to or fetch, and
        // what each configuration field is.
        "get /v1/models/catalog",
        "get /v1/config/fields",
        // Spec playground/05 follow-up 2: where each of those values comes
        // from.
        "get /v1/config/origins",
        // Spec config-v2/02: the running configuration, read and changed.
        "get /v1/config",
        "patch /v1/config",
        "post /v1/chat/completions",
        // GitHub #285: the prompt counted without being served.
        "post /v1/tokenize",
        "post /v1/detokenize",
        "post /v1/responses",
        // GitHub #282: the Responses API's WebSocket mode, deliberately.
        "get /v1/responses",
        "post /v1/decide",
        // Spec image-model/05: the image model's routes.
        "post /v1/images/generations",
        "post /v1/images/edits",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    assert_eq!(documented(), expected);
}

/// Spec playground/05 AC 9: the Server view's routes are in the document
/// with their responses, and the rows they return are described.
#[test]
fn the_server_view_routes_are_documented_with_their_responses() {
    let document = document();
    for (path, list, ok) in [("/v1/models/catalog", "CatalogList", "CatalogRow"), ("/v1/config/fields", "FieldList", "FieldDeclaration"), ("/v1/config/origins", "OriginList", "OriginRow")] {
        let get = &document["paths"][path]["get"];
        for status in ["200", "401"] {
            assert!(get["responses"][status].is_object(), "{path} documents {status}");
        }
        assert_eq!(
            get["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            format!("#/components/schemas/{list}"),
            "{path}"
        );
        assert!(document["components"]["schemas"][ok]["properties"].is_object(), "{ok}");
    }
    // An origin's source is one of six, each named; no configuration is a 501.
    assert_eq!(
        document["components"]["schemas"]["OriginSource"]["enum"],
        serde_json::json!(["default", "profile", "file", "env", "flag", "live"])
    );
    assert!(document["paths"]["/v1/config/origins"]["get"]["responses"]["501"].is_object());
    // The switch takes a model alone, and says what a 404 is.
    let switch = &document["paths"]["/v1/models/switch"]["post"];
    assert!(switch["responses"]["404"].is_object());
    let request = &document["components"]["schemas"]["SwitchRequest"];
    assert_eq!(request["required"], serde_json::json!(["model"]), "artifact is optional");
    // `GET /v1/models` carries how the last switch ended.
    assert!(document["components"]["schemas"]["ModelList"]["properties"]["last_switch"].is_object());
    assert!(document["components"]["schemas"]["LastSwitchInfo"]["properties"]["outcome"].is_object());
}

/// Spec playground/05 AC 22: `GET /v1/config` documents its `format` query,
/// the two media types it answers in, and the `400` of any other format.
#[test]
fn the_config_route_documents_its_format_and_both_media_types() {
    let document = document();
    let get = &document["paths"]["/v1/config"]["get"];
    let parameters = get["parameters"].as_array().expect("parameters");
    let format = parameters.iter().find(|parameter| parameter["name"] == "format").expect("the format query");
    assert_eq!((format["in"].clone(), format["required"].clone()), (serde_json::json!("query"), serde_json::json!(false)));
    // The two formats are the enum the handler matches on, not a sentence.
    let schema = &format["schema"];
    let listed = schema.get("enum").or_else(|| schema["oneOf"].as_array()?.iter().find_map(|any| any.get("enum")));
    assert_eq!(listed, Some(&serde_json::json!(["json", "yaml"])), "{format}");
    let content = &get["responses"]["200"]["content"];
    assert_eq!(content["application/json"]["schema"]["$ref"], "#/components/schemas/ConfigDocument");
    assert_eq!(content["application/yaml"]["schema"]["type"], "string");
    assert!(get["responses"]["400"].is_object());
}

#[test]
fn the_counting_routes_carry_their_schemas() {
    // GitHub #285: the request and response shapes are in the document, not
    // only the two paths.
    let document = document();
    for schema in ["TokenizeRequest", "TokenizeResponse", "DetokenizeRequest", "DetokenizeResponse"] {
        assert!(document["components"]["schemas"][schema].is_object(), "{schema}");
    }
    assert_eq!(
        document["paths"]["/v1/tokenize"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"],
        "#/components/schemas/TokenizeRequest"
    );
}

#[test]
fn every_refused_field_is_named_in_its_operation_description() {
    // Spec server/09: driven off the table the validator reads, so a refusal
    // cannot be added without the document saying so.
    use ignis_server::openai_fields::{table, Surface};
    let document = document();
    for (path, surface) in [
        ("/v1/chat/completions", Surface::Chat),
        ("/v1/responses", Surface::Responses),
        ("/v1/images/generations", Surface::Images),
        ("/v1/images/edits", Surface::Images),
    ] {
        let description = document["paths"][path]["post"]["description"].as_str().unwrap_or_default();
        for rule in table(surface) {
            assert!(description.contains(&format!("`{}`", rule.name)), "{path}: `{}` not named", rule.name);
            if let Some(reason) = rule.refusal {
                assert!(description.contains(reason), "{path}: `{}`'s refusal not described", rule.name);
            }
        }
    }
    // What the chat operation honours beyond the classic fields.
    let chat = document["paths"]["/v1/chat/completions"]["post"]["description"].as_str().unwrap();
    for honoured in ["`max_completion_tokens`", "`stop`", "`usage.prompt_tokens_details.cached_tokens`"] {
        assert!(chat.contains(honoured), "{honoured}");
    }
}

#[test]
fn the_document_names_what_the_server_does_not_serve() {
    let description = document()["info"]["description"].as_str().unwrap_or_default().to_owned();
    for choice in ["embeddings", "`/v1/completions`", "`/v1/batches`", "audio", "LoRA", "more than one loaded model"] {
        assert!(description.contains(choice), "`{choice}` missing: {description}");
    }
}

#[test]
fn the_document_carries_no_monitoring_path() {
    // "Not for the monitoring" (GitHub #251), made checkable: the
    // Prometheus exposition is ADR 0017's contract, served on its own
    // listener, and the Playground's pages are assets rather than API.
    for path in document()["paths"].as_object().expect("paths").keys() {
        assert!(
            !path.contains("metrics") && !path.starts_with("/ui"),
            "the document must not carry a monitoring or Playground path: {path}",
        );
    }
}

#[test]
fn the_alias_is_named_but_not_listed() {
    // `/v1/systemone` is the same handler under Jev's name. OpenAPI has no
    // alias, so it is described in prose rather than duplicated as a path
    // with every schema reference under it repeated.
    let document = document();
    assert!(document["paths"]["/v1/systemone"].is_null());
    let description = document["info"]["description"].as_str().unwrap_or_default();
    assert!(
        description.contains("/v1/systemone"),
        "the alias must be named in the document: {description}",
    );
}

#[test]
fn the_error_envelope_is_one_component() {
    // Shared, not inlined per operation: a client handles failures on this
    // surface with one type.
    let document = document();
    assert!(document["components"]["schemas"]["ApiError"].is_object());
    let four_hundred = document["paths"]["/v1/chat/completions"]["post"]["responses"]["400"]
        ["content"]["application/json"]["schema"]["$ref"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    assert_eq!(four_hundred, "#/components/schemas/ApiError");
}

#[test]
fn the_streaming_half_is_documented_as_its_own_content_type() {
    // `stream: true` answers SSE chunks, not the single JSON body: a
    // client generator that sees only one of the two writes the wrong
    // reader.
    let document = document();
    let ok = &document["paths"]["/v1/chat/completions"]["post"]["responses"]["200"]["content"];
    assert!(ok["application/json"].is_object(), "{ok}");
    assert!(ok["text/event-stream"].is_object(), "{ok}");
}

/// Spec image-model/05 AC 13: both image routes, their ignis extensions,
/// their streamed events and their errors are in the document — the edit as
/// a multipart form.
#[test]
fn the_image_routes_document_their_extensions_events_and_errors() {
    let document = document();
    for path in ["/v1/images/generations", "/v1/images/edits"] {
        let post = &document["paths"][path]["post"];
        assert_eq!(post["tags"][0], "images", "{path}");
        let ok = &post["responses"]["200"]["content"];
        assert_eq!(ok["application/json"]["schema"]["$ref"], "#/components/schemas/ImagesResponse", "{path}");
        assert_eq!(ok["text/event-stream"]["schema"]["$ref"], "#/components/schemas/ImageStreamEvent", "{path}");
        for status in ["400", "401", "404", "409", "500", "503"] {
            assert!(post["responses"][status].is_object(), "{path} {status}");
        }
        let description = post["description"].as_str().unwrap_or_default();
        let errors = post["responses"]["503"]["description"].as_str().unwrap_or_default();
        assert!(errors.contains("engine_full") && errors.contains("model_stopped"), "{path}: {errors}");
        assert!(description.contains("progress") && description.contains("completed"), "{path}: {description}");
    }
    let generation = &document["paths"]["/v1/images/generations"]["post"];
    assert_eq!(generation["requestBody"]["content"]["application/json"]["schema"]["$ref"], "#/components/schemas/GenerationRequest");
    let edit = &document["paths"]["/v1/images/edits"]["post"]["requestBody"]["content"]["multipart/form-data"];
    assert_eq!(edit["schema"]["$ref"], "#/components/schemas/EditForm", "{edit}");
    let schemas = &document["components"]["schemas"];
    for field in ["seed", "prompt_enhancement", "size", "n", "background", "stream"] {
        assert!(schemas["GenerationRequest"]["properties"][field].is_object(), "{field}");
    }
    assert!(schemas["EditForm"].to_string().contains("image[]"), "{}", schemas["EditForm"]);
    assert!(schemas["ImagesResponse"]["properties"]["seed"].is_object());
    assert!(schemas["ImagesResponse"]["properties"]["condition_sizes"].is_object());
    // AC 9: the rewrite and the warnings a fallback leaves.
    assert!(schemas["ImagesResponse"]["properties"]["warnings"].is_object());
    assert!(schemas["ImageData"]["properties"]["revised_prompt"].is_object());
    assert!(schemas["CompletedEvent"]["properties"]["revised_prompt"].is_object() && schemas["CompletedEvent"]["properties"]["warnings"].is_object());
    let generation_text = generation["description"].as_str().unwrap();
    assert!(generation_text.contains("image.prompt_enhancer") && generation_text.contains("503 model_switching"), "{generation_text}");
    let events = schemas["ImageStreamEvent"].to_string();
    for event in ["ProgressEvent", "CompletedEvent", "ErrorEvent"] {
        assert!(events.contains(event), "{events}");
    }
    let phases = schemas["Phase"].to_string();
    for phase in ["enhancing", "switching", "loading", "encoding", "denoising", "decoding"] {
        assert!(phases.contains(phase), "{phases}");
    }
}

#[test]
fn a_forced_thinking_close_is_documented_as_an_optional_field_of_each_response_shape() {
    // Spec server/08: the field is absent when the close was not forced, so
    // a generated client must not require it.
    let document = document();
    for schema in ["CompletionChoice", "ChunkChoice", "ResponseObject"] {
        let schema = &document["components"]["schemas"][schema];
        assert!(schema["properties"]["thinking_budget_forced_at"].is_object(), "{schema}");
        let required = schema["required"].as_array().cloned().unwrap_or_default();
        assert!(!required.contains(&serde_json::json!("thinking_budget_forced_at")), "{schema}");
    }
}

#[test]
fn the_decision_endpoint_carries_its_question_kinds() {
    // `/v1/decide` is the one endpoint no external documentation covers,
    // so a path entry alone would not be worth much.
    let document = document();
    let text = document["components"]["schemas"]["QuestionKind"].to_string();
    for kind in ["noul", "choice", "score", "scalar", "number", "point", "box", "locate"] {
        assert!(text.contains(kind), "QuestionKind must name {kind}: {text}");
    }
    assert!(document["components"]["schemas"]["DecideResponse"].is_object());
}

/// GitHub #260, #263 (ADR 0036): a `point`'s and a `box`'s `method`, the
/// head answer's fields and the head set's `extent` are part of the
/// contract, readable without the source.
#[test]
fn the_decision_endpoint_documents_how_a_point_and_a_box_are_answered() {
    let document = document();
    let schemas = &document["components"]["schemas"];
    let method = schemas["SpatialMethod"].to_string();
    for value in ["head", "chain"] {
        assert!(method.contains(value), "SpatialMethod must name {value}: {method}");
    }
    assert!(
        schemas["Question"]["properties"]["method"].is_object(),
        "a question documents its `method`: {}",
        schemas["Question"]
    );
    let answer = schemas["Answer"].to_string();
    for field in ["method", "region", "HeadRegion", "extent"] {
        assert!(answer.contains(field), "the point answer documents `{field}`: {answer}");
    }
    // The box variant carries `method` and the head box's `region` too.
    let variants = schemas["Answer"]["oneOf"].as_array().expect("Answer is a tagged union");
    let boxed = variants
        .iter()
        .find(|variant| variant.to_string().contains(r#""enum":["box"]"#))
        .expect("a box variant");
    for field in ["method", "region", "digits", "pixels"] {
        assert!(boxed["properties"][field].is_object(), "the box answer documents `{field}`: {boxed}");
    }
    let required = boxed["required"].to_string();
    assert!(!required.contains("digits"), "a head box has no digits: {required}");
    let region = schemas["HeadRegion"].to_string();
    for field in ["cells", "share"] {
        assert!(region.contains(field), "HeadRegion documents `{field}`: {region}");
    }
    let description = document["paths"]["/v1/decide"]["post"]["description"].to_string();
    assert!(description.contains("one pass") && description.contains("chain"), "{description}");
}

/// GitHub #275 (ADR 0036): a `locate`'s `within`, its answer's fields and
/// its refusals are part of the contract, readable without the source.
#[test]
fn the_decision_endpoint_documents_how_a_locate_is_answered() {
    let document = document();
    let schemas = &document["components"]["schemas"];
    assert!(
        schemas["Question"]["properties"]["within"].is_object(),
        "a question documents its `within`: {}",
        schemas["Question"]
    );
    let variants = schemas["Answer"]["oneOf"].as_array().expect("Answer is a tagged union");
    let located = variants
        .iter()
        .find(|variant| variant.to_string().contains(r#""enum":["locate"]"#))
        .expect("a locate variant");
    for field in ["segment", "value", "confidence", "ranking"] {
        assert!(located["properties"][field].is_object(), "the locate answer documents `{field}`: {located}");
    }
    let rank = schemas["LocateRank"].to_string();
    for field in ["segment", "share"] {
        assert!(rank.contains(field), "LocateRank documents `{field}`: {rank}");
    }
    let refused = document["paths"]["/v1/decide"]["post"]["responses"]["422"]["description"].to_string();
    for code in ["locate_uncalibrated", "locate_needs_json_state", "locate_too_long"] {
        assert!(refused.contains(code), "the 422 names `{code}`: {refused}");
    }
    let description = document["paths"]["/v1/decide"]["post"]["description"].to_string();
    assert!(description.contains("locate") && description.contains("content-free"), "{description}");
}

/// GitHub #278 (spec 22): a `locate`'s `kind`, `method` and `compression`,
/// its answer's `found` and `pointers`, and the refusals they can earn are
/// part of the contract, readable without the source.
#[test]
fn the_decision_endpoint_documents_the_shortlist() {
    let document = document();
    let schemas = &document["components"]["schemas"];
    for field in ["kind", "compression", "method"] {
        assert!(schemas["Question"]["properties"][field].is_object(), "a question documents `{field}`: {}", schemas["Question"]);
    }
    for (schema, values) in [
        ("LocateKindField", &["auto", "log", "prose", "records"][..]),
        ("QuestionMethod", &["head", "chain", "shortlist", "vote"][..]),
        ("LocateCompression", &["template_fold", "none"][..]),
        ("LocateKind", &["log", "prose", "records"][..]),
        ("LocateMethod", &["shortlist", "vote"][..]),
    ] {
        let text = schemas[schema].to_string();
        for value in values {
            assert!(text.contains(value), "{schema} names {value}: {text}");
        }
    }
    assert!(!schemas["LocateKind"]["enum"].to_string().contains("auto"), "an answer never says auto: {}", schemas["LocateKind"]);
    let variants = schemas["Answer"]["oneOf"].as_array().expect("Answer is a tagged union");
    let located = variants
        .iter()
        .find(|variant| variant.to_string().contains(r#""enum":["locate"]"#))
        .expect("a locate variant");
    for field in ["kind", "method", "compression", "found", "pointers", "segment", "value", "confidence", "ranking"] {
        assert!(located["properties"][field].is_object(), "the locate answer documents `{field}`: {located}");
    }
    assert!(!located["required"].to_string().contains("found"), "found rides three routes only: {located}");
    let pointer = schemas["LocatePointer"].to_string();
    for field in ["segment", "value", "share"] {
        assert!(pointer.contains(field), "LocatePointer documents `{field}`: {pointer}");
    }
    let refused = document["paths"]["/v1/decide"]["post"]["responses"]["422"]["description"].to_string();
    for code in [
        "kind_unknown",
        "kind_unsupported",
        "kind_mismatch",
        "method_unknown",
        "compression_unknown",
        "compression_unsupported",
        "locate_segment_too_long",
    ] {
        assert!(refused.contains(code), "the 422 names `{code}`: {refused}");
    }
    let description = document["paths"]["/v1/decide"]["post"]["description"].to_string();
    for words in ["shortlist", "template_fold", "found", "pointers", "200,000"] {
        assert!(description.contains(words), "the description says `{words}`: {description}");
    }
}

/// GitHub #270: a `state` part's reuse marker and the two refusals it can
/// earn are part of the contract, readable without the source.
#[test]
fn the_decision_endpoint_documents_the_reuse_marker_and_its_refusals() {
    let document = document();
    let state = document["components"]["schemas"]["DecideRequest"]["properties"]["state"].to_string();
    assert!(state.contains("cache_control") && state.contains("ephemeral"), "{state}");
    let refused = document["paths"]["/v1/decide"]["post"]["responses"]["422"]["description"].to_string();
    for code in ["malformed_reuse_marker", "too_many_reuse_markers"] {
        assert!(refused.contains(code), "the 422 names `{code}`: {refused}");
    }
}

// ── the routes ───────────────────────────────────────────────────────────

#[tokio::test]
async fn v1_redirects_to_the_page() {
    let app = server().app();
    for uri in ["/v1", "/v1/", "/v1/docs"] {
        let response = get(&app, uri).await;
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT, "{uri}");
        assert_eq!(header_value(&response, header::LOCATION), "/v1/docs/", "{uri}");
    }
}

#[tokio::test]
async fn the_page_and_its_assets_come_from_this_binary() {
    // The vendored distribution, not a CDN: a server with no route to the
    // internet serves the same page.
    let app = server().app();
    let page = get(&app, "/v1/docs/").await;
    assert_eq!(page.status(), StatusCode::OK);
    assert_eq!(
        header_value(&page, header::CONTENT_TYPE),
        "text/html; charset=utf-8",
    );
    let html = String::from_utf8(body_bytes(page).await).unwrap();
    assert!(html.contains("/v1/docs/swagger-ui-bundle.js"), "{html}");
    assert!(html.contains("/v1/openapi.json"), "{html}");
    assert!(!html.contains("//cdn."), "the page must name no CDN: {html}");

    for (file, content_type) in [
        ("swagger-ui-bundle.js", "text/javascript; charset=utf-8"),
        ("swagger-ui.css", "text/css; charset=utf-8"),
    ] {
        let response = get(&app, &format!("/v1/docs/{file}")).await;
        assert_eq!(response.status(), StatusCode::OK, "{file}");
        assert_eq!(header_value(&response, header::CONTENT_TYPE), content_type, "{file}");
        assert!(!body_bytes(response).await.is_empty(), "{file}");
    }

    // The bundle's own third-party notices, served beside it as its
    // licence asks.
    assert_eq!(
        get(&app, "/v1/docs/swagger-ui-bundle.js.LICENSE.txt").await.status(),
        StatusCode::OK,
    );
    // Exact table lookup: nothing else is reachable under the page.
    assert_eq!(get(&app, "/v1/docs/swagger-ui.js").await.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn the_document_is_served_as_openapi_3_1() {
    let app = server().app();
    let response = get(&app, "/v1/openapi.json").await;
    assert_eq!(response.status(), StatusCode::OK);
    assert!(header_value(&response, header::CONTENT_TYPE).starts_with("application/json"));
    let document: serde_json::Value = serde_json::from_slice(&body_bytes(response).await).unwrap();
    assert!(
        document["openapi"].as_str().unwrap_or_default().starts_with("3.1"),
        "{}",
        document["openapi"],
    );
    assert_eq!(document["info"]["version"], env!("CARGO_PKG_VERSION"));
    assert!(document["paths"]["/v1/chat/completions"]["post"].is_object());
    assert!(
        document["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer",
        "{}",
        document["components"]["securitySchemes"],
    );
}

#[tokio::test]
async fn the_reference_is_open_on_a_keyed_server() {
    // A browser pointed at `/v1` has no way to set an `Authorization`
    // header, so a keyed server that answered 401 here would publish a
    // reference nobody could read. The handler routes stay gated.
    let app = server().with_api_key(ApiKey::new(KEY.to_owned())).app();
    for uri in ["/v1", "/v1/docs/", "/v1/docs/swagger-ui.css", "/v1/openapi.json"] {
        let status = get(&app, uri).await.status();
        assert!(
            status == StatusCode::OK || status == StatusCode::TEMPORARY_REDIRECT,
            "{uri} answered {status}",
        );
    }
    assert_eq!(get(&app, "/v1/models").await.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn the_preflight_still_answers_on_every_v1_route() {
    // The `OPTIONS` handlers are attached beside the documented ones
    // rather than through `routes!`, so this is what says they survived
    // the move to `OpenApiRouter`.
    let app = server().app();
    for uri in [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/responses",
        "/v1/decide",
        "/v1/systemone",
    ] {
        let request = Request::builder()
            .method(Method::OPTIONS)
            .uri(uri)
            .body(Body::empty())
            .unwrap();
        let response = app.clone().oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK, "{uri}");
    }
}

#[test]
fn the_field_declaration_schema_names_exactly_the_keys_a_declaration_has() {
    // ADR 0046: the route builds each object with `FieldMeta::declaration`;
    // the schema restating it by hand must not drift from it silently.
    let document = document();
    let schema: BTreeSet<String> =
        document["components"]["schemas"]["FieldDeclaration"]["properties"].as_object().expect("properties").keys().cloned().collect();
    let declared: BTreeSet<String> = ignis_server::config::schema::all_fields().next().expect("a field").declaration().keys().cloned().collect();
    assert_eq!(schema, declared);
}
