# 05 — image serving: the model kind, the image routes, the queue, the Playground, prompt enhancement

GitHub: #322. ADR 0050, ADR 0049 (idle unload), ADR 0047
(catalog), ADR 0046 (config v2). Feature: `spec.md`. Needs 04 for the real
leaf; everything here is built and tested against a fake image leaf first.

## Problem

The server is shaped for text models:
- `ActiveModel` holds an `Engine`;
- the idle clock reads the engine's lanes;
- every model route renders a chat template.

An image model needs:
- its own routes, a job queue and progress events;
- PNG output;
- a refusal or a switch when a request is of the other kind;
- a Playground view;
- optionally, a text model rewriting the prompt first.

## Solution

### Model kind

- **The model and its kind.** `ModelFamily::QwenImage21` is of kind
  *image*, and `ActiveModel` holds either an engine (text) or an image runner
  (image).
- **Text-only consumers.** Everything that touched the engine directly asks
  the kind first:
  - `/metrics`;
  - telemetry;
  - the idle clock;
  - the tokenize routes;
  - `/v1/decide`.
- **Wrong kind, no model named.** A route of one kind, called while a model
  of the other kind is loaded with no `model` naming a model of its own kind,
  answers `409 model_kind_mismatch`, naming both kinds and the loaded id.
- **Wrong kind, model named.** If `model` names a known model of the route's
  kind, the request triggers the implicit switch under
  `switch.allow_implicit` (ADR 0049's rules, unchanged).

### Image routes

**`POST /v1/images/generations`** (JSON) and **`POST /v1/images/edits`**
(multipart, `image[]` from 1 to `image.max_condition_images`, PNG / JPEG /
WebP, decoded to RGBA):

| field | rule |
|---|---|
| `model` | as above |
| `prompt` | required; never altered (prompt enhancement aside) |
| `n` | 1..`image.max_n`, made one after another, one condition prefix per job |
| `size` | one of `image.sizes`, else `400` listing them. `auto` is 2048x2048 for a generation, and for an edit the preset whose aspect ratio is nearest the last condition image's |
| `response_format` | `b64_json` only; `url` is `400` |
| `output_format` | `png` only; anything else is `400` |
| `background` | `auto` / `transparent`: the alpha channel as generated. `opaque`: composited on white |
| `mask` (edits) | not supported: `400` |
| `seed` (extension) | optional. Absent: drawn, and returned |
| `prompt_enhancement` (extension) | optional bool; default `image.prompt_enhancement` |
| `stream` | `true`: Server-Sent Events, below |

**Response.**
- `{created, data: [{b64_json, revised_prompt?}], seed, condition_sizes?}`.
- The PNG is RGBA 8-bit, encoded with the `png` crate.
- `condition_sizes` lists each condition image's effective size when the
  budget scaled them down.

**Streaming.**
- `image_generation.progress` events: `{phase, image, step, total}`, where
  `phase` is `enhancing`, `switching`, `loading`, `encoding`, `denoising` or
  `decoding`.
- One `image_generation.completed` per image, with its `b64_json`.
- Partial images are not sent.

### Sizes

`image.sizes` defaults to:
- the checkpoint card's presets: 2048x2048, 2400x1792, 1792x2400, 2528x1696,
  1696x2528, 2752x1536, 1536x2752;
- 1024x1024, diffusers' own default output, which OpenAI clients often send.

All are multiples of 32.

### The condition budget

- Condition images are resized to 1024² area each, keeping their aspect
  ratio, with sides rounded to 32 (diffusers' `calculate_dimensions`).
- If the prompt plus all condition images exceed the condition budget (spec
  04), every condition image is scaled down by the same factor until they
  fit, never below `image.condition_floor_px`² area.
- Below the floor the request is `400`, naming the budget.
- `image.condition_budget_tokens` (0 = the plan's whole budget) lowers the
  budget.

### The queue

- **Running.** One image job runs at a time, and up to `image.queue_depth`
  wait in FIFO order.
- **Full.** When the queue is full, the answer is `503 engine_full` with
  `Retry-After`.
- **Cancellation.** A client that disconnects cancels its job: before it
  starts, or between denoising steps. The job's prefix is released.
- **Idle unload.** A running or queued job is in use. The idle clock asks the
  loaded model whether it is busy: lanes for text, the queue for image.

### Prompt enhancement

Optional, and off unless the request or `image.prompt_enhancement` asks for
it. The enhancer model is `image.prompt_enhancer`, a served id that must be a
known model (default the 27B's).

- **Text model loaded when the job arrives.** If it is the enhancer, the job
  is rewritten first and then the switch to the image model happens, with no
  extra switch. If it is another text model, the job switches to the
  enhancer first and then to the image model.
- **Image model loaded.** One **enhancement visit** per job: switch to the
  enhancer, rewrite, switch back.
  - During the visit other requests get `503 model_switching` with
    `Retry-After`.
  - *Last switch* is not updated, and the visit counts as use.
- **The rewrite itself:**
  - The system prompts are Qwen's official ones, kept as adapted text with
    attribution: `QwenLM/Qwen-Image` `src/examples/tools/prompt_utils_2512.py`
    for t2i, English or Chinese chosen by CJK detection, and `prompt_utils.py`
    for edits (Apache-2.0).
  - In an edit the enhancer also sees the condition images, when its vision
    tower is loaded; otherwise it sees text only, with a warning.
  - Reasoning is off.
- **Result.** The rewritten prompt is returned as `data[].revised_prompt`. If
  enhancement fails, the original prompt is used and the response carries a
  warning.

### Catalog and Make

- **Catalog.** A built-in entry `ignis-image-2.1-turbo` with spec 01's pinned
  revision and sha256s. `model download ignis-image-2.1-turbo` fetches it.
- **Make.** `MODEL=image` selects family `image` in `mk/config.mk`, with its
  artifact path and none of the text knobs (lanes, context, chunk, spec,
  vision).

### Playground

A new `image` view (`web/src/app/Header.tsx`'s `View`). It has:
- prompt, size preset, `n` and seed;
- up to 10 condition images by drop or pick, with their alpha shown;
- the prompt-enhancement toggle and the revised prompt;
- progress per denoising step and phase;
- a client-side gallery with download.

When a text model is loaded, the view shows an explicit "switch to the image
model" button. It never switches implicitly, and never sends a `model` it was
not told to.

## Seams

- **Server types:**
  - `crates/core/src/compute.rs` (`ModelFamily`, and the kind beside it);
  - `crates/server/src/active.rs` (`ActiveModel`'s engine becomes a kind
    enum);
  - `crates/server/src/model_switch.rs` (`ModelLoader`/`PreparedLoad` for the
    image family);
  - `crates/server/src/idle.rs` (`tick_once`'s busy test).
- **Routes and media:**
  - new `crates/server/src/images/` for the routes, the queue, the budget
    policy and the PNG;
  - `api.rs` for the router, the kind check and the refusals;
  - `media.rs` for decoding RGBA.
- **Config:** a new `config_group! { ImageGroup = "image" }` in
  `crates/server/src/config/schema.rs`, every field `[Visible, Patchable]`
  unless it needs a reload, each with flag and env:

  | field | default |
  |---|---|
  | `queue_depth` | 4 |
  | `max_n` | 10 |
  | `max_condition_images` | 10 |
  | `sizes` | above |
  | `condition_budget_tokens` | 0 = the plan's |
  | `condition_floor_px` | 512 |
  | `prompt_enhancement` | false |
  | `prompt_enhancer` | `qwen3.8-27b` |
- **OpenAPI:** both routes, the events and the errors in
  `crates/server/src/openapi.rs`.
- **Catalog:** `crates/server/src/download/catalog.yaml`. **Make:**
  `mk/config.mk`.
- **Playground:**
  - `web/src/app/Header.tsx` (the view);
  - a new `web/src/image/` for the view itself;
  - `web/src/api/` for the client.
- **RNG:** ignis's own, seeded per job and per image (`seed + i`). The same
  seed gives the same image on the same build and machine, not diffusers'
  image.

## Acceptance

1. With the image model loaded (fake leaf), `/v1/chat/completions`,
   `/v1/responses` and `/v1/decide` without a `model` answer `409
   model_kind_mismatch`; with the 27B loaded, `/v1/images/*` do. Naming a
   known model of the route's kind switches implicitly when
   `switch.allow_implicit` is on, and is refused by name when it is off.
2. `/metrics`, telemetry, the tokenize routes and the idle clock work with
   the image model loaded. None of them touches an engine that is not there.
3. A generation returns `n` RGBA PNGs as `b64_json` with the seed. The same
   seed twice gives the same bytes. `background=opaque` gives alpha 255,
   composited on white.
4. Size validation: a size outside `image.sizes` is `400` with the list.
   `auto` resolves as specified. `url`, a non-PNG `output_format` and a
   `mask` are `400`.
5. An edit with 1 to 10 condition images (PNG / JPEG / WebP, with and without
   alpha) is accepted; 11 is `400`. Over the budget, all images are scaled by
   one factor and `condition_sizes` reports them. Below the floor the request
   is `400`.
6. Streaming emits the phases in order, one `denoising` event per step per
   image, and one `completed` per image.
7. The queue runs one job at a time, FIFO, and answers `503 engine_full` past
   `image.queue_depth`. A disconnect cancels its job between denoising steps
   and releases its prefix.
8. Idle unload: a running or queued image job keeps the image model loaded,
   and an idle image model unloads after `model.idle_unload` like a text
   model.
9. Prompt enhancement, with a fake text engine:
   - the text-model-first path makes one switch;
   - the visit path makes two switches and leaves *Last switch* unchanged;
   - `503 model_switching` is answered during a visit;
   - `revised_prompt` is returned;
   - a failing enhancer falls back to the original prompt with a warning;
   - a CJK prompt picks the Chinese system prompt.
10. Every `image.*` field resolves from flag, env and file with config v2
    precedence, is visible in `GET /v1/config/fields`, and is refused above
    its bounds.
11. The catalog entry downloads and verifies (sha256) the artifact.
    `MODEL=image make config` prints the image model's command.
12. Playground:
    - the image view generates, edits and shows progress;
    - the switch button switches;
    - the gallery downloads;
    - no message from the view ever switches implicitly.
13. OpenAPI documents both routes, their extensions, the events and the
    errors. `docs/user/README.md` gains an image section (routes, sizes, the
    knobs, the licence's non-commercial terms). `CONTEXT.md` already has the
    terms.
14. With spec 04's leaf: one live run on the 5090 serves a 2048² generation
    and a 2-image edit from the Playground, switching from the 27B and back.
    The times are written in "As built".

## As built (322a, 2026-10-13, branch `image-serving-322`: AC 1, 2, 8, 10)

Against `FakeImageRunner`; the image routes, the queue, the PNG, prompt
enhancement and the Playground are the agents after it.

- **The kind.** `ModelKind { Text, Image }` and `ModelFamily::kind()` in
  `crates/core/src/compute.rs`. `ActiveModel.runner` is
  `Runner::Text(Engine) | Runner::Image(ImageModel)`; `kind()`, `model_id()`
  and `in_flight()` serve both, `engine()` is `None` on the image model, and
  `text_engine()` is what a text route reads past the kind check (it panics
  on the image model). The template, calibrations, labels and acquirer stay
  on `ActiveModel`: the image model carries the placeholder template and
  none of the rest.
- **The runner.** `ignis_core::image::ImageRunner`, in core so the image
  leaf's crate implements it as the text leaf implements `Compute`:
  `plan()` (condition budget, sample sigmas), `encode_condition(&ImageCondition)`
  → `ConditionId`, `denoise_step(id, &mut TargetLatents, σ, σ_next,
  velocity)`, `vae_decode` → RGBA, `release_condition`. `&mut self`: one job
  at a time. The prompt goes in as text and the condition images as RGBA at
  their budgeted size: the runner owns the tokenizer and the preprocessing
  (spec 03). The latents cross to the host between calls (≤ 4 MiB a step).
  `vae_encode` is inside `encode_condition`. `FakeImageRunner` is
  deterministic in the condition and the noise and counts its held prefixes.
- **The image model in the server** (`crates/server/src/images/mod.rs`):
  `ImageModel` holds the runner behind a lock (`with_runner`), and its
  shutdown drops it — the counterpart of joining a text model's thread.
  `ImageJobs` is the queue-busy hook: `wait()` → `Waiting`, `start()` →
  `Running`, RAII; `in_flight()` is what the idle clock and a switch's drain
  read; `close()` is a switch's or an unload's stop.
- **The kind check** (`crates/server/src/api.rs`):
  `switch_to_named_model(server, route_kind, model)` refuses
  `409 model_kind_mismatch` (`param: model`) before any load or switch when
  the model that would serve the request is of the other kind: the one
  `model` names when it is another, its kind read by the switcher
  (`Switcher::kind_of`, `ModelLoader::kind_of`: the artifact's header for a
  known model, else the id when it is a family's own), or the loaded one —
  also while unloaded, so a text request never reloads the image model only
  to be refused. `check_served_kind` runs again on the pinned server and
  refuses by name (`404 model_not_found`) a model of the route's kind that
  was not switched to; `/v1/decide` answers that with its own `422`. The
  tokenize routes never switch: `409` with the image model loaded.
- **Other consumers.** The warm-up is a text model's only; metrics install
  into an engine when there is one (`/metrics` serves either way); the
  decide heads log and the thinking checks (start, `PATCH`) skip the image
  model. `GET /v1/models` gains `kind`, and `max_model_len` is a text
  model's only.
- **The load.** `prepare_model` reads the family first: the image artifact
  is verified against its sidecar (no text frontend is read) and the
  options fitted to the family. `load_model` then loads `FakeImageRunner`
  without `cuda`, and under `cuda` refuses naming #321
  (`cuda_image_runner`, the stub the leaf fills).
- **Config** (`config_group! { ImageGroup = "image" }`): the table's fields
  and defaults, each with flag and env var, `[Visible, Patchable]`, every
  family taking them (they matter while a text model is loaded too). Bounds:
  `queue_depth` 0..=64, `max_n` 1..=64, `max_condition_images` 1..=10 (the
  encoder's), `condition_floor_px` 32..=1024; `sizes` is its own list kind,
  each side a multiple of 32; `condition_budget_tokens` and the enhancement
  fields have no bound. The live knobs carry the group
  (`server.live().image`). No `qwenimage21` family scope.
- **Tests.** `crates/server/tests/model_kind.rs` (AC 1, 2, 8 and the
  loader on the converter's synthetic artifact), `config/tests.rs` and
  `tests/config_http.rs` (AC 10), `core::image` and `server::images` units.
