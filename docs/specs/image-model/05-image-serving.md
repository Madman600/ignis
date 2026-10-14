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
  their budgeted size; the runner tokenizes and prepares them (322c's F7,
  below, says where that runner lives). The latents cross to the host
  between calls (≤ 4 MiB a step).
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

## As built (322b, 2026-10-14, branch `image-serving-322`: AC 3-7, 13's OpenAPI)

Against `FakeImageRunner`; prompt enhancement, the catalog, Make, the user
docs and the Playground are the agents after it.

- **The routes** (`crates/server/src/images/routes.rs`): `create_image`
  (JSON) and `create_image_edit` (multipart: `image[]`, or `image` for one;
  the other parts are the JSON fields as text). A request is checked whole
  before it costs anything — the field table against `image.*`, an edit's
  images decoded (`media::decode_condition_image`: PNG and WebP keep their
  alpha, JPEG is opaque, GIF is refused) — and only then
  `switch_to_named_model`, the pin and `check_served_kind`: a 400 never
  loads or switches a model. Unknown fields are ignored; OpenAI's `quality`,
  `style`, `moderation`, `output_compression`, `partial_images` and `user`
  are named inert in the operation. `auto` is the size in `image.sizes`
  whose ratio is nearest on the log scale (1 for a generation, the last
  condition image's for an edit), the larger on a tie. The response adds
  OpenAI's `size` (`auto` resolved), `background` and `output_format` to the
  spec's fields. The edit route has the media body limit (384 MiB) on every
  load.
- **Images and seeds.** RGBA 8-bit PNG (`png`). `opaque` is
  `(c·a + 255·(255 − a) + 127) / 255`, alpha 255. Image `i` starts from
  `TargetLatents::noise(size, seed + i)` (core: splitmix64's stream,
  Box–Muller). A request without a seed takes `server.seedless_seed`, else a
  drawn one, and it is returned.
- **The condition budget** (`images/budget.rs`). `condition_size` is
  diffusers' `calculate_dimensions` (halves to even; a side is never 0).
  A prefix holds what `ImageRunner::condition_tokens(prompt, sizes)`
  counts: the prompt as templated plus `(W / 16) · (H / 16)` per image
  (322c's call, below). Over the budget
  (`min(plan, image.condition_budget_tokens)`), every image gets the same
  area s², s the largest in [floor, 1024] that fits (bisected: the tokens
  only grow with s); below the floor `400 condition_budget_exceeded`
  (`param` `image`, or `prompt` for a generation). The resize is
  `media::resize_rgba`, Pillow's Lanczos on `RGBa` in Pillow's fixed point,
  byte-equal to Pillow 12.3 on three committed fixtures.
- **A new runner call:** `ImageRunner::prompt_tokens`, which 322c widened
  into `condition_tokens` (below). The runner holds the image model's
  tokenizer, so the server asks it for the prefix's tokens.
- **The queue** (`images/mod.rs`). `ImageJobs` is a ticket queue under a
  lock. `admit(depth)` admits while waiting + running ≤ `image.queue_depth`;
  past it the answer is `503 engine_full` with `Retry-After: 1`, and on a
  closed queue `503 model_stopped`. `Waiting::turn()` runs the jobs in the
  order they were admitted, not the order they ask, and gives `None` once the
  queue is closed. `wait()` / `start()` keep 322a's counting meaning.
- **The job** (`images/job.rs`). `spawn` waits for the turn and also
  watches the request: a waiting job whose request went away leaves the
  queue at once. `run` (blocking) encodes once, then for each image: noise,
  steps, decode, PNG. It holds `with_runner` for one call at a time. Between
  calls a closed queue ends it `Stopped`, and a channel with no reader ends
  it `Cancelled`. `Held` releases the prefix on every path. The job leaves
  the count before its request hears that it ended.
- **Answers.**
  - Not streaming: `200` with the JSON body; `503 model_stopped` with
    `Retry-After`; `500 engine_error` when the runner refuses a call.
  - Streaming: `event: <type>` and a `data:` object. Generations send
    `image_generation.progress` and `.completed`; edits send
    `image_edit.progress` and `.completed`, OpenAI's names for the edit
    route (the spec says `image_generation.*` for both).
  - A progress event announces a unit as it starts: encoding 1/1, then for
    each image denoising k/total and decoding 1/1. A job that ends early
    sends one `error` event (`{type, error}`). No `[DONE]`.
- **OpenAPI.** An `images` tag and both routes, the edit as a multipart
  `EditForm`. `ImageStreamEvent` is oneOf progress / completed / error, and
  `Phase` lists all six phases.
- **Not applied.** `server.request_timeout` (a text knob, 30 s by default)
  does not bound an image request. `prompt_enhancement` is parsed and has no
  effect until 322c.
- **Tests.**
  - `crates/server/tests/images_http.rs`: AC 3-7, the `409` with a text
    model loaded, and the edit's body limit.
  - Unit tests in `images::{budget, job, routes}`, the queue's in `images`,
    `media` (the Pillow fixtures), `vision::decode` (RGBA), and `core::image`
    (the noise and the prefix count).
  - `tests/openapi_http.rs`: AC 13's OpenAPI part.

## As built (322c, 2026-10-14, branch `image-serving-322`: the 322a review's fixes)

The findings of `.scratch/orchestra/reviews/review-322a.md` (F1-F8), on
322a's and 322b's work.

- **The prefix count (F1).** `ImageRunner::condition_tokens(prompt,
  &[ImageSize]) -> u64` replaces 322b's `prompt_tokens(prompt, n)`: the
  whole condition prefix, the prompt as the runner's template and tokenizer
  lay it out plus `latent_cells` per image. `budget::fit` takes the count as
  a function of the images' sizes; `prepare` holds the runner for the fit's
  few host-side counts (a bisection). **#321 implements it with the very
  count `encode_condition` refuses by**: the fit trusts it, so an
  `OverBudget` from the encode is a server bug (`500 engine_error`).
- **The largest target (F4).** `ImagePlan.max_target_cells`, what the load
  sized its arena for, and `ImagePlan::takes(size)`. The fake plans 2400x1792
  (16,800 cells), the largest default preset in latent cells. Once pinned, a
  requested size past it is `400` (`param: size`), naming the cells and
  listing the sizes served; `auto` picks only among the served sizes
  (`images::served_sizes`), and with none of them is `400`. The load warns
  (`ignis.image.sizes_not_served`) of each `image.sizes` entry past its plan.
  **Default taken:** such an entry is dropped from what this load serves,
  not from the config — `GET /v1/config` still lists it, a load that can
  make it serves it, and a `PATCH` adding one is refused per request, not at
  the patch.
- **For #321, the arena.** Spec 04 sizes the arena for "the largest preset
  (2752 × 1536, 16,512 target tokens)", but in latent cells 2400x1792
  (16,800) and 2528x1696 (16,748) are larger: an arena for 16,512 drops both
  defaults at load. Size it for 16,800, and plan `max_target_cells` as what
  it holds.
- **A name it cannot switch to (F2).** `api::unswitchable_named`: with the
  loaded model — or the unloaded one — of the other kind, a request naming a
  model the switcher does not know (implicit switching off, or an unknown
  id) is refused by name at once (`404 model_not_found`, decide's `422`),
  before `ensure_loaded` would reload the image model only to refuse it.
  With a model of the route's kind loaded, such a name passes as before.
- **Metrics on the image model (F3).** `observe_load` of a model with no
  engine calls `Metrics::set_unloaded`: the text model's reservations, KV
  pages and occupancy read zero while the image model holds the card, as
  while unloaded.
- **A runner call that panics (F5).** `with_runner` and `shut_down` take a
  poisoned lock (`PoisonError::into_inner`): the job's prefix is still
  released while it unwinds, and the shutdown still drops the runner.
- **Where the real runner lives (F7).** Spec 03 §Seams puts the tokenizer,
  the templates and the Qwen3-VL preprocessing in server-side Rust
  (`crates/server/src/media.rs`), and the trait holds: #321's runner is an
  adapter in `crates/server`, beside `cuda_image_runner` (`load.rs`), over
  the image leaf. The adapter owns the artifact's tokenizer and templates and
  `media.rs`'s preprocessing, and it hands the leaf token ids and prepared
  patches. So the routes pass the prompt as text and the images as RGBA, and
  `condition_tokens` is host-side.
- **Coverage (F6).** These tests pin behaviour already built: the switch
  drain's image arm (a running job holds the teardown back; past a short
  window the queue closes and the switch lands, the runner gone before the
  next load); the thinking checks' skip on the image model (`PATCH`); and
  the Responses socket on the image model (`409` as an `error` event, with
  the socket left open). `main.rs`'s skip of the thinking validation stays
  untested, as the review allowed.
- **322b's review notes, checked.**
  - The edit's media body limit is a layer on the edit route alone, so it
    holds on the image model and on a text load without vision; both are
    tested.
  - A job holds `with_runner` for one call at a time, on `spawn_blocking`.
  - A job that passes the gate between the drain and `close()` stops at its
    next call: `run` checks the queue before every call, and a runner taken
    by the shutdown answers `Stopped`.
  - `auto` without 2048x2048 picks the nearest ratio among the sizes served.
- **Not done.** F8, caching `kind_of` per artifact, is left out: optional in
  the review, and a header read costs nothing next to the switch it
  precedes.
- **Tests.** `core::image` (the count, `takes`), `images` (`served_sizes`,
  the poisoned lock), `tests/images_http.rs` (the size past the plan, `auto`
  among the served sizes, the body limit on a text load), and
  `tests/model_kind.rs` (the unloaded refusal, the metrics after a switch,
  the drain), plus `tests/config_http.rs` and `tests/responses_socket.rs`.
