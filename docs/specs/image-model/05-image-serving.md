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
- On `/v1/images/edits` the same events are named `image_edit.progress` and
  `image_edit.completed`. `image_edit.completed` is OpenAI's own name, so
  OpenAI SDKs read an edit stream; `image_edit.progress` is ignis's, beside
  it (coordinator's ruling on review-322b F3).
- Partial images are not sent.

### Sizes

`image.sizes` defaults to:
- the checkpoint card's presets: 2048x2048, 2400x1792, 1792x2400, 2528x1696,
  1696x2528, 2752x1536, 1536x2752;
- 1024x1024, diffusers' own default output, which OpenAI clients often send.

All are multiples of 32. A side is at most 16,384 pixels (321f). The load
sizes its arena for the largest size within `image.max_target_cells`
(16,800 cells, 2400x1792, by default; 321f). A size past it is not served,
and the load warns (`ignis.image.sizes_not_served`).

### The condition budget

- Condition images are resized to 1024² area each, keeping their aspect
  ratio, with sides rounded to 32 (diffusers' `calculate_dimensions`).
- If the prompt plus all condition images exceed the condition budget (spec
  04), or the room the runner has for them beside it (the leaf's arena, the
  RoPE positions before the target; 321f), every condition image is scaled
  down by the same factor until they fit, never below
  `image.condition_floor_px`² area before the sides are rounded to 32 (the
  rounding may take an image a little under it: 672x384 at a 512 floor).
- Below the floor the request is `400`, naming the budget (or the room).
- A condition the runner refuses at diffusers' sizes (a panorama past 200:1)
  is `400 image_refused`, in its words (321f).
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
it. The enhancer model is `image.prompt_enhancer` (default the 27B's), the
served id of a text model the server can switch to by id, or of the text
model loaded. Where the server knows its text models (it can switch), a
`PATCH` naming any other is refused and a start warns of it; a request that
meets one anyway uses its prompt as sent, with a warning.

- **Text model loaded when the job arrives.** If it is the enhancer, the job
  is rewritten first and then the switch to the image model happens, with no
  extra switch. If it is another text model, the job switches to the
  enhancer first and then to the image model: the request's switch, so
  *Last switch* records both.
- **Image model loaded.** One **enhancement visit** per job: switch to the
  enhancer, rewrite, switch back.
  - The job takes its turn in the queue first: the jobs admitted before it
    finish on the image model, and the visit begins with none running.
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
not told to. (324: pressing Generate or Edit with a text model loaded is being
told to: the request names the image model; see "As built (324 item 1)".)

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
  | `min_condition_budget_tokens` | 8191 (321e: read by the load) |
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
  encoder's), `condition_floor_px` 32..=1024,
  `min_condition_budget_tokens` 0..=2^20 (321e); `sizes` is its own list kind,
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
  `switch_to_named_model`, the pin and `check_served_kind`: those 400s never
  load or switch a model. (Review-322b L1: a size past the load's largest
  target, `auto` with no size served, and `condition_budget_exceeded` need
  the loaded image model, so they come after the switch.) Unknown fields are
  ignored; OpenAI's `quality`,
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

## As built (322d, 2026-10-14, branch `image-serving-322`: the 322b review's fixes, AC 9)

The findings of `.scratch/orchestra/reviews/review-322b.md`, then prompt
enhancement, against the mock loader's text models and `FakeImageRunner`.
The workspace on 08c23b3, which 322c never recorded, was 2896/0/6.

- **Review-322b.**
  - **F1.** 322c's poisoned-lock fix (F5) already keeps `Held::drop` from
    panicking while a runner's panic unwinds: no abort. A route test now
    drives a runner whose step panics: `500 engine_error`, the prefix
    released, the model serving the next request. The shutdown half is
    322c's unit test. (321f: the model no longer serves on; it fails, see
    "As built (321f)".)
  - **F2.** The decode cap is `image.max_condition_mipx` (16 Mi pixels by
    default, 1..=64, live), named in the 400; `Server.image_decodes`, one
    permit, holds each edit's decode.
  - **F3, F4.** §Streaming names the edit events (the coordinator's ruling).
    `media.rs` names the Pillow 12.3.0 files its resize ports and root
    `NOTICE` carries Pillow's notice. ADR 0051, amended by the owner
    (2026-10-14), admits MIT-CMU (HPND), Pillow's licence.
  - **L1-L7, N1, N4.** No doc claims a 400 never loads a model. A runner's
    own `OverBudget` is `400 condition_budget_exceeded`. OpenAI's Images
    fields go through `openai_fields` (`Surface::Images`: `partial_images`
    above 0 refused, the rest inert), the validator and the OpenAPI alike.
    `server.request_timeout`'s doc says it bounds text requests. `wait()` is
    `admit` without its bound. A mid-run stop's released prefix is unit
    tested. A stream short of its images sends `error`. `testing.md` records
    the Pillow fixtures' recipe, checked against their pixels. The floor
    holds before the sides are rounded.
- **What a request waits on, streamed.** `serve` refuses the kind
  mismatch and a name it cannot switch to as statuses, before anything
  loads. `admit` then runs the load of an unloaded model, prompt
  enhancement and the implicit switch, each sending its phases
  (`loading`, `enhancing`, `switching` with `step`/`total`), before the pin,
  the size, the queue, the fit and the job. A streamed request that waits
  on any of them starts its stream at once, and a refusal past that point
  is its `error` event; one the loaded image model takes as it is starts
  once admitted, every refusal a status. `model_switch` tells its beginner
  each step (`SwitchStep`, `implicit_switch_telling`).
- **The visit** (`model_switch::visit`). Two switches in one hold of the
  switch's slot (`Run::Visit`): to the visited model, which is handed over
  pinned (`Visit::arrived`) with the status still `switching` (visited →
  next), then on (`Visit::on`, or the beginner gone). Every other request
  meets `503 model_switching` with `Retry-After`, and a switch `409`, from
  the first step to the last. `last_switch` is never written: `run` and
  `restore` record a switch or a reload only (322g: and each switch of a
  visit that ends on another model than it left, L2). A way there refused or failed
  ends the visit with the old model serving, reloaded if need be; a way on
  that fails reloads the visited model. `SwitchRefusal::NotServing` refuses
  a visit with nothing serving.
- **Prompt enhancement** (`images::enhance`, the routes' `enhance`).
  - On when the request's `prompt_enhancement`, else
    `image.prompt_enhancement`, says so. The enhancer serving: the rewrite on
    it, then the request's own switch to the image model, one switch.
    Otherwise a visit: from the image model and back, or from another text
    model on to the image model the request names.
  - `image.prompt_enhancer` resolves as an explicit switch by id does
    (`Switcher::text_model`: the listed models, then a catalog entry on
    disk), whatever `switch.allow_implicit` says, never to an image model.
  - The rewrite: Qwen's system prompts in `images/prompts.rs`, generated
    from QwenLM/Qwen-Image 6b5e1f5 (`prompt_utils_2512.py` for a
    generation, Chinese when the prompt holds U+4E00..U+9FFF;
    `prompt_utils.py`'s edit prompt), each literal trimmed, sent as the
    system turn; upstream's user turn; root `NOTICE` names them. Reasoning
    off, the instruct sampling seeded with the job's seed (the same seed, the
    same rewrite), at most `image.prompt_enhancer_max_tokens` (new, 1024,
    16..=8192), within `server.request_timeout`. Lines are joined as upstream
    joins them; an edit's `{"Rewritten": ...}` is unwrapped.
  - An edit's images go to an enhancer with a vision tower as PNG
    `image_url` parts; without one it rewrites from the text, and says so.
  - The rewrite is each image's `revised_prompt` and each completed event's;
    `warnings`, new on the answer and the completed event, says why there
    is none: an enhancer it cannot switch to, a visit that did not get
    there, a rewrite that failed. The job then runs on the prompt as sent.
    A visit that cannot get back to the image model is `503
    model_switch_failed`. The condition budget is fitted to the final prompt.
- **Defaults taken** (owner may revisit).
  - An `image.prompt_enhancer` that names no switchable text model falls back
    with a warning rather than refusing the request; a `PATCH` naming one is
    accepted (322g: refused now, M3; the fallback stays).
  - A visit drains the image model as any switch does: a running or queued
    job gets `switch.drain_timeout`, then ends `503 model_stopped`. Taking
    the queue's turn first was considered and not built (322g: built, M1).
  - An unloaded image model loads first and is then visited: three loads.
    Loading the enhancer straight away would race the requests waiting on
    the unloaded image model.
  - The enhancer-serving path's switch to the image model is an ordinary
    implicit switch, written to `last_switch`; a visit from another text
    model is not (322g: it is now, L2).
  - The rewrite is a client request on the enhancer (logged and counted),
    not an internal one. Upstream's user turn is kept verbatim, its stray
    space before `Rewritten Prompt:` included. An edit's answer that is not
    the JSON object is taken as the prompt; a rewrite cut at the token cap
    is used as it is.
  - One decode permit, fixed: it mirrors the one-job queue rather than
    being a tunable (322g: `image.max_condition_decodes`, default 1, L1).
- **Tests.** `tests/image_enhancement.rs` (the streamed phases, the visit
  and its failures, AC 9: the enhancer-serving path's one switch and its
  Chinese system prompt, the visit's two switches and untouched
  `last_switch`, `503 model_switching` mid-visit, `revised_prompt`, the four
  fallbacks, the edit without vision); `images::enhance` units (the CJK
  pick, the turns, the answer's reading); `tests/images_http.rs` (F1, F2,
  L2, L7, no switcher); `openai_fields`, `config` and `openapi_http`.

## As built (322e, 2026-10-14, branch `image-serving-322`: AC 11, 13's user docs, the AC map)

The baseline on 96db56b (322d never recorded one): `cargo test --workspace`
2913/0/6, `cargo check -p ignis-server --features cuda --tests` clean.

- **Catalog (AC 11).** `ignis-image-2.1-turbo` is the fourth built-in entry:
  repo `gpillon/ignis-image-2.1-turbo`, the artifact
  `ignis_image_2_1_turbo-v2.ninfer` (16,721,644,544 bytes, sha256
  `4205dd19...6050af`) and its `.conversion.json` record (617,924 bytes,
  sha256 `a97f0491...0f33ca`), which the catalog's parser requires beside any
  artifact and the loader verifies against. That record is the published one:
  its `converter.command` paths are placeholders, so its bytes differ from the
  local record (618,040 bytes, `b96d32ef...13ce`). The revision is the
  upload's hub commit `d6912417aa47320cd71c78e19d99872faa5cb63b` (2026-10-14),
  pinned in `catalog.yaml` and in `catalog.rs`'s `RELEASES`. Tests: `RELEASES`
  holds the four entries value for value, `entry("Ignis-Image-2.1-Turbo")`
  resolves (what `model download` looks up), and the listing, `--all` and
  operator-catalog tests count four. No test downloads the 15.6 GiB.
- **Make (AC 11).** `MODEL=image` is family `image` in `mk/config.mk`: its
  artifact path (`F:/ai/models/ignis-image-2.1-turbo/...`, `ARTIFACT` wins),
  served id `ignis-image-2.1-turbo`, and the context, KV format, prefill
  chunk, decode share, speculation, rope scaling, vision and KV-RAM knobs
  defaulting to empty, so no text flag reaches the server. `make config`
  prints a PLAN and MODEL line for it. `mk/flags-selftest.sh` asserts both the
  flags that arrive and the eleven that do not.
- **User docs (AC 13).** `docs/user/README.md`: the catalog table, the two
  routes in the API table, an "Image generation" section (licence, routes,
  streaming, sizes, condition images, queue, the `image.*` table with
  `max_condition_mipx` and `prompt_enhancer_max_tokens`, prompt enhancement
  and its cost, the known-model rule) and a `MODEL=image` paragraph under make.
- **The acceptance map.**
  1. Done (322a, `model_kind_mismatch` and the implicit switch).
  2. Done (322a).
  3. Done (322b).
  4. Done (322b).
  5. Done (322b, with 322c's fixes).
  6. Done (322b, 322d's phases).
  7. Done (322b); an enhanced request's visit keeps the FIFO (322g, M1).
  8. Done (322a).
  9. Done (322d); 322g closes review-322cf's M1, M3, L2 and L3 on it.
  10. Done (322a); 322g adds `image.max_condition_decodes`.
  11. Done (322e); the revision is pinned to d6912417 (c81e8c6).
  12. Done (322f): the image view, its switch button and gallery; see "As built (322f)". 322g: the switch is followed by its record (M2).
  13. Done: OpenAPI (322b) and the user docs (322e), corrected by 322g (L4).
  14. Done (live run, 2026-10-14, on d81e33e): 27B → image 10.0 s, 2048² generation 7.87 s, 2-image edit 9.48 s, image → 27B 11.3 s; see "As built (AC 14, live run)".

## As built (322f, 2026-10-14, branch `image-serving-322`: AC 12)

The Playground's image view. Web only; no Rust touched.

- **Files.** `web/src/api/images.ts` (the two routes, JSON and multipart, whole
  and SSE, refusals with `Retry-After`), `web/src/image/studio.ts` (pure parts:
  limits, `performRun`, `switchToImageModel`, download), `useStudio.ts` (state),
  `ImageView.tsx`. The header's `View` gains `image`; the view stays mounted
  from its first visit, like the Server view.
- **The form.** Prompt; size (`auto` plus `image.sizes` read from
  `GET /v1/config`, the server's defaults until read); `n` (`image.max_n`);
  seed (blank is the server's draw); up to `image.max_condition_images`
  condition images by drop or pick (PNG, JPEG, WebP), shown over a checkerboard
  (`.checker`) so alpha reads; the enhancement toggle, which starts at
  `image.prompt_enhancement` and is always sent explicitly; a "Show progress"
  toggle (stream or whole answer). Any condition image makes it an edit.
- **Progress.** The stream's phase, the image of `n` and the denoising step.
  The revised prompt, the answer's `warnings` and `condition_sizes` show beside
  it. A 409 names the way out, a 503 shows its `Retry-After`, an `error` event
  keeps the pictures made before it.
- **The switch.** With any model but the image model loaded, a band offers
  "Switch to the image model": `POST /v1/models/switch` by catalog id
  (the Server view's `switchTo`), then the page follows the model each second.
  Generate stays disabled until the image model serves. No request of the view
  names a `model` except that switch; tests assert it for all four
  route/stream combinations.
- **Model kind.** `ModelState` carries `kind` and `canChat` (chat and Decide
  are not offered an image model: they would get a 409), so the context bar,
  which reads `max_model_len`, gets `null` and draws no limit; the chat's empty
  state says the image model is loaded.
- **Defaults taken.** The gallery lives in the tab (data URLs, newest first,
  removable, cleared on reload) and downloads a PNG named
  `ignis-<seed+index>-<size>.png`. No auto-retry on 503: the line says when to
  retry. The previews are object URLs, revoked on unmount.
- **Gates.** `npm test` (855 tests) and `npm run build` pass. The view was not
  looked at in a browser (the owner watches :5173).

## As built (322g, 2026-10-14, branch `image-serving-322`: the review-322cf close-out)

The findings of `.scratch/orchestra/reviews/review-322cf.md`, as listed in
`step0-322g.md`; M1 and M3 as the coordinator decided them. Commits
e74a0e6..HEAD.

- **M1, the queue turn before a visit.** An enhanced request on the loaded
  image model is admitted to its queue as a job is (`503 engine_full`,
  `model_stopped`) and waits its turn (`routes::queue_turn`). The visit
  begins once every job admitted before it has run, with none running, so
  its drain cuts none of them. The jobs admitted behind it run in the
  visit's drain, as in any switch's, and end `503 model_stopped` past
  `switch.drain_timeout`. The enhanced job then runs on the image model
  loaded again. Test: a running and a queued job, held past a 200 ms drain
  window, both complete, then the visit, then the enhanced job.
- **M2 (web).** The image view follows its switch as the Server view follows
  Load. It records the last switch at the press, stops once the image model
  serves or the record changes, and shows `lastSwitchLine` in the band when
  the switch did not land (`studio.ts` `settleSwitch`, tested). The POST's
  answer goes through `afterSwitch`, so a `409` re-reads the model.
- **M3, the enhancer's rule.** `images::enhance::unknown_enhancer` applies
  where the server can switch. The enhancer must be the text model loaded,
  or one an explicit switch by id may name (`Switcher::text_model_among`,
  the patch's own `switch.known_models` included), never the image model.
  - A `PATCH` naming `image.prompt_enhancer` otherwise is `400
    config_invalid`, nothing applied. A patch that does not name it is not
    judged by it.
  - A start warns once: `ignis.config.prompt_enhancer_unknown`, naming it.
  - The request-time fallback with a warning stays.
- **Found on the way: the start's `image.*`.** `main` never put the start's
  `image.*` (flag, env, file) into the live knobs. Every image route ran on
  the defaults until a `PATCH` or a switch applied the configuration.
  `Server::with_image` now does it at start. The start warning's test sets
  the enhancer by flag, so it covers this.
- **L1, the decode.** The decode's turn is held by the blocking task that
  decodes, so a disconnect mid-decode does not free it (`images::Decodes`).
  Their count is the new live field `image.max_condition_decodes` (1,
  1..=16). Each condition image is resized as soon as it is decoded, to its
  unscaled encoded size (`routes::Conditions`); its size as sent stays for
  the budget and `auto`.
- **L2, the visit from another text model.** A visit that ends on another
  model than it left writes each of its two switches to `last_switch` with
  that switch's own ends (`Begun::recorded`). Once it lands, the record is
  the enhancer to the image model; when the way on fails, it says how
  (restored: the enhancer serves again). A visit back to the model it left
  writes nothing, as before.
- **L3.** An enhanced request on the enhancer the idle clock unloaded loads
  the enhancer, rewrites on it and switches once: two loads, not three.
- **L4, the user docs.** The README now:
  - links the weights' `LICENSE` and `NOTICE` on the hub at the pinned
    revision, with root `NOTICE` for the third-party code and text;
  - describes the queue turn and the drain behind it, the visit's records
    and the enhancer's rule;
  - gives `max_condition_mipx` in Mi pixels per image, lists every
    condition-image refusal, and adds `max_condition_decodes`;
  - names the past-the-plan `400` under Sizes.
- **L5.** `tools/image-prompts/gen_prompts.py`, whose docstring is the
  recipe, regenerates `prompts.rs` byte for byte; the header names it.
- **L6.** An edit's images are PNG-encoded for the enhancer on
  `spawn_blocking`. The vision path, untested before, now has a test.
- **L7.**
  - `make smoke` under `MODEL=image` makes one 1024x1024 image and prints
    its status and size; `flags-selftest` checks both families.
  - The web's `DEFAULT_LIMITS.maxN` is 10, and a test pins the defaults to
    the server's.
  - The chat's empty state names the image view for an unloaded image model
    too, tested.
  - A gallery picture's data URL is built once, when it arrives.
- **Nits.** `loading` is streamed from any load a request waits on, another
  request's included. `enhance` checks and rewrites on one pinned server.
  `Waiting::start` is gone (review-322b L5).
- **Defaults taken** (owner may revisit).
  - **M1, the jobs behind.** They run in the visit's drain rather than being
    cut at once: they cannot move to the image model loaded again, and the
    drain window is what any switch gives them. The enhanced job runs after
    them.
  - **L1, scaled images.** An image the budget scales is resized again from
    its 1024²-area copy, not from the source: two Lanczos passes where there
    was one. diffusers has no scaled case to match. The unscaled case is the
    single pass from the source it always was. A source smaller than 1024² of
    area is upscaled at once too (a 64x32 PNG is held as 1440x736, about
    4 MiB): what an edit holds while it waits is bounded by 10 such images,
    not 10 frames of up to 64 MiB.
  - **L2, the records.** Each leg is recorded with its own ends, not one
    record from the first model to the last, so that a restored or failed
    outcome names the model actually loaded again.
  - **M3, without a switcher.** A server that cannot switch judges no
    enhancer: there are no text models to judge it against.
- **For #321.** A runner that panicked keeps serving: the job is `500
  engine_error` and its prefix is released. With the real leaf, a panic may
  leave device state behind, so the adapter should then fail the model
  (status `failed`, with a switch or reload as the way back). (321f: done,
  as an unload the next request loads back from; see "As built (321f)".)
- **Also.** `tests/idle_unload.rs`' `unloaded()` waits for the unload to
  free the switch's slot. Under CPU load, four tests met the slot still held
  (`409 switch_in_progress`).
- **Gates.** `CARGO_INCREMENTAL=0 cargo test --workspace` 2921/0/6 (322e's
  2913 plus eight new tests); `cargo check -p ignis-server --features cuda
  --tests` clean; in `web/`, `npm test` 860 and `npm run build` green;
  `mk/flags-selftest.sh` ok.

## As built (321d, 2026-10-14, branch `image-leaf-321`: the real runner behind these routes)

Spec 04's "As built (321d)" has the adapter itself. What it settles on this
side of the seam:

- **The runner.** Under `cuda`, `load_model` loads `images::leaf::LeafRunner`
  (the stub `cuda_image_runner` is gone): the artifact's front end, then the
  image leaf on device 0 in `vram`'s budget. A plan that does not fit is the
  load's refusal (`ignis.model.load_failed`), naming the shortfall.
- **`condition_tokens` is the leaf's own count** (322c's F1), and more: within
  the plan's budget, a condition the leaf's arena has no room for, or a
  condition size the encoder's processor would not keep (past 200:1, off the
  32-pixel grid, under its pixel floor), counts one past the plan's budget.
  `budget::fit` then scales the images for those limits as for the budget.
  (321f: superseded. The count now says why it has none: no room is scaled
  for, and a size the processor would not keep is `400 image_refused`.)
- **`max_target_cells`** is the largest of `image.sizes` at load, and the
  leaf's arena is sized for it: 16,800 cells by default. So at load no
  configured size is past the plan, and the `ignis.image.sizes_not_served`
  warning fires only for a size a live `PATCH` added; that size is served
  after the next load. (321f: superseded. The arena takes the largest size
  within `image.max_target_cells`, and the warning fires at load for every
  size past it.)
- **A refusal of the runner's own** is `ImageError::Refused`, answered
  `400 invalid_request_error` with code `image_refused` in the runner's words
  (whole, or as the stream's `error` event), logged `ignis.image.job_refused`
  at info. Examples: a prompt past the RoPE table (about 8,000 tokens), a
  condition size the processor would not keep. `500 engine_error` is now a
  failure of the leaf, not a refusal.
- **A prefix outlives no runner.** The runner frees every prefix it still
  holds when it is dropped: a switch or an idle unload that ends a running
  job between two calls drops the runner first, and `Held`'s release then
  finds none (measured: +0.0 MiB, against −1,042 MiB without it).
- **A corner for the fit.** The processor's 200:1 limit is not monotone in
  the area, and `fit` reads the floor first: ten 16380x100 panoramas round to
  6560x32 at the 512 floor and are refused `400 condition_budget_exceeded`,
  though 12800x64 would serve. Not changed here. (321f: fixed. The fit no
  longer reads the floor first.)
- **AC 14.** The routes now run on the real leaf through the router
  (`tests/image_runner_gpu.rs`: a 1024² generation in 1.70 s and an edit in
  2.25 s, the same bytes twice each). The live run from the Playground,
  switching from the 27B and back, remains the owner's.

## As built (321f, 2026-10-14, branch `image-leaf-321`: the review-321d close-out)

Spec 04's "As built (321f)" has the details. On this side of the seam:

- **Refusals.** Two requests are `400 image_refused` in the model's words,
  at the fit, before anything is encoded:
  - a condition image the model refuses at its condition size (a panorama
    past 200:1);
  - a prompt with no image that runs past the model's positions.

  When the condition images have no room even at the floor, the answer is
  `400 condition_budget_exceeded`, naming the room. The OpenAPI descriptions
  and the user README say so.
- **The fit's corner is fixed** (321d's): the fit no longer reads the floor
  first.
- **`image.max_target_cells`** is a new live field: 16,800 cells by default,
  1..=2^20, read by the load. The arena is sized for the largest of
  `image.sizes` within it. A size past it is not served, and the load warns
  `ignis.image.sizes_not_served`. `image.sizes` takes sides up to 16,384.
- **A runner that panics fails its model.** Its request gets `500
  engine_error`, and at once the queue closes and the runner is dropped.
  Then the model is unloaded, and the next request loads it again. Without
  a switcher the answer is `503 model_stopped`.
- **AC 14** remains the owner's. Spec 04's "As built (321f)" has the notes
  for it.

## As built (AC 14, live run, 2026-10-14, branch `image-live-322` on d81e33e)

One live run on the 5090, switching from the 27B to the image model and back.
The run uses the Playground's own HTTP calls, in its order.

**Setup.**
- The build: `make build` (release, CUDA leaf, embedded UI).
- The start: `make start` with `ARTIFACT=F:/ai/q38/ninfer-models/qwen3_8_27b_nvfp4full-v2.ninfer`,
  `IDLE_UNLOAD=0`, `METRICS=1` and
  `ARGS='--switch-known-models ignis-image-2.1-turbo=F:/ai/models/ignis-image-2.1-turbo/ignis_image_2_1_turbo-v2.ninfer'`.
  The rest is the make defaults: context 524,288, `hq-e8-2b`, dflash2/7,
  yarn:2, host pool 8G.
- The process was ready in 12.7 s.
- The image load dropped `--spec-backend` and `--spec-draft-tokens`
  (`ignis.model.switch_flags_dropped`).
- The image plan: weights 14.40 GiB, arena 1.23 GiB, condition budget 43,584
  tokens; budget 26.96 of 28.46 GiB free; the leaf's load took 8.17-8.47 s.

**The calls.** The driver is `.scratch/live/drive.py` (log `drive.jsonl`,
server log `ignis-server.log`, both in `.scratch/live/`). Its bodies are the
Playground's:
- the chat is `buildChatRequest` (streamed, no `model`);
- the switch is `switchTo` (`POST /v1/models/switch {model}`, with the id
  `imageModelRow` picks from `GET /v1/models/catalog`), then `GET /v1/models`
  polled every 250 ms until it serves with a new `last_switch`;
- the image requests are `generationBody` and `editForm` (streamed, no
  `model`).

The times are client-side, from each request's send. The phase marks are the
first SSE event of each phase. "Server" is `ignis.model.switched`.

| step | call | answer | time |
|---|---|---|---|
| a | chat on the 27B, 23 prompt tokens | 29 tokens | first token 0.216 s, done 0.403 s |
| b | switch 27B → `ignis-image-2.1-turbo` | 202 in 16 ms | serving at **10.02 s**. Server: 9,962 ms (teardown 908, load 9,021, warm-up 0) |
| c | 2048x2048 generation, seed 2, streamed | 200 in 15 ms | encoding 0.015 → denoising 0.188 → decoding 6.100 → completed 7.847; **7.87 s** total |
| d | edit, 2 images, size `auto` (→ 2048x2048), seed 3, streamed | 200 in 48 ms | encoding 0.048 → denoising 0.999 → decoding 7.872 → completed 9.466; **9.48 s** total |
| e | switch `ignis-image-2.1-turbo` → 27B (the Server view's Load) | 202 in 16 ms | serving at **11.30 s**. Server: 11,246 ms (teardown 123, load 10,570, warm-up 124) |
| f | chat on the 27B again | 41 tokens | first token 0.050 s, done 0.331 s |

Steps a to f took 39.5 s end to end.

**The image times, by phase.**
- **The generation (c).**
  - Encoding: 0.17 s.
  - Denoising: 8 steps in 5.91 s, 0.731-0.742 s each (321e's in-process
    step: 747-802 ms).
  - Decoding to completed: 1.75 s. The VAE decode alone is 646 ms (321e).
    The other ~1.1 s is the PNG, its base64 and the send; the events do not
    split them.
  - Against 321e's in-process 2048² job (6.79 s), the request takes 1.08 s
    more, all of it in that last phase.
- **The edit (d).**
  - Encoding: 0.95 s, between 321e's one image (469 ms) and four (2,162 ms).
  - Denoising: 8 steps in 6.87 s, 0.855-0.861 s each (321e: 809-816 ms with
    one image, 982-991 with four).
  - Decoding to completed: 1.59 s.
  - 9.48 s in all, inside the 7.6-10.7 s that spec 04's 321f notes expect.

**The PNGs**, in `F:/ai/models/ignis-image-2.1-turbo/freerun/live/`:
- **`t2i-2048-seed2-0.png`** (2048x2048 RGBA, 8,711,736 bytes). The prompt is
  the oracle's t2i prompt: a red fox in snow at dawn, holding a sign reading
  "ignis", oil painting.
  - It looks right: a sharp, coherent fox, the sign spelled "ignis", an
    oil-paint texture, a misty treeline.
  - Alpha is 250-255.
  - Byte-identical to `attempt1-t2i-2048-seed2-0.png`, the same request on
    an earlier load.
- **`edit-2img-auto-seed3-0.png`** (2048x2048 RGBA, 4,922,151 bytes). The
  condition images, in this order: the oracle's `edit/condition.png` (shapes
  on a transparent background) and `freerun/t2i/ignis.png` (the 1024² fox).
  The prompt: "Place the colourful shapes from the first image as polished
  glass ornaments in the snow around the fox from the second image".
  - It is coherent, with no noise or banding to the eye.
  - The circle, square and triangle are now glossy glass.
  - The first image's layout and alpha are kept: transparent above, opaque
    below.
  - The opaque band is the second image's snow. Under the triangle are the
    fox's tail and fur and the sign's post.
  - The model followed the first image's layout more than "around the fox":
    the fox is mostly hidden. That is the model's reading of the prompt, not
    a fault of the run.

**Defaults taken.**
- **Calls, not clicks.** The Playground's calls were sent by a script, not by
  clicking in the browser: the same routes, bodies and order.
- **The chat.** It is the Playground's body with `reasoning_effort: "none"`
  (its "Off") and `max_tokens: 256`, instead of the defaults xhigh/16384, so
  the chat stays short.
- **The 27B's artifact.** It is named outright. The worktree has no
  `models/` link, and the default `./models/...` points there from the main
  checkout.
- **The known model.** The image model is known through `ARGS`
  (`--switch-known-models`). Make has no knob for it.
- **Prompts and seeds.** The generation reuses the oracle's prompt, with
  seed 2 as the oracle's 2K case. The edit prompt is agent-live's: the oracle's
  "keep the transparent background" shows W8A8's faint band (321e).
- **Attempt 1.** The driver failed on its own log call after the
  generation's image had arrived (a keyword clash). In that attempt:
  - 27B → image took 9.86 s (server 9,810 ms: teardown 1,065, load 8,711);
  - the 2048² phases matched the clean run (decoding at 6.185 s);
  - it went back to the 27B in 11.61 s (server 11,569 ms).

  The table above is the second, clean run of the whole sequence. Its logs
  are `attempt1.jsonl` and `back.jsonl`.

**Anything odd.**
- **The 27B comes back with a smaller KV pool.** After a switch away and
  back, it planned 10,428 pages (667,392 tokens) against 11,357 (726,848)
  at the cold start: 8.2% less.
  - The cause: its `free_at_start` was 30,556,864,512 bytes against
    31,104,512,000, so 547,647,488 bytes less.
  - The plan charges the CUDA context (452,595,712 bytes) as if it did not
    exist yet (`crates/runtime/src/lib.rs`, `CUDA_CONTEXT_BYTES`). But a
    switch keeps the process's context alive across the teardown, so the
    context is counted twice. The other ~91 MiB is held across the teardown
    as well.
  - It leaves 1.78 GB free after the load, against 1.24 GB cold.
  - The image plan reads the same smaller free figure but charges no
    context; its budget is 43,584 tokens here against 321f's 43,840.
  - This is not image-specific: any switch or reload does it. It is not
    changed here.
- **The catalog's `on_disk` reads `absent` for both built-in rows.** That
  covers the 27B, which is loaded, and the image model, which is switchable
  through `--switch-known-models`. `on_disk` counts only `download.path`,
  and neither model is under it. The switch by id follows `switchable` and
  works; only the badge misleads.
- **A successful image job leaves no log line and no metric.** The text
  requests log `admitted`/`ttft`/`done`. An image job logs only when refused,
  failed or panicked. `/metrics` has no image series, so the phase times
  above exist only in the stream.

## As built (324 item 1, 2026-10-15, branch `followups-324-serving`: Generate with a text model loaded)

The owner's report (2026-10-15): with the 27B loaded, Generate and Edit could
not be pressed; with enhancement on, the only way was image → 27B → image,
three loads, where the server already had the text-first path (one load).

- **The press is the choice.** With a text model loaded (or unloaded while
  idle) and the catalog listing the image model, Generate and Edit are on: the
  request carries `model: <the image model's catalog id>`, so the server does
  §Prompt enhancement's text-first path. `studio.ts` `runPlan` says what the
  press does and the button reads it:
  - enhancement off: "Switch & generate" (or "& edit"), "Unloads <id> and
    loads the image model first." (just "Loads …" when it is unloaded);
  - enhancement on, the enhancer loaded (`image.prompt_enhancer` read from
    `GET /v1/config`): "Enhance with <id>, then switch & generate", the rewrite
    on it, then one switch;
  - enhancement on, another text model: "Switch & generate", "Switches to
    <enhancer> to rewrite the prompt, then to the image model.";
  - image model loaded: "Generate", no `model`, as before.
- **Refusals.** `switch.allow_implicit` off (read from `GET /v1/config`): the
  button stays disabled and says so. At run time, a `404 model_not_found` (the
  server's answer when it will not switch for a request: implicit switching
  off, or a model it cannot find) reads plainly and points to the switch
  button. The standalone "Switch to the image model" stays; it is disabled
  while a run is under way.
- **Progress.** The phases are those the stream already sends (`enhancing`,
  `switching`, `loading`, then the job's). On the first `switching` or
  `loading` event the view reads `GET /v1/models` once, and `useModel`
  follows the switch from there as any switch; a run that named a model reads
  it again at its end. Without "Show progress" the line says the image model
  is loaded before the job. With `switch.allow_implicit` off the band no
  longer says Generate switches.
- **Untouched.** The chat and Decide never send `model` (the #305 lesson); the
  view's own requests on the image model send none either.
- **Server.** No change: three router tests through the mock loader in
  `tests/image_enhancement.rs` pin the path, and passed as written:
  - enhancer loaded, enhancement on, streamed: `enhancing 1/1`, `switching
    1/1`, `loading 1/1`, then the job; one load (the image model); the
    rewrite prefilled once on the enhancer; `revised_prompt` on the completed
    event; `last_switch` enhancer → image;
  - enhancement off (the request's `false` over the setting's `true`): one
    switch, nothing prefilled on the text model, no `revised_prompt`;
  - `switch.allow_implicit` off: `404 model_not_found`, a status even when
    streamed, before any rewrite or load; the text model serves on.
- **Defaults taken** (owner may revisit).
  - "Known" is the catalog's image row (`GET /v1/models/catalog`), not
    `GET /v1/models`, which lists only the loaded model. `on_disk` is not
    consulted: it read `absent` for a switchable known model when it counted
    only `download.path` (AC 14's note; item 3 below made a listed id read
    its own file); a model the server cannot find answers the `404` above.
  - The id sent is the catalog row's, the one the switch button sends.
  - With enhancement on and another text model loaded, the label is the
    plain "Switch & generate" with the two switches in the line beneath,
    rather than a third, longer label.
- **Gates** (on 4bd8923). `CARGO_INCREMENTAL=0 cargo test --workspace`
  2983/0/6 (321f's 2980 plus the three tests above); in `web/`, `npm test`
  870 and `npm run build` green (again on b52e530, the band's copy);
  `cargo check -p ignis-server --features cuda --tests` clean (its one
  warning, `openapi.rs`' unused import, is older). The view was not looked at
  in a browser (the owner watches :5173).

## As built (324 items 3-4, 2026-10-15, branch `followups-324-serving`: the catalog's `on_disk`, and what an image job leaves behind)

- **Item 3, `on_disk`.** A catalog row is `complete` when the file the
  server would load the model from is there: the artifact a listed model
  names (`switch.known_models`, including the model the server started on)
  wins over `download.path`, exactly as `Switcher::resolve` lets it win for a
  switch. `bytes` is then that file's size and `family` its header's. The
  rule is one `metadata` call per row, no hash (the sha256 stays `model
  verify`'s). A listed model whose file is missing reads `absent`, as
  `resolve` refuses it (324 close-out F5). Tests:
  `a_known_model_that_is_also_a_catalog_entry_is_one_row`,
  `a_listed_model_whose_file_is_gone_reads_absent_though_download_path_has_the_entry`.
- **324ui's default revisited, and kept.** The image view still takes "known"
  to mean the catalog row, not `on_disk` or `switchable`: the server's `404
  model_not_found` stays the one place that decides, and the row is right for
  a model the server cannot find yet. Nothing on the web side changed for
  item 3.
- **Item 4, the log line.** One `ignis.image.job` INFO line per ended job,
  every outcome: `job_id` (`img-<n>`), `model`, `kind`
  (`generation|edit`), `outcome`, `n`, `images` made, `size`,
  `condition_images`, `enhancement` (`off|rewritten|as_sent`), and in
  milliseconds `enhance_ms`, `switch_ms` (loads and switches before the job),
  `queued_ms`, `encode_ms`, `denoise_ms`, `decode_ms`, `run_ms` (turn to end)
  and `total_ms` (admission to end). A job whose request left the queue is
  `cancelled` with zeros for the phases it never reached.
- **Item 4, the metrics.** Five `ignis_image_*` families (ADR 0017's
  amendment of 2026-10-15): jobs by outcome, images, a duration histogram of
  the completed jobs, seconds per phase, and the queue gauge. Absent until
  the image model admitted a job. Recorded by `job::spawn` as the job ends;
  the queue gauge is set by the queue itself (see the close-out below).
- **Item 4, the Monitor.** `snapshot.ts` reads the families (all in the
  contract, the restart check counts them), `derive.ts` gives `images` (the
  window's jobs by outcome, images, duration quantiles, each phase's share),
  and an *Image jobs* card draws them once they exist.
- **Defaults taken.** Outcomes are the five of `ImageOutcome`; a refused
  runner call (`ImageError::Refused`) is `refused`, not `failed`. The
  duration histogram reuses the request duration's bounds. `queued` is a
  phase of the counter and not of the histogram. The id is a process
  counter, not an id the client sees.
- **Tests.** `metrics.rs`: absent then counted, declared once.
  `tests/images_observability.rs`, a binary of its own (the line is written
  on the blocking thread, so it installs the process subscriber): one fake
  runner job, one line with its fields, the families on `/metrics`.
  Monitor: `snapshot.test.ts`, `derive.test.ts`, `MonitorView.test.tsx`.

## As built (324 close-out, 2026-10-15, branch `followups-324-serving`: review-324serving F1 to F8)

- **F1, the queue gauge.** `ImageJobs` publishes the gauge itself, under its
  lock, at every admission, turn and departure (`Waiting`/`Running` drop):
  a request refused after admission (condition budget, `model_stopped`)
  leaves `waiting=0`, and a stale snapshot cannot land last. `admit` hands
  the queue the metrics (`report_to`) before it asks for a place; the
  enhancer's visit turn counts as any job in the queue does. Test: an
  over-budget request, then `waiting 0`, in `tests/images_observability.rs`.
- **F2, three loads to two.** An idle-unloaded text model that is not the
  enhancer, enhancement on, a request naming the image model: `admit` loads
  the enhancer first (the rewrite on it, then one switch). Test:
  `an_enhanced_request_on_another_unloaded_text_model_loads_the_enhancer_first`
  (`loads()` is `[enhancer, IMAGE]`). `runPlan` says "Loads X to rewrite
  the prompt, then switches" for an unloaded model, "Switches to X" for a
  loaded one: two loads either way.
- **F3.** The observability test runs a job whose decode fails (a `Flaky`
  wrapper of the fake runner): one more log line, `failed`, one more count.
- **F4.** The duration histogram observes `ran` (turn to last image), as the
  HELP and ADR 0017 say; `ImageJobEnd` carries it.
- **F5.** See item 3 above. **F6.** `queued_ms` and the `queued` phase run
  from admission (`JobContext::queued_at`), through `prepare`. **F7.**
  `TokenizeRoute`'s doc line back. **F8.** The notes about `on_disk` above.
- **Nits.** The Playground reads the model once per request that switches
  (not on every event); while a request runs the form no longer flips to
  "Needs the image model" (`runPlan`'s `running`); the Monitor's list of
  non-completed outcomes is `notCompleted`; the log field is `job_id`
  (ADR 0012's span field is `request_id`); `job::run` is test-only.
- **Defaults taken.** The enhancer is loaded first only when the server can
  switch to it (`Switcher::text_model`); otherwise the old path (the image
  model loads, the rewrite is skipped with a warning). The view was not
  looked at in a browser.

## As built (324 image VRAM, 2026-10-15, branch `image-vram-324`: the image model's VRAM on `/metrics` and the Monitor)

- **Why it read empty.** 322c F3 had `observe_load` zero a text model's
  reservations for a model with no engine, and the image load reported none
  of its own (`LoadedModel::reservations` was `None`): with the image model
  loaded the VRAM plan read no load.
- **The plan's lines.** `LoadReservations::image` maps the leaf's
  `ImagePlan` into ADR 0030's family: weights as `line="weights"`, the arena
  as `line="workspace"`, the condition budget as a new
  `line="condition_budget"` (always rendered, 0 on a text load), inside
  `ignis_vram_budget_bytes`. No KV pool, no `cuda_context` or `residual`
  line: the image plan charges neither (ADR 0030's #324 amendment).
  `cuda_image_runner` returns them beside the runner (`LeafRunner::leaf_plan`),
  so the start (`with_load_reservations`) and a switch (`observe_load`, after
  its `set_unloaded`) publish them by the paths a text load takes. The fake
  runner plans no device memory and reports none, as `MockCompute`.
- **What is in use.** `ignis_condition_budget_used_bytes` (ADR 0017): the
  bytes of the condition prefixes held, the counterpart of
  `ignis_kv_pool_used_pages`. `ImageRunner::condition_bytes` gives them (the
  leaf runner sums `Condition::bytes`; the fake charges
  `FAKE_PREFIX_TOKEN_BYTES`, 256 KiB, a token); the job publishes them
  through the queue's gauge (`ImageJobs::publish_conditions`) with the runner
  held, right after `encode_condition` and after the release in `Held`'s drop.
  Every `set_load_reservations` resets the gauge to 0, so a load either way
  starts clean, and a job of a model shut down cannot publish after it (its
  runner is gone). After a failed drain or restore the gauge can stay stale
  until the next load, as the text model's KV occupancy already does.
- **The Monitor.** `snapshot.ts` reads the line and the gauge (both in the
  contract); `derive.ts` counts `condition_budget` as a line a server may
  predate (as Flash-Next's two) and gives `conditionBudgetBytes` and a
  `conditionInUse` meter on a load with a condition budget. On an image load
  the *VRAM plan* card lays out weights, workspace and the condition budget,
  says the budget took what the weights and the arena left (not a KV pool),
  and *In use now* has a *Condition prefixes* row. A text load shows neither.
- **Defaults taken.** The arena is `workspace`, not a new line; the condition
  budget is reported in bytes only (its tokens stay in `ignis.image.plan`).
  The `condition_budget` line is the plan's, set at load: a live
  `image.condition_budget_tokens` lowers what each request may hold
  (`images::budget`), not the line, so a PATCH of it leaves the Monitor's plan
  as it was. The gauge is rendered on every load, 0 on a text one, so its name is not an
  `ignis_image_*` one: those families are absent until a job. The Monitor
  keeps the KV pool, KV-RAM and retained-slot rows on an image load; they
  read "this load has none of it to give".
- **Tests.** `metrics.rs`:
  `an_image_load_reserves_its_plan_lines_and_its_prefixes_read_against_the_condition_budget`
  (the lines, the gauge, a text load's and an unload's reset), the line order
  and the declared families. `tests/model_kind.rs`:
  `the_image_models_vram_plan_and_the_prefixes_its_jobs_hold_show_on_metrics_and_go_with_it`
  (start on the image model, a gated job holds its prefix, the end releases
  it, a switch to a text model and back; `MockLoader::reserve` gives each id
  its reservations). `image.rs`: the fake's bytes follow encode and release.
  Monitor: `snapshot.test.ts`, `derive.test.ts`, `MonitorView.test.tsx`
  (`IMAGE_LOAD_EXPOSITION`). The real leaf's numbers were not seen live here:
  no GPU run (another agent held the card).
