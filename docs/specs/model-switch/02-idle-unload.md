# 02 — idle unload: the model leaves the GPU when nobody uses it

GitHub: #317. ADR 0049. Builds on spec 01 (runtime model switch) and on
config v2 (spec `docs/specs/config-v2/`).

## Problem

A loaded model holds the 5090 (VRAM, pinned host pools, KV-RAM arena) for
as long as the process runs, used or not. Giving the card back means
stopping the server, which also takes the API, `/metrics` and the
Playground down.

## Solution

After `model.idle_unload` seconds of no use (default 300, 0 = never), the
server tears the model down the way a switch does, and stays up with
`status: "unloaded"`. The next request that needs a model loads it again
from its artifact and is served once it is warm; requests arriving during
that load wait for it. Decisions and trade-offs: ADR 0049.

## Seams

- **Config** (`crates/server/src/config/schema.rs`, `ModelGroup`): one new
  declaration, `idle_unload: Secs = 300`, `[Visible, Patchable]`, not
  `ReloadRequired`, all families. Flag `--model-idle-unload`, env
  `IGNIS_MODEL_IDLE_UNLOAD`, file key `model.idle_unload`. A ceiling like
  the other `Secs` fields (e.g. 7 days); 0 never unloads.
- **Status** (`crates/server/src/active.rs`, `ModelStatus`): two new
  variants, `Unloaded` (`"unloaded"`) and `Loading` (`"loading"`, the load
  that leaves `Unloaded`). `active` keeps the torn-down model's
  `ActiveModel` (its id and `source`), so everything that reads the id
  still can; nothing may call its engine. Audit what touches the active
  engine outside the gate (`/metrics`, telemetry, `observe_load`) and make
  each one safe on a shut-down engine.
- **The gate** (`crates/server/src/api.rs`, `require_serving`): in
  `Unloaded` and `Loading`, every route passes to its handler (the load
  happens behind auth, below). It must **not** load here: this layer sits
  outside `require_api_key`, and an unauthenticated request must never
  cost a load.
- **The load on demand** (`ensure_loaded`): where the model is first
  needed, after auth and before `Server::pinned()` — at the three
  `switch_to_named_model` call sites (`api.rs` chat completions,
  `responses/mod.rs`, and **per job** in `responses/socket.rs`, since an
  open socket never re-enters the middleware), and on every other model
  route (`decide`, tokenize/detokenize if they need the engine — audit the
  route list in `api.rs`), e.g. as a layer *after* `require_api_key`. From
  `Unloaded` it loads the model the request names when that is a known one
  (resolve the implicit switch's target first: one load, never two), else
  the unloaded model's `source`. Waiters need a signal, not polling of
  `status`: a `tokio::sync::watch` (or `Notify`) beside `status`, fired on
  every store; waiters loop until `Serving` (go on) or `Failed` (503 with
  the reason). Concurrent arrivals share one load.
- **The switch module** (`crates/server/src/model_switch.rs`): unload and
  the load after it reuse the switch's slot (`Switcher::running`, so an
  unload, a load, a switch and a reload never overlap), its teardown and
  its `load_and_warm`/`publish`. From `Unloaded`, `run()` must **skip the
  drain and the teardown** (the engine is already shut down) and, on a
  failed load, **not** `restore()`: it leaves `ModelStatus::Failed`.
  `POST /v1/models/switch` and a reloading `PATCH /v1/config` work from
  `Unloaded` (they load). A switch to the unloaded model's own id loads it.
- **The idle clock**: a task owned by the server, ticking about once a
  second. Use an **in-flight count**, not a timestamp of admission alone:
  incremented when a model route admits a request, decremented when its
  response body / stream ends (per job on the socket, not per connection,
  so an idle socket does not pin the model). Without it the window between
  admission and submit (template render, media decode) looks idle. "Last
  use" moves while the count is above 0 or the engine has work waiting or
  running (`Engine::interval_counters`, a plain read). Unload when
  `idle_unload > 0`, the status is `Serving`, count 0, engine empty, and
  `now - last use >= idle_unload`. The value reaches the clock through
  `Server::apply_config` (the path a live non-reload `PATCH` already takes
  for `switch.drain_timeout`), read at each tick, never a copy captured at
  start.
- **Telemetry**: `ignis.model.unloaded` (model, idle_secs) and
  `ignis.model.reloaded` (model, took_ms, load_ms, warm_up_ms). Not written
  to `last_switch`. If a metric exports the model status, it gains the
  unloaded state.
- **Playground** (`web/src/api/model.ts`, the header chip, the Server
  view): `unloaded` is shown as such (the model loads on the next
  message), not as an error; polling does not wake it. The Server view's
  "Load" on the unloaded model loads it.
- **Make**: the targets that start a server for a measurement (bench, gate)
  pass `--model-idle-unload 0`.

## Acceptance

1. `model.idle_unload` resolves from flag, env and file with the config v2
   precedence; default 300; 0 accepted; above the ceiling refused naming it;
   visible in `GET /v1/config/fields`, patchable, no reload.
2. With `idle_unload = N > 0` and no requests, the model is torn down after
   N s (± one tick): `GET /v1/models` reports `status: "unloaded"` with the
   same model id, and `ignis.model.unloaded` is logged once.
3. With `idle_unload = 0` the model is never unloaded.
4. A request in flight (waiting or running, e.g. a long stream) keeps the
   model loaded past N s; the clock starts when the engine is empty again.
5. Polling `GET /v1/models`, `GET /v1/config*`, `GET /v1/models/catalog`
   and `/metrics` neither resets the clock nor loads an unloaded model.
6. A chat request while `unloaded` loads the model (`status: "loading"`
   meanwhile), is served (200 with a normal body), and `GET /v1/models`
   reports `serving` after; `ignis.model.reloaded` is logged. Two requests
   arriving together while unloaded cause one load and are both served.
   The same holds for a `/v1/responses` job on an already-open socket.
7. A load that fails while leaving `unloaded` leaves `status: "failed"` and
   the requests waiting on it get `503` with the reason; a later request or
   switch can try again.
8. An unauthenticated request (with `--server-api-key` set) while
   `unloaded` gets `401` and loads nothing.
9. A request naming another known model while `unloaded` loads that model
   only (one load).
10. A request admitted but not yet submitted to the engine (e.g. a slow
    template render or media decode, simulated in the test) is not unloaded
    under; the clock starts when its response has ended.
11. `POST /v1/models/switch` while unloaded loads the named model (the same
    id included); a `PATCH /v1/config` that needs a reload, while unloaded,
    loads the model with the new configuration.
12. An unload never overlaps a switch, a reload or a load (they share the
    switch's slot); an unload due while one runs waits for the next tick.
13. A live `PATCH` of `model.idle_unload` takes effect without a reload
    (e.g. 0 → 2 unloads ~2 s later; N → 0 stops a pending unload).
14. `last_switch` is not changed by an unload or the load after it.
15. Playground: `unloaded` and `loading` are shown as states, not errors;
    `unloaded` does not keep polling fast; a message sent while unloaded is
    answered once the model has loaded.
16. Docs: `docs/user/README.md` (what idle unload does, the knob, 0),
    `CONTEXT.md` (Idle unload), the config reference if generated.
17. Tests use a fake loader and an idle value of 1–2 s; no test waits
    minutes. One live check on the 5090 with a few seconds' value after
    the merge (VRAM freed after N s, the next request served).

## As built (2026-10-12, branch `idle-unload`)

Where the implementation chose what the sections above leave open.

- **The count is a layer.** `api.rs`'s `count_use`, inside the key layer,
  counts `POST` on chat completions, responses, `/v1/decide` (and
  `/v1/systemone`), `/v1/tokenize` and `/v1/detokenize` from admission until
  the response body is dropped (`idle::Counted`), streams included; the
  socket counts per job in `serve()`. The tokenize routes count as use (ADR
  0049 §2: the gate holds them) but never load: they read the kept
  template and `max_model_len`, not the engine.
- **No race with the unload.** The clock decides and stores `unloaded` while
  holding the count's lock (`Usage::when_idle` → `model_switch::claim_unload`),
  which an admission takes too: a request admitted around the unload either
  kept it from happening or reads `unloaded` and loads.
- **The load on demand** is `model_switch::ensure_loaded`, called first in
  `api::switch_to_named_model` (chat, responses, the socket's jobs) and in
  `/v1/decide`. A request naming a model the server does not know loads the
  unloaded one (its source) and is then refused by name as before: one load,
  never two, but not none.
- **Wire.** A failed load out of `unloaded` answers its waiters `503
  model_load_failed` with the reason, no `Retry-After`. `GET /v1/models`
  carries `switching: {from, to}` while `loading` too, naming what loads.
  The catalog lists no row as loaded while unloaded or loading, and the
  unloaded model as switchable (the Server view's "Load").
- **What is recorded.** The load a request begins is `ignis.model.reloaded`
  and leaves `last_switch` alone; an explicit `POST /v1/models/switch` or a
  reloading `PATCH /v1/config` out of `unloaded` is still a switch or a
  reload, logged and recorded as one (its status is `loading`, since nothing
  is drained). A failed load out of `unloaded` sets `failed`; the serving
  gate lets requests through then (the next one tries again) only while the
  switcher says nothing is resident — a failed switch with nothing to reload
  is still held as before.
- **Off unless configured.** `Live::default()` has `idle_unload` zero: only
  `main` (the config) and a live `PATCH` turn it on, so tests and in-process
  harnesses never meet an unload they did not ask for. The clock runs in the
  serve loops (`Server::spawn_idle_clock`, one-second tick); a test passes a
  shorter tick. `model_switch_gpu.rs` passes `--model-idle-unload 0`.
- **Make.** No Make target starts a server for a measurement by itself; the
  knob is `IDLE_UNLOAD` (passed as `--model-idle-unload`), and
  `scripts/swarm-ab.sh` and `scripts/vision-ttft-session.sh` pass 0.
