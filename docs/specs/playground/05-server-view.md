# playground 05 — the Server view: change the running configuration and the model from the page

GitHub: #316

## Problem Statement

The server can change model while it runs (`POST /v1/models/switch`, spec
model-switch/01, #305) and change its own configuration while it runs
(`GET`/`PATCH /v1/config`, spec config-v2/02, #311), and the catalog (ADR 0047,
#313) says which models it can fetch and which are on disk. None of it is
reachable from the Playground: an operator types `curl` against the API, reads
the field table in `ignis-server help --fields`, and needs the artifact's path on
the server's disk to switch to it.

A page cannot be built on today's API alone:

- `GET /v1/config` returns values only. Whether a field may be changed, whether
  a change reloads the model, what values it takes and what it means are in each
  field's declaration (`FieldMeta`, ADR 0046) and served nowhere.
- No route lists the models a switch can go to. `switch.known_models` is in
  `GET /v1/config`, but a catalog entry on disk (a known model too, ADR 0047) is
  not, and `model list` is a CLI command.
- `POST /v1/models/switch` needs the artifact's path on the server's machine,
  which a browser has no way to know for a catalog entry.
- A switch refused or rolled back says why only for one tick of
  `GET /v1/models` (`status: failed`, then `serving` again), so a page polling
  it learns that the switch did not take but not why.

## Solution

A fourth view in the Playground header, **Server**, beside Playground, Decide
and Monitor. It leads with the loaded model and the models it can switch to,
and below them the running configuration as a form. The server gains what the
page needs, each as an ordinary `/v1` route any client can use:

- `GET /v1/config/fields` — every visible field's declaration;
- `GET /v1/models/catalog` — every model the server can switch to or fetch,
  with its state on disk;
- `POST /v1/models/switch` with `model` alone — the id is resolved as a known
  model or a catalog entry on disk;
- `last_switch` on `GET /v1/models` — how the last switch or reload ended, and
  why.

Downloading a model from the page is not part of this spec: an absent model
shows the `model download` command to copy.

## Implementation Decisions

### `GET /v1/config/fields`

- `{ "object": "list", "data": [ ... ] }`, the envelope of `GET /v1/models`:
  one object per **visible** field, in declaration order — the same whitelist
  as `GET /v1/config`, so a hidden field (the API key, the download token) is
  absent here too.
- Built from `all_fields()`, like `help --fields --format json`
  (`config/cli.rs` `fields_json`); the two share whatever they can, so the CLI
  table and the route cannot drift.
- Keys: `group`, `name`, `file_key`, `flag`, `env`, `kind`, `nullable`,
  `default`, `rule`, `families`, `scoped`, `patchable`, `reload_required`,
  `description`.
  - `rule` is machine-readable: `null`, `{ "min": n|null, "max": n|null }`,
    `{ "multiple_of": n }` or `{ "one_of": ["…"] }`, plus `rule_text` (the
    sentence `help --fields` prints). `one_of` lists each value once in the
    form the field writes it (`canonical`): an alias (`hq_e8_2b`, `hq` for
    `hq-e8-2b`) is accepted by `PATCH` but not offered.
  - `families` is `null` for a field every family takes, else the list of
    scope names (`qwen38`, `qwen38flashnext`) that take it.
  - `nullable` says the field may be unset (its kind is `Opt<…>`), so a page
    can show "auto" for a `null` value.
- Static: it answers whatever the model's state, during a switch and on a
  server built with no running configuration too. Behind the API key like every
  other `/v1` route.

### `GET /v1/models/catalog`

- `{ "object": "list", "data": [ ... ] }`, one row per model the server knows of: every catalog entry (built-in, then
  operator), then every known model (`switch.known_models` and the start model)
  that is not a catalog entry.
- Row keys:
  - `id`, the served id;
  - `source`: `built-in`, `operator` or `known`;
  - `on_disk`: `complete`, `partial` or `absent`, as `model list` computes it
    for an entry (sizes only, never a hash); a known model is `complete` when
    its artifact is a file, else `absent`;
  - `bytes`: the entry's total, or the artifact's size for a known model on
    disk; the key is omitted otherwise (as `family` is when unknown);
  - `family`: the scope name the artifact's header names, when it is on disk;
    for the loaded model always the running family;
  - `loaded`: true for exactly the model serving now (the start model before
    any switch); no row is `loaded` while nothing serves (a switch whose
    restore failed), and the model torn down is then `switchable` again;
  - `switchable`: whether `POST /v1/models/switch` with this `model` alone
    would begin a switch now (resolvable, on disk, not the loaded one).
- No path is in a row: the page switches by id.
- Answers during a switch and while failed, like `GET /v1/models`.

### Switch by id

- `artifact` becomes optional in `POST /v1/models/switch`. With no `artifact`,
  `model` is resolved like this:
  1. the known models (`switch.known_models`, the start model — the start
     model stays one after any number of switches) — an explicit entry wins,
     as for an implicit switch;
  2. a catalog entry whose artifact is a file under `download.path`.
- `switch.allow_implicit` does not apply: it gates the switch a request
  triggers by naming a model, not an explicit one.
- An id that resolves to nothing, or to an entry not on disk, is `404
  model_not_found`, naming the id and, for an absent entry, the `model
  download` command that fetches it. A switch never starts a download (ADR
  0047).
- With both fields given, behaviour is as today.

### `last_switch` on `GET /v1/models`

- After the first switch or reload since start, `GET /v1/models` carries
  `last_switch: { from, to, outcome, reason? }`.
  - `outcome` is `switched`, `refused` (nothing was torn down), `restored`
    (the old model was reloaded) or `failed` (nothing serves).
  - `reason` is present for every outcome but `switched`.
- A reload from `PATCH /v1/config` is reported the same way, `from` equal to
  `to`. Additive: no existing key changes.

### The Server view (`web/src/server/`)

The page talks only to the routes above, `GET /v1/models`, `GET`/`PATCH
/v1/config`. Chat requests still name no model (#305's ping-pong fix stands);
the only switch the Playground makes is the explicit one on this page.

**Model band** (top, the kiln palette the Monitor uses in both schemes):

- The loaded id, large, in the display face; under it its state and its
  context (`max_model_len`).
- While switching: "Switching from A to B" and the seconds elapsed. The
  header's heat bar and flame run, through the header's existing `busy`.
- One row per `GET /v1/models/catalog` row: id, source, on-disk state with GiB.
  - The action is "Load" when `switchable`, "Loaded" text on the loaded row.
  - An absent or partial model shows "Copy download command", which copies
    `ignis-server model download <id>`.
- The band names how the last switch ended from `last_switch`: a `refused`,
  `restored` or `failed` outcome shows its reason in the fault colour until
  the next switch.
- When a chat reply is streaming, the band says the switch gives it
  `switch.drain_timeout` seconds to finish.

**Configuration** (below the band):

- One section per group, in declaration order, every visible field a row; on
  `lg` a list of the groups beside it jumps to each.
- A row: the field's name as the config file spells it, its description
  (first sentence, the rest behind "More"), and its control.
  - The control is chosen by the declaration: a `bool` is the existing
    `Switch`; a `one_of` with up to four values the existing `Segmented`,
    more a `<select>`; anything else a text input showing the kind and
    `rule_text` as its hint and "auto" as the placeholder of a `null` value.
  - The server is the judge of a value. The page sends what was typed and
    shows the refusal.
- A field that is not `patchable` shows its value with "Set at start" and no
  control. A field whose `families` excludes the running family shows "Not
  used by <family>" and no control. `switch.known_models` is shown read-only
  (a patch only adds to it).
- An editable `reload_required` field carries a small heat-gradient mark and the word
  "reload": the mark is the same heat as the header's bar, the one place
  this page spends colour.
- **The value in force.** For a `scoped` field, the running family's section
  value when `GET /v1/config` has one, else the group's. An edit goes where
  the value came from (the family's section or the group). A row reading a
  family section says "<family>'s own value".

**Applying** (a dock at the foot, shown while the draft holds a change):

- "N changes", "Discard", and the primary action, whose label says what will
  happen before the click:
  - "Apply" when no changed field is `reload_required`;
  - "Apply and reload <id>" when any is, since spec config-v2/02 then sends
    the whole patch through a reload. In that case the button carries the heat
    gradient and, when a chat reply is streaming, the dock says the reload gives
    it `switch.drain_timeout` seconds to finish.
- No confirmation dialog.
- One `PATCH` with only the changed fields, nested as the config file nests
  them.
- Outcomes:
  - `200`: the draft clears and the form shows the response's `config`; "Applied
    and written to <path>" or "Applied, not written to a file: <reason>".
  - `202`: the band shows the reload as a switch. Once `GET /v1/models` reads
    `serving` with `last_switch.outcome` `switched`: "Reloaded <id> and written
    to <path>" (or the no-file reason), and the configuration is read again.
    Any other outcome shows its reason, and the draft is kept.
  - `shadowed_at_restart`, when present: "Overridden at the next start:
    <key> (set by <spelling>)", one per key.
  - `400`: the server's message in the dock, and every row whose file key the
    message names is marked; nothing was applied, the draft is kept.
  - `409`: the switch under way is named; the action is disabled until it lands.
- No config (`501 config_unavailable`): the section says the server has no
  running configuration to show and the band still works.

**The model state the whole page shares** (`api/model.ts`):

- `ModelState` gains `switching` (`from`, `to`). `readModel` maps
  `status: switching` to it and `status: failed` to `error` with the reason.
- While `switching`, `useModel` re-reads every second. The chat's composer is
  not ready meanwhile (every other `/v1` route answers `503`), and the header's
  model status reads "Switching to <id>…".
- When the state returns to `ready`, the Server view reads the configuration
  and the catalog again.

**Look.** The page uses the Playground's brand as it is, with no new palette or
typeface:

- the kiln band, ember, `--heat`, Chakra Petch for display text and Instrument
  Sans for the rest, `.cut` chamfers;
- the form sits on one surface with group headings, not a card per field or per
  group;
- the one animation is the switch, through the header's heat bar and flame,
  which already exist;
- copy is English and sentence case like the rest of the Playground: no all-caps
  labels, no middle-dot meta strings;
- any popup inside a `.cut` box goes in a portal (the clip-path clips
  descendants).

### Mock (`web/mock.ts`)

`dev:mock` serves `GET /v1/config/fields`, `GET`/`PATCH /v1/config` (`200` live,
`202` when a changed field is `reload_required`), `GET /v1/models/catalog`,
`POST /v1/models/switch`, and `GET /v1/models` reporting `switching` for about
three seconds after a switch or reload, then `last_switch`.
- A patch whose value is the text "/error" is refused with a `400` naming that
  field's key.
- A switch to an id containing "fail" ends `restored` with a reason.

## Acceptance

### Server

1. `GET /v1/config/fields` lists every visible field once, in declaration
   order, with the keys above; a field not marked `visible` is absent; `rule`
   is structured for each `Validator` variant; `families` and `nullable` match
   the declaration.
2. `GET /v1/config/fields` answers `200` during warm-up, during a switch and
   on a server with no running configuration; `401` without the API key when
   one is set.
3. `GET /v1/models/catalog` lists every catalog entry (built-in then operator)
   and every known model that is not one, each once, with `source`, `on_disk`
   (`complete`/`partial`/`absent`), `bytes`, `family`, `loaded` and
   `switchable` as defined; exactly one row is `loaded` while a model serves,
   none while nothing does; no row carries a path.
4. `GET /v1/models/catalog` answers during a switch, and `loaded` follows a
   completed switch.
5. `POST /v1/models/switch` with `model` alone switches to a known model and
   to a catalog entry on disk, with `switch.allow_implicit` on or off; a known
   model's entry wins over a catalog entry of the same id; the start model
   stays switchable by id after switches through a loader that re-applies the
   config (not only the mock).
6. `POST /v1/models/switch` with `model` alone for an unknown id, or a catalog
   entry not on disk, is `404 model_not_found` naming the id (and the `model
   download` command for an entry), and nothing is unloaded or downloaded.
7. `POST /v1/models/switch` with `artifact` and `model` behaves as before
   (the existing switch tests pass unchanged).
8. `GET /v1/models` carries no `last_switch` before the first switch, and
   after each switch or reload carries its `from`, `to`, `outcome` and, for
   `refused`/`restored`/`failed`, `reason`; a `PATCH /v1/config` reload
   reports `from` = `to`.
9. The new routes are in the OpenAPI document (`/v1/docs`) with their
   responses, and answer the CORS preflight like the other `/v1` routes.

### Playground

10. The header offers a fourth view, Server, always present; it has no side
    drawers. Below `sm` the views are one compact menu, so the header fits a
    phone with the model status, Memory and Lock visible.
11. The band shows the loaded id, its state and context, and one row per
    catalog row with the actions defined above; "Load" sends
    `POST /v1/models/switch { model }` and the band follows the switch to its
    end, showing `last_switch`'s reason when it did not take.
12. While switching, every view sees it: the header's flame and heat bar run,
    the model status reads "Switching to <id>…", and the composer is not ready.
13. The form shows every visible field in its group with the control its
    declaration calls for; not-patchable, not-applicable and `known_models`
    rows are read-only as defined; editable `reload_required` rows carry the
    mark.
14. A scoped field shows the running family's section value when there is
    one, and an edit to it is sent under that family's section; otherwise
    under the group.
15. The primary action reads "Apply" for a draft of live fields and "Apply
    and reload <id>" as soon as one changed field is `reload_required`; the
    streaming-reply warning appears only while a reply streams.
16. `PATCH` carries only the changed fields; `200`, `202`, `400`, `409` and
    `501` are each handled as defined, including `persisted`/`path`/`reason`
    and `shadowed_at_restart`.
17. `dev:mock` serves every route above as defined, so the view runs without
    the GPU.
18. `npm test` and `npm run build` pass in `web/`; `cargo test --workspace`
    passes.

## Testing Decisions

- Server: the new routes against the test server the existing
  `tests/config_http.rs` and `tests/model_switch.rs` drive (no GPU), the
  catalog rows with a temporary `download.path` holding complete, partial
  and absent entries; `fields_json` and the route's shared part as a unit test.
- Playground: pure functions first, each a vitest file:
  - draft → `PATCH` body, scoped sections included;
  - draft → primary label;
  - declaration → control;
  - value in force;
  - response → outcome text;
  - `readModel`'s `switching`/`failed` mapping.
  Then a `renderToStaticMarkup` test of the view in its states, as
  `MonitorView.test.tsx` does.
- No headless screenshots: the owner watches the page live on `:5173`.

## Out of Scope

- Downloading a model from the page (an absent model shows the command).
- Editing `switch.known_models`, profiles, or any field not `patchable`.
- ~~Returning a nullable field to unset~~: superseded by the follow-up below,
  where `null` removes a key.
- Arbitration between clients switching the same server (spec
  model-switch/01: no router).

## As built (2026-10-12)

Branch `server-view`, `0a899a2..` (server `d5dee2f`, `5969862`, `1da20a2`,
`466e128`; Playground `1230c01`, `6d8cea6`, `4ce7e04`, `e9749a8`, `d7f5ae1`,
`987112d`, `6aec2c2`, `8cc2771`, `c86e29d`, `c4352ad`).

- Wire shapes settled after the spec was written:
  - both new routes answer `{ "object": "list", "data": [...] }`;
  - a catalog row omits `bytes` / `family` when there is none;
  - `last_switch` is absent before the first switch;
  - `switchable` is false on every row while a switch runs, and true for the
    torn-down model while nothing serves.
- `help --fields --format json` keeps `rule` as a sentence and `families` as
  comma-joined names, and gains `nullable`, `rule_text` and `scoped`. One builder
  (`FieldMeta::declaration`) serves both, and a test holds the OpenAPI schema to
  its keys.
- The start model is remembered by the `Switcher` for the server's life
  (`note_start`). `main` and `apply_config` build the explicit and implicit
  tables in one place (`Switcher::tables`).
- Ratified deviations:
  - The reload button uses the bright core of `--heat` (`.heat-fill`) with dark
    text, because no text colour has enough contrast on the whole gradient.
  - Only editable rows carry the reload mark.
  - An absent known model (not a catalog entry) shows no download command,
    since there is nothing to fetch it from.
- The Playground keeps reading `GET /v1/models` every second while the state is
  `failed` too, until a model serves.
- A draft is dropped when another model id serves.
- Below `sm` the header's views are a native `<select>`, and the wordmark is
  hidden.
- The CORS `Access-Control-Allow-Methods` header gains `PATCH`, so a
  cross-origin `PATCH /v1/config` passes its preflight.

## Follow-up: per-model overrides and the rendered file (2026-10-12)

The owner, trying the view, asked two things:
- where a field both models take is set for one model only;
- a way to see the configuration as the file it amounts to.

As built, the form edits the value in force and cannot create or remove a
family's own value. The server keeps a value for the family not loaded, but it
still reloads the loaded model for it, and a `PATCH` cannot remove anything.

### Decisions

**Server**

- **A reload only for what the loaded model runs with.** A patch needs a
  reload iff, after it, a `reload_required` field resolves differently for the
  running family.
  - A value scoped to the other family, or a group key the running family
    overrides in its own section, changes nothing running. It is a live
    change: applied, written to the file, and used when that family next
    loads.
  - `applied` is `live` for it. A patch that mixes such values with a
    reload-requiring change for the running model reloads as before, and every
    value in it lands.
- **A patched group value yields to the file's own family section.** What
  runs is what a start from the rewritten file resolves, so a group key that
  the running family overrides in its own section stays that section's value.
  A patch's own family value still wins over the file.
- **`GET /v1/config` answers in the file's shape**: each group key is the
  shared value; a section for each family, the running one included, holds
  every value where that family differs from the shared one. The value in
  force is the running family's section, else the group's.
- **`null` removes**, with JSON merge patch semantics.
  - `null` inside a family section removes that family's own value, so the
    group's applies to it again.
  - `null` at a group key removes the key, so the next source down applies (a
    profile, then the hardcoded default).
  - After a removal, the running value is the one a start with the rewritten
    file would resolve. A flag or env var that sets the field still outranks
    it and is named in `shadowed_at_restart`.
  - The config file loses the key, and a family section or group left empty
    goes with it.
  - A removal of a key no source sets is accepted and changes nothing. A body
    of only removals is a patch.
- **The rendered file.** `GET /v1/config?format=yaml` answers `application/yaml`:
  the YAML document `ignis-server config print` writes for the running
  configuration's visible fields, with the same `ignis-model-status` header.
  - `format=json`, or no `format`, is today's answer.
  - Any other value is `400 config_invalid`.

**Playground**

- **Scope selector** at the top of the configuration: "All models", then one
  entry per family. It opens on the running family, which is marked.
  - In a family's scope:
    - each scoped field shows that family's effective value, marked "Own value"
      when the family has its own section entry, else "Shared";
    - an edit writes that family's section;
    - "Use shared value (<v>)" sends `null` there;
    - a field only that family takes is editable, and its edit goes to its
      family's section;
    - a field every family shares (no scope) is editable and marked "Same for
      every model".
  - In "All models":
    - the group values are shown and edited;
    - a scoped field that a family overrides says "Qwen3.8-27B has its own
      value: <v>";
    - a family-only field is listed under its family's name and edits that
      family's section.
- **The dock** labels reloads by what reaches the running model.
  - It reads "Apply and reload <id>" only when a changed or removed
    `reload_required` value resolves differently for the running family.
  - Otherwise it reads "Apply". A value that waits for the other model is
    listed as "Takes effect when <family> loads".
- **"View as config file"** sits beside the configuration's heading. It opens
  a modal holding the YAML from `GET /v1/config?format=yaml`, in the code face,
  with "Copy" and "Download" (`ignis.yaml`).
  - One line under it says hidden fields (the API key, the download token) are
    not in it, and that it is the configuration in force, which includes live
    changes and not the comments of the file on disk.
  - A pending draft is not in it.
  - The modal is portalled, traps focus, closes on Esc, and returns focus to
    its button.
- **Mock.** `dev:mock` serves all of the above:
  - a `null` that removes;
  - a value scoped to the family not loaded, answered `200` live;
  - `?format=yaml`.

### Acceptance (follow-up)

19. A `PATCH` holding only values that change nothing for the running family
    is `200` live, even when they are `reload_required`. It is written to the
    file, and after a switch to that family the load runs with them (driven
    through a loader that applies the config, not only the mock).
20. A `PATCH` mixing such values with a reload-requiring change for the running
    family reloads, and both land.
21. `null` in a family section removes that family's value, so the group's
    applies. `null` at a group key falls back to the profile or the default.
    The file loses the key, and an emptied section or group goes with it. A
    removal no source sets is a no-op `200`. A flag-set field is named in
    `shadowed_at_restart`.
22. `GET /v1/config?format=yaml` answers `application/yaml` equal to what
    `config print` writes for the same configuration, minus hidden fields;
    an unknown `format` is `400`.
23. The scope selector shows "All models" and each family, and opens on the
    running one. Each scope shows and edits the values defined above, with
    "Own value", "Shared" and "Same for every model" marked. "Use shared value"
    sends `null` in that family's section.
24. A family-only field is editable while the other family runs, and its edit
    goes to its family's section.
25. The primary action's label follows what reaches the running model.
    "Takes effect when <family> loads" lists the values that wait.
26. "View as config file" opens a modal with the rendered YAML, "Copy" and
    "Download". It is keyboard-operable (focus trap, Esc, focus return).
27. `dev:mock` serves every behaviour above. `npm test`, `npm run build` and
    `cargo test --workspace` pass.

Known limit: `GET /v1/config` shows a family's section only where its value
differs from the group's. An own value equal to the shared one therefore reads
as "Shared" and cannot be removed from the page until it differs. Lifted by
the origins below (follow-up 2).

## Follow-up 2: where each value comes from (2026-10-12)

The second review found that a family's section in `GET /v1/config` mixes two
kinds of value:
- **The operator's own**, written in a family section or in a live family
  value. A shared edit does not reach it, and a `null` removes it.
- **A family-scoped value from a source that is not the operator's**:
  - the profile's per-family numbers, e.g. `rtx5090`'s `vram.headroom_bytes`
    1200M for the 27B;
  - a `--<family>-…` flag or `IGNIS_<FAMILY>_…` env var.

  A shared edit does reach the profile's value, because a profile is a
  default, never an override (ADR 0046). A `null` cannot remove any of them.

The page could not tell the two apart, so it labelled a reload "Apply" and
offered a "Use shared value" that changed nothing. The owner chose to show each
value's origin. The precedence is unchanged.

### Decisions

**Server**

- **`GET /v1/config/origins`** answers `{ "object": "list", "data": [row, ...] }`.
  - There is one row per visible field and per scope: `null` for the shared
    value, then each family that takes the field.
  - A row is `{ key: "<group>.<name>", scope: null | "<family>", source,
    scoped, spelling? }`:
    - `source` is the source of the value that resolves for that scope:
      `default`, `profile`, `file`, `env`, `flag` or `live` (a `PATCH`);
    - `scoped` is true when that value was set for this family specifically
      (a family section, a family flag or env var, a profile's family value,
      a live family value);
    - `spelling` says where, as the startup source log spells it: the flag,
      the env var, the file key and path, or the profile's name.
  - It answers like `GET /v1/config`: during a switch, with the API key, and
    `501` with no running configuration. Hidden fields are absent.
- **A removal goes where a value goes.** A `null` for a family-only field at a
  group key is homed to that family's section, as a value is. What runs after
  any removal is what a start from the rewritten file resolves; a test drives
  the family-only case both ways.
- **`shadowed_at_restart` names a family-scoped flag or env var** that
  outranks a patched group value or family value at the next start.
- **A YAML test on a fitted configuration.** `GET /v1/config?format=yaml` on a
  running, family-fitted configuration is held equal to the file's shape that
  `config print` writes for the same sources.
- **A whole group or section cannot be removed.** `{group: null}` and
  `{group: {family: null}}` are `400`: a removal names its keys.

**Playground**

- **Labels from the origins.** A family's value:
  - with `scoped` and `source` `file` or `live` is "Own value", offering "Use
    shared value";
  - with `scoped` and `source` `profile` reads "From profile <name>", and
    has no remove;
  - with `scoped` and `source` `flag` or `env` reads "Set by <spelling>".
    An edit applies until restart, and the answer's `shadowed_at_restart`
    says so. It has no remove.
  - without `scoped` is "Shared".

  An own value equal to the shared one is now shown as own.
- **"All models" notes** what a shared edit does to each family with a scoped
  value:
  - "<family> has its own value: <v>" when the shared edit does not reach it;
  - "Replaces the profile's <v> for <family>" when it does;
  - "<family> is set by <spelling>" for a flag or env var.
- **The reload prediction** follows the server's rule. A group edit reaches a
  family unless that family's value is `scoped` with `source` `file` or
  `live`.
- **Families** are derived from the declarations (`families`) and the catalog,
  never listed by hand. The modal's note says "Hidden fields are not in it"
  without naming them (ADR 0046).
- **The modal** focuses its close button, or the dialog itself, on open, never
  a disabled button.
- **Mock.** `dev:mock` serves `GET /v1/config/origins`, including a
  profile-sourced family value for the 27B. It refuses a group-key value for
  a field only the other family takes, as the server does.

### Acceptance (follow-up 2)

28. `GET /v1/config/origins` lists every visible field once per scope with
    `source`, `scoped` and `spelling` as defined. Rows are right for a
    default, a profile family value, a file group and a file family value,
    a flag, a family flag, an env var and a live change. It answers during a
    switch, behind the key, and `501` without a configuration.
29. A `null` for a family-only field at a group key removes it from its
    family's section. Afterwards the running value equals a start from the
    rewritten file. Both orders are tested: set then remove, file then remove.
30. `shadowed_at_restart` names a family-scoped flag or env var that outranks
    the patched value at restart.
31. The fitted-config YAML test holds. `{group: null}` and
    `{group: {family: null}}` are `400`.
32. The page labels each family value by its origin and offers "Use shared
    value" only for `file`/`live` own values. Each "All models" note and the
    reload label follow the server's rule. With the `rtx5090` profile, a
    shared `vram.headroom_bytes` edit on the 27B reads "Apply and reload".
33. No hand-kept family list or hidden-field names in `web/`. The modal
    focuses a focusable control on open. `dev:mock` serves the origins.
    `npm test`, `npm run build` and `cargo test --workspace` pass.
