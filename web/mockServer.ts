import type { LastSwitch } from "./src/api/model.ts";
import { CATALOG, CONFIG, FIELDS } from "./src/server/fixture.ts";
import type { CatalogRow, ConfigDocument, FieldDecl } from "./src/server/routes.ts";

// A fake of the routes the Server view talks to, for `npm run dev:mock`
// (spec playground/05 §Mock): GET /v1/config/fields, GET and PATCH
// /v1/config, GET /v1/models/catalog, POST /v1/models/switch and the model
// state GET /v1/models reports. Development only — never part of the build.
//
// It starts as the 27B under the `rtx5090` profile (src/server/fixture.ts). A
// switch, or a PATCH naming a field that needs a reload, reads `switching`
// for three seconds, then `last_switch` says how it ended; meanwhile a PATCH
// gets the gate's 503 and a second switch a 409, as the server answers them.
//
// Triggers: a value "/error" is refused with a 400 naming that field's key;
// a switch to an id containing "fail" reads `failed` for a second (the old
// model being loaded again) and ends `restored` with a reason; one to an id
// containing "dead" ends `failed` for good, the old model gone and nothing
// serving (no `loaded` row, the torn-down model `switchable` again) until a
// switch to a model that loads; a reload whose
// patch holds a value "/fail" ends the same way, the configuration unchanged.
// A patch of `model.thinking_budget` is shadowed at restart, as if the mock
// had started with IGNIS_MODEL_THINKING_BUDGET set.
//
// Per-model values (spec playground/05 §Follow-up): a patch reloads only if a
// `reload_required` field resolves differently for the loaded family, so a
// value in the other family's section, or a group key the loaded family
// overrides, is a `200` live. `null` removes: a family's own value (the group's
// applies again) or a group key (back to the field's default). `GET /v1/config`
// shows a family's value only where it differs from the group's, as the
// server does. `?format=yaml` renders the configuration as a YAML document.

const SWITCH_MS = 3_000;
const RESTORE_MS = 1_000;
const CONFIG_PATH = "ignis.yaml";
const SCOPES = ["qwen38", "qwen38flashnext"];
const COUNTS = new Set(["tokens", "seconds", "slots", "percent", "lanes", "MiB", "rows", "tokens|off"]);

/** One value of a patch; `remove` is its `null`. */
type Entry = { decl: FieldDecl; scope: string | null; value: unknown; remove?: true };
type Running = { from: string; to: string; until: number; failure?: string; dead?: string; patch?: Entry[] };
export type MockAnswer = { status: number; body: unknown };

const isObject = (value: unknown): value is Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value);

const refuse = (status: number, type: string, code: string, message: string): MockAnswer => ({ status, body: { error: { message, type, code } } });

const keyOf = (entry: Entry) => (entry.scope === null ? entry.decl.file_key : `${entry.decl.group}.${entry.scope}.${entry.decl.name}`);

/** A value as the config file writes it back: a bool, a count, a choice in its own spelling, else the text. */
function canonical(decl: FieldDecl, value: unknown): unknown {
  if (typeof value !== "string") return value;
  const text = value.trim();
  if (decl.kind === "bool") {
    const lower = text.toLowerCase();
    if (["true", "on", "1"].includes(lower)) return true;
    if (["false", "off", "0"].includes(lower)) return false;
  }
  if (decl.rule && "one_of" in decl.rule) return decl.rule.one_of.find((name) => name.toLowerCase() === text.toLowerCase()) ?? text;
  if (COUNTS.has(decl.kind) && /^\d+$/.test(text)) return Number(text);
  return text;
}

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

const section = (doc: ConfigDocument, group: string, scope: string) => {
  const found = doc[group]?.[scope];
  return isObject(found) ? found : undefined;
};

/** The value a family runs a field with: its own section's, else the group's. */
function resolved(doc: ConfigDocument, decl: FieldDecl, family: string): unknown {
  const own = section(doc, decl.group, family);
  return own && decl.name in own ? own[decl.name] : doc[decl.group]?.[decl.name];
}

const plain = (text: string) => /^[A-Za-z0-9_./~-][A-Za-z0-9_./:~-]*$/.test(text) && !/^(true|false|null|yes|no|on|off|~|[-+]?[0-9.]+)$/i.test(text);

const scalar = (value: unknown) => (typeof value === "string" ? (plain(value) ? value : JSON.stringify(value)) : JSON.stringify(value ?? null));

/** A configuration document as YAML, two spaces to a level. */
export function yamlOf(value: Record<string, unknown>, depth = 0): string {
  const pad = "  ".repeat(depth);
  return Object.entries(value)
    .map(([key, entry]) => {
      if (!isObject(entry)) return `${pad}${key}: ${scalar(entry)}\n`;
      return Object.keys(entry).length === 0 ? `${pad}${key}: {}\n` : `${pad}${key}:\n${yamlOf(entry, depth + 1)}`;
    })
    .join("");
}

export function createServerSim() {
  const doc = structuredClone(CONFIG) as ConfigDocument;
  let loaded = "qwen3.8-27b";
  let running: Running | null = null;
  // A target that did not load: `failed` while the old model is loaded again.
  let restoring: { reason: string; until: number; last: LastSwitch } | null = null;
  let last: LastSwitch | undefined;
  // A switch that left nothing serving: `loaded` is then only the model torn down.
  let dead: { reason: string } | null = null;

  const familyOf = (id: string) => CATALOG.find((row) => row.id === id)?.family ?? "qwen38";

  function advance(now: number) {
    if (running && now >= running.until) {
      const ended = running;
      running = null;
      if (ended.dead) {
        dead = { reason: ended.dead };
        last = { from: ended.from, to: ended.to, outcome: "failed", reason: ended.dead };
      } else if (ended.failure) {
        restoring = { reason: ended.failure, until: ended.until + RESTORE_MS, last: { from: ended.from, to: ended.to, outcome: "restored", reason: ended.failure } };
      } else {
        loaded = ended.to;
        dead = null;
        if (ended.patch) apply(ended.patch);
        last = { from: ended.from, to: ended.to, outcome: "switched" };
      }
    }
    if (restoring && now >= restoring.until) {
      last = restoring.last;
      restoring = null;
    }
  }

  const status = () => (running ? "switching" : restoring || dead ? "failed" : "serving");

  function apply(entries: Entry[], into: ConfigDocument = doc) {
    for (const { decl, scope, value, remove } of entries) {
      const group = (into[decl.group] ??= {});
      if (remove) {
        if (scope === null) group[decl.name] = decl.default;
        else if (section(into, decl.group, scope)) delete (group[scope] as Record<string, unknown>)[decl.name];
      } else {
        const at = scope === null ? group : ((group[scope] ??= {}) as Record<string, unknown>);
        if (decl.file_key === "switch.known_models" && isObject(value)) at[decl.name] = { ...(at[decl.name] as object), ...value };
        else at[decl.name] = canonical(decl, value);
      }
      // The server shows a family's value only where it differs from the group's; an emptied section goes.
      const own = scope === null ? undefined : section(into, decl.group, scope);
      if (own) {
        if (decl.scoped && decl.name in own && sameValue(own[decl.name], group[decl.name])) delete own[decl.name];
        if (Object.keys(own).length === 0) delete group[scope!];
      }
    }
  }

  /** The changes a PATCH body names, or the 400 the server answers it with. */
  function readPatch(body: unknown): Entry[] | MockAnswer {
    const invalid = (message: string) => refuse(400, "invalid_request_error", "config_invalid", message);
    if (!isObject(body)) return invalid("the body must be a config document: `{ <group>: { <field>: value } }`");
    const find = (group: string, name: string) => FIELDS.find((decl) => decl.group === group && decl.name === name);
    const entries: Entry[] = [];
    for (const [group, fields] of Object.entries(body)) {
      if (!isObject(fields)) return invalid(`\`${group}\` must be a map of its fields`);
      for (const [key, value] of Object.entries(fields)) {
        if (SCOPES.includes(key) && isObject(value)) {
          for (const [name, scoped] of Object.entries(value)) {
            const decl = find(group, name);
            if (!decl) return invalid(`unknown field \`${group}.${key}.${name}\``);
            if (!decl.scoped && !decl.families?.includes(key)) return invalid(`\`${group}.${key}.${name}\`: \`${group}.${name}\` has no family sections`);
            entries.push({ decl, scope: key, value: scoped, ...(scoped === null ? { remove: true as const } : {}) });
          }
          continue;
        }
        const decl = find(group, key);
        if (!decl) return invalid(`unknown field \`${group}.${key}\``);
        // A field only one family takes lives in that family's section, as the server writes it.
        entries.push({ decl, scope: decl.families?.length === 1 ? decl.families[0] : null, value, ...(value === null ? { remove: true as const } : {}) });
      }
    }
    if (entries.length === 0) return invalid("the patch names no field");
    const fixed = entries.filter((entry) => !entry.decl.patchable).map((entry) => `\`${keyOf(entry)}\``);
    if (fixed.length > 0) {
      return refuse(400, "invalid_request_error", "config_field_not_patchable", `${fixed.join(", ")} cannot be changed while the server runs (not patchable); nothing in the patch was applied`);
    }
    const refused = entries.find((entry) => entry.value === "/error");
    if (refused) return invalid(`\`${keyOf(refused)}\` expects a ${refused.decl.kind} value, got \`/error\` (the mock refuses "/error"); nothing in the patch was applied`);
    return entries;
  }

  return {
    fields: () => ({ object: "list" as const, data: FIELDS }),

    /** GET /v1/config: the running configuration, and the model status its header carries. */
    config(now: number): { body: ConfigDocument; status: string } {
      advance(now);
      return { body: doc, status: status() };
    },

    /** GET /v1/config?format=...: `json` (or none) is the document, `yaml` the file it amounts to, any other format a 400. */
    configAs(format: string | null, now: number): { modelStatus: string; http: number; type: string; text: string } {
      const { body, status } = this.config(now);
      if (format === null || format === "json") return { modelStatus: status, http: 200, type: "application/json", text: JSON.stringify(body) };
      if (format === "yaml") return { modelStatus: status, http: 200, type: "application/yaml", text: yamlOf(body) };
      const refused = refuse(400, "invalid_request_error", "config_invalid", `unknown format \`${format}\`: use \`json\` or \`yaml\``);
      return { modelStatus: status, http: 400, type: "application/json", text: JSON.stringify(refused.body) };
    },

    models(now: number): Record<string, unknown> {
      advance(now);
      const family = familyOf(loaded);
      const context = Number((doc.model?.[family] as Record<string, unknown> | undefined)?.max_context ?? doc.model?.max_context);
      return {
        object: "list",
        data: dead ? [] : [{ id: loaded, object: "model", owned_by: "ignis", max_model_len: context }],
        status: status(),
        ...(running ? { switching: { from: running.from, to: running.to } } : {}),
        ...(restoring || dead ? { reason: (restoring ?? dead)!.reason } : {}),
        ...(last ? { last_switch: last } : {}),
      };
    },

    catalog(now: number): { object: "list"; data: CatalogRow[] } {
      advance(now);
      return {
        object: "list",
        data: CATALOG.map((row) => {
          const isLoaded = !dead && row.id === loaded;
          // A switch can begin while serving or after a failed one, never during one.
          return { ...row, loaded: isLoaded, switchable: !running && !isLoaded && row.on_disk === "complete" };
        }),
      };
    },

    patch(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return refuse(503, "server_error", "model_switching", `the server is switching models from ${running.from} to ${running.to}; retry once GET /v1/models reports \`serving\``);
      }
      const unready = restoring ?? dead;
      if (unready) return refuse(503, "server_error", "server_not_ready", `a model switch failed and no model is serving (${unready.reason}); retry shortly`);
      const entries = readPatch(body);
      if (!Array.isArray(entries)) return entries;
      const shadowed = entries.filter((entry) => keyOf(entry) === "model.thinking_budget").map((entry) => `${keyOf(entry)} (set by IGNIS_MODEL_THINKING_BUDGET)`);
      const shadowing = shadowed.length > 0 ? { shadowed_at_restart: shadowed } : {};
      // A reload only for what the loaded model runs with: apply the patch to a copy and compare what the loaded family resolves.
      const family = familyOf(loaded);
      const after = structuredClone(doc) as ConfigDocument;
      apply(entries, after);
      const reloads = entries.some((entry) => entry.decl.reload_required && !sameValue(resolved(doc, entry.decl, family), resolved(after, entry.decl, family)));
      if (reloads) {
        const failure = entries.some((entry) => entry.value === "/fail")
          ? `the reload of ${loaded} failed: the mock refuses a value "/fail"; the previous configuration was loaded again`
          : undefined;
        running = { from: loaded, to: loaded, until: now + SWITCH_MS, patch: entries, failure };
        return {
          status: 202,
          body: {
            applied: "reload",
            persisted: false,
            path: CONFIG_PATH,
            reason: "written once the reloaded model serves; not if the reload fails",
            ...shadowing,
            switching: { from: loaded, to: loaded },
          },
        };
      }
      apply(entries);
      return { status: 200, body: { applied: "live", persisted: true, path: CONFIG_PATH, ...shadowing, config: doc } };
    },

    /** POST /v1/models/switch by id alone (an `artifact` beside it is ignored here). */
    switchTo(body: unknown, now: number): MockAnswer {
      advance(now);
      if (running) {
        return refuse(409, "invalid_request_error", "switch_in_progress", `a switch from ${running.from} to ${running.to} is already under way; switches are not queued`);
      }
      const id = isObject(body) && typeof body.model === "string" ? body.model.trim() : "";
      if (!id) return refuse(400, "invalid_request_error", "invalid_request_error", "`model` must name the id to serve it under");
      const row = CATALOG.find((r) => r.id === id);
      if (!row) {
        return refuse(404, "model_not_found", "model_not_found", `unknown model: ${id} is neither in switch.known_models nor a catalog entry; GET /v1/models/catalog lists the models this server can switch to`);
      }
      if (row.on_disk !== "complete") {
        return refuse(
          404,
          "model_not_found",
          "model_not_found",
          row.source === "known"
            ? `model ${id} is in switch.known_models but its artifact is not a file; fix the path or put the file there`
            : `model ${id} is in the catalog but not on disk (download.path); fetch it with \`ignis-server model download ${id}\`, a switch never starts a download`,
        );
      }
      const failure = id.includes("fail") ? `loading ${id} failed: the mock refuses every id containing "fail"; ${loaded} was loaded again` : undefined;
      const gone = id.includes("dead") ? `loading ${id} failed and ${loaded} could not be loaded again: the mock leaves nothing serving for every id containing "dead"` : undefined;
      running = { from: loaded, to: id, until: now + SWITCH_MS, failure: gone ? undefined : failure, dead: gone };
      restoring = null;
      return { status: 202, body: { status: "switching", from: loaded, to: id } };
    },
  };
}
