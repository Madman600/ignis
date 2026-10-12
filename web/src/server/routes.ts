import { authHeaders, keyRequired } from "../api/auth.ts";

// The routes the Server view talks to (spec playground/05, GitHub #316),
// beside GET /v1/models (api/model.ts): what each config field is, the
// running configuration and a change to it, the models a switch can go to,
// and the switch itself, by id. Each answer is read into a value or the
// server's own message; a 401 shows the key prompt, as everywhere else.

/** The rule a value must meet, as `GET /v1/config/fields` structures it; `rule_text` says it in words. */
export type Rule = null | { min: number | null; max: number | null } | { multiple_of: number } | { one_of: string[] };

/** One visible field's declaration (ADR 0046's `FieldMeta`). */
export type FieldDecl = {
  group: string;
  name: string;
  /** `<group>.<name>`, as the config file nests it and a `400` names it. */
  file_key: string;
  flag: string;
  env: string;
  /** `bool`, `tokens`, `bytes|tokens`, … */
  kind: string;
  /** The field may be unset: `null`, shown as "auto". */
  nullable: boolean;
  default: unknown;
  rule: Rule;
  /** The rule as `help --fields` prints it; empty with none. */
  rule_text: string;
  /** `null` for every family, else the scope names (`qwen38`, `qwen38flashnext`) that take it. */
  families: string[] | null;
  /** It may be given per family, in a `<group>.<family>` section. */
  scoped: boolean;
  patchable: boolean;
  reload_required: boolean;
  description: string;
};

/** A config document in the file's own shape: `<group>: { <field>: value, <family>: { <field>: value } }`. */
export type ConfigDocument = Record<string, Record<string, unknown>>;

/** One model the server can switch to or fetch (`GET /v1/models/catalog`). No path: the page switches by id. */
export type CatalogRow = {
  id: string;
  source: "built-in" | "operator" | "known";
  on_disk: "complete" | "partial" | "absent";
  /** An entry's total, or a known model's artifact on disk; absent otherwise. */
  bytes?: number;
  /** The scope name the artifact's header names, when it is on disk; the running family for the loaded row. */
  family?: string;
  loaded: boolean;
  switchable: boolean;
};

/** A `PATCH /v1/config` that was applied (`200`) or began a reload (`202`). */
export type PatchApplied = {
  applied: "live" | "reload";
  persisted: boolean;
  path?: string;
  reason?: string;
  /** `<key> (set by <spelling>)`: a flag or an env var outranks the file at the next start. */
  shadowed_at_restart?: string[];
  switching?: { from: string; to: string };
  /** The configuration now running, after a live change. */
  config?: ConfigDocument;
};

/** A refusal: the HTTP status, and the server's own `error.message` and `error.code`. */
export type Refusal = { status: number; message: string; code?: string };

export type Answer<T> = { ok: true; status: number; value: T } | ({ ok: false } & Refusal);

/** The server's `{"error": {message, code}}`, or the raw body, or the status alone. */
export function refusalOf(status: number, body: string): Refusal {
  try {
    const error = (JSON.parse(body) as { error?: { message?: unknown; code?: unknown } }).error;
    if (typeof error?.message === "string" && error.message) {
      return { status, message: error.message, ...(typeof error.code === "string" ? { code: error.code } : {}) };
    }
  } catch {
    // Not JSON: the raw body below.
  }
  const text = body.trim();
  return { status, message: text || `${status}` };
}

async function callText<T>(path: string, init: RequestInit, read: (text: string) => T, fetcher: typeof fetch): Promise<Answer<T>> {
  try {
    const res = await fetcher(path, { ...init, headers: { ...authHeaders(), ...(init.headers as Record<string, string> | undefined) } });
    if (res.status === 401) keyRequired();
    const text = await res.text();
    if (!res.ok) return { ok: false, ...refusalOf(res.status, text) };
    return { ok: true, status: res.status, value: read(text) };
  } catch (err) {
    return { ok: false, status: 0, message: `Could not reach ignis: ${String(err)}` };
  }
}

const call = <T,>(path: string, init: RequestInit, read: (body: unknown) => T, fetcher: typeof fetch) =>
  callText(path, init, (text) => read(JSON.parse(text)), fetcher);

/** The list envelope both new routes answer with, as `GET /v1/models` does. */
const listData = <T,>(body: unknown) => ((body as { data?: T[] }).data ?? []) as T[];

export const readFields = (fetcher: typeof fetch = fetch) => call("/v1/config/fields", {}, listData<FieldDecl>, fetcher);

export const readConfig = (fetcher: typeof fetch = fetch) => call("/v1/config", {}, (body) => body as ConfigDocument, fetcher);

/** The configuration in force as the file it amounts to (`?format=yaml`), as text. */
export const readConfigFile = (fetcher: typeof fetch = fetch) => callText("/v1/config?format=yaml", {}, (text) => text, fetcher);

export const readCatalog = (fetcher: typeof fetch = fetch) => call("/v1/models/catalog", {}, listData<CatalogRow>, fetcher);

/** Switch to `id` by id alone: the server resolves it to a known model or a catalog entry on disk. */
export const switchTo = (id: string, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/models/switch",
    { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: id }) },
    (body) => body as { from: string; to: string },
    fetcher,
  );

export const patchConfig = (body: object, fetcher: typeof fetch = fetch) =>
  call(
    "/v1/config",
    { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) },
    (answer) => answer as PatchApplied,
    fetcher,
  );
