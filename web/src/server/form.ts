import type { ConfigDocument, FieldDecl } from "./routes.ts";

// The Server view's form (spec playground/05 §Configuration, §Applying), as
// pure functions over the declarations and the running configuration: which
// control a field gets, which value is in force and where it was read from,
// and the draft of changes that becomes one PATCH /v1/config. The server is
// the judge of a value: nothing here checks one, a row sends what was typed.

/** A family's scope name as people say it. */
export function familyName(scope: string): string {
  return scope === "qwen38" ? "Qwen3.8-27B" : scope === "qwen38flashnext" ? "Qwen3.8-Flash-Next" : scope;
}

const GROUP_TITLES: Record<string, string> = {
  server: "Server",
  model: "Model",
  vram: "VRAM",
  reuse: "Prompt reuse",
  switch: "Model switch",
  spec: "Speculative decoding",
  vision: "Vision",
  media: "Media",
  download: "Download",
  ngram: "N-gram table",
  kv_disk: "KV-disk",
};

/** A group's heading; a group this page does not know yet is shown by its key. */
export function groupTitle(group: string): string {
  return GROUP_TITLES[group] ?? group;
}

export type Control =
  | { kind: "switch" }
  | { kind: "segmented"; options: string[] }
  | { kind: "select"; options: string[] }
  /** `hint`: the kind and the rule; `placeholder`: "auto" for an unset value. */
  | { kind: "text"; hint: string; placeholder: string };

/** The control a declaration calls for, for a field whose value in force is `value`. */
export function controlFor(decl: FieldDecl, value: unknown): Control {
  if (decl.kind === "bool") return { kind: "switch" };
  if (decl.rule && "one_of" in decl.rule) {
    // The server lists canonical forms; a value in force it spells otherwise (an alias set from the file) is listed too, to show as chosen.
    const named = decl.rule.one_of;
    const shown = typeof value === "string" ? value : null;
    const options = shown !== null && !named.some((option) => option.toLowerCase() === shown.toLowerCase()) ? [...named, shown] : named;
    return options.length <= 4 ? { kind: "segmented", options } : { kind: "select", options };
  }
  const hint = decl.rule_text ? `${decl.kind}, ${decl.rule_text}` : decl.kind;
  return { kind: "text", hint, placeholder: value === null || value === undefined ? "auto" : textOf(value) };
}

/**
 * How a row shows its field: with a control, or read-only because it is set
 * at start (not patchable), because the running family does not take it, or
 * because it is `switch.known_models`, which a patch only adds to.
 */
export type RowMode = { kind: "edit" } | { kind: "fixed" } | { kind: "unused"; family: string } | { kind: "list" };

export function rowMode(decl: FieldDecl, family: string | null): RowMode {
  if (family !== null && decl.families !== null && !decl.families.includes(family)) return { kind: "unused", family };
  if (!decl.patchable) return { kind: "fixed" };
  if (decl.file_key === "switch.known_models") return { kind: "list" };
  return { kind: "edit" };
}

/** A value in force and where it was read from: a family's section, or the group (`null`). */
export type InForce = { value: unknown; scope: string | null };

const section = (doc: ConfigDocument, group: string, scope: string) => {
  const found = doc[group]?.[scope];
  return found !== null && typeof found === "object" ? (found as Record<string, unknown>) : undefined;
};

/**
 * The value a field runs with: the running family's own when its section
 * has one — a scoped field's override, or a field only that family takes,
 * which is never in the group — else the group's. With no family known, a
 * field only one family takes is still read from that family's section.
 */
export function valueInForce(decl: FieldDecl, doc: ConfigDocument | null, family: string | null): InForce {
  if (!doc) return { value: undefined, scope: null };
  const scope = family ?? (decl.families?.length === 1 ? decl.families[0] : null);
  const own = scope !== null ? section(doc, decl.group, scope) : undefined;
  if (own && decl.name in own) return { value: own[decl.name], scope };
  return { value: doc[decl.group]?.[decl.name], scope: null };
}

/** A value as a text field shows it: unset is empty. */
export function textOf(value: unknown): string {
  if (value === null || value === undefined) return "";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
}

/** A value shown with no control. */
export function shownOf(value: unknown): string {
  if (value === null || value === undefined) return "Not set";
  if (typeof value === "boolean") return value ? "On" : "Off";
  return textOf(value);
}

/** One edited field: where it goes, what was typed or chosen, and whether it needs a reload. */
export type DraftEntry = { group: string; name: string; scope: string | null; value: string | boolean; reload: boolean };

/** The edits, by file key. A cleared text field stays here so the field can show it empty. */
export type Draft = Record<string, DraftEntry>;

const same = (decl: FieldDecl, value: string | boolean, current: unknown) => {
  if (typeof value === "boolean") return value === current;
  const typed = value.trim();
  const shown = textOf(current);
  return decl.rule && "one_of" in decl.rule ? typed.toLowerCase() === shown.toLowerCase() : typed === shown;
};

/**
 * The draft with `decl` set to `value`. The edit goes where the value in
 * force came from (the family's section or the group); a value equal to the
 * one in force drops the entry, so the draft holds only real changes.
 */
export function editDraft(draft: Draft, decl: FieldDecl, inForce: InForce, value: string | boolean): Draft {
  const { [decl.file_key]: _dropped, ...rest } = draft;
  if (same(decl, value, inForce.value)) return rest;
  return { ...rest, [decl.file_key]: { group: decl.group, name: decl.name, scope: inForce.scope, value, reload: decl.reload_required } };
}

/**
 * The entries a PATCH carries: every one but a cleared text field — PATCH
 * reads an unset value as no change, so there is no way to send one (spec
 * playground/05, Out of Scope).
 */
export function changes(draft: Draft): Draft {
  return Object.fromEntries(Object.entries(draft).filter(([, entry]) => typeof entry.value === "boolean" || entry.value.trim() !== ""));
}

/** The PATCH /v1/config body: only the changed fields, nested as the config file nests them. */
export function patchBody(draft: Draft): Record<string, Record<string, unknown>> {
  const body: Record<string, Record<string, unknown>> = {};
  for (const entry of Object.values(changes(draft))) {
    const group = (body[entry.group] ??= {});
    const at = entry.scope === null ? group : ((group[entry.scope] ??= {}) as Record<string, unknown>);
    at[entry.name] = typeof entry.value === "string" ? entry.value.trim() : entry.value;
  }
  return body;
}

/** Whether applying the draft reloads the model: any changed field needs it, and then the whole patch goes through the reload. */
export function needsReload(draft: Draft): boolean {
  return Object.values(changes(draft)).some((entry) => entry.reload);
}

/** The primary action's label, which says what will happen before the click. */
export function primaryLabel(draft: Draft, id: string): string {
  return needsReload(draft) ? `Apply and reload ${id}` : "Apply";
}

const escaped = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/**
 * The file keys of the fields a refusal names — as `<group>.<name>` or in a
 * family section, `<group>.<family>.<name>` — never a key that only begins
 * like a longer one (`server.metrics` in `server.metrics_bind`).
 */
export function namedKeys(message: string, fields: FieldDecl[]): Set<string> {
  const named = new Set<string>();
  for (const decl of fields) {
    const key = new RegExp(`(?<![\\w.])${escaped(decl.group)}\\.(?:\\w+\\.)?${escaped(decl.name)}(?!\\w)`);
    if (key.test(message)) named.add(decl.file_key);
  }
  return named;
}

/** A description's first sentence, and the rest (empty when there is none). */
export function firstSentence(text: string): [string, string] {
  const split = /^(.+?\.)\s+(?=[A-Z`])(.*)$/s.exec(text);
  return split ? [split[1], split[2]] : [text, ""];
}

/** `switch.drain_timeout` in force: the seconds a switch or a reload gives a running reply. */
export function drainTimeout(doc: ConfigDocument | null): number | null {
  const raw = doc?.switch?.drain_timeout;
  if (raw === null || raw === undefined) return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}
