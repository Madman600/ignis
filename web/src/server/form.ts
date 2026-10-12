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

/** A value in force and where an edit of it goes: a family's section, or the group (`null`). */
export type InForce = { value: unknown; scope: string | null };

/** The families this page knows by name; others come from the declarations. */
const FAMILIES = ["qwen38", "qwen38flashnext"];

/** Every family a scope selector offers: the known ones, any a declaration names, and the running one. */
export function familiesOf(fields: FieldDecl[], running: string | null): string[] {
  const named = fields.flatMap((decl) => decl.families ?? []);
  return [...new Set([...FAMILIES, ...named, ...(running ? [running] : [])])];
}

const section = (doc: ConfigDocument, group: string, scope: string) => {
  const found = doc[group]?.[scope];
  return found !== null && typeof found === "object" ? (found as Record<string, unknown>) : undefined;
};

/** The one family a field is only taken by; `null` for a field every family takes. */
const onlyFamily = (decl: FieldDecl) => (decl.families?.length === 1 ? decl.families[0] : null);

/** The value a family runs a field with: its own section's when it has one, else the group's. */
function resolved(doc: ConfigDocument, decl: FieldDecl, family: string): unknown {
  const own = section(doc, decl.group, family);
  return own && decl.name in own ? own[decl.name] : doc[decl.group]?.[decl.name];
}

/**
 * What a row shows in a scope (`null` is "All models"), spec playground/05
 * §Follow-up. `inForce` is the value the control starts from and the section
 * an edit goes to; `shared` is the group's value, which "Use shared value"
 * falls back to; `note` says what the value is: the family's own or the
 * shared one, the same for every model, or who overrides it.
 */
export type RowView = {
  mode: RowMode;
  inForce: InForce;
  shared: unknown;
  note:
    | { kind: "scoped"; own: boolean }
    | { kind: "every" }
    | { kind: "only"; family: string }
    | { kind: "overridden"; by: { family: string; value: unknown }[] }
    | null;
};

export function rowView(decl: FieldDecl, doc: ConfigDocument | null, scope: string | null): RowView {
  const mode = rowMode(decl, scope);
  const shared = doc?.[decl.group]?.[decl.name];
  const only = onlyFamily(decl);
  if (only !== null) {
    // Only ever in its family's section, whichever scope shows it.
    const own = doc ? section(doc, decl.group, only) : undefined;
    return { mode, shared, inForce: { value: own?.[decl.name], scope: only }, note: scope === null ? { kind: "only", family: only } : null };
  }
  if (!decl.scoped) {
    return { mode, shared, inForce: { value: shared, scope: null }, note: scope === null || !decl.patchable ? null : { kind: "every" } };
  }
  if (scope === null) {
    const by = (doc ? Object.keys(doc[decl.group] ?? {}) : [])
      .map((family) => ({ family, own: section(doc!, decl.group, family) }))
      .filter(({ own }) => own !== undefined && decl.name in own)
      .map(({ family, own }) => ({ family, value: own![decl.name] }));
    return { mode, shared, inForce: { value: shared, scope: null }, note: by.length > 0 ? { kind: "overridden", by } : null };
  }
  const own = doc ? section(doc, decl.group, scope) : undefined;
  const has = own !== undefined && decl.name in own;
  return { mode, shared, inForce: { value: has ? own[decl.name] : shared, scope }, note: { kind: "scoped", own: has } };
}

/** A value as a note quotes it: unset is "auto". */
export function labelOf(value: unknown): string {
  return value === null || value === undefined ? "auto" : shownOf(value);
}

/** The lines a row says about its value, given its pending edit (`entry`) in this scope, if any. */
export function noteLines(view: RowView, entry: DraftEntry | undefined): string[] {
  const note = view.note;
  if (note === null) return [];
  switch (note.kind) {
    case "scoped":
      if (entry?.removal) return ["Back to the shared value once applied"];
      return [note.own || entry ? "Own value" : "Shared"];
    case "every":
      return ["Same for every model"];
    case "only":
      return [`${familyName(note.family)} only`];
    case "overridden":
      return note.by.map(({ family, value }) => `${familyName(family)} has its own value: ${labelOf(value)}`);
  }
}

/** Whether "Use shared value" is on offer: the family has its own value and none is on its way out. */
export function canUseShared(view: RowView, entry: DraftEntry | undefined): boolean {
  return view.note?.kind === "scoped" && view.note.own && !entry?.removal;
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

/**
 * One edited field: where it goes (a family's section, or the group), what
 * was typed or chosen, whether it needs a reload, and whether it removes the
 * family's own value instead (`null` in the patch).
 */
export type DraftEntry = { group: string; name: string; scope: string | null; value: string | boolean; reload: boolean; removal?: true };

/** The edits, by `draftKey`. A cleared text field stays here so the field can show it empty. */
export type Draft = Record<string, DraftEntry>;

/** A field's key in the draft: the group's and a family's section hold separate edits of one field. */
export const draftKey = (group: string, name: string, scope: string | null) => (scope === null ? `${group}.${name}` : `${group}.${scope}.${name}`);

const same = (decl: FieldDecl, value: string | boolean, current: unknown) => {
  if (typeof value === "boolean") return value === current;
  const typed = value.trim();
  const shown = textOf(current);
  return decl.rule && "one_of" in decl.rule ? typed.toLowerCase() === shown.toLowerCase() : typed === shown;
};

/**
 * The draft with `decl` set to `value`. The edit goes where `inForce` says
 * (the family's section or the group); a value equal to the one in force
 * drops the entry, so the draft holds only real changes.
 */
export function editDraft(draft: Draft, decl: FieldDecl, inForce: InForce, value: string | boolean): Draft {
  const key = draftKey(decl.group, decl.name, inForce.scope);
  const { [key]: _dropped, ...rest } = draft;
  if (same(decl, value, inForce.value)) return rest;
  return { ...rest, [key]: { group: decl.group, name: decl.name, scope: inForce.scope, value, reload: decl.reload_required } };
}

/** The draft with the family's own value of `decl` on its way out: `null` in its section. */
export function removeDraft(draft: Draft, decl: FieldDecl, scope: string): Draft {
  return { ...draft, [draftKey(decl.group, decl.name, scope)]: { group: decl.group, name: decl.name, scope, value: "", reload: decl.reload_required, removal: true } };
}

/** The draft without the edit or removal of `decl` in `scope`. */
export function dropDraft(draft: Draft, decl: FieldDecl, scope: string | null): Draft {
  const { [draftKey(decl.group, decl.name, scope)]: _dropped, ...rest } = draft;
  return rest;
}

/**
 * The entries a PATCH carries: every one but a cleared text field — PATCH
 * reads an unset value as no change, so a removal is `null`, never empty
 * (spec playground/05, Out of Scope).
 */
export function changes(draft: Draft): Draft {
  return Object.fromEntries(Object.entries(draft).filter(([, entry]) => entry.removal || typeof entry.value === "boolean" || entry.value.trim() !== ""));
}

/** The PATCH /v1/config body: only the changed fields, nested as the config file nests them; a removal is `null` in the family's section. */
export function patchBody(draft: Draft): Record<string, Record<string, unknown>> {
  const body: Record<string, Record<string, unknown>> = {};
  for (const entry of Object.values(changes(draft))) {
    const group = (body[entry.group] ??= {});
    const at = entry.scope === null ? group : ((group[entry.scope] ??= {}) as Record<string, unknown>);
    at[entry.name] = entry.removal ? null : typeof entry.value === "string" ? entry.value.trim() : entry.value;
  }
  return body;
}

/** The configuration after the draft, as far as the page can tell: a removal drops the family's own value. */
function applied(doc: ConfigDocument, draft: Draft): ConfigDocument {
  const after: ConfigDocument = Object.fromEntries(Object.entries(doc).map(([group, fields]) => [group, { ...fields }]));
  for (const entry of Object.values(changes(draft))) {
    const group = (after[entry.group] ??= {});
    const at = entry.scope === null ? group : ((group[entry.scope] = { ...(group[entry.scope] as object | undefined) }) as Record<string, unknown>);
    if (entry.removal) delete at[entry.name];
    else at[entry.name] = entry.value;
  }
  return after;
}

const norm = (value: unknown) => textOf(value).trim().toLowerCase();

/** Whether an entry changes what the running family runs with. With no family or configuration known, it is assumed to. */
function reaches(entry: DraftEntry, before: ConfigDocument | null, after: ConfigDocument | null, family: string | null, fields: FieldDecl[]): boolean {
  const decl = fields.find((d) => d.group === entry.group && d.name === entry.name);
  if (!before || !after || family === null || !decl) return true;
  return norm(resolved(before, decl, family)) !== norm(resolved(after, decl, family));
}

/**
 * Whether applying the draft reloads the running model: a changed or removed
 * `reload_required` value that resolves differently for the running family.
 * A value for the other family, or a group key the running family overrides
 * in its own section, is a live change; and when any value does reload, the
 * whole patch goes through the reload.
 */
export function needsReload(draft: Draft, fields: FieldDecl[], doc: ConfigDocument | null, family: string | null): boolean {
  const after = doc ? applied(doc, draft) : null;
  return Object.values(changes(draft)).some((entry) => entry.reload && reaches(entry, doc, after, family, fields));
}

/**
 * The values that wait for another model, by family, as `Takes effect when
 * <family> loads` lists them: a value in a family's section waits for that
 * family; a group key the running family overrides waits for each other
 * family that does not.
 */
export function waiting(draft: Draft, fields: FieldDecl[], doc: ConfigDocument | null, family: string | null): { family: string; keys: string[] }[] {
  if (!doc || family === null) return [];
  const after = applied(doc, draft);
  const others = familiesOf(fields, family).filter((f) => f !== family);
  const by = new Map<string, string[]>();
  for (const entry of Object.values(changes(draft))) {
    if (reaches(entry, doc, after, family, fields)) continue;
    const goes = entry.scope !== null ? [entry.scope] : others.filter((f) => !(entry.name in (section(after, entry.group, f) ?? {})));
    for (const f of goes) by.set(f, [...(by.get(f) ?? []), draftKey(entry.group, entry.name, entry.scope)]);
  }
  return [...by].map(([f, keys]) => ({ family: f, keys }));
}

/** The dock's line for values that wait for another model. */
export const waitingLine = (entry: { family: string; keys: string[] }) => `Takes effect when ${familyName(entry.family)} loads: ${entry.keys.join(", ")}`;

/** The primary action's label, which says what will happen before the click. */
export function primaryLabel(reload: boolean, id: string): string {
  return reload ? `Apply and reload ${id}` : "Apply";
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
