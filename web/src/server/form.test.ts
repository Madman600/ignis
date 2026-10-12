import { describe, expect, it } from "vitest";
import { CONFIG, FIELDS } from "./fixture.ts";
import {
  changes,
  controlFor,
  type Draft,
  drainTimeout,
  editDraft,
  familyName,
  firstSentence,
  namedKeys,
  patchBody,
  canUseShared,
  dropDraft,
  needsReload,
  noteLines,
  primaryLabel,
  removeDraft,
  rowMode,
  rowView,
  shownOf,
  waiting,
  waitingLine,
} from "./form.ts";
import type { FieldDecl } from "./routes.ts";

// The Server view's form (spec playground/05 §Configuration, §Applying): the
// control a declaration calls for, the value in force, and the draft that
// becomes one PATCH.

const field = (key: string): FieldDecl => {
  const found = FIELDS.find((f) => f.file_key === key);
  if (!found) throw new Error(key);
  return found;
};

/** Edit `key` to `value` in `scope` (the 27B's by default; `null` is "All models"), as a row does. */
const edit = (draft: Draft, key: string, value: string | boolean, scope: string | null = "qwen38") =>
  editDraft(draft, field(key), rowView(field(key), CONFIG, scope).inForce, value);

const view = (key: string, scope: string | null) => rowView(field(key), CONFIG, scope);

describe("controlFor", () => {
  it("is the switch for a bool, segments up to four values, a select past that, else a text field", () => {
    expect(controlFor(field("model.enable_thinking"), true)).toEqual({ kind: "switch" });
    expect(controlFor(field("server.system_message_policy"), "merge")).toEqual({ kind: "segmented", options: ["merge", "strict"] });
    expect(controlFor(field("model.kv_format"), "hq-e8-2b")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b"] });
    expect(controlFor(field("server.developer_message_policy"), "inplace")).toEqual({
      kind: "select",
      options: ["inplace", "into-system", "after-system", "one-after-system", "reject"],
    });
    expect(controlFor(field("model.reasoning_effort"), null)).toMatchObject({ kind: "select" });
  });

  it("lists a value in force the rule does not name (an alias set from the file), so it shows as the choice", () => {
    expect(controlFor(field("model.kv_format"), "hq_e8_2b")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b", "hq_e8_2b"] });
    // A spelling that differs in case only is the same choice.
    expect(controlFor(field("model.kv_format"), "BF16")).toEqual({ kind: "segmented", options: ["bf16", "hq-e8-2b"] });
    // Unset is no value to list.
    expect(controlFor(field("model.reasoning_effort"), null)).toMatchObject({ options: expect.not.arrayContaining([null]) });
  });

  it("gives a text field the kind and the rule as its hint, and \"auto\" as the placeholder of an unset value", () => {
    expect(controlFor(field("server.request_timeout"), 30)).toEqual({ kind: "text", hint: "seconds, 1..=3600", placeholder: "30" });
    expect(controlFor(field("model.prefill_chunk"), 1024)).toMatchObject({ hint: "tokens, nonzero multiple of 128" });
    expect(controlFor(field("model.max_context"), 786432)).toMatchObject({ hint: "tokens, >= 1" });
    expect(controlFor(field("vram.kv_pool_bytes"), null)).toEqual({ kind: "text", hint: "bytes|tokens", placeholder: "auto" });
  });
});

describe("rowMode", () => {
  it("reads a field that is not patchable as set at start, with no control", () => {
    expect(rowMode(field("server.bind"), "qwen38")).toEqual({ kind: "fixed" });
    expect(rowMode(field("download.path"), null)).toEqual({ kind: "fixed" });
  });

  it("reads a field the running family does not take as not used by it", () => {
    expect(rowMode(field("vision.enabled"), "qwen38flashnext")).toEqual({ kind: "unused", family: "qwen38flashnext" });
    expect(rowMode(field("spec.decode_lanes"), "qwen38")).toEqual({ kind: "unused", family: "qwen38" });
    expect(rowMode(field("spec.decode_lanes"), "qwen38flashnext")).toEqual({ kind: "edit" });
    // With no family known, the server judges: the control stays.
    expect(rowMode(field("vision.enabled"), null)).toEqual({ kind: "edit" });
  });

  it("shows the known models read-only: a patch only adds to them", () => {
    expect(rowMode(field("switch.known_models"), "qwen38")).toEqual({ kind: "list" });
  });
});

describe("rowView", () => {
  it("in a family's scope reads a scoped field from its section when it has one, else the group's, and says which", () => {
    expect(view("model.max_context", "qwen38")).toMatchObject({ inForce: { value: 786432, scope: "qwen38" }, shared: 40960, note: { kind: "scoped", own: true } });
    expect(view("model.max_context", "qwen38flashnext").inForce).toEqual({ value: 589824, scope: "qwen38flashnext" });
    // No own value: the group's, and an edit still goes to the family's section.
    expect(view("model.kv_format", "qwen38")).toMatchObject({ inForce: { value: "hq-e8-2b", scope: "qwen38" }, note: { kind: "scoped", own: false } });
    expect(view("spec.backend", "qwen38flashnext")).toMatchObject({ inForce: { value: null, scope: "qwen38flashnext" }, note: { kind: "scoped", own: false } });
  });

  it("in All models shows and edits the group value, and names each family that has its own", () => {
    const context = view("model.max_context", null);
    expect(context.inForce).toEqual({ value: 40960, scope: null });
    expect(noteLines(context, undefined)).toEqual(["Qwen3.8-27B has its own value: 786432", "Qwen3.8-Flash-Next has its own value: 589824"]);
    expect(noteLines(view("spec.backend", null), undefined)).toEqual(["Qwen3.8-27B has its own value: dflash2"]);
    expect(noteLines(view("model.kv_format", null), undefined)).toEqual([]);
  });

  it("marks a field every family shares as the same for every model, in a family's scope", () => {
    expect(view("server.request_timeout", "qwen38")).toMatchObject({ inForce: { value: 30, scope: null }, note: { kind: "every" } });
    expect(noteLines(view("server.request_timeout", "qwen38"), undefined)).toEqual(["Same for every model"]);
    expect(noteLines(view("server.request_timeout", null), undefined)).toEqual([]);
    // A field with no control says nothing about its sharing.
    expect(noteLines(view("server.bind", "qwen38"), undefined)).toEqual([]);
  });

  it("reads a field only one family takes from its family's section, in any scope that shows it", () => {
    expect(view("vision.enabled", "qwen38").inForce).toEqual({ value: true, scope: "qwen38" });
    expect(view("vision.enabled", null).inForce).toEqual({ value: true, scope: "qwen38" });
    expect(noteLines(view("vision.enabled", null), undefined)).toEqual(["Qwen3.8-27B only"]);
    expect(noteLines(view("vision.enabled", "qwen38"), undefined)).toEqual([]);
    expect(view("vision.enabled", "qwen38flashnext").mode).toEqual({ kind: "unused", family: "qwen38flashnext" });
    // It is editable while the other family runs: the section is its family's.
    expect(view("spec.decode_lanes", null)).toMatchObject({ mode: { kind: "edit" }, inForce: { value: null, scope: "qwen38flashnext" } });
    expect(noteLines(view("spec.decode_lanes", null), undefined)).toEqual(["Qwen3.8-Flash-Next only"]);
  });

  it("has no value without a configuration", () => {
    expect(rowView(field("server.request_timeout"), null, "qwen38").inForce).toEqual({ value: undefined, scope: null });
    expect(rowView(field("model.max_context"), null, null).note).toBeNull();
  });

  it("marks a pending edit as an own value, and a pending removal as a return to the shared one", () => {
    const kv = view("model.kv_format", "qwen38");
    const edited = edit({}, "model.kv_format", "bf16")["model.qwen38.kv_format"];
    expect(noteLines(kv, edited)).toEqual(["Own value"]);
    expect(canUseShared(kv, edited)).toBe(false);
    const context = view("model.max_context", "qwen38");
    expect(canUseShared(context, undefined)).toBe(true);
    const removal = removeDraft({}, field("model.max_context"), "qwen38")["model.qwen38.max_context"];
    expect(noteLines(context, removal)).toEqual(["Back to the shared value once applied"]);
    expect(canUseShared(context, removal)).toBe(false);
  });
});

describe("the draft", () => {
  it("holds only what differs from the value in force, compared as the field reads it", () => {
    let draft: Draft = {};
    draft = edit(draft, "server.request_timeout", "60");
    expect(Object.keys(changes(draft))).toEqual(["server.request_timeout"]);
    // Typed back to the value in force, it is no change at all.
    draft = edit(draft, "server.request_timeout", "30");
    expect(draft).toEqual({});
    // A choice is compared without case, as the server compares it.
    expect(edit({}, "server.system_message_policy", "MERGE")).toEqual({});
    expect(edit({}, "model.enable_thinking", true)).toEqual({});
  });

  it("keeps a cleared text field on screen but sends nothing for it: a PATCH cannot unset a field", () => {
    const draft = edit({}, "server.request_timeout", "  ");
    expect(draft["server.request_timeout"]?.value).toBe("  ");
    expect(changes(draft)).toEqual({});
    expect(patchBody(draft)).toEqual({});
  });
});

describe("patchBody", () => {
  it("nests only the changed fields as the config file nests them, a scoped edit under the family it was made for", () => {
    let draft: Draft = {};
    draft = edit(draft, "server.request_timeout", " 60 ");
    draft = edit(draft, "model.enable_thinking", false);
    draft = edit(draft, "model.max_context", "524288");
    draft = edit(draft, "model.kv_format", "bf16");
    draft = edit(draft, "vision.max_tokens", "16384");
    expect(patchBody(draft)).toEqual({
      server: { request_timeout: "60" },
      model: { enable_thinking: false, qwen38: { max_context: "524288", kv_format: "bf16" } },
      vision: { qwen38: { max_tokens: "16384" } },
    });
  });

  it("holds edits of one field from several scopes at once, and a removal as null in its family's section", () => {
    let draft: Draft = {};
    draft = edit(draft, "model.max_context", "50000", null);
    draft = edit(draft, "model.kv_format", "bf16", "qwen38");
    draft = removeDraft(draft, field("model.max_context"), "qwen38flashnext");
    draft = edit(draft, "spec.decode_lanes", "4", null);
    expect(Object.keys(draft)).toEqual(["model.max_context", "model.qwen38.kv_format", "model.qwen38flashnext.max_context", "spec.qwen38flashnext.decode_lanes"]);
    expect(Object.keys(changes(draft))).toHaveLength(4);
    expect(patchBody(draft)).toEqual({
      model: { max_context: "50000", qwen38: { kv_format: "bf16" }, qwen38flashnext: { max_context: null } },
      spec: { qwen38flashnext: { decode_lanes: "4" } },
    });
    expect(Object.keys(dropDraft(draft, field("model.max_context"), "qwen38flashnext"))).not.toContain("model.qwen38flashnext.max_context");
  });

  it("replaces a removal by an edit, and an edit by a removal, of the same field in the same scope", () => {
    const removed = removeDraft({}, field("model.max_context"), "qwen38");
    expect(patchBody(edit(removed, "model.max_context", "524288"))).toEqual({ model: { qwen38: { max_context: "524288" } } });
    expect(patchBody(removeDraft(edit({}, "model.max_context", "524288"), field("model.max_context"), "qwen38"))).toEqual({ model: { qwen38: { max_context: null } } });
  });

  it("sends a scoped field under the group when it is edited in All models", () => {
    expect(patchBody(edit({}, "spec.backend", "mtp", null))).toEqual({ spec: { backend: "mtp" } });
  });
});

describe("the reload a draft causes", () => {
  const reload = (draft: Draft, family: string | null = "qwen38") => needsReload(draft, FIELDS, CONFIG, family);
  const waits = (draft: Draft, family: string | null = "qwen38") => waiting(draft, FIELDS, CONFIG, family);

  it("reloads for a reload_required value that resolves differently for the running family", () => {
    expect(reload(edit({}, "model.max_context", "524288"))).toBe(true);
    expect(reload(edit({}, "model.kv_format", "bf16", null))).toBe(true);
    expect(reload(edit({}, "vision.enabled", false))).toBe(true);
    // A live field never does.
    expect(reload(edit({}, "server.request_timeout", "60"))).toBe(false);
    expect(reload({})).toBe(false);
  });

  it("does not reload for a value of the other family, which waits for it", () => {
    const draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft)).toBe(false);
    expect(waits(draft)).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
    expect(waitingLine(waits(draft)[0])).toBe("Takes effect when Qwen3.8-Flash-Next loads: model.qwen38flashnext.max_context");
    // A field only that family takes is the same.
    const lanes = edit({}, "spec.decode_lanes", "4", null);
    expect(reload(lanes)).toBe(false);
    expect(waits(lanes)).toEqual([{ family: "qwen38flashnext", keys: ["spec.qwen38flashnext.decode_lanes"] }]);
    // And from the other side.
    expect(reload(edit({}, "model.max_context", "524288", "qwen38"), "qwen38flashnext")).toBe(false);
    expect(waits(edit({}, "model.max_context", "524288", "qwen38"), "qwen38flashnext")).toEqual([{ family: "qwen38", keys: ["model.qwen38.max_context"] }]);
  });

  it("does not reload for a group key the running family overrides, which reaches only a family that does not", () => {
    // The 27B and Flash-Next both have their own max_context: the shared one is run by neither.
    const both = edit({}, "model.max_context", "50000", null);
    expect(reload(both)).toBe(false);
    expect(waits(both)).toEqual([]);
    // Only the 27B overrides the backend: Flash-Next takes the shared one when it loads.
    const backend = edit({}, "spec.backend", "mtp", null);
    expect(reload(backend)).toBe(false);
    expect(waits(backend)).toEqual([{ family: "qwen38flashnext", keys: ["spec.backend"] }]);
    // Seen from Flash-Next, which has no own backend, the same edit reloads.
    expect(reload(backend, "qwen38flashnext")).toBe(true);
  });

  it("reloads for a removal of the running family's own value when the shared one differs", () => {
    expect(reload(removeDraft({}, field("model.max_context"), "qwen38"))).toBe(true);
    expect(reload(removeDraft({}, field("spec.backend"), "qwen38"))).toBe(true);
    expect(reload(removeDraft({}, field("model.max_context"), "qwen38flashnext"))).toBe(false);
    expect(waits(removeDraft({}, field("model.max_context"), "qwen38flashnext"))).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
    // A removal against a group edit in the same draft resolves to the edited value.
    let draft = edit({}, "model.max_context", "786432", null);
    draft = removeDraft(draft, field("model.max_context"), "qwen38");
    expect(reload(draft)).toBe(false);
  });

  it("reloads when any value reaches the running model, and still lists what waits", () => {
    let draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft)).toBe(false);
    draft = edit(draft, "model.kv_format", "bf16");
    expect(reload(draft)).toBe(true);
    expect(waits(draft)).toEqual([{ family: "qwen38flashnext", keys: ["model.qwen38flashnext.max_context"] }]);
  });

  it("assumes a reload when the running family or the configuration is not known, and lists nothing as waiting", () => {
    const draft = edit({}, "model.max_context", "524288", "qwen38flashnext");
    expect(reload(draft, null)).toBe(true);
    expect(waits(draft, null)).toEqual([]);
    expect(needsReload(draft, FIELDS, null, "qwen38")).toBe(true);
  });
});

describe("primaryLabel", () => {
  it("is Apply, and names the reload when the draft causes one", () => {
    expect(primaryLabel(false, "qwen3.8-27b")).toBe("Apply");
    expect(primaryLabel(true, "qwen3.8-27b")).toBe("Apply and reload qwen3.8-27b");
    // A cleared reload field is no change, so no reload either.
    expect(needsReload(edit({}, "model.max_context", ""), FIELDS, CONFIG, "qwen38")).toBe(false);
  });
});

describe("namedKeys", () => {
  it("marks every row whose file key the refusal names, in a family section too, and no row a longer key contains", () => {
    const message =
      "`model.qwen38.max_context` must be at least 1, got 0; `server.metrics_bind` cannot be changed while the server runs; `server.request_timeout` expects a second count, got `x`";
    expect([...namedKeys(message, FIELDS)].sort()).toEqual(["model.max_context", "server.metrics_bind", "server.request_timeout"]);
  });
});

describe("text", () => {
  it("splits a description after its first sentence, for the rest behind More", () => {
    expect(firstSentence(field("model.id").description)).toEqual([
      "The id the model is served under.",
      "Unset, a load is served under its own model's id (qwen3.8-27b or qwen3.8-flash-next); without an artifact, qwen3.8-27b. An id naming the other model than the artifact's is refused.",
    ]);
    expect(firstSentence(field("model.enable_thinking").description)).toEqual(["Whether a request that says nothing thinks.", ""]);
  });

  it("names the families as people do, and shows a value with no control as words", () => {
    expect(familyName("qwen38")).toBe("Qwen3.8-27B");
    expect(familyName("qwen38flashnext")).toBe("Qwen3.8-Flash-Next");
    expect(shownOf(null)).toBe("Not set");
    expect(shownOf(true)).toBe("On");
    expect(shownOf("127.0.0.1:8000")).toBe("127.0.0.1:8000");
    expect(shownOf(40960)).toBe("40960");
  });
});

describe("drainTimeout", () => {
  it("is switch.drain_timeout in force, or unknown without a configuration", () => {
    expect(drainTimeout(CONFIG)).toBe(30);
    expect(drainTimeout({ switch: { drain_timeout: 0 } })).toBe(0);
    expect(drainTimeout(null)).toBeNull();
  });
});
