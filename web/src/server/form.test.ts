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
  primaryLabel,
  rowMode,
  shownOf,
  valueInForce,
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

/** Edit `key` to `value` on the 27B, as a row does. */
const edit = (draft: Draft, key: string, value: string | boolean, family: string | null = "qwen38") =>
  editDraft(draft, field(key), valueInForce(field(key), CONFIG, family), value);

describe("controlFor", () => {
  it("is the switch for a bool, segments up to four values, a select past that, else a text field", () => {
    expect(controlFor(field("model.enable_thinking"), true)).toEqual({ kind: "switch" });
    expect(controlFor(field("server.system_message_policy"), "merge")).toEqual({ kind: "segmented", options: ["merge", "strict"] });
    // Four exactly is still segments.
    expect(controlFor(field("model.kv_format"), "hq-e8-2b")).toMatchObject({ kind: "segmented" });
    expect(controlFor(field("server.developer_message_policy"), "inplace")).toEqual({
      kind: "select",
      options: ["inplace", "into-system", "after-system", "one-after-system", "reject"],
    });
    expect(controlFor(field("model.reasoning_effort"), null)).toMatchObject({ kind: "select" });
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

describe("valueInForce", () => {
  it("reads a scoped field from the running family's section when it has one, else from the group", () => {
    expect(valueInForce(field("model.max_context"), CONFIG, "qwen38")).toEqual({ value: 786432, scope: "qwen38" });
    expect(valueInForce(field("model.max_context"), CONFIG, "qwen38flashnext")).toEqual({ value: 589824, scope: "qwen38flashnext" });
    expect(valueInForce(field("model.kv_format"), CONFIG, "qwen38")).toEqual({ value: "hq-e8-2b", scope: null });
    expect(valueInForce(field("spec.backend"), CONFIG, "qwen38")).toEqual({ value: "dflash2", scope: "qwen38" });
    expect(valueInForce(field("spec.backend"), CONFIG, "qwen38flashnext")).toEqual({ value: null, scope: null });
  });

  it("reads a field only one family takes from that family's section, the only place it is", () => {
    expect(valueInForce(field("vision.enabled"), CONFIG, "qwen38")).toEqual({ value: true, scope: "qwen38" });
    expect(valueInForce(field("spec.decode_lanes"), CONFIG, "qwen38flashnext")).toEqual({ value: null, scope: "qwen38flashnext" });
    // Not knowing the family, it is still where its one family keeps it.
    expect(valueInForce(field("vision.enabled"), CONFIG, null)).toEqual({ value: true, scope: "qwen38" });
  });

  it("reads the group for every other field, and nothing without a configuration", () => {
    expect(valueInForce(field("server.request_timeout"), CONFIG, "qwen38")).toEqual({ value: 30, scope: null });
    expect(valueInForce(field("model.max_context"), CONFIG, null)).toEqual({ value: 40960, scope: null });
    expect(valueInForce(field("server.request_timeout"), null, "qwen38")).toEqual({ value: undefined, scope: null });
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
  it("nests only the changed fields as the config file nests them, a scoped edit under the family it was read from", () => {
    let draft: Draft = {};
    draft = edit(draft, "server.request_timeout", " 60 ");
    draft = edit(draft, "model.enable_thinking", false);
    draft = edit(draft, "model.max_context", "524288");
    draft = edit(draft, "model.kv_format", "bf16");
    draft = edit(draft, "vision.max_tokens", "16384");
    expect(patchBody(draft)).toEqual({
      server: { request_timeout: "60" },
      model: { enable_thinking: false, kv_format: "bf16", qwen38: { max_context: "524288" } },
      vision: { qwen38: { max_tokens: "16384" } },
    });
  });

  it("sends a scoped field under the group when no family section held its value", () => {
    const draft = edit({}, "spec.backend", "mtp", "qwen38flashnext");
    expect(patchBody(draft)).toEqual({ spec: { backend: "mtp" } });
  });
});

describe("primaryLabel", () => {
  it("is Apply for live fields and names the reload as soon as one changed field needs it", () => {
    let draft = edit({}, "server.request_timeout", "60");
    draft = edit(draft, "switch.drain_timeout", "10");
    expect(primaryLabel(draft, "qwen3.8-27b")).toBe("Apply");
    draft = edit(draft, "model.max_context", "524288");
    expect(primaryLabel(draft, "qwen3.8-27b")).toBe("Apply and reload qwen3.8-27b");
    // A cleared reload field is no change, so no reload either.
    expect(primaryLabel(edit({}, "model.max_context", ""), "qwen3.8-27b")).toBe("Apply");
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
