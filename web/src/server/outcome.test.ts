import { describe, expect, it } from "vitest";
import type { LastSwitch, ModelState } from "../api/model.ts";
import { FIELDS } from "./fixture.ts";
import { afterPatch, afterSwitch, appliedLines, drainNote, lastSwitchLine, pendingReload, refusalLines, settleReload } from "./outcome.ts";

// What the Server view says once a change or a switch has an answer (spec
// playground/05 §Applying, §Model band), worded as the spec words it.

describe("appliedLines (200)", () => {
  it("says where a live change was written, or why it was not", () => {
    expect(appliedLines({ applied: "live", persisted: true, path: "F:/ignis/ignis.yaml" })).toEqual([
      { text: "Applied and written to F:/ignis/ignis.yaml", tone: "done" },
    ]);
    expect(appliedLines({ applied: "live", persisted: false, reason: "no config file is in use" })).toEqual([
      { text: "Applied, not written to a file: no config file is in use", tone: "warn" },
    ]);
  });

  it("adds one line per key a flag or an env var overrides at the next start", () => {
    const lines = appliedLines({
      applied: "live",
      persisted: true,
      path: "c.yaml",
      shadowed_at_restart: ["model.thinking_budget (set by IGNIS_MODEL_THINKING_BUDGET)", "server.request_timeout (set by --server-request-timeout)"],
    });
    expect(lines.slice(1)).toEqual([
      { text: "Overridden at the next start: model.thinking_budget (set by IGNIS_MODEL_THINKING_BUDGET)", tone: "warn" },
      { text: "Overridden at the next start: server.request_timeout (set by --server-request-timeout)", tone: "warn" },
    ]);
  });
});

describe("the reload a 202 began", () => {
  const accepted = {
    applied: "reload" as const,
    persisted: false,
    path: "c.yaml",
    reason: "written once the reloaded model serves; not if the reload fails",
    shadowed_at_restart: ["model.max_context (set by --model-max-context)"],
    switching: { from: "qwen3.8-27b", to: "qwen3.8-27b" },
  };
  const before: LastSwitch = { from: "qwen3.8-27b", to: "qwen3.8-27b", outcome: "switched" };
  const ready = (lastSwitch?: LastSwitch): ModelState => ({ state: "ready", id: "qwen3.8-27b", contextLimit: 1, lastSwitch });
  const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-27b", since: 0, lastSwitch: before };

  it("waits until it has seen the switch run, even when the last switch already reads `switched`", () => {
    const pending = pendingReload(accepted, before);
    // A read before the switch shows: the same record as before, so not this reload's.
    expect(settleReload(pending, ready(before))).toEqual({ kind: "wait", pending });
    const seen = settleReload(pending, switching);
    expect(seen).toEqual({ kind: "wait", pending: { ...pending, seen: true } });
  });

  it("follows a reload asked of an unloaded model through its load (spec model-switch/02)", () => {
    const pending = pendingReload(accepted, before);
    const waking: ModelState = { state: "waking", to: "qwen3.8-27b", since: 0, lastSwitch: before };
    const seen = settleReload(pending, waking);
    expect(seen).toEqual({ kind: "wait", pending: { ...pending, seen: true } });
    expect(settleReload({ ...pending, seen: true }, ready(before))).toMatchObject({ kind: "done", ok: true });
  });

  it("reports the reload once the model serves with `switched`, naming the file it was written to", () => {
    const pending = { ...pendingReload(accepted, before), seen: true };
    expect(settleReload(pending, ready(before))).toEqual({
      kind: "done",
      ok: true,
      lines: [
        { text: "Reloaded qwen3.8-27b and written to c.yaml", tone: "done" },
        { text: "Overridden at the next start: model.max_context (set by --model-max-context)", tone: "warn" },
      ],
    });
    const noFile = { ...pendingReload({ ...accepted, path: undefined, reason: "no config file is in use", shadowed_at_restart: undefined }), seen: true };
    expect(settleReload(noFile, ready(before))).toMatchObject({ lines: [{ text: "Reloaded qwen3.8-27b, not written to a file: no config file is in use" }] });
  });

  it("settles on a record it has not seen before even when the switch ran between two reads", () => {
    const pending = pendingReload(accepted, undefined);
    expect(settleReload(pending, ready(before))).toMatchObject({ kind: "done", ok: true });
  });

  it("reports any other outcome with its reason, so the draft is kept", () => {
    const pending = { ...pendingReload(accepted, before), seen: true };
    const restored: LastSwitch = { from: "qwen3.8-27b", to: "qwen3.8-27b", outcome: "restored", reason: "out of memory" };
    expect(settleReload(pending, ready(restored))).toEqual({
      kind: "done",
      ok: false,
      lines: [{ text: "The reload of qwen3.8-27b failed, the previous configuration was loaded again: out of memory", tone: "fault" }],
    });
  });

  it("waits through `failed` while the old model is loaded again, and ends on a failure that leaves nothing serving", () => {
    const pending = { ...pendingReload(accepted, before), seen: true };
    const restoring: ModelState = { state: "error", message: "Switch failed: x", failed: true, lastSwitch: before };
    expect(settleReload(pending, restoring)).toMatchObject({ kind: "wait" });
    const failed: LastSwitch = { from: "qwen3.8-27b", to: "qwen3.8-27b", outcome: "failed", reason: "the artifact is gone" };
    expect(settleReload(pending, { ...restoring, lastSwitch: failed })).toMatchObject({ kind: "done", ok: false });
  });
});

describe("refusalLines", () => {
  it("shows the server's own message, its first letter raised", () => {
    expect(refusalLines({ status: 400, message: "`server.request_timeout` must be in 1..=3600, got 0", code: "config_invalid" })).toEqual([
      { text: "`server.request_timeout` must be in 1..=3600, got 0", tone: "fault" },
    ]);
    expect(refusalLines({ status: 409, message: "a switch from a to b is already under way; switches are not queued", code: "switch_in_progress" })).toEqual([
      { text: "A switch from a to b is already under way; switches are not queued", tone: "fault" },
    ]);
  });
});

describe("lastSwitchLine", () => {
  it("names how the last switch or reload ended, a failure in the fault colour with its reason", () => {
    expect(lastSwitchLine({ from: "a", to: "b", outcome: "switched" })).toEqual({ text: "Switched from a to b", tone: "done" });
    expect(lastSwitchLine({ from: "a", to: "a", outcome: "switched" })).toEqual({ text: "Reloaded a", tone: "done" });
    expect(lastSwitchLine({ from: "a", to: "b", outcome: "refused", reason: "r" })).toEqual({ text: "The switch to b was refused, a kept serving: r", tone: "fault" });
    expect(lastSwitchLine({ from: "a", to: "b", outcome: "restored", reason: "r" })).toEqual({ text: "The switch to b failed, a was loaded again: r", tone: "fault" });
    expect(lastSwitchLine({ from: "a", to: "b", outcome: "failed", reason: "r" })).toEqual({ text: "The switch to b failed and no model is serving: r", tone: "fault" });
    expect(lastSwitchLine({ from: "a", to: "a", outcome: "refused", reason: "r" })).toEqual({ text: "The reload of a was refused, nothing changed: r", tone: "fault" });
  });
});

describe("drainNote", () => {
  it("says how long a switch gives the reply streaming now", () => {
    expect(drainNote(30, "A switch")).toBe("A reply is streaming: a switch gives it 30 seconds to finish, then cancels it.");
    expect(drainNote(1, "The reload")).toBe("A reply is streaming: the reload gives it 1 second to finish, then cancels it.");
    expect(drainNote(0, "The reload")).toBe("A reply is streaming: the reload cancels it at once.");
    expect(drainNote(null, "A switch")).toBe("A reply is streaming: a switch lets it finish first, up to switch.drain_timeout.");
  });
});

describe("afterPatch", () => {
  const fields = FIELDS;
  const before: LastSwitch = { from: "qwen3.8-27b", to: "qwen3.8-27b", outcome: "switched" };

  it("200: the draft clears and the form shows the configuration the answer carries", () => {
    const config = { server: { request_timeout: 60 } };
    const after = afterPatch({ ok: true, status: 200, value: { applied: "live", persisted: true, path: "c.yaml", config } }, fields, before);
    expect(after).toEqual({
      clearDraft: true,
      result: [{ text: "Applied and written to c.yaml", tone: "done" }],
      marked: [],
      pending: null,
      config,
      refreshModel: false,
    });
  });

  it("202: the draft is kept and the reload followed, the model read again so the band shows it", () => {
    const after = afterPatch(
      { ok: true, status: 202, value: { applied: "reload", persisted: false, path: "c.yaml", reason: "later", switching: { from: "a", to: "a" } } },
      fields,
      before,
    );
    expect(after).toEqual({
      clearDraft: false,
      result: null,
      marked: [],
      pending: { id: "a", path: "c.yaml", reason: "later", shadowed: [], seen: false, before },
      config: null,
      refreshModel: true,
    });
  });

  it("400: the message, and every row it names marked; nothing applied, the draft kept", () => {
    const after = afterPatch(
      { ok: false, status: 400, code: "config_invalid", message: "`model.qwen38.max_context (the request body)` must be at least 1, got 0" },
      fields,
      before,
    );
    expect(after).toMatchObject({ clearDraft: false, marked: ["model.max_context"], pending: null, refreshModel: false });
    expect(after.result?.[0].tone).toBe("fault");
  });

  it("409 and 503: the switch under way is named and the model read again, so the action waits for it", () => {
    for (const status of [409, 503]) {
      const after = afterPatch({ ok: false, status, message: "a switch from a to b is already under way; switches are not queued" }, fields, before);
      expect(after).toMatchObject({ clearDraft: false, refreshModel: true, result: [{ text: "A switch from a to b is already under way; switches are not queued", tone: "fault" }] });
    }
  });

  it("501: the server's reason, nothing else", () => {
    const after = afterPatch({ ok: false, status: 501, code: "config_unavailable", message: "this server was built without a running configuration" }, fields, before);
    expect(after).toMatchObject({ clearDraft: false, refreshModel: false, marked: [], result: [{ tone: "fault" }] });
  });
});

describe("afterSwitch", () => {
  it("follows a switch that began, and names a refusal in the band", () => {
    expect(afterSwitch({ ok: true, status: 202, value: { from: "a", to: "b" } })).toEqual({ refusal: null, follow: true, refreshModel: true });
    const absent = afterSwitch({ ok: false, status: 404, code: "model_not_found", message: "model x is in the catalog but not on disk" });
    expect(absent).toEqual({ refusal: "Model x is in the catalog but not on disk", follow: false, refreshModel: false });
    expect(afterSwitch({ ok: false, status: 409, message: "a switch from a to b is already under way" })).toMatchObject({ refreshModel: true });
  });
});
