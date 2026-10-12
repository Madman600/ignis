import { describe, expect, it } from "vitest";
import { createServerSim } from "./mockServer.ts";
import { FIELDS } from "./src/server/fixture.ts";
import type { CatalogRow, PatchApplied } from "./src/server/routes.ts";

// The mock's Server view routes (spec playground/05 §Mock): the shapes the
// real ones answer with, a switch or a reload that reads `switching` for
// about three seconds, and the two triggers — a value "/error", an id
// containing "fail".

const T = 1_000_000;

describe("createServerSim", () => {
  it("serves the fields and the catalog in the list envelope, exactly one row loaded", () => {
    const sim = createServerSim();
    expect(sim.fields()).toEqual({ object: "list", data: FIELDS });
    const rows = sim.catalog(T).data as CatalogRow[];
    expect(rows.filter((row) => row.loaded).map((row) => row.id)).toEqual(["qwen3.8-27b"]);
    expect(rows.find((row) => row.id === "qwen3.8-flash-next")).toMatchObject({ switchable: true, family: "qwen38flashnext" });
    expect(rows.find((row) => row.id === "qwen3.8-27b-abliterated")).toMatchObject({ on_disk: "partial", switchable: false });
    expect(rows.find((row) => row.id === "qwen3.8-27b-abliterated")).not.toHaveProperty("family");
    expect(sim.models(T)).toMatchObject({ data: [{ id: "qwen3.8-27b", max_model_len: 786432 }], status: "serving" });
    expect(sim.models(T)).not.toHaveProperty("last_switch");
  });

  it("applies a live patch at once (200), the running configuration in the answer", () => {
    const sim = createServerSim();
    const answer = sim.patch({ server: { request_timeout: "60" }, model: { thinking_budget: "off" } }, T);
    expect(answer.status).toBe(200);
    const body = answer.body as PatchApplied;
    expect(body).toMatchObject({ applied: "live", persisted: true, path: "ignis.yaml" });
    expect(body.config?.server.request_timeout).toBe(60);
    expect(body.shadowed_at_restart).toEqual(["model.thinking_budget (set by IGNIS_MODEL_THINKING_BUDGET)"]);
    expect(sim.config(T).body.server.request_timeout).toBe(60);
  });

  it("refuses a value \"/error\" with a 400 naming the field's key, in a family section too, and applies nothing", () => {
    const sim = createServerSim();
    const plain = sim.patch({ server: { request_timeout: "90", developer_message_policy: "/error" } }, T);
    expect(plain.status).toBe(400);
    expect(JSON.stringify(plain.body)).toContain("`server.developer_message_policy`");
    expect(sim.config(T).body.server.request_timeout).toBe(30);
    const scoped = sim.patch({ model: { qwen38: { max_context: "/error" } } }, T);
    expect(JSON.stringify(scoped.body)).toContain("`model.qwen38.max_context`");
  });

  it("refuses a field that is not patchable, or not a field, as the server does", () => {
    const sim = createServerSim();
    const fixed = sim.patch({ server: { bind: "0.0.0.0:8000" } }, T);
    expect(fixed).toMatchObject({ status: 400, body: { error: { code: "config_field_not_patchable" } } });
    expect(sim.patch({ server: { bnd: "x" } }, T)).toMatchObject({ status: 400, body: { error: { code: "config_invalid" } } });
    expect(sim.patch({}, T)).toMatchObject({ status: 400 });
  });

  it("reloads for a field that needs it (202), reads `switching` for about three seconds, then `last_switch`", () => {
    const sim = createServerSim();
    const answer = sim.patch({ model: { qwen38: { max_context: "524288" } }, server: { request_timeout: "45" } }, T);
    expect(answer.status).toBe(202);
    expect(answer.body).toMatchObject({ applied: "reload", persisted: false, path: "ignis.yaml", switching: { from: "qwen3.8-27b", to: "qwen3.8-27b" } });
    expect(sim.models(T + 1_000)).toMatchObject({ status: "switching", switching: { from: "qwen3.8-27b", to: "qwen3.8-27b" } });
    // The gate holds every other route meanwhile.
    expect(sim.patch({ server: { request_timeout: "50" } }, T + 1_000)).toMatchObject({ status: 503, body: { error: { code: "model_switching" } } });
    const after = sim.models(T + 3_500);
    expect(after).toMatchObject({ status: "serving", data: [{ max_model_len: 524288 }], last_switch: { from: "qwen3.8-27b", to: "qwen3.8-27b", outcome: "switched" } });
    expect(sim.config(T + 3_500).body.server.request_timeout).toBe(45);
  });

  it("switches by id, and the catalog's loaded row follows", () => {
    const sim = createServerSim();
    expect(sim.switchTo({ model: "qwen3.8-flash-next" }, T)).toEqual({ status: 202, body: { status: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next" } });
    expect((sim.catalog(T + 500).data as CatalogRow[]).every((row) => !row.switchable)).toBe(true);
    expect(sim.switchTo({ model: "qwen3.8-27b-fail" }, T + 500)).toMatchObject({ status: 409, body: { error: { code: "switch_in_progress" } } });
    expect(sim.models(T + 3_500)).toMatchObject({ data: [{ id: "qwen3.8-flash-next", max_model_len: 589824 }], last_switch: { outcome: "switched" } });
    const loaded = (sim.catalog(T + 3_500).data as CatalogRow[]).find((row) => row.loaded);
    expect(loaded).toMatchObject({ id: "qwen3.8-flash-next", family: "qwen38flashnext" });
  });

  it("ends a switch to an id containing \"fail\" restored, through a brief `failed`, with a reason", () => {
    const sim = createServerSim();
    sim.switchTo({ model: "qwen3.8-27b-fail" }, T);
    expect(sim.models(T + 3_200)).toMatchObject({ status: "failed", reason: expect.stringContaining("qwen3.8-27b-fail") });
    const back = sim.models(T + 4_500);
    expect(back).toMatchObject({ status: "serving", data: [{ id: "qwen3.8-27b" }], last_switch: { from: "qwen3.8-27b", to: "qwen3.8-27b-fail", outcome: "restored" } });
    expect((back as { last_switch: { reason: string } }).last_switch.reason).toContain("qwen3.8-27b-fail");
  });

  it("ends a switch to an id containing \"dead\" with nothing serving: no loaded row, the torn-down model switchable again", () => {
    const sim = createServerSim();
    sim.switchTo({ model: "qwen3.8-27b-dead" }, T);
    const dead = sim.models(T + 5_000);
    expect(dead).toMatchObject({ status: "failed", data: [], reason: expect.stringContaining("qwen3.8-27b-dead"), last_switch: { outcome: "failed" } });
    const rows = sim.catalog(T + 5_000).data as CatalogRow[];
    expect(rows.filter((row) => row.loaded)).toEqual([]);
    expect(rows.find((row) => row.id === "qwen3.8-27b")).toMatchObject({ switchable: true });
    expect(sim.patch({ server: { request_timeout: "60" } }, T + 5_000)).toMatchObject({ status: 503 });
    // A switch to a model that loads is the way out.
    sim.switchTo({ model: "qwen3.8-flash-next" }, T + 5_000);
    expect(sim.models(T + 9_000)).toMatchObject({ status: "serving", data: [{ id: "qwen3.8-flash-next" }] });
  });

  it("answers 404 for an id it cannot switch to, naming the download command for an entry not on disk", () => {
    const sim = createServerSim();
    const absent = sim.switchTo({ model: "qwen3.8-27b-custom" }, T);
    expect(absent).toMatchObject({ status: 404, body: { error: { code: "model_not_found" } } });
    expect(JSON.stringify(absent.body)).toContain("ignis-server model download qwen3.8-27b-custom");
    expect(sim.switchTo({ model: "nope" }, T)).toMatchObject({ status: 404 });
    expect(sim.models(T)).toMatchObject({ status: "serving" });
  });
});
