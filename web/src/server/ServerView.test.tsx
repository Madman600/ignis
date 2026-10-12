import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { CATALOG, CONFIG, FIELDS } from "./fixture.ts";
import { type Draft, editDraft, valueInForce } from "./form.ts";
import type { FieldDecl } from "./routes.ts";
import { ServerView } from "./ServerView.tsx";
import type { ServerActions, ServerState } from "./useServer.ts";

// The Server view (spec playground/05) rendered in its states, over the real
// field declarations: serving, switching, after a switch that did not take,
// on a server with no configuration, and with a draft of live or reload
// changes in the dock.

const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432 };

const actions: ServerActions = { load: () => {}, edit: () => {}, discard: () => {}, apply: () => {}, dismiss: () => {} };

const base: ServerState = {
  fields: { state: "ok", value: FIELDS },
  config: { state: "ok", value: CONFIG },
  catalog: { state: "ok", value: CATALOG },
  draft: {},
  marked: [],
  result: null,
  pending: null,
  applying: false,
  loadingId: null,
  loadRefusal: null,
};

const render = (state: Partial<ServerState> = {}, model: ModelState = ready, streaming = false, now = 0) =>
  renderToStaticMarkup(<ServerView model={model} state={{ ...base, ...state }} actions={actions} streaming={streaming} now={now} />);

const decl = (key: string): FieldDecl => FIELDS.find((f) => f.file_key === key)!;
const edit = (draft: Draft, key: string, value: string | boolean) => editDraft(draft, decl(key), valueInForce(decl(key), CONFIG, "qwen38"), value);

/** The markup of the configuration row of `key`, from its name to the next row. */
const rowOf = (html: string, key: string) => {
  const start = html.indexOf(`id="server-field-${key}-name"`);
  expect(start, key).toBeGreaterThan(-1);
  const names = /id="server-field-[\w.]+-name"/g;
  names.lastIndex = start + 1;
  const next = names.exec(html);
  return html.slice(start, next ? next.index : undefined);
};

describe("ServerView: the model band", () => {
  it("shows the loaded id, its state, context and family, and one row per catalog row with its action", () => {
    const html = render();
    expect(html).toContain('aria-label="Model"');
    expect(html).toContain(">qwen3.8-27b</h2>");
    expect(html).toContain("Serving");
    expect(html).toContain("786,432 tokens");
    expect(html).toContain("Qwen3.8-27B");
    const rows = html.match(/<li class="grid/g) ?? [];
    expect(rows).toHaveLength(CATALOG.length);
    expect(html).toContain(">Loaded<");
    expect(html).toContain('aria-label="Load qwen3.8-flash-next"');
    expect(html).toContain('aria-label="Load qwen3.8-27b-fail"');
    expect(html).toContain('aria-label="Copy download command for qwen3.8-27b-abliterated"');
    expect(html).toContain('aria-label="Copy download command for qwen3.8-27b-custom"');
    expect(html).toContain("On disk, 67.8 GiB");
    expect(html).toContain("Partly downloaded, 18.07 GiB in all");
  });

  it("has no Loaded row while nothing serves, and offers Load for the retry, the torn-down model's included", () => {
    const dead: ModelState = { state: "error", message: "Switch failed: x", failed: true };
    const torn = CATALOG.map((r) => ({ ...r, loaded: false, switchable: r.on_disk === "complete" }));
    const html = render({ catalog: { state: "ok", value: torn } }, dead);
    expect(html).not.toContain(">Loaded<");
    expect(html).toContain("No model is serving");
    expect(html).toContain('aria-label="Load qwen3.8-27b"');
    expect(html).not.toMatch(/aria-label="Load qwen3\.8-27b" disabled=""/);
    expect(html).not.toMatch(/aria-label="Load qwen3\.8-flash-next" disabled=""/);
  });

  it("follows a switch: both ids, the seconds it has run, the model it loads, and no Load in reach", () => {
    const switching: ModelState = { state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 1_000 };
    const html = render({}, switching, false, 13_500);
    expect(html).toContain("Switching from <span");
    expect(html).toMatch(/qwen3\.8-27b<\/span> to qwen3\.8-flash-next/);
    expect(html).toContain("12 s");
    expect(html).toContain(">Loading…<");
    expect(html).toMatch(/aria-label="Load qwen3\.8-27b-fail" disabled=""/);
  });

  it("names how the last switch ended, a failure in the fault colour with its reason", () => {
    const restored: ModelState = {
      ...ready,
      lastSwitch: { from: "qwen3.8-27b", to: "qwen3.8-27b-fail", outcome: "restored", reason: "the artifact failed its checksum" },
    };
    const html = render({}, restored);
    expect(html).toContain('class="text-fault [overflow-wrap:anywhere]">The switch to qwen3.8-27b-fail failed, qwen3.8-27b was loaded again: the artifact failed its checksum<');
    expect(render({ loadRefusal: "Model x is in the catalog but not on disk" })).toContain("Model x is in the catalog but not on disk");
  });

  it("says what a switch does to a reply streaming now, only while one streams", () => {
    expect(render({}, ready, true)).toContain("A reply is streaming: a switch gives it 30 seconds to finish, then cancels it.");
    expect(render()).not.toContain("A reply is streaming");
  });
});

describe("ServerView: the configuration", () => {
  it("shows every visible field once, in its group, in declaration order", () => {
    const html = render();
    const names = [...html.matchAll(/id="server-field-([\w.]+)-name"/g)].map((m) => m[1]);
    expect(names).toEqual(FIELDS.map((f) => f.file_key));
    for (const group of new Set(FIELDS.map((f) => f.group))) expect(html).toContain(`id="server-group-${group}"`);
    expect(html).toContain('aria-label="Configuration groups"');
  });

  it("gives each field the control its declaration calls for", () => {
    const html = render();
    expect(rowOf(html, "model.enable_thinking")).toContain('role="switch"');
    expect(rowOf(html, "model.kv_format")).toMatch(/type="radio"[^>]*name="model\.kv_format"/);
    expect(rowOf(html, "server.developer_message_policy")).toMatch(/<select[^>]*name="server\.developer_message_policy"/);
    const timeout = rowOf(html, "server.request_timeout");
    expect(timeout).toMatch(/<input[^>]*name="server\.request_timeout"[^>]*value="30"/);
    expect(timeout).toContain("seconds, 1..=3600");
    expect(rowOf(html, "vram.kv_pool_bytes")).toContain('placeholder="auto"');
    // An unset choice of more than four reads "auto" too.
    expect(rowOf(html, "model.reasoning_effort")).toMatch(/<option value="" disabled="" selected="">auto<\/option>/);
  });

  it("shows set-at-start, not-used and the known models read-only, and marks the fields a change reloads", () => {
    const html = render();
    const bind = rowOf(html, "server.bind");
    expect(bind).toContain("127.0.0.1:8000");
    expect(bind).toContain("Set at start");
    expect(bind).not.toMatch(/<input|<select|role="switch"/);
    const lanes = rowOf(html, "spec.decode_lanes");
    expect(lanes).toContain("Not used by Qwen3.8-27B");
    expect(lanes).not.toMatch(/<input|<select/);
    const known = rowOf(html, "switch.known_models");
    expect(known).toContain("qwen3.8-27b-fail");
    expect(known).toContain("Read-only here");
    expect(known).not.toMatch(/<input|<select/);
    expect(rowOf(html, "model.max_context")).toContain(">reload<");
    expect(rowOf(html, "server.request_timeout")).not.toContain(">reload<");
  });

  it("reads a scoped field from the running family's section and says so", () => {
    const html = render();
    const context = rowOf(html, "model.max_context");
    expect(context).toContain('value="786432"');
    expect(context).toContain("Qwen3.8-27B&#x27;s own value");
    expect(rowOf(html, "model.kv_format")).not.toContain("own value");
  });

  it("says the server has no running configuration on a 501, and the band still works", () => {
    const html = render({ config: { state: "error", refusal: { status: 501, message: "no config", code: "config_unavailable" } } });
    expect(html).toContain("This server has no running configuration to show.");
    expect(html).toContain('aria-label="Load qwen3.8-flash-next"');
    expect(html).not.toContain('id="server-field-');
  });

  it("gives every form field an id or a name", () => {
    const fields = render().match(/<(input|textarea|select)\b[^>]*>/g) ?? [];
    expect(fields.length).toBeGreaterThan(0);
    for (const field of fields) expect(field).toMatch(/\s(id|name)="[^"]+"/);
  });
});

describe("ServerView: the dock", () => {
  it("is not there without a change or an answer", () => {
    expect(render()).not.toContain('aria-label="Changes"');
  });

  it("offers Apply for live changes and names the reload, in the heat, once a changed field needs one", () => {
    const live = render({ draft: edit(edit({}, "server.request_timeout", "60"), "switch.drain_timeout", "5") });
    expect(live).toContain("2 changes");
    expect(live).toContain(">Discard<");
    expect(live).toMatch(/>Apply<\/button>/);
    expect(live).not.toContain("heat-fill");

    const reload = render({ draft: edit(edit({}, "server.request_timeout", "60"), "model.max_context", "524288") });
    expect(reload).toMatch(/class="[^"]*heat-fill[^"]*">Apply and reload qwen3\.8-27b<\/button>/);
  });

  it("warns a streaming reply about the reload only while one streams, and only for a reload", () => {
    const reload = { draft: edit({}, "model.max_context", "524288") };
    expect(render(reload, ready, true)).toContain("A reply is streaming: the reload gives it 30 seconds to finish, then cancels it.");
    expect(render(reload, ready, false)).not.toContain("the reload gives it");
    expect(render({ draft: edit({}, "server.request_timeout", "60") }, ready, true)).not.toContain("the reload gives it");
  });

  it("shows a refusal and marks the rows it names, the draft kept", () => {
    const html = render({
      draft: edit({}, "server.request_timeout", "/error"),
      marked: ["server.request_timeout"],
      result: [{ text: "`server.request_timeout` expects a second count, got `/error`", tone: "fault" }],
    });
    expect(html).toContain("expects a second count, got `/error`");
    expect(rowOf(html, "server.request_timeout")).toContain("Refused by the server");
    expect(rowOf(html, "server.request_timeout")).toContain('aria-invalid="true"');
    expect(html).toContain("1 change");
  });

  it("holds the action while a switch runs or a reload is followed", () => {
    const draft = edit({}, "server.request_timeout", "60");
    const switching: ModelState = { state: "switching", from: "a", to: "b", since: 0 };
    expect(render({ draft }, switching)).toContain("A switch from a to b is under way. Apply once it lands.");
    expect(render({ draft }, switching)).toMatch(/disabled=""[^>]*>Apply<\/button>/);
    const reloading = render({ draft, pending: { id: "qwen3.8-27b", shadowed: [], seen: true } });
    expect(reloading).toContain("Reloading qwen3.8-27b…");
  });

  it("keeps an answer on screen once the draft is applied, until dismissed", () => {
    const html = render({ result: [{ text: "Applied and written to c.yaml", tone: "done" }] });
    expect(html).toContain("Applied and written to c.yaml");
    expect(html).toContain('aria-label="Dismiss"');
  });
});
