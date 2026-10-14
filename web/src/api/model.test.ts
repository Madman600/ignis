import { describe, expect, it } from "vitest";
import { afterRefresh, canChat, canSend, imageLoaded, loadAwaited, loadUnderWay, readModel, switchFollowed, type ModelState } from "./model.ts";
import { keyRequired, saveKey } from "./auth.ts";

const listing = (id: string, max?: number) =>
  (async () => new Response(JSON.stringify({ data: [{ id, max_model_len: max }] }))) as typeof fetch;

const answering = (body: object) => (async () => new Response(JSON.stringify(body))) as typeof fetch;

describe("readModel", () => {
  it("reads the loaded model and its context limit from GET /v1/models", async () => {
    saveKey("sk");
    expect(await readModel(listing("qwen3.8-27b", 786432))).toEqual({ state: "ready", id: "qwen3.8-27b", contextLimit: 786432 });
  });

  it("reads again what is loaded now, so the page follows a switch another client made", async () => {
    saveKey("sk");
    const first = await readModel(listing("qwen3.8-27b", 786432));
    const second = await readModel(listing("flash-next", 589824));
    expect(first).toMatchObject({ id: "qwen3.8-27b" });
    expect(second).toEqual({ state: "ready", id: "flash-next", contextLimit: 589824 });
  });

  it("is an error when nothing is listed or ignis refuses", async () => {
    saveKey("sk");
    const empty = (async () => new Response(JSON.stringify({ data: [] }))) as typeof fetch;
    expect(await readModel(empty)).toEqual({ state: "error", message: "Error: GET /v1/models: no model listed" });
    const down = (async () => new Response("", { status: 503 })) as typeof fetch;
    expect(await readModel(down)).toEqual({ state: "error", message: "Error: GET /v1/models: 503" });
    const unauthorized = (async () => new Response("", { status: 401 })) as typeof fetch;
    expect((await readModel(unauthorized)).state).toBe("error");
    keyRequired();
  });
});

describe("readModel during a switch (spec playground/05)", () => {
  it("reads `switching` as the switch under way, stamped with when this page first saw it", async () => {
    saveKey("sk");
    // The entry is still the old model until the new one serves.
    const body = { data: [{ id: "qwen3.8-27b", max_model_len: 786432 }], status: "switching", switching: { from: "qwen3.8-27b", to: "qwen3.8-flash-next" } };
    expect(await readModel(answering(body), 1000)).toEqual({ state: "switching", from: "qwen3.8-27b", to: "qwen3.8-flash-next", since: 1000 });
  });

  it("reads `failed` as an error carrying the reason, marked as the server's own report", async () => {
    saveKey("sk");
    const body = { data: [{ id: "qwen3.8-27b" }], status: "failed", reason: "the artifact is not a file" };
    expect(await readModel(answering(body))).toEqual({ state: "error", message: "Switch failed: the artifact is not a file", failed: true });
  });

  it("carries how the last switch ended, on every state", async () => {
    saveKey("sk");
    const last = { from: "qwen3.8-27b", to: "qwen3.8-27b-fail", outcome: "restored", reason: "the load failed" };
    const ready = await readModel(answering({ data: [{ id: "qwen3.8-27b", max_model_len: 1 }], status: "serving", last_switch: last }));
    expect(ready).toEqual({ state: "ready", id: "qwen3.8-27b", contextLimit: 1, lastSwitch: last });
    const switching = await readModel(answering({ data: [{ id: "a" }], status: "switching", switching: { from: "a", to: "b" }, last_switch: last }), 5);
    expect(switching).toMatchObject({ state: "switching", lastSwitch: last });
    // Before the first switch the key is absent, and so is the field.
    expect(await readModel(listing("qwen3.8-27b", 1))).not.toHaveProperty("lastSwitch");
  });
});

describe("afterRefresh", () => {
  const ready: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: null };
  it("takes the model read", () => {
    const next: ModelState = { state: "ready", id: "flash-next", contextLimit: 1 };
    expect(afterRefresh(ready, next)).toBe(next);
  });
  it("keeps a model already shown when the re-read fails, as a turn must not lose it to a blip", () => {
    expect(afterRefresh(ready, { state: "error", message: "x" })).toBe(ready);
    const error: ModelState = { state: "error", message: "x" };
    expect(afterRefresh({ state: "reading" }, error)).toBe(error);
  });

  const switching: ModelState = { state: "switching", from: "a", to: "b", since: 1000 };

  it("keeps a switch shown through a failed read too", () => {
    expect(afterRefresh(switching, { state: "error", message: "x" })).toBe(switching);
  });

  it("shows a failed switch the server reports, never masking it as a blip", () => {
    const failed: ModelState = { state: "error", message: "Switch failed: r", failed: true };
    expect(afterRefresh(ready, failed)).toBe(failed);
    expect(afterRefresh(switching, failed)).toBe(failed);
  });

  it("keeps when the page first saw a switch while the same one runs, and starts over for another", () => {
    expect(afterRefresh(switching, { state: "switching", from: "a", to: "b", since: 4000 })).toEqual(switching);
    expect(afterRefresh(switching, { state: "switching", from: "b", to: "c", since: 4000 })).toEqual({ state: "switching", from: "b", to: "c", since: 4000 });
  });
});

describe("switchFollowed", () => {
  // useModel re-reads each second while this holds: a switch runs, or the
  // server reports one failed — briefly, while the old model is loaded again,
  // so the page has to see it through to `serving`.
  it("holds while switching and while the server reports a failed switch, not for a failed read", () => {
    expect(switchFollowed({ state: "switching", from: "a", to: "b", since: 0 })).toBe(true);
    expect(switchFollowed({ state: "error", message: "m", failed: true })).toBe(true);
    expect(switchFollowed({ state: "error", message: "m" })).toBe(false);
    expect(switchFollowed({ state: "ready", id: "a", contextLimit: null })).toBe(false);
    expect(switchFollowed({ state: "reading" })).toBe(false);
  });
});

describe("readModel with idle unload (spec model-switch/02)", () => {
  it("reads `unloaded` as a state naming the model and its context, never an error", async () => {
    saveKey("sk");
    const body = { data: [{ id: "qwen3.8-27b", max_model_len: 786432 }], status: "unloaded" };
    expect(await readModel(answering(body))).toEqual({ state: "unloaded", id: "qwen3.8-27b", contextLimit: 786432 });
  });

  it("reads `loading` as the load under way, naming what loads, stamped with when this page first saw it", async () => {
    saveKey("sk");
    const body = { data: [{ id: "qwen3.8-27b" }], status: "loading", switching: { from: "qwen3.8-27b", to: "qwen3.8-flash-next" } };
    expect(await readModel(answering(body), 7)).toEqual({ state: "loading", to: "qwen3.8-flash-next", since: 7 });
    // Without the ids, what loads is the model listed.
    expect(await readModel(answering({ data: [{ id: "qwen3.8-27b" }], status: "loading" }), 7)).toEqual({ state: "loading", to: "qwen3.8-27b", since: 7 });
  });

  it("lets a message go while unloaded, which loads the model, and not while one loads or switches", () => {
    expect(canSend({ state: "ready", id: "a", contextLimit: null })).toBe(true);
    expect(canSend({ state: "unloaded", id: "a", contextLimit: null })).toBe(true);
    expect(canSend({ state: "loading", to: "a", since: 0 })).toBe(false);
    expect(canSend({ state: "switching", from: "a", to: "b", since: 0 })).toBe(false);
    expect(canSend({ state: "reading" })).toBe(false);
  });

  it("says a model is being loaded while a switch or a load out of unloaded runs, and only then", () => {
    expect(loadUnderWay({ state: "switching", from: "a", to: "b", since: 0 })).toBe(true);
    expect(loadUnderWay({ state: "loading", to: "a", since: 0 })).toBe(true);
    expect(loadUnderWay({ state: "unloaded", id: "a", contextLimit: null })).toBe(false);
    expect(loadUnderWay({ state: "ready", id: "a", contextLimit: null })).toBe(false);
    expect(loadUnderWay({ state: "reading" })).toBe(false);
    expect(loadUnderWay({ state: "error", message: "m", failed: true })).toBe(false);
  });

  it("follows a load each second until it serves, and leaves an unloaded model alone", () => {
    expect(switchFollowed({ state: "loading", to: "a", since: 0 })).toBe(true);
    expect(switchFollowed({ state: "unloaded", id: "a", contextLimit: null })).toBe(false);
  });

  it("reads an unloaded model each second while a turn of the page's own is in flight, which loads it", () => {
    const unloaded: ModelState = { state: "unloaded", id: "a", contextLimit: null };
    expect(loadAwaited(unloaded, true)).toBe(true);
    expect(loadAwaited(unloaded, false)).toBe(false);
    // The read that says `loading` hands the load to switchFollowed; the one that says `ready` ends both.
    const loading: ModelState = { state: "loading", to: "a", since: 0 };
    expect(loadAwaited(loading, true) || switchFollowed(loading)).toBe(true);
    const ready: ModelState = { state: "ready", id: "a", contextLimit: null };
    expect(loadAwaited(ready, true) || switchFollowed(ready)).toBe(false);
  });

  it("keeps an unloaded model or a load shown through a failed read, and when the page first saw the load", () => {
    const unloaded: ModelState = { state: "unloaded", id: "a", contextLimit: null };
    const loading: ModelState = { state: "loading", to: "a", since: 1000 };
    expect(afterRefresh(unloaded, { state: "error", message: "x" })).toBe(unloaded);
    expect(afterRefresh(loading, { state: "error", message: "x" })).toBe(loading);
    expect(afterRefresh(loading, { state: "loading", to: "a", since: 4000 })).toEqual(loading);
    expect(afterRefresh(loading, { state: "loading", to: "b", since: 4000 })).toEqual({ state: "loading", to: "b", since: 4000 });
  });
});

describe("the model's kind (spec image-model/05)", () => {
  const imageListing = answering({ data: [{ id: "ignis-image-2.1-turbo", kind: "image" }] });

  it("reads kind, and no context limit for the image model", async () => {
    saveKey("sk");
    expect(await readModel(imageListing)).toEqual({ state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" });
    expect(await readModel(answering({ data: [{ id: "qwen3.8-27b", kind: "text", max_model_len: 5 }] }))).toEqual({
      state: "ready",
      id: "qwen3.8-27b",
      contextLimit: 5,
      kind: "text",
    });
  });

  it("keeps the kind through an unloaded read, and ignores one it does not know", async () => {
    saveKey("sk");
    expect(await readModel(answering({ status: "unloaded", data: [{ id: "x", kind: "image" }] }))).toMatchObject({ state: "unloaded", kind: "image" });
    expect(await readModel(answering({ data: [{ id: "x", kind: "audio" }] }))).not.toHaveProperty("kind");
  });

  it("lets the chat use any model but an image one, and the image view use only that", () => {
    const text: ModelState = { state: "ready", id: "a", contextLimit: 1, kind: "text" };
    const old: ModelState = { state: "ready", id: "a", contextLimit: 1 };
    const image: ModelState = { state: "ready", id: "b", contextLimit: null, kind: "image" };
    expect([canChat(text), canChat(old), canChat(image)]).toEqual([true, true, false]);
    expect([imageLoaded(text), imageLoaded(old), imageLoaded(image)]).toEqual([false, false, true]);
    expect(canSend(image)).toBe(true);
    expect(imageLoaded({ state: "unloaded", id: "b", contextLimit: null, kind: "image" })).toBe(true);
    expect(canChat({ state: "reading" })).toBe(false);
  });
});
