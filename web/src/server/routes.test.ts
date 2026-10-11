import { describe, expect, it } from "vitest";
import { getAuth, saveKey } from "../api/auth.ts";
import { patchConfig, readCatalog, readFields, refusalOf, switchTo } from "./routes.ts";

// The Server view's requests (spec playground/05): what each sends, and how
// each answer is read.

type Sent = { url: string; init: RequestInit };

/** A fetch that records what was sent and answers `status` with `body`. */
function answering(status: number, body: unknown, sent: Sent[] = []): typeof fetch {
  return (async (url: string, init: RequestInit) => {
    sent.push({ url, init });
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  }) as typeof fetch;
}

describe("switchTo", () => {
  it("sends POST /v1/models/switch with the id alone, never a path, and the key", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const answer = await switchTo("qwen3.8-flash-next", answering(202, { status: "switching", from: "a", to: "qwen3.8-flash-next" }, sent));
    expect(answer).toEqual({ ok: true, status: 202, value: { status: "switching", from: "a", to: "qwen3.8-flash-next" } });
    expect(sent[0].url).toBe("/v1/models/switch");
    expect(sent[0].init.method).toBe("POST");
    expect(JSON.parse(sent[0].init.body as string)).toEqual({ model: "qwen3.8-flash-next" });
    expect(sent[0].init.headers).toMatchObject({ Authorization: "Bearer sk-test", "Content-Type": "application/json" });
  });

  it("reads a refusal as the server's message and code", async () => {
    saveKey("sk-test");
    const body = { error: { message: "model x is in the catalog but not on disk", type: "model_not_found", code: "model_not_found" } };
    expect(await switchTo("x", answering(404, body))).toEqual({ ok: false, status: 404, message: "model x is in the catalog but not on disk", code: "model_not_found" });
  });
});

describe("patchConfig", () => {
  it("sends PATCH /v1/config with the body as given", async () => {
    saveKey("sk-test");
    const sent: Sent[] = [];
    const body = { model: { qwen38: { max_context: "524288" } } };
    const answer = await patchConfig(body, answering(202, { applied: "reload", persisted: false }, sent));
    expect(answer).toMatchObject({ ok: true, status: 202, value: { applied: "reload" } });
    expect(sent[0]).toMatchObject({ url: "/v1/config", init: { method: "PATCH" } });
    expect(JSON.parse(sent[0].init.body as string)).toEqual(body);
  });
});

describe("the list routes", () => {
  it("read the `data` of the list envelope", async () => {
    saveKey("sk-test");
    const fields = await readFields(answering(200, { object: "list", data: [{ file_key: "server.bind" }] }));
    expect(fields).toMatchObject({ ok: true, value: [{ file_key: "server.bind" }] });
    const catalog = await readCatalog(answering(200, { object: "list", data: [{ id: "a", loaded: true }] }));
    expect(catalog).toMatchObject({ ok: true, value: [{ id: "a", loaded: true }] });
  });

  it("show the key prompt on a 401, and say when ignis cannot be reached", async () => {
    saveKey("sk-test");
    const refused = await readFields(answering(401, { error: { message: "invalid api key", code: "invalid_api_key" } }));
    expect(refused).toMatchObject({ ok: false, status: 401 });
    expect(getAuth().needsKey).toBe(true);
    saveKey("sk-test");
    const down = await readCatalog((async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch);
    expect(down).toEqual({ ok: false, status: 0, message: "Could not reach ignis: TypeError: fetch failed" });
  });
});

describe("refusalOf", () => {
  it("is the error's message and code, else the raw body, else the status", () => {
    expect(refusalOf(400, JSON.stringify({ error: { message: "m", code: "c" } }))).toEqual({ status: 400, message: "m", code: "c" });
    expect(refusalOf(502, "Bad gateway")).toEqual({ status: 502, message: "Bad gateway" });
    expect(refusalOf(503, "")).toEqual({ status: 503, message: "503" });
  });
});
