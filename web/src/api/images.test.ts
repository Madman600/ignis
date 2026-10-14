import { describe, expect, it } from "vitest";
import { EDITS_PATH, editForm, generate, generateStreaming, GENERATIONS_PATH, generationBody, type ImageParams, type ImageProgress } from "./images.ts";

// The image routes' client (spec image-model/05 §Playground): the request
// shapes, the answer, the stream's events and the refusals, over a mocked
// fetch. No request carries a `model` but one whose run names the image
// model (GitHub #324).

const params: ImageParams = { prompt: "a lantern", n: 2, size: "1024x1024", seed: 7, enhance: false, images: [] };
const png = (name: string) => new File([new Uint8Array([137, 80, 78, 71])], name, { type: "image/png" });

type Call = { path: string; init: RequestInit };

function mock(respond: () => Response) {
  const calls: Call[] = [];
  const fetcher = (async (path: string, init: RequestInit) => {
    calls.push({ path, init });
    return respond();
  }) as unknown as typeof fetch;
  return { calls, fetcher };
}

const sse = (events: [string, object][]) =>
  new Response(events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join(""), {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });

const json = (body: object, status = 200, headers: Record<string, string> = {}) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json", ...headers } });

describe("the request bodies", () => {
  it("writes a generation as JSON with no model, the seed only when given, and stream only when asked", () => {
    expect(generationBody(params, false)).toEqual({ prompt: "a lantern", n: 2, size: "1024x1024", seed: 7, prompt_enhancement: false });
    expect(generationBody({ ...params, seed: null, enhance: true }, true)).toEqual({
      prompt: "a lantern",
      n: 2,
      size: "1024x1024",
      prompt_enhancement: true,
      stream: true,
    });
  });

  it("writes an edit as multipart: one image[] part per condition image, the rest as text, no model", () => {
    const form = editForm({ ...params, images: [png("a.png"), png("b.png")] }, true);
    expect(form.getAll("image[]").map((f) => (f as File).name)).toEqual(["a.png", "b.png"]);
    expect(form.get("prompt")).toBe("a lantern");
    expect(form.get("n")).toBe("2");
    expect(form.get("seed")).toBe("7");
    expect(form.get("prompt_enhancement")).toBe("false");
    expect(form.get("stream")).toBe("true");
    expect(form.has("model")).toBe(false);
  });

  it("names the image model when the run does (GitHub #324), in either shape", () => {
    const named = { ...params, model: "ignis-image-2.1-turbo" };
    expect(generationBody(named, true)).toMatchObject({ model: "ignis-image-2.1-turbo", prompt: "a lantern", stream: true });
    expect(editForm({ ...named, images: [png("a.png")] }, false).get("model")).toBe("ignis-image-2.1-turbo");
  });
});

describe("generate (answered whole)", () => {
  it("posts JSON to the generations route and reads the pictures, the revised prompt, the warnings and the seed", async () => {
    const { calls, fetcher } = mock(() =>
      json({
        created: 1,
        data: [{ b64_json: "AAA", revised_prompt: "a brass lantern" }, { b64_json: "BBB" }],
        seed: 7,
        size: "1024x1024",
        warnings: ["the enhancer could not be had"],
      }),
    );
    const outcome = await generate(params, undefined, fetcher);
    expect(calls).toHaveLength(1);
    expect(calls[0].path).toBe(GENERATIONS_PATH);
    expect((calls[0].init.headers as Record<string, string>)["Content-Type"]).toBe("application/json");
    expect(JSON.parse(calls[0].init.body as string)).not.toHaveProperty("model");
    expect(outcome).toEqual({
      ok: true,
      images: [
        { b64: "AAA", index: 0, seed: 7, size: "1024x1024", revisedPrompt: "a brass lantern" },
        { b64: "BBB", index: 1, seed: 7, size: "1024x1024" },
      ],
      extras: { seed: 7, size: "1024x1024", warnings: ["the enhancer could not be had"], conditionSizes: [] },
    });
  });

  it("sends an edit to the edits route as multipart, the browser setting the boundary", async () => {
    const { calls, fetcher } = mock(() => json({ data: [{ b64_json: "AAA" }], condition_sizes: ["1024x768"] }));
    const outcome = await generate({ ...params, images: [png("a.png")] }, undefined, fetcher);
    expect(calls[0].path).toBe(EDITS_PATH);
    expect(calls[0].init.body).toBeInstanceOf(FormData);
    expect((calls[0].init.headers as Record<string, string>)["Content-Type"]).toBeUndefined();
    expect(outcome.ok && outcome.extras.conditionSizes).toEqual(["1024x768"]);
  });

  it("reads a 409 model_kind_mismatch", async () => {
    const { fetcher } = mock(() =>
      json({ error: { message: "this route serves image models (loaded: qwen3.8-27b, a text model)", code: "model_kind_mismatch" } }, 409),
    );
    const outcome = await generate(params, undefined, fetcher);
    expect(outcome).toMatchObject({ ok: false, status: 409, code: "model_kind_mismatch", message: expect.stringContaining("qwen3.8-27b") });
  });

  it("reads a 503 with its Retry-After", async () => {
    const { fetcher } = mock(() => json({ error: { message: "the queue is full", code: "engine_full" } }, 503, { "Retry-After": "1" }));
    const outcome = await generate(params, undefined, fetcher);
    expect(outcome).toMatchObject({ ok: false, status: 503, code: "engine_full", retryAfter: 1 });
  });

  it("says when ignis cannot be reached", async () => {
    const fetcher = (async () => {
      throw new TypeError("failed to fetch");
    }) as unknown as typeof fetch;
    expect(await generate(params, undefined, fetcher)).toMatchObject({ ok: false, status: 0, message: expect.stringContaining("Could not reach ignis") });
  });
});

describe("generateStreaming", () => {
  const events: [string, object][] = [
    ["image_generation.progress", { type: "image_generation.progress", phase: "enhancing", image: 0, step: 1, total: 1 }],
    ["image_generation.progress", { type: "image_generation.progress", phase: "encoding", image: 0, step: 1, total: 1 }],
    ["image_generation.progress", { type: "image_generation.progress", phase: "denoising", image: 0, step: 3, total: 8 }],
    [
      "image_generation.completed",
      { type: "image_generation.completed", image: 0, b64_json: "AAA", seed: 7, size: "1024x1024", revised_prompt: "a brass lantern", warnings: ["w"] },
    ],
    ["image_generation.completed", { type: "image_generation.completed", image: 1, b64_json: "BBB", seed: 7, size: "1024x1024" }],
  ];

  it("posts stream: true and hands over each phase, step and finished picture as it comes", async () => {
    const { calls, fetcher } = mock(() => sse(events));
    const progress: ImageProgress[] = [];
    const seen: string[] = [];
    const outcome = await generateStreaming(params, { onProgress: (p) => progress.push(p), onImage: (i) => seen.push(i.b64) }, undefined, fetcher);
    const body = JSON.parse(calls[0].init.body as string);
    expect(body).toMatchObject({ stream: true });
    expect(body).not.toHaveProperty("model");
    expect(progress.map((p) => `${p.phase} ${p.step}/${p.total}`)).toEqual(["enhancing 1/1", "encoding 1/1", "denoising 3/8"]);
    expect(seen).toEqual(["AAA", "BBB"]);
    expect(outcome).toMatchObject({ ok: true, extras: { seed: 7, warnings: ["w"] } });
    expect(outcome.ok && outcome.images[0].revisedPrompt).toBe("a brass lantern");
  });

  it("reads the edit stream's image_edit.* events from the edits route", async () => {
    const { calls, fetcher } = mock(() =>
      sse([
        ["image_edit.progress", { type: "image_edit.progress", phase: "denoising", image: 0, step: 1, total: 4 }],
        ["image_edit.completed", { type: "image_edit.completed", image: 0, b64_json: "AAA", seed: 3, size: "1024x1024" }],
      ]),
    );
    const progress: ImageProgress[] = [];
    const outcome = await generateStreaming(
      { ...params, images: [png("a.png"), png("b.png")] },
      { onProgress: (p) => progress.push(p), onImage: () => {} },
      undefined,
      fetcher,
    );
    expect(calls[0].path).toBe(EDITS_PATH);
    expect((calls[0].init.body as FormData).get("stream")).toBe("true");
    expect(progress).toHaveLength(1);
    expect(outcome.ok && outcome.images).toHaveLength(1);
  });

  it("ends in an error event with the pictures made before it", async () => {
    const { fetcher } = mock(() => sse([events[3], ["error", { type: "error", error: { message: "the model was stopped", code: "model_stopped" } }]]));
    const outcome = await generateStreaming(params, { onProgress: () => {}, onImage: () => {} }, undefined, fetcher);
    expect(outcome).toMatchObject({ ok: false, code: "model_stopped", message: "the model was stopped" });
    expect(!outcome.ok && outcome.partial.map((i) => i.b64)).toEqual(["AAA"]);
  });

  it("reads a refusal before the stream began as a status", async () => {
    const { fetcher } = mock(() => json({ error: { message: "a model switch is under way", code: "model_switching" } }, 503, { "Retry-After": "2" }));
    const outcome = await generateStreaming(params, { onProgress: () => {}, onImage: () => {} }, undefined, fetcher);
    expect(outcome).toMatchObject({ ok: false, status: 503, code: "model_switching", retryAfter: 2 });
  });
});
