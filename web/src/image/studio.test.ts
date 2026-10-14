import { describe, expect, it } from "vitest";
import type { ImageParams } from "../api/images.ts";
import type { LastSwitch, ModelState } from "../api/model.ts";
import type { CatalogRow } from "../server/routes.ts";
import {
  DEFAULT_LIMITS,
  downloadImage,
  downloadName,
  type GalleryItem,
  imageModelRow,
  limitsOf,
  parseSeed,
  performRun,
  progressLine,
  progressShare,
  refusalLine,
  type RunSink,
  settleSwitch,
  switchToImageModel,
} from "./studio.ts";

// The image view's behaviour over a mocked server (spec image-model/05 AC 12):
// it generates (JSON and streamed) and edits (multipart), shows progress, the
// switch button switches, the gallery downloads, and no request carries a
// `model` unless the button was pressed.

const params: ImageParams = { prompt: "a lantern", n: 2, size: "auto", seed: null, enhance: true, images: [] };
const png = (name: string) => new File([new Uint8Array([1])], name, { type: "image/png" });

type Call = { path: string; init: RequestInit };

/** A server that records each request and answers by path. */
function server(answers: Record<string, () => Response>) {
  const calls: Call[] = [];
  const fetcher = (async (path: string, init: RequestInit) => {
    calls.push({ path, init });
    const answer = answers[path];
    if (!answer) throw new Error(`unexpected request to ${path}`);
    return answer();
  }) as unknown as typeof fetch;
  return { calls, fetcher };
}

const json = (body: object, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
const sse = (events: object[]) =>
  new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(""), { status: 200, headers: { "Content-Type": "text/event-stream" } });

/** A sink that keeps what it was told. */
function sink() {
  const log = { progress: [] as string[], items: [] as GalleryItem[], end: null as Parameters<RunSink["done"]>[0] | null };
  const s: RunSink = {
    progress: (p) => log.progress.push(progressLine(p, 2)),
    image: (item) => log.items.push(item),
    done: (end) => {
      log.end = end;
    },
  };
  return { log, s };
}
let ids = 0;
const nextId = () => ++ids;

/** Whether a request body carries a `model`, JSON or multipart. */
function names(call: Call): boolean {
  const body = call.init.body;
  if (body instanceof FormData) return body.has("model");
  return typeof body === "string" && "model" in JSON.parse(body);
}

describe("performRun", () => {
  it("generates as JSON: the pictures land in the gallery with the revised prompt, size and seed", async () => {
    const { calls, fetcher } = server({
      "/v1/images/generations": () =>
        json({ data: [{ b64_json: "AAA", revised_prompt: "a brass lantern" }, { b64_json: "BBB" }], seed: 10, size: "2048x2048", warnings: ["w"] }),
    });
    const { log, s } = sink();
    await performRun(params, false, s, nextId, undefined, fetcher);
    expect(calls.map((c) => c.path)).toEqual(["/v1/images/generations"]);
    expect(log.items.map((i) => [i.url, i.seed, i.size, i.revisedPrompt, i.edit])).toEqual([
      ["data:image/png;base64,AAA", 10, "2048x2048", "a brass lantern", false],
      ["data:image/png;base64,BBB", 11, "2048x2048", undefined, false],
    ]);
    expect(log.end).toEqual({ notes: ["w"], error: null, reread: false });
  });

  it("generates streamed: progress per phase and step arrives with the pictures", async () => {
    const { fetcher } = server({
      "/v1/images/generations": () =>
        sse([
          { type: "image_generation.progress", phase: "encoding", image: 0, step: 1, total: 1 },
          { type: "image_generation.progress", phase: "denoising", image: 1, step: 5, total: 20 },
          { type: "image_generation.completed", image: 0, b64_json: "AAA", seed: 4, size: "1024x1024" },
        ]),
    });
    const { log, s } = sink();
    await performRun(params, true, s, nextId, undefined, fetcher);
    expect(log.progress).toEqual(["Encoding the prompt", "Denoising · image 2 of 2 · step 5/20"]);
    expect(log.items).toHaveLength(1);
    expect(log.end?.error).toBeNull();
  });

  it("edits with condition images as multipart, and notes the sizes the budget scaled them to", async () => {
    const { calls, fetcher } = server({
      "/v1/images/edits": () => json({ data: [{ b64_json: "AAA" }], seed: 1, size: "1024x1024", condition_sizes: ["512x512", "768x512"] }),
    });
    const { log, s } = sink();
    await performRun({ ...params, images: [png("a.png"), png("b.png")] }, false, s, nextId, undefined, fetcher);
    expect(calls.map((c) => c.path)).toEqual(["/v1/images/edits"]);
    expect((calls[0].init.body as FormData).getAll("image[]")).toHaveLength(2);
    expect(log.items[0].edit).toBe(true);
    expect(log.end?.notes).toEqual(["Condition images used at 512x512, 768x512."]);
  });

  it("names no model in any request", async () => {
    const { calls, fetcher } = server({
      "/v1/images/generations": () => json({ data: [{ b64_json: "A" }] }),
      "/v1/images/edits": () => sse([{ type: "image_edit.completed", image: 0, b64_json: "A" }]),
    });
    await performRun(params, false, sink().s, nextId, undefined, fetcher);
    await performRun(params, true, sink().s, nextId, undefined, fetcher);
    await performRun({ ...params, images: [png("a.png")] }, true, sink().s, nextId, undefined, fetcher);
    await performRun({ ...params, images: [png("a.png")] }, false, sink().s, nextId, undefined, fetcher);
    expect(calls).toHaveLength(4);
    expect(calls.some(names)).toBe(false);
  });

  it("shows a 409 with the way out and asks for the model to be read again", async () => {
    const { fetcher } = server({
      "/v1/images/generations": () => json({ error: { message: "this route serves image models (loaded: qwen3.8-27b)", code: "model_kind_mismatch" } }, 409),
    });
    const { log, s } = sink();
    await performRun(params, false, s, nextId, undefined, fetcher);
    expect(log.end?.error).toContain("loaded: qwen3.8-27b");
    expect(log.end?.error).toContain("Switch to the image model");
    expect(log.end?.reread).toBe(true);
    expect(log.items).toEqual([]);
  });

  it("shows a 503 with its Retry-After, and an error event in a stream", async () => {
    const busy = () =>
      new Response(JSON.stringify({ error: { message: "the queue is full", code: "engine_full" } }), { status: 503, headers: { "Retry-After": "1" } });
    const full = server({ "/v1/images/generations": busy });
    const first = sink();
    await performRun(params, true, first.s, nextId, undefined, full.fetcher);
    expect(first.log.end?.error).toBe("The queue is full: the queue is full Try again in 1 s.");
    expect(first.log.end?.reread).toBe(true);

    const broken = server({
      "/v1/images/generations": () => sse([{ type: "error", error: { message: "the model was stopped", code: "model_stopped" } }]),
    });
    const second = sink();
    await performRun(params, true, second.s, nextId, undefined, broken.fetcher);
    expect(second.log.end).toMatchObject({ error: "the model was stopped", reread: false });
  });

  it("ends a stopped run without an error", async () => {
    const controller = new AbortController();
    const fetcher = (async () => {
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    }) as unknown as typeof fetch;
    const { log, s } = sink();
    await performRun(params, false, s, nextId, controller.signal, fetcher);
    expect(log.end?.error).toBeNull();
  });
});

describe("the switch to the image model", () => {
  const catalog: CatalogRow[] = [
    { id: "qwen3.8-27b", source: "built-in", on_disk: "complete", family: "qwen38", loaded: true, switchable: true },
    { id: "ignis-image-2.1-turbo", source: "built-in", on_disk: "complete", family: "qwenimage21", loaded: false, switchable: true },
  ];

  it("finds the image model's row", () => {
    expect(imageModelRow(catalog)?.id).toBe("ignis-image-2.1-turbo");
    expect(imageModelRow([{ ...catalog[1], family: undefined, on_disk: "absent" }])?.id).toBe("ignis-image-2.1-turbo");
    expect(imageModelRow(catalog.slice(0, 1))).toBeNull();
    expect(imageModelRow(null)).toBeNull();
  });

  it("switches by id through the Server view's route, and that is the one request that names a model", async () => {
    const { calls, fetcher } = server({ "/v1/models/switch": () => json({ from: "qwen3.8-27b", to: "ignis-image-2.1-turbo" }, 202) });
    const answer = await switchToImageModel(catalog, fetcher);
    expect(answer).toMatchObject({ ok: true });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toMatchObject({ path: "/v1/models/switch", init: { method: "POST" } });
    expect(JSON.parse(calls[0].init.body as string)).toEqual({ model: "ignis-image-2.1-turbo" });
  });

  it("sends nothing when the server lists no image model", async () => {
    const { calls, fetcher } = server({});
    expect(await switchToImageModel(catalog.slice(0, 1), fetcher)).toBeNull();
    expect(calls).toEqual([]);
  });

  const earlier: LastSwitch = { from: "ignis-image-2.1-turbo", to: "qwen3.8-27b", outcome: "switched" };
  const text = (lastSwitch?: LastSwitch): ModelState => ({ state: "ready", id: "qwen3.8-27b", contextLimit: 786432, kind: "text", ...(lastSwitch ? { lastSwitch } : {}) });

  it("follows the switch until the image model serves", () => {
    expect(settleSwitch(earlier, text(earlier))).toEqual({ kind: "wait" });
    expect(settleSwitch(earlier, { state: "switching", from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", since: 0, lastSwitch: earlier })).toEqual({ kind: "wait" });
    const served: ModelState = { state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" };
    expect(settleSwitch(earlier, { ...served, lastSwitch: { from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", outcome: "switched" } })).toEqual({ kind: "done", line: null });
    expect(settleSwitch(earlier, { state: "error", message: "GET /v1/models: 502" })).toEqual({ kind: "done", line: null });
  });

  it("ends on a switch refused at prepare, which only the record shows: the text model never stopped serving", () => {
    const refused: LastSwitch = { from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", outcome: "refused", reason: "the checksum is not clean" };
    expect(settleSwitch(undefined, text(refused))).toEqual({
      kind: "done",
      line: "The switch to ignis-image-2.1-turbo was refused, qwen3.8-27b kept serving: the checksum is not clean",
    });
    expect(settleSwitch(earlier, text(refused))).toMatchObject({ kind: "done" });
  });

  it("waits through the reload of the text model after a failed load, and ends when nothing serves", () => {
    const failing: ModelState = { state: "error", message: "Switch failed: out of memory", failed: true, lastSwitch: earlier };
    expect(settleSwitch(earlier, failing)).toEqual({ kind: "wait" });
    const failed: LastSwitch = { from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", outcome: "failed", reason: "out of memory" };
    expect(settleSwitch(earlier, { ...failing, lastSwitch: failed })).toEqual({
      kind: "done",
      line: "The switch to ignis-image-2.1-turbo failed and no model is serving: out of memory",
    });
    const restored: LastSwitch = { ...failed, outcome: "restored" };
    expect(settleSwitch(earlier, text(restored))).toEqual({
      kind: "done",
      line: "The switch to ignis-image-2.1-turbo failed, qwen3.8-27b was loaded again: out of memory",
    });
  });
});

describe("the gallery's download", () => {
  const item: GalleryItem = { id: 3, url: "data:image/png;base64,AAA", prompt: "a lantern", size: "1024x1024", seed: 12, edit: false };

  it("saves a PNG named for the seed and the size, from the picture's data URL", () => {
    const made: { href?: string; download?: string; clicked: boolean }[] = [];
    const doc = {
      createElement: () => {
        const anchor = {
          clicked: false,
          click() {
            this.clicked = true;
          },
        };
        made.push(anchor);
        return anchor;
      },
    } as unknown as Document;
    downloadImage(item, doc);
    expect(made).toHaveLength(1);
    expect(made[0]).toMatchObject({ href: "data:image/png;base64,AAA", download: "ignis-12-1024x1024.png", clicked: true });
  });

  it("names a picture without a seed by its place in the gallery", () => {
    expect(downloadName({ ...item, seed: undefined })).toBe("ignis-image-3-1024x1024.png");
  });
});

describe("the form's limits and words", () => {
  it("offers the server's sizes, counts and enhancement setting, and its defaults while unread", () => {
    expect(limitsOf({ image: { sizes: ["1024x1024", "2048x2048"], max_n: 3, max_condition_images: 4, prompt_enhancement: true } })).toEqual({
      sizes: ["1024x1024", "2048x2048"],
      maxN: 3,
      maxImages: 4,
      enhance: true,
    });
    expect(limitsOf(null).sizes).toContain("2752x1536");
    expect(limitsOf(null).maxImages).toBe(10);
  });

  it("falls back to the server's own defaults for image.* while the configuration is unread", () => {
    expect(DEFAULT_LIMITS).toEqual({
      sizes: ["2048x2048", "2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752", "1024x1024"],
      maxN: 10,
      maxImages: 10,
      enhance: false,
    });
  });

  it("reads a seed: blank is the server's draw, a whole number is itself, anything else is refused", () => {
    expect([parseSeed(""), parseSeed(" 42 "), parseSeed("-1"), parseSeed("1.5"), parseSeed("x")]).toEqual([null, 42, undefined, undefined, undefined]);
  });

  it("says the phase, the image and the step", () => {
    expect(progressLine({ phase: "denoising", image: 0, step: 3, total: 30 }, 1)).toBe("Denoising · step 3/30");
    expect(progressLine({ phase: "decoding", image: 1, step: 1, total: 1 }, 3)).toBe("Decoding · image 2 of 3");
    expect(progressLine({ phase: "loading", image: 0, step: 1, total: 2 }, 1)).toBe("Loading the model · 1/2");
    expect(progressShare({ phase: "denoising", image: 1, step: 11, total: 20 }, 2)).toBeCloseTo(0.75);
    expect(progressShare({ phase: "encoding", image: 0, step: 1, total: 1 }, 2)).toBe(0);
  });

  it("reads each refusal", () => {
    const base = { ok: false as const, partial: [], extras: { warnings: [], conditionSizes: [] } };
    expect(refusalLine({ ...base, status: 503, code: "model_switching", message: "switching", retryAfter: 1 })).toBe(
      "A model switch is under way: switching Try again in 1 s.",
    );
    expect(refusalLine({ ...base, status: 400, message: "size must be one of 1024x1024" })).toBe("size must be one of 1024x1024");
  });
});
