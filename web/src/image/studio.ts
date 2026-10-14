import { generate, generateStreaming, type ImageOutcome, type ImageParams, type ImageProgress, isEdit, type MadeImage, STOPPED } from "../api/images.ts";
import { canSend, imageLoaded, type LastSwitch, type ModelState } from "../api/model.ts";
import { lastSwitchLine } from "../server/outcome.ts";
import { type CatalogRow, type ConfigDocument, switchTo } from "../server/routes.ts";

// The image view's pure parts (spec image-model/05 §Playground, GitHub #322):
// what the server's configuration allows, which catalog row is the image
// model, what Generate does with the model loaded now (GitHub #324), how
// progress and refusals read, and the gallery's download. The view and its
// hook only compose these.

/** `auto` first, then the server's `image.sizes`: the form offers exactly what the server will accept. */
export const AUTO_SIZE = "auto";

/**
 * What the form may ask for, from the running configuration; the server's own defaults while it is unread. `enhancer`
 * (`image.prompt_enhancer`) and `allowImplicit` (`switch.allow_implicit`) say what Generate does with a text model loaded.
 */
export type Limits = { sizes: string[]; maxN: number; maxImages: number; enhance: boolean; enhancer: string; allowImplicit: boolean };

export const DEFAULT_LIMITS: Limits = {
  sizes: ["2048x2048", "2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752", "1024x1024"],
  maxN: 10,
  maxImages: 10,
  enhance: false,
  enhancer: "qwen3.8-27b",
  allowImplicit: true,
};

export function limitsOf(config: ConfigDocument | null): Limits {
  const image = config?.image ?? {};
  const enhancer = image.prompt_enhancer;
  const sizes = Array.isArray(image.sizes) ? image.sizes.filter((s): s is string => typeof s === "string") : [];
  const count = (value: unknown, fallback: number) => (typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : fallback);
  return {
    sizes: sizes.length > 0 ? sizes : DEFAULT_LIMITS.sizes,
    maxN: count(image.max_n, DEFAULT_LIMITS.maxN),
    maxImages: count(image.max_condition_images, DEFAULT_LIMITS.maxImages),
    enhance: image.prompt_enhancement === true,
    enhancer: typeof enhancer === "string" && enhancer !== "" ? enhancer : DEFAULT_LIMITS.enhancer,
    allowImplicit: config?.switch?.allow_implicit !== false,
  };
}

/** The catalog row the switch button loads: the image family's, whether it is on disk or not (the server refuses what it cannot load). */
export function imageModelRow(catalog: CatalogRow[] | null): CatalogRow | null {
  const rows = (catalog ?? []).filter((row) => row.family === "qwenimage21" || /^ignis-image/.test(row.id));
  return rows.find((row) => row.switchable) ?? rows[0] ?? null;
}

/**
 * What pressing Generate or Edit does with the model loaded now (GitHub #324): the button's words, a line on what
 * happens first or why it cannot, and whether it can. With a text model loaded the press is the owner's choice of the
 * image model: the request names it (`model`), and the server switches — after the rewrite, when enhancement is on and
 * the enhancer is the model loaded (spec image-model/05 §Prompt enhancement). `running`: the request is under way, and
 * the model reads switching or loading because of it, which is not a missing image model.
 */
export type RunPlan = { label: string; note: string | null; model?: string; ready: boolean };

export function runPlan(model: ModelState, limits: Limits, imageModel: CatalogRow | null, enhance: boolean, edit: boolean, running = false): RunPlan {
  const verb = edit ? "edit" : "generate";
  const plain = edit ? "Edit" : "Generate";
  if (imageLoaded(model)) return { label: plain, note: null, ready: true };
  if (!canSend(model) || !imageModel) return { label: plain, note: running ? null : "Needs the image model.", ready: false };
  if (!limits.allowImplicit) {
    return { label: plain, note: "switch.allow_implicit is off, so a request does not switch: switch to the image model first.", ready: false };
  }
  const switching = { model: imageModel.id, ready: true };
  if (enhance && model.id === limits.enhancer) {
    const note =
      model.state === "ready"
        ? `${model.id} rewrites the prompt, then the image model replaces it.`
        : `Loads ${model.id} to rewrite the prompt, then the image model replaces it.`;
    return { label: `Enhance with ${model.id}, then switch & ${verb}`, note, ...switching };
  }
  // Two loads whichever text model is there: the enhancer for the rewrite, then the image model.
  const note = enhance
    ? model.state === "ready"
      ? `Switches to ${limits.enhancer} to rewrite the prompt, then to the image model.`
      : `Loads ${limits.enhancer} to rewrite the prompt, then switches to the image model.`
    : model.state === "ready"
      ? `Unloads ${model.id} and loads the image model first.`
      : "Loads the image model first.";
  return { label: `Switch & ${verb}`, note, ...switching };
}

const PHASE: Record<ImageProgress["phase"], string> = {
  switching: "Switching models",
  loading: "Loading the model",
  enhancing: "Rewriting the prompt",
  encoding: "Encoding the prompt",
  denoising: "Denoising",
  decoding: "Decoding",
};

export const phaseLabel = (phase: ImageProgress["phase"]) => PHASE[phase] ?? phase;

/** The progress line: the phase, and for the per-image phases which image and, while denoising, which step. */
export function progressLine(progress: ImageProgress, n: number): string {
  const label = phaseLabel(progress.phase);
  const perImage = progress.phase === "denoising" || progress.phase === "decoding";
  const which = perImage && n > 1 ? ` · image ${progress.image + 1} of ${n}` : "";
  const step = progress.phase === "denoising" && progress.total > 0 ? ` · step ${progress.step}/${progress.total}` : "";
  const unit = (progress.phase === "switching" || progress.phase === "loading") && progress.total > 1 ? ` · ${progress.step}/${progress.total}` : "";
  return `${label}${which}${step}${unit}`;
}

/** How far through the request the stream says it is, 0..1: each image weighs the same, its denoising steps most of it. */
export function progressShare(progress: ImageProgress, n: number): number {
  const images = Math.max(1, n);
  const perImage = progress.phase === "denoising" || progress.phase === "decoding";
  if (!perImage) return 0;
  const within = progress.phase === "decoding" ? 1 : progress.total > 0 ? (progress.step - 1) / progress.total : 0;
  return Math.min(1, (progress.image + Math.min(1, Math.max(0, within))) / images);
}

/** A refused or broken request in words, with what to do about it. */
export function refusalLine(outcome: Extract<ImageOutcome, { ok: false }>): string {
  const wait = outcome.retryAfter ? ` Try again in ${outcome.retryAfter} s.` : "";
  switch (outcome.code) {
    case "model_kind_mismatch":
      return `${outcome.message} Switch to the image model, then make the request again.`;
    case "engine_full":
      return `The queue is full: ${outcome.message}${wait}`;
    case "model_switching":
      return `A model switch is under way: ${outcome.message}${wait}`;
    case "model_not_found":
      return `The server would not switch for this request (${outcome.message}): switch.allow_implicit is off, or the image model is not where it looks. Switch to the image model with the button above.`;
    default:
      return outcome.status === 503 ? `${outcome.message}${wait}` : outcome.message;
  }
}

/** One picture in the gallery. */
export type GalleryItem = {
  id: number;
  /** The PNG as a data URL, built once when the picture arrives: the gallery re-renders on every progress event. */
  url: string;
  prompt: string;
  revisedPrompt?: string;
  size: string;
  /** The image's own seed (the job's seed plus its index), when the server said. */
  seed?: number;
  edit: boolean;
};

export const dataUrl = (b64: string) => `data:image/png;base64,${b64}`;

/** A made image as a gallery picture; `size` is what was asked, used when the server did not say what it made. */
export function galleryItem(image: MadeImage, meta: { id: number; prompt: string; size: string; edit: boolean }): GalleryItem {
  return {
    id: meta.id,
    url: dataUrl(image.b64),
    prompt: meta.prompt,
    ...(image.revisedPrompt ? { revisedPrompt: image.revisedPrompt } : {}),
    size: image.size ?? meta.size,
    ...(image.seed !== undefined ? { seed: image.seed + image.index } : {}),
    edit: meta.edit,
  };
}

export const downloadName = (item: GalleryItem) => `ignis-${item.seed ?? `image-${item.id}`}-${item.size}.png`;

/** Saves one gallery picture as a PNG file: the browser's own download, from the picture's data URL. */
export function downloadImage(item: GalleryItem, doc: Pick<Document, "createElement"> = document): void {
  const anchor = doc.createElement("a");
  anchor.href = item.url;
  anchor.download = downloadName(item);
  anchor.click();
}

/** The seed field's text as a seed: blank leaves it to the server; otherwise a whole number the server accepts, else `undefined`. */
export function parseSeed(text: string): number | null | undefined {
  const t = text.trim();
  if (t === "") return null;
  if (!/^\d+$/.test(t)) return undefined;
  const n = Number(t);
  return Number.isSafeInteger(n) ? n : undefined;
}

/** What a run reports as it goes; the view's hook turns each into state. */
export type RunSink = {
  progress: (progress: ImageProgress) => void;
  image: (item: GalleryItem) => void;
  /** The run ended: the lines the server added, the error if it failed (`null` for a stop), and whether the model should be read again. */
  done: (end: { notes: string[]; error: string | null; reread: boolean }) => void;
};

/**
 * One request from the form to the gallery. Streamed or whole, it goes to the generations route, or the edits route when
 * condition images are given. It names the image model only when the params do (`runPlan`'s switch); else whichever
 * model is loaded answers, or says with a 409 that it is the wrong kind.
 */
export async function performRun(
  params: ImageParams,
  stream: boolean,
  sink: RunSink,
  nextId: () => number,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<void> {
  const edit = isEdit(params);
  const add = (image: MadeImage) => sink.image(galleryItem(image, { id: nextId(), prompt: params.prompt, size: params.size, edit }));
  const outcome = stream
    ? await generateStreaming(params, { onProgress: sink.progress, onImage: add }, signal, fetcher)
    : await generate(params, signal, fetcher);
  if (outcome.ok) {
    // A whole answer arrives all at once; a stream's images are in the gallery already.
    if (!stream) outcome.images.forEach(add);
    const scaled = outcome.extras.conditionSizes.length > 0 ? [`Condition images used at ${outcome.extras.conditionSizes.join(", ")}.`] : [];
    // A request that named the image model switched to it.
    sink.done({ notes: [...outcome.extras.warnings, ...scaled], error: null, reread: params.model !== undefined });
    return;
  }
  const stopped = outcome.status === 0 && outcome.message === STOPPED;
  sink.done({
    notes: outcome.extras.warnings,
    error: stopped ? null : refusalLine(outcome),
    // A 409 or a 503 may mean the model changed under the page; a request that named a model may have switched it.
    reread: params.model !== undefined || outcome.status === 409 || outcome.status === 503,
  });
}

/**
 * The owner's "Switch to the image model": the Server view's switch by id, the switch on its own. The other requests
 * of the view that carry a `model` are Generate and Edit pressed with a text model loaded (`runPlan`).
 */
export function switchToImageModel(catalog: CatalogRow[] | null, fetcher: typeof fetch = fetch) {
  const row = imageModelRow(catalog);
  return row ? switchTo(row.id, fetcher) : Promise.resolve(null);
}

/** The record of the last switch or reload the server reports, if the page has read one. */
export const lastSwitchOf = (model: ModelState) => (model.state === "reading" ? undefined : model.lastSwitch);

const sameSwitch = (a: LastSwitch | undefined, b: LastSwitch | undefined) => JSON.stringify(a) === JSON.stringify(b);

/** Where the owner's switch stands: still to follow, or over — with what to say when it did not land. */
export type SwitchSettled = { kind: "wait" } | { kind: "done"; line: string | null };

/**
 * Where the owner's switch stands, from the model state the page reads each second and `before`, the record it
 * read when the button was pressed (as the Server view follows "Load"). Over once the image model serves, or once
 * the record changes: a switch refused at prepare shows only there, since the text model never stopped serving and
 * the `failed` it reports for one tick is missed between two reads. Through `failed` it waits while the text model
 * is loaded again, unless the record already says how it ended.
 */
export function settleSwitch(before: LastSwitch | undefined, model: ModelState): SwitchSettled {
  if (imageLoaded(model)) return { kind: "done", line: null };
  const last = lastSwitchOf(model);
  if (last !== undefined && !sameSwitch(last, before)) {
    return { kind: "done", line: last.outcome === "switched" ? null : lastSwitchLine(last).text };
  }
  if (model.state === "error" && !model.failed) return { kind: "done", line: null };
  return { kind: "wait" };
}
