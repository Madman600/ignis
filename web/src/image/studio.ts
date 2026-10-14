import { generate, generateStreaming, type ImageOutcome, type ImageParams, type ImageProgress, isEdit, type MadeImage, STOPPED } from "../api/images.ts";
import { type CatalogRow, type ConfigDocument, switchTo } from "../server/routes.ts";

// The image view's pure parts (spec image-model/05 §Playground, GitHub #322):
// what the server's configuration allows, which catalog row is the image
// model, how progress and refusals read, and the gallery's download. The view
// and its hook only compose these.

/** `auto` first, then the server's `image.sizes`: the form offers exactly what the server will accept. */
export const AUTO_SIZE = "auto";

/** What the form may ask for, from the running configuration; the server's own defaults while it is unread. */
export type Limits = { sizes: string[]; maxN: number; maxImages: number; enhance: boolean };

export const DEFAULT_LIMITS: Limits = {
  sizes: ["2048x2048", "2400x1792", "1792x2400", "2528x1696", "1696x2528", "2752x1536", "1536x2752", "1024x1024"],
  maxN: 4,
  maxImages: 10,
  enhance: false,
};

export function limitsOf(config: ConfigDocument | null): Limits {
  const image = config?.image ?? {};
  const sizes = Array.isArray(image.sizes) ? image.sizes.filter((s): s is string => typeof s === "string") : [];
  const count = (value: unknown, fallback: number) => (typeof value === "number" && Number.isInteger(value) && value >= 1 ? value : fallback);
  return {
    sizes: sizes.length > 0 ? sizes : DEFAULT_LIMITS.sizes,
    maxN: count(image.max_n, DEFAULT_LIMITS.maxN),
    maxImages: count(image.max_condition_images, DEFAULT_LIMITS.maxImages),
    enhance: image.prompt_enhancement === true,
  };
}

/** The catalog row the switch button loads: the image family's, whether it is on disk or not (the server refuses what it cannot load). */
export function imageModelRow(catalog: CatalogRow[] | null): CatalogRow | null {
  const rows = (catalog ?? []).filter((row) => row.family === "qwenimage21" || /^ignis-image/.test(row.id));
  return rows.find((row) => row.switchable) ?? rows[0] ?? null;
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
    default:
      return outcome.status === 503 ? `${outcome.message}${wait}` : outcome.message;
  }
}

/** One picture in the gallery. */
export type GalleryItem = {
  id: number;
  b64: string;
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
    b64: image.b64,
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
  anchor.href = dataUrl(item.b64);
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
 * condition images are given, and it names no model: whichever model is loaded answers, or says with a 409 that it
 * is the wrong kind.
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
    sink.done({ notes: [...outcome.extras.warnings, ...scaled], error: null, reread: false });
    return;
  }
  const stopped = outcome.status === 0 && outcome.message === STOPPED;
  sink.done({
    notes: outcome.extras.warnings,
    error: stopped ? null : refusalLine(outcome),
    // A 409 or a 503 may mean the model changed under the page.
    reread: outcome.status === 409 || outcome.status === 503,
  });
}

/**
 * The owner's "Switch to the image model": the Server view's switch by id. This is the one request of the view
 * that carries a `model`, and only the button sends it.
 */
export function switchToImageModel(catalog: CatalogRow[] | null, fetcher: typeof fetch = fetch) {
  const row = imageModelRow(catalog);
  return row ? switchTo(row.id, fetcher) : Promise.resolve(null);
}
