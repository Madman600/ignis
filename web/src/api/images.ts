import { authHeaders, keyRequired } from "./auth.ts";
import { createSseParser } from "./sse.ts";

/** The message of a request the owner stopped. */
export const STOPPED = "Stopped.";

// The image routes the Playground's image view calls (spec image-model/05,
// GitHub #322): `POST /v1/images/generations` (JSON) and `/v1/images/edits`
// (multipart, `image[]`), answered whole or as Server-Sent Events.
//
// A request names a model only when its params say so: the image view sets
// the image model's id when the owner pressed Generate or Edit with a text
// model loaded, and the server switches to it (`switch.allow_implicit`) —
// after the rewrite, when prompt enhancement is on and the enhancer is the
// model loaded (GitHub #324). A bare request on a text model answers
// `409 model_kind_mismatch`.

export const GENERATIONS_PATH = "/v1/images/generations";
export const EDITS_PATH = "/v1/images/edits";

/** The work a request is in, as the stream announces it. */
export type ImagePhase = "switching" | "loading" | "enhancing" | "encoding" | "denoising" | "decoding";

export type ImageParams = {
  prompt: string;
  n: number;
  /** One of the server's sizes, or `auto`. */
  size: string;
  /** `null` leaves the seed to the server, which returns the one it drew. */
  seed: number | null;
  /** Sent as written: the toggle starts at what `image.prompt_enhancement` says, and the request repeats it. */
  enhance: boolean;
  /** Condition images: any makes the request an edit. */
  images: File[];
  /** The image model's id when the run is the owner's switch to it (GitHub #324); absent, the loaded model answers. */
  model?: string;
};

/** One finished image, base64 PNG. `seed` is the job's: this image started from `seed + index`. */
export type MadeImage = { b64: string; index: number; seed?: number; size?: string; revisedPrompt?: string };

export type ImageProgress = { phase: ImagePhase; image: number; step: number; total: number };

/** What came back besides the pictures. */
export type ImageExtras = { seed?: number; size?: string; warnings: string[]; conditionSizes: string[] };

export type ImageOutcome =
  | { ok: true; images: MadeImage[]; extras: ImageExtras }
  /** `retryAfter` is the seconds a 503 asked for. `partial` are the images made before an `error` event ended a stream. */
  | { ok: false; status: number; message: string; code?: string; retryAfter?: number; partial: MadeImage[]; extras: ImageExtras };

export const isEdit = (params: ImageParams) => params.images.length > 0;

/** The JSON body of a generation; a `model` only when the params name one. */
export function generationBody(params: ImageParams, stream: boolean): object {
  return {
    ...(params.model ? { model: params.model } : {}),
    prompt: params.prompt,
    n: params.n,
    size: params.size,
    ...(params.seed !== null ? { seed: params.seed } : {}),
    prompt_enhancement: params.enhance,
    ...(stream ? { stream: true } : {}),
  };
}

/** The multipart body of an edit; a `model` only when the params name one. */
export function editForm(params: ImageParams, stream: boolean): FormData {
  const form = new FormData();
  for (const image of params.images) form.append("image[]", image, image.name);
  if (params.model) form.append("model", params.model);
  form.append("prompt", params.prompt);
  form.append("n", String(params.n));
  form.append("size", params.size);
  if (params.seed !== null) form.append("seed", String(params.seed));
  form.append("prompt_enhancement", String(params.enhance));
  if (stream) form.append("stream", "true");
  return form;
}

function requestOf(params: ImageParams, stream: boolean): [string, RequestInit] {
  const headers: Record<string, string> = { ...authHeaders() };
  if (isEdit(params)) return [EDITS_PATH, { method: "POST", headers, body: editForm(params, stream) }];
  return [
    GENERATIONS_PATH,
    { method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify(generationBody(params, stream)) },
  ];
}

const empty = (): ImageExtras => ({ warnings: [], conditionSizes: [] });

type Wire = {
  data?: { b64_json?: string; revised_prompt?: string }[];
  seed?: number;
  size?: string;
  warnings?: string[];
  condition_sizes?: string[];
};

function extrasOf(body: Wire, into: ImageExtras = empty()): ImageExtras {
  return {
    seed: body.seed ?? into.seed,
    size: body.size ?? into.size,
    warnings: [...new Set([...into.warnings, ...(body.warnings ?? [])])],
    conditionSizes: body.condition_sizes ?? into.conditionSizes,
  };
}

/** A refusal's words: the server's `error.message` and `error.code`, else the body, else the status. */
async function refusal(res: Response): Promise<ImageOutcome> {
  if (res.status === 401) keyRequired();
  const text = await res.text().catch(() => "");
  let message = text.trim() || `${res.status}`;
  let code: string | undefined;
  try {
    const error = (JSON.parse(text) as { error?: { message?: unknown; code?: unknown } }).error;
    if (typeof error?.message === "string" && error.message) message = error.message;
    if (typeof error?.code === "string") code = error.code;
  } catch {
    // Not JSON: the raw text above.
  }
  const retry = Number(res.headers.get("Retry-After"));
  return { ok: false, status: res.status, message, ...(code ? { code } : {}), ...(retry > 0 ? { retryAfter: retry } : {}), partial: [], extras: empty() };
}

const unreachable = (err: unknown, partial: MadeImage[] = [], extras: ImageExtras = empty()): ImageOutcome => ({
  ok: false,
  status: 0,
  message: `Could not reach ignis: ${String(err)}`,
  partial,
  extras,
});

/** One request answered whole: the pictures arrive together, and nothing shows progress meanwhile. */
export async function generate(params: ImageParams, signal?: AbortSignal, fetcher: typeof fetch = fetch): Promise<ImageOutcome> {
  try {
    const [path, init] = requestOf(params, false);
    const res = await fetcher(path, { ...init, signal });
    if (!res.ok) return await refusal(res);
    const body = (await res.json()) as Wire;
    const images = (body.data ?? []).flatMap((d, index): MadeImage[] =>
      d.b64_json
        ? [
            {
              b64: d.b64_json,
              index,
              ...(body.seed !== undefined ? { seed: body.seed } : {}),
              ...(body.size ? { size: body.size } : {}),
              ...(d.revised_prompt ? { revisedPrompt: d.revised_prompt } : {}),
            },
          ]
        : [],
    );
    return { ok: true, images, extras: extrasOf(body) };
  } catch (err) {
    return signal?.aborted ? { ok: false, status: 0, message: STOPPED, partial: [], extras: empty() } : unreachable(err);
  }
}

export type StreamHandlers = {
  onProgress: (progress: ImageProgress) => void;
  /** Each image as it is made. */
  onImage: (image: MadeImage) => void;
};

/** One request streamed: progress per phase and denoising step, each image as it finishes. */
export async function generateStreaming(
  params: ImageParams,
  handlers: StreamHandlers,
  signal?: AbortSignal,
  fetcher: typeof fetch = fetch,
): Promise<ImageOutcome> {
  const images: MadeImage[] = [];
  let extras = empty();
  try {
    const [path, init] = requestOf(params, true);
    const res = await fetcher(path, { ...init, signal });
    if (!res.ok) return await refusal(res);
    if (!res.body) throw new Error("the response has no body");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    const parser = createSseParser();
    for (;;) {
      const { done, value } = await reader.read();
      const events = parser.push(done ? `${decoder.decode()}\n\n` : decoder.decode(value, { stream: true }));
      for (const data of events) {
        let event: Record<string, unknown>;
        try {
          event = JSON.parse(data) as Record<string, unknown>;
        } catch {
          continue;
        }
        const type = String(event.type ?? "");
        if (type === "error") {
          const error = (event.error ?? {}) as { message?: unknown; code?: unknown };
          return {
            ok: false,
            status: 200,
            message: typeof error.message === "string" ? error.message : "the job ended before its last image",
            ...(typeof error.code === "string" ? { code: error.code } : {}),
            partial: images,
            extras,
          };
        }
        if (type.endsWith(".progress")) {
          handlers.onProgress({
            phase: event.phase as ImagePhase,
            image: Number(event.image ?? 0),
            step: Number(event.step ?? 0),
            total: Number(event.total ?? 0),
          });
        } else if (type.endsWith(".completed") && typeof event.b64_json === "string") {
          const image: MadeImage = {
            b64: event.b64_json,
            index: Number(event.image ?? images.length),
            ...(typeof event.seed === "number" ? { seed: event.seed } : {}),
            ...(typeof event.size === "string" ? { size: event.size } : {}),
            ...(typeof event.revised_prompt === "string" ? { revisedPrompt: event.revised_prompt } : {}),
          };
          images.push(image);
          extras = extrasOf(event as Wire, extras);
          handlers.onImage(image);
        }
      }
      if (done) break;
    }
    return { ok: true, images, extras };
  } catch (err) {
    if (signal?.aborted) return { ok: false, status: 0, message: STOPPED, partial: images, extras };
    return unreachable(err, images, extras);
  }
}
