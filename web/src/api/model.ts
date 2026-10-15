import { useCallback, useEffect, useState } from "react";
import { authHeaders, keyRequired, useAuth } from "./auth.ts";

// The model ignis serves, asked from GET /v1/models when the page opens,
// whenever the API key changes, and when a turn starts: another client may
// have switched the server since. The Playground only follows it — its
// requests name no model, so none of them can switch it. The one switch it
// makes is the Server view's explicit "Load" (spec playground/05).
//
// While a switch runs, GET /v1/models says so (`status: switching`, both
// ids) and every other /v1 route answers 503; the page reads it again each
// second until the switch lands, so every view sees it — the header's flame
// and heat bar, its model status, the composer that is not ready meanwhile.
//
// A model nobody used for `model.idle_unload` seconds is unloaded (spec
// model-switch/02): `status: unloaded`, the same id listed. That is a state,
// not an error, and it is not followed — polling GET /v1/models neither keeps
// a model loaded nor loads it — so the page shows it until a turn starts and
// reads again. A message sent meanwhile loads the model and is answered once
// it is warm; while that load runs (`status: loading`) the page follows it
// each second, as it follows a switch.

/** How the last switch or reload since start ended (`last_switch`, spec playground/05). */
export type LastSwitch = {
  from: string;
  /** Equal to `from` for a reload from `PATCH /v1/config`. */
  to: string;
  /** `refused`: nothing was torn down; `restored`: the old model was loaded again; `failed`: nothing serves. */
  outcome: "switched" | "refused" | "restored" | "failed";
  /** Present for every outcome but `switched`: for a person, in GiB, with the shortfall. */
  reason?: string;
  /** The host plan that refused the target in exact bytes, when that is what failed it (Flash-Next short of RAM). */
  host_plan?: HostPlan;
};

/** A refused host plan (`last_switch.host_plan`): every figure in bytes. */
export type HostPlan = {
  available_bytes: number;
  margin_bytes: number;
  planned_bytes: number;
  needed_bytes: number;
  short_bytes: number;
  /** The first line, in plan order, the available memory could not hold. */
  crossing_line: "expert_pool" | "ngram_hot_rows" | "staging" | "retained_host_slots" | "kv_ram_arena";
  lines: { name: string; bytes: number }[];
};

/** What a model makes (`kind` of `GET /v1/models`, spec image-model/05): text answers chat, image answers the image routes. */
export type ModelKind = "text" | "image";

export type ModelState =
  /** The page's own first read of the model, under way. */
  | { state: "reading" }
  | { state: "ready"; id: string; contextLimit: number | null; kind?: ModelKind; lastSwitch?: LastSwitch }
  /** `since`: when this page first saw this switch, for the seconds it has run. */
  | { state: "switching"; from: string; to: string; since: number; lastSwitch?: LastSwitch }
  /** Unloaded while idle (`status: unloaded`): the next message loads `id` again. */
  | { state: "unloaded"; id: string; contextLimit: number | null; kind?: ModelKind; lastSwitch?: LastSwitch }
  /** A model loads while none is loaded (`status: loading`): `to` is the one loading, `since` as for a switch. */
  | { state: "loading"; to: string; since: number; lastSwitch?: LastSwitch }
  /** `failed`: the server reports a switch that failed (`status: failed`), not a read that did. */
  | { state: "error"; message: string; failed?: true; lastSwitch?: LastSwitch };

type ModelList = {
  data?: { id: string; max_model_len?: number; kind?: string }[];
  status?: string;
  switching?: { from: string; to: string };
  reason?: string;
  last_switch?: LastSwitch;
};

/** The model loaded now, or why it could not be read. `now` stamps a switch first seen by this read. */
export async function readModel(fetcher: typeof fetch = fetch, now = Date.now()): Promise<ModelState> {
  try {
    const res = await fetcher("/v1/models", { headers: authHeaders() });
    if (res.status === 401) {
      keyRequired();
      throw new Error("GET /v1/models: ignis wants an API key");
    }
    if (!res.ok) throw new Error(`GET /v1/models: ${res.status}`);
    const body = (await res.json()) as ModelList;
    const last = body.last_switch ? { lastSwitch: body.last_switch } : {};
    // The entry is still the old model while switching; the ids are in `switching`.
    if (body.status === "switching" && body.switching) {
      return { state: "switching", from: body.switching.from, to: body.switching.to, since: now, ...last };
    }
    if (body.status === "failed") {
      return { state: "error", message: `Switch failed: ${body.reason ?? "no reason given"}`, failed: true, ...last };
    }
    const id = body.data?.[0]?.id;
    if (!id) throw new Error("GET /v1/models: no model listed");
    // The image model reports no context at all; an older server no kind (a text model).
    const contextLimit = body.data?.[0]?.max_model_len ?? null;
    const reported = body.data?.[0]?.kind;
    const kind: { kind?: ModelKind } = reported === "image" || reported === "text" ? { kind: reported } : {};
    if (body.status === "unloaded") return { state: "unloaded", id, contextLimit, ...kind, ...last };
    if (body.status === "loading") return { state: "loading", to: body.switching?.to ?? id, since: now, ...last };
    return { state: "ready", id, contextLimit, ...kind, ...last };
  } catch (err) {
    return { state: "error", message: String(err) };
  }
}

/**
 * What a re-read leaves on screen: a model or a switch already shown is not
 * lost to a read that failed, and a switch still running keeps the moment it
 * was first seen. A failed switch the server reports always shows.
 */
export function afterRefresh(shown: ModelState, read: ModelState): ModelState {
  if (read.state === "error" && !read.failed && shown.state !== "reading" && shown.state !== "error") return shown;
  if (read.state === "switching" && shown.state === "switching" && read.from === shown.from && read.to === shown.to) {
    return { ...read, since: shown.since };
  }
  if (read.state === "loading" && shown.state === "loading" && read.to === shown.to) return { ...read, since: shown.since };
  return read;
}

/**
 * Whether a message may be sent now: a model serves, or one was unloaded while
 * idle and the message loads it again (spec model-switch/02). Not while a
 * switch or a load runs: the composer opens once the model serves.
 */
export function canSend(model: ModelState): model is Extract<ModelState, { state: "ready" | "unloaded" }> {
  return model.state === "ready" || model.state === "unloaded";
}

/**
 * Whether the chat can use the model loaded: `canSend`, and not an image model, whose routes are the image view's
 * (a chat sent to it is a `409 model_kind_mismatch`).
 */
export function canChat(model: ModelState): model is Extract<ModelState, { state: "ready" | "unloaded" }> {
  return canSend(model) && model.kind !== "image";
}

/** Whether the image model is the one loaded (or unloaded while idle). */
export function imageLoaded(model: ModelState): boolean {
  return (model.state === "ready" || model.state === "unloaded") && model.kind === "image";
}

/**
 * Whether a model is being loaded: a switch replaces the one serving, or one loads while none is loaded (spec
 * model-switch/02). The header's flame runs and the Server view counts the seconds meanwhile, and no other switch
 * may begin.
 */
export function loadUnderWay(model: ModelState): model is Extract<ModelState, { state: "switching" | "loading" }> {
  return model.state === "switching" || model.state === "loading";
}

/**
 * Whether the page follows a switch, re-reading each second: one runs, a model
 * loads while none is loaded, or the server reports one failed — briefly,
 * while the old model is loaded again, or until a switch succeeds — so the
 * page sees it through to `serving`. An unloaded model is not followed: it
 * stays so until something sends a request.
 */
export function switchFollowed(model: ModelState): boolean {
  return loadUnderWay(model) || (model.state === "error" && model.failed === true);
}

/**
 * Whether the page reads the model each second for a turn of its own: one is in flight while the model reads
 * `unloaded`, so that turn's request is loading it (spec model-switch/02). The read that says `loading` hands the
 * load to `switchFollowed`, and the one that says the model serves ends both, so a long reply is not followed.
 */
export function loadAwaited(model: ModelState, sending: boolean): boolean {
  return sending && model.state === "unloaded";
}

export function useModel(): [ModelState, () => void] {
  const [model, setModel] = useState<ModelState>({ state: "reading" });
  const { key, needsKey } = useAuth();

  useEffect(() => {
    if (needsKey) return;
    setModel({ state: "reading" });
    void readModel().then(setModel);
  }, [key, needsKey]);

  const refresh = useCallback(() => {
    if (needsKey) return;
    void readModel().then((read) => setModel((shown) => afterRefresh(shown, read)));
  }, [needsKey]);

  const following = switchFollowed(model);
  useEffect(() => {
    if (!following) return;
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [following, refresh]);

  return [model, refresh];
}
