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

/** How the last switch or reload since start ended (`last_switch`, spec playground/05). */
export type LastSwitch = {
  from: string;
  /** Equal to `from` for a reload from `PATCH /v1/config`. */
  to: string;
  /** `refused`: nothing was torn down; `restored`: the old model was loaded again; `failed`: nothing serves. */
  outcome: "switched" | "refused" | "restored" | "failed";
  /** Present for every outcome but `switched`. */
  reason?: string;
};

export type ModelState =
  | { state: "loading" }
  | { state: "ready"; id: string; contextLimit: number | null; lastSwitch?: LastSwitch }
  /** `since`: when this page first saw this switch, for the seconds it has run. */
  | { state: "switching"; from: string; to: string; since: number; lastSwitch?: LastSwitch }
  /** `failed`: the server reports a switch that failed (`status: failed`), not a read that did. */
  | { state: "error"; message: string; failed?: true; lastSwitch?: LastSwitch };

type ModelList = {
  data?: { id: string; max_model_len?: number }[];
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
    return { state: "ready", id, contextLimit: body.data?.[0]?.max_model_len ?? null, ...last };
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
  if (read.state === "error" && !read.failed && (shown.state === "ready" || shown.state === "switching")) return shown;
  if (read.state === "switching" && shown.state === "switching" && read.from === shown.from && read.to === shown.to) {
    return { ...read, since: shown.since };
  }
  return read;
}

/**
 * Whether the page follows a switch, re-reading each second: one runs, or the
 * server reports one failed — briefly, while the old model is loaded again,
 * or until a switch succeeds — so the page sees it through to `serving`.
 */
export function switchFollowed(model: ModelState): boolean {
  return model.state === "switching" || (model.state === "error" && model.failed === true);
}

export function useModel(): [ModelState, () => void] {
  const [model, setModel] = useState<ModelState>({ state: "loading" });
  const { key, needsKey } = useAuth();

  useEffect(() => {
    if (needsKey) return;
    setModel({ state: "loading" });
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
