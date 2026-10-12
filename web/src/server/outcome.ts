import { type LastSwitch, loadUnderWay, type ModelState } from "../api/model.ts";
import { namedKeys } from "./form.ts";
import type { Answer, ConfigDocument, FieldDecl, PatchApplied, Refusal } from "./routes.ts";

// What the Server view says once a change or a switch has an answer (spec
// playground/05 §Applying, §Model band). The result lines are the spec's own
// words; the action keeps its name through the flow ("Apply" → "Applied").

/** One line of a result: `done` as it was asked, `warn` applied with a caveat, `fault` not applied. */
export type Line = { text: string; tone: "done" | "warn" | "fault" };

/** The server's sentences start in lower case; a line on the page does not. */
const sentence = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

const shadowed = (keys: string[] | undefined): Line[] =>
  (keys ?? []).map((key) => ({ text: `Overridden at the next start: ${key}`, tone: "warn" }));

/** A `200`: the change runs; where it was written down, and what a restart would override. */
export function appliedLines(applied: PatchApplied): Line[] {
  const written: Line =
    applied.persisted && applied.path
      ? { text: `Applied and written to ${applied.path}`, tone: "done" }
      : { text: `Applied, not written to a file: ${applied.reason ?? "no reason given"}`, tone: "warn" };
  return [written, ...shadowed(applied.shadowed_at_restart)];
}

/** A refusal (`400`, `409`, `501`, `503`, or no answer at all): the server's message; nothing was applied. */
export function refusalLines(refusal: Refusal): Line[] {
  return [{ text: sentence(refusal.message), tone: "fault" }];
}

/**
 * A reload a `202` began, followed until it ends: the file it will be
 * written to (or why not), what a restart would override, whether this page
 * has seen it run, and the last switch recorded before it.
 */
export type PendingReload = { id: string; path?: string; reason?: string; shadowed: string[]; seen: boolean; before?: LastSwitch };

export function pendingReload(applied: PatchApplied, before?: LastSwitch): PendingReload {
  return {
    id: applied.switching?.to ?? "",
    ...(applied.path ? { path: applied.path } : {}),
    ...(applied.reason ? { reason: applied.reason } : {}),
    shadowed: applied.shadowed_at_restart ?? [],
    seen: false,
    ...(before ? { before } : {}),
  };
}

export type Settled = { kind: "wait"; pending: PendingReload } | { kind: "done"; ok: boolean; lines: Line[] };

const sameSwitch = (a: LastSwitch | undefined, b: LastSwitch | undefined) => JSON.stringify(a) === JSON.stringify(b);

/**
 * Where a reload stands, from the model state the page reads each second.
 * Two reloads of one model leave the same `last_switch`, so it settles only
 * once it has seen this one run — or, when it ran entirely between two
 * reads, on a record unlike the one before it. Through `failed` it waits
 * while the old model is loaded again, unless the record says nothing serves.
 */
export function settleReload(pending: PendingReload, model: ModelState): Settled {
  if (loadUnderWay(model)) return { kind: "wait", pending: pending.seen ? pending : { ...pending, seen: true } };
  const last = model.state === "ready" || model.state === "error" ? model.lastSwitch : undefined;
  const fresh = last !== undefined && !sameSwitch(last, pending.before);
  if (!last || !(pending.seen || fresh)) return { kind: "wait", pending };
  if (model.state === "ready" && last.outcome === "switched") {
    const written: Line = pending.path
      ? { text: `Reloaded ${pending.id} and written to ${pending.path}`, tone: "done" }
      : { text: `Reloaded ${pending.id}, not written to a file: ${pending.reason ?? "no reason given"}`, tone: "warn" };
    return { kind: "done", ok: true, lines: [written, ...shadowed(pending.shadowed)] };
  }
  if (model.state === "ready" || (fresh && last.outcome === "failed")) return { kind: "done", ok: false, lines: [lastSwitchLine(last)] };
  return { kind: "wait", pending };
}

/** What an answer to "Apply" does to the view. */
export type AfterPatch = {
  /** The change runs: the draft is empty again. */
  clearDraft: boolean;
  /** The dock's lines; none while a reload is followed. */
  result: Line[] | null;
  /** The file keys of the rows a refusal names. */
  marked: string[];
  /** The reload a `202` began. */
  pending: PendingReload | null;
  /** The configuration now running, when the answer carries it. */
  config: ConfigDocument | null;
  /** Read the model state again: a switch began, or one is under way. */
  refreshModel: boolean;
};

/**
 * An answer to PATCH /v1/config (spec playground/05 §Applying). `200`: the
 * draft clears and the form shows the answer's configuration. `202`: the
 * whole patch goes through a reload, followed until it ends, the draft kept
 * until then. `400`: the server's message, and the rows it names marked.
 * `409` (a switch under way) and the gate's `503`: named, and the model read
 * again, so the action waits for the switch to land. Anything else
 * (`501`): the server's reason. Nothing refused was applied.
 */
export function afterPatch(answer: Answer<PatchApplied>, fields: FieldDecl[] | null, before?: LastSwitch): AfterPatch {
  const kept = { clearDraft: false, marked: [], pending: null, config: null, refreshModel: false };
  if (answer.ok && answer.value.applied === "reload") {
    return { ...kept, result: null, pending: pendingReload(answer.value, before), refreshModel: true };
  }
  if (answer.ok) return { ...kept, clearDraft: true, result: appliedLines(answer.value), config: answer.value.config ?? null };
  const marked = answer.status === 400 && fields ? [...namedKeys(answer.message, fields)] : [];
  return { ...kept, result: refusalLines(answer), marked, refreshModel: answer.status === 409 || answer.status === 503 };
}

/**
 * An answer to "Load" (POST /v1/models/switch): a switch that began is
 * followed — the state still reads `ready` until the next read — and a
 * refusal (`404 model_not_found`, `409`, `501`) is named in the band; a
 * `409` reads the model again, so the band shows the switch under way.
 */
export function afterSwitch(answer: Answer<{ from: string; to: string }>): { refusal: string | null; follow: boolean; refreshModel: boolean } {
  if (answer.ok) return { refusal: null, follow: true, refreshModel: true };
  return { refusal: refusalLines(answer)[0].text, follow: false, refreshModel: answer.status === 409 };
}

/** How the last switch or reload ended, as the band names it: a failure with its reason. */
export function lastSwitchLine(last: LastSwitch): Line {
  const reload = last.from === last.to;
  const reason = last.reason ?? "no reason given";
  switch (last.outcome) {
    case "switched":
      return { text: reload ? `Reloaded ${last.to}` : `Switched from ${last.from} to ${last.to}`, tone: "done" };
    case "refused":
      return {
        text: reload ? `The reload of ${last.to} was refused, nothing changed: ${reason}` : `The switch to ${last.to} was refused, ${last.from} kept serving: ${reason}`,
        tone: "fault",
      };
    case "restored":
      return {
        text: reload
          ? `The reload of ${last.to} failed, the previous configuration was loaded again: ${reason}`
          : `The switch to ${last.to} failed, ${last.from} was loaded again: ${reason}`,
        tone: "fault",
      };
    case "failed":
      return { text: `The ${reload ? "reload of" : "switch to"} ${last.to} failed and no model is serving: ${reason}`, tone: "fault" };
  }
}

/**
 * What a switch or a reload does to the reply streaming now: it gets
 * `switch.drain_timeout` seconds to finish, then is cancelled. `subject` is
 * "A switch" or "The reload".
 */
export function drainNote(seconds: number | null, subject: string): string {
  const what = subject.charAt(0).toLowerCase() + subject.slice(1);
  if (seconds === null) return `A reply is streaming: ${what} lets it finish first, up to switch.drain_timeout.`;
  if (seconds === 0) return `A reply is streaming: ${what} cancels it at once.`;
  return `A reply is streaming: ${what} gives it ${seconds} second${seconds === 1 ? "" : "s"} to finish, then cancels it.`;
}
