import type { LastSwitch, ModelState } from "../api/model.ts";
import type { PatchApplied, Refusal } from "./routes.ts";

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
  if (model.state === "switching") return { kind: "wait", pending: pending.seen ? pending : { ...pending, seen: true } };
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
