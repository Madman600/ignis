import type { ModelState } from "../api/model.ts";
import { formatBytes, formatSeconds } from "../monitor/format.ts";
import type { CatalogRow } from "./routes.ts";

// The model band's rows (spec playground/05 §Model band): one per
// GET /v1/models/catalog row, switched to by id alone. An absent model is
// not fetched from the page: the row copies the command that fetches it.

const SOURCES: Record<CatalogRow["source"], string> = {
  "built-in": "Built-in catalog",
  operator: "Operator catalog",
  known: "Known model",
};

export function sourceLabel(source: CatalogRow["source"]): string {
  return SOURCES[source] ?? source;
}

/** What of a model is on disk; an entry's size is its total, whatever is there. */
export function diskLabel(row: CatalogRow): string {
  const size = row.bytes === undefined ? "" : formatBytes(row.bytes);
  if (row.on_disk === "complete") return size ? `On disk, ${size}` : "On disk";
  if (row.on_disk === "partial") return size ? `Partly downloaded, ${size} in all` : "Partly downloaded";
  // A known model is a path, not a catalog entry: there is nothing to download.
  if (row.source === "known") return "Artifact not found";
  return size ? `Not downloaded, ${size}` : "Not downloaded";
}

export function downloadCommand(id: string): string {
  return `ignis-server model download ${id}`;
}

export type RowAction =
  | { kind: "loaded" }
  /** The model the switch under way is loading. */
  | { kind: "loading" }
  | { kind: "load"; enabled: boolean }
  /** A catalog entry not (wholly) on disk: copy `model download`. */
  | { kind: "download" }
  /** A known model whose artifact is not there. */
  | { kind: "missing" };

/**
 * The row's one action. "Load" follows the server's `switchable`, and is out
 * of reach while a switch or a load runs whatever the last catalog read said;
 * a failed switch is left by switching again, so it stays in reach then, and
 * so does an unloaded model (spec model-switch/02), which "Load" loads.
 */
export function rowAction(row: CatalogRow, model: ModelState): RowAction {
  const loading = model.state === "switching" || model.state === "waking";
  if (loading && row.id === model.to) return { kind: "loading" };
  if (row.loaded) return { kind: "loaded" };
  // `switchable` decides first: the server knows a switch by id alone can begin
  // (a catalog entry fetched since the last read, the model torn down by a
  // failed restore), whatever `on_disk` last said.
  if (row.switchable || row.on_disk === "complete") {
    return { kind: "load", enabled: row.switchable && !loading && model.state !== "loading" };
  }
  return row.source === "known" ? { kind: "missing" } : { kind: "download" };
}

/** The family the loaded model runs as (its scope name), from the catalog's loaded row. */
export function runningFamily(catalog: CatalogRow[] | null): string | null {
  return catalog?.find((row) => row.loaded)?.family ?? null;
}

/** Whole seconds since a switch was first seen: `12 s`, `1m 12s`. */
export function elapsed(since: number, now: number): string {
  const seconds = Math.max(0, Math.floor((now - since) / 1000));
  return seconds < 60 ? `${seconds} s` : formatSeconds(seconds);
}
