import { useState } from "react";
import type { ModelState } from "../api/model.ts";
import { diskLabel, downloadCommand, elapsed, rowAction, runningFamily, sourceLabel } from "./band.ts";
import { familyName } from "./form.ts";
import { drainNote, lastSwitchLine, type Line } from "./outcome.ts";
import type { CatalogRow } from "./routes.ts";
import type { Loaded } from "./useServer.ts";

// The model band (spec playground/05 §Model band): the loaded model on the
// kiln palette the Monitor's band uses in both schemes, how the last switch
// ended, and one row per model the server can switch to or fetch. The only
// motion of a switch is the header's flame and heat bar, run by the page's
// shared model state; the band says what is happening in words.

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

export function ModelBand({
  model,
  catalog,
  loadingId,
  refusal,
  streaming,
  drain,
  now,
  onLoad,
}: {
  model: ModelState;
  catalog: Loaded<CatalogRow[]>;
  loadingId: string | null;
  refusal: string | null;
  /** A chat reply is streaming: say what a switch does to it. */
  streaming: boolean;
  /** `switch.drain_timeout` in force, when the configuration was read. */
  drain: number | null;
  now: number;
  onLoad: (id: string) => void;
}) {
  const rows = catalog.state === "ok" ? catalog.value : null;
  const family = runningFamily(rows);
  // The last switch's line stays until the next switch begins.
  const last =
    model.state !== "loading" && model.state !== "switching" && model.state !== "waking" && model.lastSwitch ? lastSwitchLine(model.lastSwitch) : null;
  return (
    <section aria-label="Model" className="monitor-kiln cut overflow-hidden text-[#eae8e4] [--cut-size:20px]">
      <div className="flex flex-col gap-4 p-5 md:p-6">
        <Headline model={model} />
        <dl className="flex flex-wrap gap-x-8 gap-y-3">
          <Figure label="State" value={stateWord(model)} dot={stateColor(model)} />
          {(model.state === "ready" || model.state === "unloaded") && (
            <Figure label="Context" value={model.contextLimit === null ? "Not reported" : `${model.contextLimit.toLocaleString("en-US")} tokens`} />
          )}
          {model.state === "ready" && family && <Figure label="Family" value={familyName(family)} />}
          {(model.state === "switching" || model.state === "waking") && <Figure label="Elapsed" value={elapsed(model.since, now)} />}
        </dl>
        {(last || refusal || streaming || model.state === "error" || model.state === "unloaded") && (
          <div className="flex flex-col gap-1 text-sm" aria-live="polite">
            {model.state === "error" && <p className="text-fault [overflow-wrap:anywhere]">{model.message}</p>}
            {model.state === "unloaded" && (
              <p className="text-[#b9bec4]">Unloaded after going unused (model.idle_unload). The next message loads it again; Load does it now.</p>
            )}
            {last && <p className={`${TONE[last.tone]} [overflow-wrap:anywhere]`}>{last.text}</p>}
            {refusal && <p className="text-fault [overflow-wrap:anywhere]">{refusal}</p>}
            {streaming && <p className="text-[#b9bec4]">{drainNote(drain, model.state === "switching" ? "The switch" : "A switch")}</p>}
          </div>
        )}
      </div>
      <Catalog catalog={catalog} model={model} loadingId={loadingId} onLoad={onLoad} />
    </section>
  );
}

function Headline({ model }: { model: ModelState }) {
  if (model.state === "switching") {
    return (
      <h2 className="font-display text-2xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-[30px]">
        Switching from <span className="text-[#b9bec4]">{model.from}</span> to {model.to}
      </h2>
    );
  }
  if (model.state === "waking") {
    return (
      <h2 className="font-display text-2xl leading-tight font-semibold text-white [overflow-wrap:anywhere] sm:text-[30px]">
        Loading {model.to}
      </h2>
    );
  }
  // A reported failure lasts while the old model is loaded again, or until a
  // switch succeeds: either way nothing serves meanwhile (a refused switch
  // reads `failed` for one tick only).
  const text =
    model.state === "ready" || model.state === "unloaded"
      ? model.id
      : model.state === "loading"
        ? "Reading the model…"
        : model.failed
          ? "No model is serving"
          : "ignis is not answering";
  const muted = model.state !== "ready";
  return (
    <h2
      className={`font-display leading-none font-semibold [overflow-wrap:anywhere] ${muted ? "text-2xl text-[#b9bec4] sm:text-[30px]" : "text-[32px] text-white sm:text-[44px]"}`}
    >
      {text}
    </h2>
  );
}

function stateWord(model: ModelState): string {
  if (model.state === "ready") return "Serving";
  if (model.state === "switching") return "Switching";
  if (model.state === "waking") return "Loading";
  if (model.state === "unloaded") return "Unloaded";
  if (model.state === "loading") return "Reading";
  return model.failed ? "Not serving" : "Unreachable";
}

function stateColor(model: ModelState): string {
  if (model.state === "ready") return "var(--good)";
  if (model.state === "switching" || model.state === "waking") return "#ff5a1f";
  return model.state === "loading" || model.state === "unloaded" ? "var(--ash)" : "var(--fault)";
}

function Figure({ label, value, dot }: { label: string; value: string; dot?: string }) {
  return (
    <div>
      <dt className="text-[11px] text-[#939ba4]">{label}</dt>
      <dd className="flex items-center gap-2 font-display text-lg leading-tight font-semibold text-[#eae8e4]">
        {dot && <span className="cut size-2.5 shrink-0 [--cut-size:4px]" style={{ background: dot }} aria-hidden />}
        {value}
      </dd>
    </div>
  );
}

function Catalog({ catalog, model, loadingId, onLoad }: { catalog: Loaded<CatalogRow[]>; model: ModelState; loadingId: string | null; onLoad: (id: string) => void }) {
  if (catalog.state !== "ok") {
    return (
      <p className="border-t border-kiln-line px-5 py-3 text-sm text-[#939ba4] md:px-6">
        {catalog.state === "loading" ? "Reading the models this server can load…" : `The models this server can load could not be read: ${catalog.refusal.message}`}
      </p>
    );
  }
  return (
    <ul aria-label="Models this server can load" className="border-t border-kiln-line">
      {catalog.value.map((row) => (
        <li
          key={row.id}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-0.5 border-t border-kiln-line px-5 py-3 first:border-t-0 sm:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)_minmax(0,1.3fr)_auto] md:px-6"
        >
          <span className="font-display text-base font-semibold text-white [overflow-wrap:anywhere]">{row.id}</span>
          <span className="col-start-1 text-xs text-[#939ba4] sm:col-start-auto sm:text-sm">{sourceLabel(row.source)}</span>
          <span className="col-start-1 text-xs text-[#939ba4] sm:col-start-auto sm:text-sm">{diskLabel(row)}</span>
          <div className="col-start-2 row-span-3 row-start-1 justify-self-end sm:col-start-auto sm:row-span-1 sm:row-start-auto">
            <RowActionView row={row} model={model} loadingId={loadingId} onLoad={onLoad} />
          </div>
        </li>
      ))}
    </ul>
  );
}

function RowActionView({ row, model, loadingId, onLoad }: { row: CatalogRow; model: ModelState; loadingId: string | null; onLoad: (id: string) => void }) {
  const action = rowAction(row, model);
  switch (action.kind) {
    case "loaded":
      return <span className="font-display text-[13px] font-semibold text-good">Loaded</span>;
    case "loading":
      return <span className="font-display text-[13px] font-semibold text-[#ff8a4c]">Loading…</span>;
    case "load":
      return (
        <button
          type="button"
          aria-label={`Load ${row.id}`}
          disabled={!action.enabled || loadingId !== null}
          onClick={() => onLoad(row.id)}
          className="border border-[#4a525c] px-3 py-1.5 font-display text-[13px] font-semibold text-white hover:border-[#ff5a1f] hover:bg-[#ff5a1f]/15 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-[#4a525c] disabled:hover:bg-transparent"
        >
          {loadingId === row.id ? "Sending…" : "Load"}
        </button>
      );
    case "download":
      return <CopyCommand id={row.id} />;
    case "missing":
      return null;
  }
}

/** Copies `model download <id>`; where the clipboard is out of reach (a page not on https or localhost) it shows the command to select instead. */
function CopyCommand({ id }: { id: string }) {
  const [copy, setCopy] = useState<"idle" | "copied" | "failed">("idle");
  const command = downloadCommand(id);
  if (copy === "failed") return <code className="block font-mono text-xs break-all text-[#eae8e4] select-all">{command}</code>;
  return (
    <button
      type="button"
      aria-label={`Copy download command for ${id}`}
      onClick={() => {
        const written = navigator.clipboard?.writeText(command);
        if (!written) return setCopy("failed");
        written.then(
          () => {
            setCopy("copied");
            setTimeout(() => setCopy("idle"), 2000);
          },
          () => setCopy("failed"),
        );
      }}
      className="px-3 py-1.5 font-display text-[13px] font-semibold text-[#b9bec4] hover:bg-kiln-line hover:text-white"
    >
      {copy === "copied" ? "Copied" : "Copy download command"}
    </button>
  );
}
