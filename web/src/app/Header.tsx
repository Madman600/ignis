import type { ReactNode } from "react";
import type { ModelState } from "../api/model.ts";
import flame from "../brand/flame.webp";
import wordmark from "../brand/wordmark-light.webp";
import { MonitorPulse } from "../monitor/MonitorPulse.tsx";
import { useMemoryNotes } from "../tools/local/memory.ts";
import { IconSessions, IconSliders } from "../ui/icons.tsx";

export type View = "chat" | "decide" | "monitor" | "server" | "image";

/** The kiln bar: brand, the view switch, model status, and the drawer buttons below `lg`. */
export function Header({
  model,
  busy,
  onOpen,
  onForgetKey,
  onOpenMemory,
  view = "chat",
  onView,
  monitorAvailable = false,
}: {
  model: ModelState;
  busy: boolean;
  onOpen: (drawer: "sessions" | "settings") => void;
  /** Present when an API key is in use: drops it and returns to the key prompt. */
  onForgetKey?: () => void;
  onOpenMemory: () => void;
  view?: View;
  /** Switches views. The Decide tab and the Server view need no metrics, so this is always present in the app. */
  onView?: (view: View) => void;
  /** ignis serves metrics, so the Monitor is one of the views on offer. */
  monitorAvailable?: boolean;
}) {
  const chat = view === "chat";
  // The chat and the bench both have two side columns that are drawers below
  // `lg`; the Monitor, the Server view and the Image view have none.
  const sided = view === "chat" || view === "decide";
  return (
    <header className="z-20 shrink-0 bg-kiln text-[#eae8e4]">
      <div className="flex items-center gap-1.5 px-3 py-3 sm:gap-4 sm:px-4 md:px-6">
        {sided && (
          <HeaderButton label={chat ? "Sessions" : "Decisions"} onClick={() => onOpen("sessions")}>
            <IconSessions />
          </HeaderButton>
        )}
        <img src={flame} alt="" className="flame h-9 w-auto" data-busy={busy} />
        <img src={wordmark} alt="ignis" className="hidden h-[18px] w-auto sm:block" />
        <span className="hidden h-6 w-px bg-kiln-line sm:block" aria-hidden />
        {onView ? (
          <ViewSwitch view={view} onView={onView} monitorAvailable={monitorAvailable} />
        ) : (
          <span className="hidden font-display text-[15px] font-medium tracking-wide text-[#b9bec4] sm:block">Playground</span>
        )}
        <ModelStatus model={model} busy={busy} />
        {onView && monitorAvailable && chat && <MonitorPulse onOpen={() => onView("monitor")} />}
        <MemoryButton onOpen={onOpenMemory} />
        {onForgetKey && (
          <button
            type="button"
            title="Forget the API key"
            className="shrink-0 px-2 py-1 font-display text-[13px] font-medium text-[#939ba4] hover:bg-kiln-line hover:text-white"
            onClick={onForgetKey}
          >
            Lock
          </button>
        )}
        {sided && (
          <HeaderButton label={chat ? "Settings" : "Questions"} onClick={() => onOpen("settings")}>
            <IconSliders />
          </HeaderButton>
        )}
      </div>
      <div className="heat" data-busy={busy} aria-hidden />
    </header>
  );
}

function ViewSwitch({ view, onView, monitorAvailable }: { view: View; onView: (view: View) => void; monitorAvailable: boolean }) {
  // Below `sm` the views are one native menu: four tabs and the rest of the
  // bar do not fit a phone. (A native menu, since `.cut` is a clip-path: a
  // popup of our own inside a clipped box would be cut.)
  // The Image view talks to /v1/images/* (spec image-model/05) and offers the switch to the image model itself.
  // The Monitor is only on offer when ignis serves metrics; the Decide tab
  // talks to /v1/decide and the Server view to /v1/models and /v1/config
  // (spec playground/05), so they are there either way.
  const views: [View, string][] = [
    ["chat", "Playground"],
    ["decide", "Decide"],
    ["image", "Image"],
    ...(monitorAvailable ? ([["monitor", "Monitor"]] as [View, string][]) : []),
    ["server", "Server"],
  ];
  return (
    <>
      <select
        name="view"
        aria-label="Choose a view"
        value={view}
        onChange={(e) => onView(e.target.value as View)}
        className="min-w-0 rounded-[2px] border border-kiln-line bg-kiln py-1 pr-1 pl-1.5 font-display text-[15px] font-medium text-white [color-scheme:dark] focus-visible:outline-2 focus-visible:outline-[#ff5a1f] sm:hidden"
      >
        {views.map(([v, label]) => (
          <option key={v} value={v}>
            {label}
          </option>
        ))}
      </select>
      <nav aria-label="View" className="hidden items-center gap-1 self-stretch sm:flex">
        {views.map(([v, label]) => (
          <button
            key={v}
            type="button"
            aria-current={view === v ? "page" : undefined}
            onClick={() => onView(v)}
            className={`relative px-2 font-display text-[15px] font-medium tracking-wide ${view === v ? "text-white" : "text-[#939ba4] hover:text-white"}`}
          >
            {label}
            {view === v && <span className="absolute inset-x-2 -bottom-3 h-[2px] bg-[#ff5a1f]" aria-hidden />}
          </button>
        ))}
      </nav>
    </>
  );
}

/** Opens what ignis remembers; the count is how many notes it keeps. */
function MemoryButton({ onOpen }: { onOpen: () => void }) {
  const count = useMemoryNotes().length;
  return (
    <button
      type="button"
      title="What ignis remembers"
      className="flex shrink-0 items-baseline gap-1.5 px-2 py-1 font-display text-[13px] font-medium text-[#939ba4] hover:bg-kiln-line hover:text-white"
      onClick={onOpen}
    >
      Memory
      {count > 0 && <span className="text-[#ff8a4c] tabular-nums">{count}</span>}
    </button>
  );
}

function HeaderButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactNode }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className="grid size-8 shrink-0 place-items-center sm:size-9 text-[#b9bec4] hover:bg-kiln-line hover:text-white lg:hidden"
      onClick={onClick}
    >
      {children}
    </button>
  );
}

function ModelStatus({ model, busy }: { model: ModelState; busy: boolean }) {
  return (
    <span className="ml-auto flex min-w-0 items-center gap-2 font-display text-sm">
      {model.state === "ready" && (
        <>
          <span className={`cut size-2.5 shrink-0 [--cut-size:4px] ${busy ? "bg-[#ff5a1f]" : "bg-[#3ecf8e]"}`} aria-hidden />
          <span className="hidden truncate text-[#eae8e4] sm:inline">{model.id}</span>
          <span className="sr-only">{busy ? "generating" : "ready"}</span>
        </>
      )}
      {model.state === "reading" && <span className="text-[#939ba4]">Loading model…</span>}
      {model.state === "unloaded" && (
        <>
          <span className="cut size-2.5 shrink-0 bg-[#939ba4] [--cut-size:4px]" aria-hidden />
          <span className="hidden truncate text-[#b9bec4] sm:inline" title="Unloaded while idle: the next message loads it again">
            {model.id} · unloaded
          </span>
          <span className="sr-only">unloaded</span>
        </>
      )}
      {model.state === "loading" && (
        <>
          <span className="cut size-2.5 shrink-0 bg-[#ff5a1f] [--cut-size:4px]" aria-hidden />
          <span className="truncate text-[#eae8e4]">Loading {model.to}…</span>
        </>
      )}
      {model.state === "switching" && (
        <>
          <span className="cut size-2.5 shrink-0 bg-[#ff5a1f] [--cut-size:4px]" aria-hidden />
          <span className="truncate text-[#eae8e4]">Switching to {model.to}…</span>
        </>
      )}
      {model.state === "error" && <span className="truncate text-[#ff7b6b]">{model.message}</span>}
    </span>
  );
}
