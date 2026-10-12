import { canSend, type ModelState } from "../api/model.ts";
import { IconClose } from "../ui/icons.tsx";
import { ConfigForm } from "./ConfigForm.tsx";
import { actionInk, actionShape } from "../ui/classes.ts";
import { changes, drainTimeout, familiesOf, needsReload, primaryLabel, waitingForOthers, waitingLine } from "./form.ts";
import { ModelBand } from "./ModelBand.tsx";
import { drainNote, type Line } from "./outcome.ts";
import { runningFamily } from "./band.ts";
import type { ServerActions, ServerState } from "./useServer.ts";

// The Server view (spec playground/05, GitHub #316): the loaded model and the
// models it can switch to, then the running configuration as a form, and a
// dock at the foot while the draft holds a change. It has no side drawers.
// It renders `state` and nothing else; useServer owns it.

export function ServerView({
  model,
  state,
  actions,
  streaming,
  now,
}: {
  model: ModelState;
  state: ServerState;
  actions: ServerActions;
  /** A chat reply is streaming somewhere on the page. */
  streaming: boolean;
  now: number;
}) {
  const config = state.config.state === "ok" ? state.config.value : null;
  const family = runningFamily(state.catalog.state === "ok" ? state.catalog.value : null);
  const drain = drainTimeout(config);
  const count = Object.keys(changes(state.draft)).length;
  const docked = Object.keys(state.draft).length > 0 || state.result !== null || state.pending !== null;
  return (
    <main aria-label="Server" className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-[1180px] flex-1 flex-col gap-6 px-4 py-5 md:px-6">
        <ModelBand
          model={model}
          catalog={state.catalog}
          loadingId={state.loadingId}
          refusal={state.loadRefusal}
          streaming={streaming}
          drain={drain}
          now={now}
          onLoad={actions.load}
        />
        <ConfigForm
          fields={state.fields}
          config={state.config}
          origins={state.origins}
          catalog={state.catalog}
          family={family}
          draft={state.draft}
          marked={state.marked}
          onEdit={actions.edit}
          onRemove={actions.remove}
          onUndo={actions.undo}
        />
      </div>
      {docked && <Dock model={model} state={state} actions={actions} count={count} streaming={streaming} drain={drain} family={family} />}
    </main>
  );
}

const TONE: Record<Line["tone"], string> = { done: "text-good", warn: "text-warn", fault: "text-fault" };

/**
 * The dock: how many changes, "Discard", and the action whose label says what
 * it will do — "Apply", or "Apply and reload <id>" in the heat of the
 * header's bar when a changed or removed value reaches the running model. A
 * value that waits for the other model is listed, not counted as a reload.
 * No confirmation: the label is the confirmation. Then the answer, in the
 * spec's words.
 */
function Dock({
  model,
  state,
  actions,
  count,
  streaming,
  drain,
  family,
}: {
  model: ModelState;
  state: ServerState;
  actions: ServerActions;
  count: number;
  streaming: boolean;
  drain: number | null;
  family: string | null;
}) {
  const fields = state.fields.state === "ok" ? state.fields.value : [];
  const config = state.config.state === "ok" ? state.config.value : null;
  const origins = state.origins.state === "ok" ? state.origins.value : null;
  const catalog = state.catalog.state === "ok" ? state.catalog.value : null;
  const basis = { fields, doc: config, origins, family, families: familiesOf(fields, catalog, family) };
  const reload = needsReload(state.draft, basis);
  const later = waitingForOthers(state.draft, basis);
  const id =
    model.state === "ready" || model.state === "unloaded"
      ? model.id
      : model.state === "switching"
        ? model.from
        : model.state === "waking"
          ? model.to
          : (state.pending?.id ?? "the model");
  // Every route but GET answers 503 while a switch runs, so the action waits for it to land.
  const waiting =
    state.pending !== null
      ? `Reloading ${state.pending.id}…`
      : model.state === "switching"
        ? `A switch from ${model.from} to ${model.to} is under way. Apply once it lands.`
        : !canSend(model)
          ? "Apply once a model serves."
          : null;
  const blocked = count === 0 || state.applying || waiting !== null;
  return (
    <div role="region" aria-label="Changes" className="sticky bottom-0 z-10 border-t border-line bg-ground/95 backdrop-blur-md">
      <div className="mx-auto flex max-w-[1180px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 md:px-6">
        <div className="flex min-w-0 flex-1 basis-64 flex-col gap-0.5 text-sm">
          {count > 0 && <p className="font-display font-semibold">{count === 1 ? "1 change" : `${count} changes`}</p>}
          {waiting && <p className="text-ash">{waiting}</p>}
          {later.map((entry) => (
            <p key={entry.family} className="text-ash [overflow-wrap:anywhere]">
              {waitingLine(entry)}
            </p>
          ))}
          {reload && streaming && state.pending === null && <p className="text-warn">{drainNote(drain, "The reload")}</p>}
          {state.result && (
            <ul aria-live="polite" className="flex flex-col gap-0.5">
              {state.result.map((line) => (
                <li key={line.text} className={`${TONE[line.tone]} [overflow-wrap:anywhere]`}>
                  {line.text}
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {Object.keys(state.draft).length > 0 && (
            <button
              type="button"
              onClick={actions.discard}
              disabled={state.applying || state.pending !== null}
              className="px-3 py-2 font-display text-[13px] font-semibold text-ash hover:text-ink disabled:opacity-40"
            >
              Discard
            </button>
          )}
          {count > 0 && (
            <button
              type="button"
              onClick={actions.apply}
              disabled={blocked}
              className={`${actionShape} px-4 py-2 [--cut-size:8px] ${reload ? "heat-fill hover:brightness-110" : actionInk}`}
            >
              {state.applying ? "Applying…" : primaryLabel(reload, id)}
            </button>
          )}
          {count === 0 && state.result && state.pending === null && (
            <button type="button" aria-label="Dismiss" title="Dismiss" onClick={actions.dismiss} className="grid size-8 place-items-center text-ash hover:bg-surface hover:text-ink">
              <IconClose />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
