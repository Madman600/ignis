import { useEffect, useId, useRef, useState } from "react";
import type { ImageParams } from "../api/images.ts";
import { imageLoaded, loadUnderWay, type ModelState } from "../api/model.ts";
import { IconClose, IconImage, IconTrash } from "../ui/icons.tsx";
import { Switch } from "../ui/Switch.tsx";
import { actionInk, actionShape, caption, field } from "../ui/classes.ts";
import {
  AUTO_SIZE,
  downloadImage,
  type GalleryItem,
  parseSeed,
  progressLine,
  progressShare,
  runPlan,
} from "./studio.ts";
import type { StudioActions, StudioState } from "./useStudio.ts";

// The image view (spec image-model/05 §Playground, GitHub #322): a prompt,
// the sizes the server offers, how many, a seed, up to ten condition images
// with their alpha shown, the prompt-enhancement toggle, progress as the
// stream reports it, and a gallery that lives in this tab. With a text model
// loaded, Generate and Edit are the owner's choice of the image model: the
// request names it and the server switches, after the rewrite when the
// enhancer is the model loaded (GitHub #324); the button says which. The
// switch on its own stays offered beside. It has no side drawers. useStudio
// owns the state.

const ACCEPTED = ["image/png", "image/jpeg", "image/webp"];

/** A condition image picked or dropped, with the URL its preview shows. */
type Condition = { key: number; file: File; url: string };

export function ImageView({ model, state, actions }: { model: ModelState; state: StudioState; actions: StudioActions }) {
  const ready = imageLoaded(model);
  return (
    <main aria-label="Image" className="flex min-h-0 flex-1 flex-col overflow-y-auto">
      <div className="mx-auto flex w-full max-w-[1180px] flex-1 flex-col gap-6 px-4 py-5 md:px-6">
        {!ready && <SwitchBand model={model} state={state} onSwitch={actions.switchToImage} />}
        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
          <Form model={model} state={state} actions={actions} />
          <Results state={state} actions={actions} />
        </div>
      </div>
    </main>
  );
}

/** The explicit switch: shown while anything but the image model serves. */
export function SwitchBand({ model, state, onSwitch }: { model: ModelState; state: StudioState; onSwitch: () => void }) {
  const under = loadUnderWay(model) || state.switching;
  const loaded = model.state === "ready" || model.state === "unloaded" ? model.id : null;
  const row = state.imageModel;
  return (
    <section aria-label="Image model" className="monitor-kiln cut text-[#eae8e4] [--cut-size:20px]">
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 p-5 md:p-6">
        <div className="flex min-w-0 flex-1 basis-72 flex-col gap-1">
          <h2 className="font-display text-xl leading-tight font-semibold text-white [overflow-wrap:anywhere]">
            {under ? "Switching to the image model…" : loaded ? `${loaded} is loaded` : "No model is loaded"}
          </h2>
          <p className="text-sm text-[#b9bec4]">
            {under
              ? "The page follows the switch and opens the form once the image model serves."
              : row
                ? `Images are made by the image model. ${state.limits.allowImplicit ? "Generate switches to it on its own, or switch now: either" : "Switching"} unloads the model serving and loads it, and the chat waits until a text model is loaded again.`
                : "ignis lists no image model. Fetch it with `ignis-server model download ignis-image-2.1-turbo`, then reload this page."}
          </p>
          {state.switchRefusal && <p className="text-sm text-[#ff7b6b] [overflow-wrap:anywhere]">{state.switchRefusal}</p>}
        </div>
        <button
          type="button"
          className={`${actionShape} bg-[#ff5a1f] px-4 py-2.5 text-[#1c2026] hover:bg-[#ff8a4c]`}
          disabled={under || !row || state.run !== null || model.state === "reading" || model.state === "error"}
          onClick={onSwitch}
        >
          Switch to the image model
        </button>
      </div>
    </section>
  );
}

function Form({ model, state, actions }: { model: ModelState; state: StudioState; actions: StudioActions }) {
  const { limits } = state;
  const [prompt, setPrompt] = useState("");
  const [size, setSize] = useState(AUTO_SIZE);
  const [count, setCount] = useState(1);
  const [seedText, setSeedText] = useState("");
  // Starts at the server's own setting once it is read, and is the owner's after a touch.
  const [enhanceTouched, setEnhanceTouched] = useState<boolean | null>(null);
  const enhance = enhanceTouched ?? limits.enhance;
  const [stream, setStream] = useState(true);
  const [conditions, setConditions] = useState<Condition[]>([]);
  const [notice, setNotice] = useState<string | null>(null);
  const keys = useRef(0);
  const enhanceLabel = useId();
  const streamLabel = useId();
  const busy = state.run !== null;

  // The previews are object URLs: let them go with the view.
  const urls = useRef<string[]>([]);
  useEffect(
    () => () => {
      for (const url of urls.current) URL.revokeObjectURL(url);
    },
    [],
  );

  function add(files: File[]) {
    const images = files.filter((f) => ACCEPTED.includes(f.type));
    const skipped = files.length - images.length;
    const room = Math.max(0, limits.maxImages - conditions.length);
    const taken = images.slice(0, room);
    const notes = [
      skipped > 0 ? `${skipped} ${skipped === 1 ? "file is" : "files are"} not PNG, JPEG or WebP.` : "",
      images.length > room ? `At most ${limits.maxImages} condition images.` : "",
    ].filter(Boolean);
    setNotice(notes.length > 0 ? notes.join(" ") : null);
    if (taken.length === 0) return;
    const added = taken.map((file) => {
      const url = URL.createObjectURL(file);
      urls.current.push(url);
      return { key: ++keys.current, file, url };
    });
    setConditions((current) => [...current, ...added]);
  }

  function remove(key: number) {
    setConditions((current) => {
      const gone = current.find((c) => c.key === key);
      if (gone) URL.revokeObjectURL(gone.url);
      return current.filter((c) => c.key !== key);
    });
  }

  const seed = parseSeed(seedText);
  const n = Math.min(limits.maxN, Math.max(1, count));
  const plan = runPlan(model, limits, state.imageModel, enhance, conditions.length > 0, busy);
  const canRun = plan.ready && !busy && prompt.trim() !== "" && seed !== undefined;

  function submit() {
    if (!canRun || seed === undefined) return;
    const params: ImageParams = {
      prompt: prompt.trim(),
      n,
      size,
      seed,
      enhance,
      images: conditions.map((c) => c.file),
      ...(plan.model ? { model: plan.model } : {}),
    };
    actions.run(params, stream);
  }

  return (
    <form
      aria-label="Request"
      className="flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <label className="flex flex-col gap-1.5">
        <span className={caption}>Prompt</span>
        <textarea
          name="prompt"
          rows={5}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder={conditions.length > 0 ? "What should change in these images" : "Describe the picture"}
          className={`${field} resize-y`}
        />
      </label>

      <div className="flex flex-wrap gap-4">
        <label className="flex min-w-0 flex-1 basis-40 flex-col gap-1.5">
          <span className={caption}>Size</span>
          <select name="size" value={size} onChange={(e) => setSize(e.target.value)} className={field}>
            <option value={AUTO_SIZE}>auto</option>
            {limits.sizes.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <label className="flex w-24 flex-col gap-1.5">
          <span className={caption}>Images</span>
          <input
            name="n"
            type="number"
            min={1}
            max={limits.maxN}
            value={count}
            onChange={(e) => setCount(Number(e.target.value) || 1)}
            className={`${field} tabular-nums`}
          />
        </label>
        <label className="flex w-36 flex-col gap-1.5">
          <span className={caption}>Seed</span>
          <input
            name="seed"
            inputMode="numeric"
            value={seedText}
            onChange={(e) => setSeedText(e.target.value)}
            placeholder="random"
            aria-invalid={seed === undefined}
            className={`${field} tabular-nums`}
          />
        </label>
      </div>

      <ConditionPicker conditions={conditions} max={limits.maxImages} notice={notice} onAdd={add} onRemove={remove} />

      <div className="flex items-start gap-3">
        <Switch on={enhance} onChange={setEnhanceTouched} labelledBy={enhanceLabel} />
        <div className="flex flex-col gap-0.5 text-sm">
          <span id={enhanceLabel} className="font-display font-medium">
            Enhance the prompt
          </span>
          <span className="text-ash">A text model rewrites it first. The server may visit it, so the first request takes longer.</span>
        </div>
      </div>
      <div className="flex items-start gap-3">
        <Switch on={stream} onChange={setStream} labelledBy={streamLabel} />
        <div className="flex flex-col gap-0.5 text-sm">
          <span id={streamLabel} className="font-display font-medium">
            Show progress
          </span>
          <span className="text-ash">Streams the phases and denoising steps, and each picture as it is made.</span>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-3">
          <button type="submit" disabled={!canRun} className={`${actionShape} ${actionInk} px-5 py-2.5`}>
            {plan.label}
          </button>
          {busy && (
            <button type="button" className={`${actionShape} border border-line px-4 py-2.5 hover:border-ember`} onClick={actions.stop}>
              Stop
            </button>
          )}
        </div>
        {plan.note && <p className="text-sm text-ash [overflow-wrap:anywhere]">{plan.note}</p>}
      </div>
    </form>
  );
}

/** Drop or pick condition images; each shows over a checkerboard, so transparent pixels are seen as such. */
function ConditionPicker({
  conditions,
  max,
  notice,
  onAdd,
  onRemove,
}: {
  conditions: Condition[];
  max: number;
  notice: string | null;
  onAdd: (files: File[]) => void;
  onRemove: (key: number) => void;
}) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  return (
    <div className="flex flex-col gap-2">
      <span className={caption}>
        Condition images <span className="text-ash/80">({conditions.length}/{max}) — any makes the request an edit</span>
      </span>
      <div
        aria-label="Condition images"
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          onAdd(Array.from(e.dataTransfer.files));
        }}
        className={`flex flex-col gap-3 rounded-[2px] border border-dashed p-3 ${over ? "border-ember bg-surface" : "border-line"}`}
      >
        {conditions.length > 0 && (
          <ul className="grid grid-cols-3 gap-2 sm:grid-cols-4 lg:grid-cols-3">
            {conditions.map((c) => (
              <li key={c.key} className="relative">
                <div className="checker aspect-square overflow-hidden rounded-[2px] border border-line">
                  <img src={c.url} alt={c.file.name} className="size-full object-contain" />
                </div>
                <button
                  type="button"
                  aria-label={`Remove ${c.file.name}`}
                  className="cut absolute top-0 right-0 grid size-6 place-items-center bg-kiln text-[#eae8e4] hover:bg-[#ff5a1f] [--cut-size:6px]"
                  onClick={() => onRemove(c.key)}
                >
                  <IconClose />
                </button>
              </li>
            ))}
          </ul>
        )}
        <button
          type="button"
          disabled={conditions.length >= max}
          className="flex items-center gap-2 self-start text-sm text-ash hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
          onClick={() => input.current?.click()}
        >
          <IconImage />
          Drop images here, or pick them
        </button>
        <input
          ref={input}
          type="file"
          name="condition"
          accept={ACCEPTED.join(",")}
          multiple
          hidden
          onChange={(e) => {
            onAdd(Array.from(e.target.files ?? []));
            e.target.value = "";
          }}
        />
      </div>
      {notice && <p className="text-sm text-warn">{notice}</p>}
    </div>
  );
}

/** Progress, what went wrong or what the server added, and the gallery. */
function Results({ state, actions }: { state: StudioState; actions: StudioActions }) {
  const { run } = state;
  return (
    <section aria-label="Results" className="flex min-w-0 flex-col gap-4">
      {run && <Progress run={run} />}
      {state.error && (
        <p role="alert" className="border-l-2 border-fault pl-3 text-sm text-fault [overflow-wrap:anywhere]">
          {state.error}
        </p>
      )}
      {state.notes.map((note) => (
        <p key={note} className="border-l-2 border-warn pl-3 text-sm text-warn [overflow-wrap:anywhere]">
          {note}
        </p>
      ))}
      {state.revised && (
        <div aria-label="Revised prompt" className="flex flex-col gap-1 border border-line bg-surface p-3">
          <span className={caption}>Revised prompt</span>
          <p className="text-sm [overflow-wrap:anywhere]">{state.revised}</p>
        </div>
      )}
      <Gallery items={state.gallery} onClear={actions.clearGallery} onRemove={actions.removeFromGallery} />
    </section>
  );
}

function Progress({ run }: { run: NonNullable<StudioState["run"]> }) {
  const progress = run.progress;
  const share = progress ? progressShare(progress, run.n) : 0;
  return (
    <div role="status" aria-live="polite" className="flex flex-col gap-2 border border-line bg-surface p-3">
      <p className="font-display text-sm font-medium">
        {progress ? progressLine(progress, run.n) : `Sending the ${run.edit ? "edit" : "request"}${run.switches ? ": the image model is loaded before the job…" : "…"}`}
      </p>
      <div className="h-1.5 overflow-hidden bg-line" aria-hidden>
        <div className="h-full bg-ember transition-[width] motion-reduce:transition-none" style={{ width: `${Math.round(share * 100)}%` }} />
      </div>
    </div>
  );
}

function Gallery({ items, onClear, onRemove }: { items: GalleryItem[]; onClear: () => void; onRemove: (id: number) => void }) {
  if (items.length === 0) {
    return <p className="text-sm text-ash">Nothing made yet. The pictures stay in this tab until it reloads; download the ones you keep.</p>;
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-baseline justify-between">
        <h2 className="font-display text-base font-semibold">Gallery · {items.length}</h2>
        <button type="button" className="flex items-center gap-1.5 text-sm text-ash hover:text-ink" onClick={onClear}>
          <IconTrash />
          Clear
        </button>
      </div>
      <ul aria-label="Gallery" className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {items.map((item) => (
          <li key={item.id} className="flex min-w-0 flex-col gap-2">
            <div className="checker overflow-hidden rounded-[2px] border border-line">
              <img src={item.url} alt={item.revisedPrompt ?? item.prompt} className="block h-auto w-full" />
            </div>
            <p className="line-clamp-2 text-sm [overflow-wrap:anywhere]" title={item.revisedPrompt ?? item.prompt}>
              {item.revisedPrompt ?? item.prompt}
            </p>
            <div className="flex items-center gap-3 text-[13px] text-ash">
              <span className="tabular-nums">
                {item.size}
                {item.seed !== undefined && ` · seed ${item.seed}`}
                {item.edit && " · edit"}
              </span>
              <button type="button" className="ml-auto font-display font-medium text-ink hover:text-ember" onClick={() => downloadImage(item)}>
                Download PNG
              </button>
              <button type="button" aria-label="Remove from the gallery" className="hover:text-ink" onClick={() => onRemove(item.id)}>
                <IconClose />
              </button>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}
