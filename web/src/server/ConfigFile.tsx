import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { actionInk, actionShape } from "../ui/classes.ts";
import { FOCUSABLE, trapIndex } from "../ui/focus.ts";
import { IconClose } from "../ui/icons.tsx";
import { readConfigFile } from "./routes.ts";
import type { Loaded } from "./useServer.ts";

// "View as config file" (spec playground/05 §Follow-up): the configuration in
// force as the YAML document `GET /v1/config?format=yaml` renders, in a modal
// with "Copy" and "Download". The modal is portalled — the form's surface is
// a clip-path (`.cut`), which would clip a popup inside it — traps Tab, closes
// on Esc and gives focus back to the button that opened it.

export const FILE_NAME = "ignis.yaml";

/** What the file leaves out, in one line. */
export const FILE_NOTE =
  "This is the configuration in force, live changes included, without the comments of the file on disk. Hidden fields are not in it, and neither is a pending draft.";

const action = `${actionShape} ${actionInk} px-3 py-1.5 [--cut-size:6px]`;

/** What takes focus when the dialog opens: its close button, which is never disabled. */
export const INITIAL_FOCUS = '[aria-label="Close"]';

/** The dialog's markup alone; `ConfigFile` gives it its behaviour. */
export function ConfigFileDialog({
  file,
  copied,
  onCopy,
  onDownload,
  onClose,
  panelRef,
}: {
  file: Loaded<string>;
  /** "Copy" read as done, or `failed`. */
  copied: boolean | "failed";
  onCopy: () => void;
  onDownload: () => void;
  onClose: () => void;
  panelRef?: RefObject<HTMLDivElement | null>;
}) {
  const ready = file.state === "ok";
  return (
    <div className="fixed inset-0 z-50 grid place-items-center bg-kiln/85 p-4" onClick={onClose} role="presentation">
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="config-file-title"
        tabIndex={-1}
        // The height is bound to the viewport (the overlay's 2rem of padding
        // off), not `max-h-full`: the overlay's grid row grows with the file,
        // so a percentage never limits it, and the `<pre>` below never scrolls.
        className="cut flex max-h-[calc(100dvh-2rem)] w-full max-w-3xl outline-none flex-col gap-3 bg-surface p-4 [--cut-size:14px] md:p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2">
          <h2 id="config-file-title" className="min-w-0 flex-1 font-display text-lg font-semibold">
            Configuration as a file
          </h2>
          <button type="button" onClick={onCopy} disabled={!ready} className={action}>
            {copied === true ? "Copied" : copied === "failed" ? "Could not copy" : "Copy"}
          </button>
          <button type="button" onClick={onDownload} disabled={!ready} className={action}>
            Download
          </button>
          <button type="button" onClick={onClose} aria-label="Close" title="Close" className="grid size-8 shrink-0 place-items-center text-ash hover:bg-line hover:text-ink">
            <IconClose />
          </button>
        </div>
        {file.state === "loading" && <p className="py-6 text-sm text-ash">Reading the file…</p>}
        {file.state === "error" && <p className="py-6 text-sm text-fault [overflow-wrap:anywhere]">The file could not be read: {file.refusal.message}</p>}
        {file.state === "ok" && (
          <pre
            tabIndex={0}
            aria-label={FILE_NAME}
            className="min-h-0 flex-1 overflow-auto border border-line bg-ground p-3 font-mono text-[12px] leading-relaxed text-ink"
          >
            {file.value}
          </pre>
        )}
        <p className="text-xs text-ash">{FILE_NOTE}</p>
      </div>
    </div>
  );
}

/** Reads the file when it opens, and runs the dialog. */
function ConfigFileModal({ onClose, opener }: { onClose: () => void; opener: RefObject<HTMLButtonElement | null> }) {
  const [file, setFile] = useState<Loaded<string>>({ state: "loading" });
  const [copied, setCopied] = useState<boolean | "failed">(false);
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    void readConfigFile().then((answer) => live && setFile(answer.ok ? { state: "ok", value: answer.value } : { state: "error", refusal: answer }));
    return () => {
      live = false;
    };
  }, []);

  // Focus goes into the dialog, Tab stays in it, Esc closes it, and the button that opened it gets focus back.
  useEffect(() => {
    const button = opener.current;
    (panel.current?.querySelector<HTMLElement>(INITIAL_FOCUS) ?? panel.current)?.focus();
    const keys = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "Tab" && panel.current) {
        const items = [...panel.current.querySelectorAll<HTMLElement>(FOCUSABLE)];
        const next = trapIndex(items.indexOf(document.activeElement as HTMLElement), items.length, e.shiftKey);
        e.preventDefault();
        items[next]?.focus();
      }
    };
    document.addEventListener("keydown", keys);
    return () => {
      document.removeEventListener("keydown", keys);
      button?.focus();
    };
  }, [onClose, opener]);

  const text = file.state === "ok" ? file.value : "";
  const copy = () => {
    const done = (ok: boolean) => setCopied(ok ? true : "failed");
    if (!navigator.clipboard) return done(false);
    navigator.clipboard.writeText(text).then(() => done(true), () => done(false));
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([text], { type: "application/yaml" }));
    const link = Object.assign(document.createElement("a"), { href: url, download: FILE_NAME });
    link.click();
    URL.revokeObjectURL(url);
  };
  return createPortal(<ConfigFileDialog file={file} copied={copied} onCopy={copy} onDownload={download} onClose={onClose} panelRef={panel} />, document.body);
}

/** The "View as config file" button and the modal it opens. */
export function ConfigFileButton() {
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  // Stable: the modal's focus effect depends on it, and runs again (taking focus back to the button) if it changes.
  const close = useCallback(() => setOpen(false), []);
  return (
    <>
      <button
        ref={button}
        type="button"
        onClick={() => setOpen(true)}
        aria-haspopup="dialog"
        className="shrink-0 px-3 py-1.5 font-display text-[13px] font-semibold text-ember hover:underline"
      >
        View as config file
      </button>
      {open && <ConfigFileModal onClose={close} opener={button} />}
    </>
  );
}
