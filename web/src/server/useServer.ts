import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { LastSwitch, ModelState } from "../api/model.ts";
import { dropDraft, type Draft, editDraft, type InForce, patchBody, removeDraft } from "./form.ts";
import { afterPatch, afterSwitch, type Line, type PendingReload, settleReload } from "./outcome.ts";
import {
  type Answer,
  type CatalogRow,
  type ConfigDocument,
  type FieldDecl,
  patchConfig,
  readCatalog,
  readConfig,
  readFields,
  type Refusal,
  switchTo,
} from "./routes.ts";

// The Server view's state (spec playground/05): the declarations, the
// running configuration and the catalog as read, the draft of changes, and
// the answers to "Apply" and "Load". The view renders it and nothing else,
// so each of its states is a plain value a test can render.
//
// The model state is the page's (api/model.ts), which re-reads it each
// second while a switch runs. This hook only starts that following after a
// 202 — the state still reads `ready` until the next read — and reads the
// configuration and the catalog again once the model serves again.

export type Loaded<T> = { state: "loading" } | { state: "ok"; value: T } | { state: "error"; refusal: Refusal };

export type ServerState = {
  fields: Loaded<FieldDecl[]>;
  config: Loaded<ConfigDocument>;
  catalog: Loaded<CatalogRow[]>;
  draft: Draft;
  /** The file keys of the rows the last refusal named. */
  marked: string[];
  /** The dock's lines: the answer to the last "Apply". */
  result: Line[] | null;
  /** A reload a `202` began, followed until it ends. */
  pending: PendingReload | null;
  applying: boolean;
  /** The id whose "Load" is being sent. */
  loadingId: string | null;
  /** Why the server refused the last "Load". */
  loadRefusal: string | null;
};

export type ServerActions = {
  load: (id: string) => void;
  /** Set `decl` where `inForce` says: a family's section, or the group. */
  edit: (decl: FieldDecl, inForce: InForce, value: string | boolean) => void;
  /** Take the family's own value of `decl` away, so the shared one applies to it again. */
  remove: (decl: FieldDecl, scope: string) => void;
  /** Take a pending removal back. */
  undo: (decl: FieldDecl, scope: string) => void;
  discard: () => void;
  apply: () => void;
  dismiss: () => void;
};

const loaded = <T,>(answer: Answer<T>): Loaded<T> => (answer.ok ? { state: "ok", value: answer.value } : { state: "error", refusal: answer });

const sameSwitch = (a: LastSwitch | undefined, b: LastSwitch | undefined) => JSON.stringify(a) === JSON.stringify(b);

export function useServer(model: ModelState, refreshModel: () => void, visible: boolean): [ServerState, ServerActions] {
  const [fields, setFields] = useState<Loaded<FieldDecl[]>>({ state: "loading" });
  const [config, setConfig] = useState<Loaded<ConfigDocument>>({ state: "loading" });
  const [catalog, setCatalog] = useState<Loaded<CatalogRow[]>>({ state: "loading" });
  const [draft, setDraft] = useState<Draft>({});
  const [marked, setMarked] = useState<string[]>([]);
  const [result, setResult] = useState<Line[] | null>(null);
  const [pending, setPending] = useState<PendingReload | null>(null);
  const [applying, setApplying] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [loadRefusal, setLoadRefusal] = useState<string | null>(null);
  // A 202 answered: follow the model until it leaves `ready` or records another switch.
  const [awaiting, setAwaiting] = useState<{ before?: LastSwitch } | null>(null);
  const modelNow = useRef(model);
  modelNow.current = model;

  // The declarations never change while the server runs: read until one read
  // answers (the warm-up holds every /v1 route but GET /v1/models).
  const fieldsRead = useRef(false);
  const reread = useCallback(() => {
    if (!fieldsRead.current) {
      void readFields().then((answer) => {
        fieldsRead.current = answer.ok;
        setFields(loaded(answer));
      });
    }
    void readConfig().then((answer) => setConfig(loaded(answer)));
    void readCatalog().then((answer) => setCatalog(loaded(answer)));
  }, []);

  // Read on opening the view, and again on each return to it: another client
  // may have switched, or a download landed, since.
  useEffect(() => {
    if (visible) reread();
  }, [visible, reread]);

  // The model serves again after a switch, a reload or a failure: what it runs with is read again.
  const lastState = useRef(model.state);
  useEffect(() => {
    const before = lastState.current;
    lastState.current = model.state;
    if (model.state === "ready" && (before === "switching" || before === "error")) reread();
  }, [model.state, reread]);

  // Another model now serves: a draft made against the old one's values is
  // dropped, not applied to the new one. A reload keeps the id, and its draft.
  const readyId = model.state === "ready" ? model.id : null;
  const lastReadyId = useRef(readyId);
  useEffect(() => {
    if (readyId === null) return;
    if (lastReadyId.current !== null && lastReadyId.current !== readyId) {
      setDraft({});
      setMarked([]);
    }
    lastReadyId.current = readyId;
  }, [readyId]);

  useEffect(() => {
    if (!awaiting) return;
    if (model.state !== "ready" || !sameSwitch(model.lastSwitch, awaiting.before)) {
      setAwaiting(null);
      return;
    }
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [awaiting, model, refreshModel]);

  // A reload ends: the draft clears once it took, and is kept when it did not.
  useEffect(() => {
    if (!pending) return;
    const settled = settleReload(pending, model);
    if (settled.kind === "wait") {
      if (settled.pending !== pending) setPending(settled.pending);
      return;
    }
    setPending(null);
    setResult(settled.lines);
    if (settled.ok) {
      setDraft({});
      setMarked([]);
    }
  }, [pending, model]);

  const fieldList = fields.state === "ok" ? fields.value : null;

  const actions = useMemo<ServerActions>(
    () => ({
      load(id) {
        setLoadRefusal(null);
        setLoadingId(id);
        void switchTo(id).then((answer) => {
          const after = afterSwitch(answer);
          setLoadingId(null);
          setLoadRefusal(after.refusal);
          if (after.follow) setAwaiting({ before: lastSwitchOf(modelNow.current) });
          if (after.refreshModel) refreshModel();
        });
      },
      edit(decl, inForce, value) {
        setDraft((current) => editDraft(current, decl, inForce, value));
        setMarked((current) => current.filter((key) => key !== decl.file_key));
      },
      remove(decl, scope) {
        setDraft((current) => removeDraft(current, decl, scope));
        setMarked((current) => current.filter((key) => key !== decl.file_key));
      },
      undo(decl, scope) {
        setDraft((current) => dropDraft(current, decl, scope));
      },
      discard() {
        setDraft({});
        setMarked([]);
        setResult(null);
      },
      apply() {
        const body = patchBody(draft);
        if (Object.keys(body).length === 0) return;
        setApplying(true);
        setResult(null);
        setMarked([]);
        void patchConfig(body).then((answer) => {
          const before = lastSwitchOf(modelNow.current);
          const after = afterPatch(answer, fieldList, before);
          setApplying(false);
          setResult(after.result);
          setMarked(after.marked);
          if (after.clearDraft) setDraft({});
          if (after.config) setConfig({ state: "ok", value: after.config });
          else if (after.clearDraft) reread();
          if (after.pending) {
            // The whole patch goes through a reload; the band shows it as a switch.
            setPending(after.pending);
            setAwaiting({ before });
          }
          if (after.refreshModel) refreshModel();
        });
      },
      dismiss() {
        setResult(null);
      },
    }),
    [draft, fieldList, refreshModel, reread],
  );

  return [{ fields, config, catalog, draft, marked, result, pending, applying, loadingId, loadRefusal }, actions];
}

const lastSwitchOf = (model: ModelState) => (model.state === "loading" ? undefined : model.lastSwitch);

/** `Date.now()`, ticking each second while `running` — the switch's elapsed seconds — and still otherwise. */
export function useTick(running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [running]);
  return now;
}
