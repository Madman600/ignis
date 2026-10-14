import { useCallback, useEffect, useRef, useState } from "react";
import { type ImageParams, type ImageProgress, isEdit } from "../api/images.ts";
import { imageLoaded, type LastSwitch, type ModelState } from "../api/model.ts";
import { afterSwitch } from "../server/outcome.ts";
import { type CatalogRow, type ConfigDocument, readCatalog, readConfig } from "../server/routes.ts";
import { type GalleryItem, imageModelRow, lastSwitchOf, type Limits, limitsOf, performRun, settleSwitch, switchToImageModel } from "./studio.ts";

// The image view's state (spec image-model/05 §Playground): what the server
// allows, the run under way with its progress, the gallery, and the explicit
// switch to the image model. The view renders it and nothing else.
//
// A `model` is sent only where the owner chose the image model: the switch
// button, to the Server view's `/v1/models/switch`, and a run whose params name
// it — Generate or Edit pressed with a text model loaded (`runPlan`, GitHub
// #324). Any other run goes to the image routes bare.

export type Run = {
  edit: boolean;
  n: number;
  /** The request names the image model: the server switches to it before the job. */
  switches: boolean;
  progress: ImageProgress | null;
};

export type StudioState = {
  limits: Limits;
  /** The image model's catalog row, for the switch button. */
  imageModel: CatalogRow | null;
  run: Run | null;
  /** What the last run ended in, when it was a refusal or a failure. */
  error: string | null;
  /** Said by the server about the last run: prompt enhancement that did not rewrite, a condition image scaled. */
  notes: string[];
  gallery: GalleryItem[];
  /** The prompt as prompt enhancement rewrote it, for the last run. */
  revised: string | null;
  switching: boolean;
  switchRefusal: string | null;
};

export type StudioActions = {
  /** Make the images: streamed when `stream`, else answered whole. */
  run: (params: ImageParams, stream: boolean) => void;
  stop: () => void;
  /** The owner pressed "Switch to the image model". */
  switchToImage: () => void;
  clearGallery: () => void;
  removeFromGallery: (id: number) => void;
};

export function useStudio(model: ModelState, refreshModel: () => void, visible: boolean): [StudioState, StudioActions] {
  const [config, setConfig] = useState<ConfigDocument | null>(null);
  const [catalog, setCatalog] = useState<CatalogRow[] | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  const [gallery, setGallery] = useState<GalleryItem[]>([]);
  const [revised, setRevised] = useState<string | null>(null);
  // The owner's switch under way, with the last switch recorded when the button was pressed.
  const [following, setFollowing] = useState<{ before?: LastSwitch } | null>(null);
  const [switchRefusal, setSwitchRefusal] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(1);
  const modelNow = useRef(model);
  modelNow.current = model;

  // Read on opening the view and on each return to it, and again once the image model serves: the sizes
  // and counts it offers are the running server's.
  const served = imageLoaded(model);
  useEffect(() => {
    if (!visible) return;
    void readConfig().then((answer) => answer.ok && setConfig(answer.value));
    void readCatalog().then((answer) => answer.ok && setCatalog(answer.value));
  }, [visible, served]);

  // After the owner's switch the page follows the model until it serves as an image model, or the switch's record
  // says it did not land (settleSwitch), and then says why.
  useEffect(() => {
    if (!following) return;
    const settled = settleSwitch(following.before, model);
    if (settled.kind === "done") {
      setFollowing(null);
      if (settled.line) setSwitchRefusal(settled.line);
      return;
    }
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [following, model, refreshModel]);

  useEffect(() => () => abort.current?.abort(), []);

  const start = useCallback((params: ImageParams, stream: boolean) => {
    if (abort.current) return;
    const controller = new AbortController();
    abort.current = controller;
    const switches = params.model !== undefined;
    // The page reads the switch once, when the request begins it: the model follows from there (useModel).
    let refreshed = false;
    setRun({ edit: isEdit(params), n: params.n, switches, progress: null });
    setError(null);
    setNotes([]);
    setRevised(null);
    void performRun(
      params,
      stream,
      {
        progress: (progress) => {
          setRun({ edit: isEdit(params), n: params.n, switches, progress });
          if (switches && !refreshed && (progress.phase === "switching" || progress.phase === "loading")) {
            refreshed = true;
            refreshModel();
          }
        },
        image: (item) => {
          setGallery((current) => [item, ...current]);
          if (item.revisedPrompt) setRevised(item.revisedPrompt);
        },
        done: (end) => {
          abort.current = null;
          setRun(null);
          setError(end.error);
          setNotes(end.notes);
          if (end.reread) refreshModel();
        },
      },
      () => nextId.current++,
      controller.signal,
    );
  }, [refreshModel]);

  const switchToImage = useCallback(() => {
    if (!imageModelRow(catalog) || following) return;
    setSwitchRefusal(null);
    setFollowing({ before: lastSwitchOf(modelNow.current) });
    void switchToImageModel(catalog).then((answer) => {
      if (!answer) {
        setFollowing(null);
        return;
      }
      // A refusal is named in the band; a 409 (a switch already under way) reads the model again.
      const after = afterSwitch(answer);
      if (!after.follow) {
        setFollowing(null);
        setSwitchRefusal(after.refusal);
      }
      if (after.refreshModel) refreshModel();
    });
  }, [catalog, following, refreshModel]);

  const state: StudioState = {
    limits: limitsOf(config),
    imageModel: imageModelRow(catalog),
    run,
    error,
    notes,
    gallery,
    revised,
    switching: following !== null,
    switchRefusal,
  };
  const actions: StudioActions = {
    run: start,
    stop: () => abort.current?.abort(),
    switchToImage,
    clearGallery: () => setGallery([]),
    removeFromGallery: (id) => setGallery((current) => current.filter((item) => item.id !== id)),
  };
  return [state, actions];
}
