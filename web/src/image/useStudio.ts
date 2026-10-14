import { useCallback, useEffect, useRef, useState } from "react";
import { type ImageParams, type ImageProgress, isEdit } from "../api/images.ts";
import { imageLoaded, type ModelState } from "../api/model.ts";
import { type CatalogRow, type ConfigDocument, readCatalog, readConfig } from "../server/routes.ts";
import { type GalleryItem, imageModelRow, type Limits, limitsOf, performRun, switchToImageModel } from "./studio.ts";

// The image view's state (spec image-model/05 §Playground): what the server
// allows, the run under way with its progress, the gallery, and the explicit
// switch to the image model. The view renders it and nothing else.
//
// Nothing here names a model. A run goes to the image routes bare, and the
// one `model` this page sends is the id in the switch the owner asked for, to
// the Server view's `/v1/models/switch`.

export type Run = {
  edit: boolean;
  n: number;
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
  const [switching, setSwitching] = useState(false);
  const [switchRefusal, setSwitchRefusal] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const nextId = useRef(1);

  // Read on opening the view and on each return to it, and again once the image model serves: the sizes
  // and counts it offers are the running server's.
  const served = imageLoaded(model);
  useEffect(() => {
    if (!visible) return;
    void readConfig().then((answer) => answer.ok && setConfig(answer.value));
    void readCatalog().then((answer) => answer.ok && setCatalog(answer.value));
  }, [visible, served]);

  // After the owner's switch the page follows the model until it serves as an image model, or the switch fails.
  useEffect(() => {
    if (!switching) return;
    if (served || model.state === "error") {
      setSwitching(false);
      return;
    }
    const timer = setInterval(refreshModel, 1000);
    return () => clearInterval(timer);
  }, [switching, served, model.state, refreshModel]);

  useEffect(() => () => abort.current?.abort(), []);

  const start = useCallback((params: ImageParams, stream: boolean) => {
    if (abort.current) return;
    const controller = new AbortController();
    abort.current = controller;
    setRun({ edit: isEdit(params), n: params.n, progress: null });
    setError(null);
    setNotes([]);
    setRevised(null);
    void performRun(
      params,
      stream,
      {
        progress: (progress) => setRun({ edit: isEdit(params), n: params.n, progress }),
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
    if (!imageModelRow(catalog) || switching) return;
    setSwitchRefusal(null);
    setSwitching(true);
    void switchToImageModel(catalog).then((answer) => {
      if (answer && !answer.ok) {
        setSwitching(false);
        setSwitchRefusal(answer.message);
        return;
      }
      refreshModel();
    });
  }, [catalog, switching, refreshModel]);

  const state: StudioState = {
    limits: limitsOf(config),
    imageModel: imageModelRow(catalog),
    run,
    error,
    notes,
    gallery,
    revised,
    switching,
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
