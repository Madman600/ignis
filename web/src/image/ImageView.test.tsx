import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelState } from "../api/model.ts";
import { ImageView } from "./ImageView.tsx";
import { DEFAULT_LIMITS } from "./studio.ts";
import type { StudioActions, StudioState } from "./useStudio.ts";

// The image view rendered in its states (spec image-model/05 AC 12): a text
// model loaded, the explicit switch and Generate's own (GitHub #324), the
// image model serving, a run under way, a refusal, and the gallery.

const text: ModelState = { state: "ready", id: "qwen3.8-27b", contextLimit: 786432, kind: "text" };
const image: ModelState = { state: "ready", id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" };

const actions: StudioActions = { run: () => {}, stop: () => {}, switchToImage: () => {}, clearGallery: () => {}, removeFromGallery: () => {} };

const base: StudioState = {
  limits: DEFAULT_LIMITS,
  imageModel: { id: "ignis-image-2.1-turbo", source: "built-in", on_disk: "complete", family: "qwenimage21", loaded: false, switchable: true },
  run: null,
  error: null,
  notes: [],
  gallery: [],
  revised: null,
  switching: false,
  switchRefusal: null,
};

const render = (model: ModelState, state: Partial<StudioState> = {}) =>
  renderToStaticMarkup(<ImageView model={model} state={{ ...base, ...state }} actions={actions} />);

describe("ImageView", () => {
  it("offers the explicit switch while a text model is loaded, and Generate switches too", () => {
    const html = render(text);
    expect(html).toContain("qwen3.8-27b is loaded");
    expect(html).toMatch(/<button[^>]*>Switch to the image model<\/button>/);
    expect(html).not.toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model/);
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Switch &amp; generate<\/button>/); // no prompt yet
    expect(html).toContain("Unloads qwen3.8-27b and loads the image model first.");
    expect(html).not.toContain("Needs the image model.");
  });

  it("with enhancement on and the enhancer loaded, Generate says the rewrite comes first (GitHub #324)", () => {
    expect(render(text, { limits: { ...DEFAULT_LIMITS, enhance: true } })).toContain("Enhance with qwen3.8-27b, then switch &amp; generate");
  });

  it("says why Generate cannot switch when implicit switching is off", () => {
    const html = render(text, { limits: { ...DEFAULT_LIMITS, allowImplicit: false } });
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Generate<\/button>/);
    expect(html).toContain("switch.allow_implicit");
    expect(html).not.toContain("Generate switches to it on its own");
    expect(render(text)).toContain("Generate switches to it on its own");
  });

  it("does not take the switch button while a run switches, and shows the run's phase", () => {
    const html = render(text, { run: { edit: false, n: 1, switches: true, progress: { phase: "switching", image: 0, step: 1, total: 1 } } });
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
    expect(html).toContain("Switching models");
    expect(render(text, { run: { edit: false, n: 1, switches: true, progress: null } })).toContain("the image model is loaded before the job");
  });

  it("shows the switch under way, not pressable", () => {
    const html = render(text, { switching: true });
    expect(html).toContain("Switching to the image model…");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
    const loading = render({ state: "switching", from: "qwen3.8-27b", to: "ignis-image-2.1-turbo", since: 0 });
    expect(loading).toContain("Switching to the image model…");
  });

  it("says why the switch is refused, and where the model comes from when none is listed", () => {
    expect(render(text, { switchRefusal: "the artifact is missing" })).toContain("the artifact is missing");
    const none = render(text, { imageModel: null });
    expect(none).toContain("ignis lists no image model");
    expect(none).toMatch(/<button[^>]*disabled=""[^>]*>Switch to the image model<\/button>/);
  });

  it("has no switch with the image model serving, and the form is open with the server's sizes and auto", () => {
    const html = render(image);
    expect(html).not.toContain("Switch to the image model");
    expect(html).toMatch(/<button[^>]*disabled=""[^>]*>Generate<\/button>/); // no prompt yet
    expect(html).not.toContain("Needs the image model.");
    expect(html).toMatch(/<option value="auto"[^>]*>auto<\/option>/);
    for (const size of DEFAULT_LIMITS.sizes) expect(html).toContain(`<option value="${size}">${size}</option>`);
    expect(html).toContain("Enhance the prompt");
    expect(html).toContain("Condition images");
    expect(html).toContain('accept="image/png,image/jpeg,image/webp"');
    expect(html).toContain(`max="${DEFAULT_LIMITS.maxN}"`);
  });

  it("shows progress per phase and per step while a run is under way, with a way to stop it", () => {
    const html = render(image, { run: { edit: false, n: 2, switches: false, progress: { phase: "denoising", image: 1, step: 5, total: 20 } } });
    expect(html).toContain("Denoising · image 2 of 2 · step 5/20");
    expect(html).toContain(">Stop<");
    expect(render(image, { run: { edit: true, n: 1, switches: false, progress: null } })).toContain("Sending the edit…");
  });

  it("shows a refusal, the server's notes and the revised prompt", () => {
    const html = render(text, {
      error: "this route serves image models. Switch to the image model, then make the request again.",
      notes: ["the enhancer could not be had"],
      revised: "a brass lantern on a stone ledge",
    });
    expect(html).toContain('role="alert"');
    expect(html).toContain("then make the request again");
    expect(html).toContain("the enhancer could not be had");
    expect(html).toContain("Revised prompt");
    expect(html).toContain("a brass lantern on a stone ledge");
  });

  it("lists the gallery over a checkerboard, each picture with its download", () => {
    const html = render(image, {
      gallery: [
        { id: 2, url: "data:image/png;base64,BBB", prompt: "a lantern", size: "1024x1024", seed: 8, edit: true },
        { id: 1, url: "data:image/png;base64,AAA", prompt: "a lantern", revisedPrompt: "a brass lantern", size: "1024x1024", seed: 7, edit: false },
      ],
    });
    expect(html).toContain("Gallery · 2");
    expect(html).toContain('src="data:image/png;base64,AAA"');
    expect(html).toContain("1024x1024 · seed 8 · edit");
    expect(html.match(/Download PNG/g)).toHaveLength(2);
    expect(html).toContain("checker");
  });

  it("says the gallery is empty, and that it lives in the tab", () => {
    expect(render(image)).toContain("Nothing made yet.");
  });
});
