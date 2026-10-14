import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { EmptyState } from "./EmptyState.tsx";

// What an empty session says while the model is not there to talk to.

describe("EmptyState", () => {
  it("says a switch is loading the next model, and points at the Server view when one failed (spec playground/05)", () => {
    const switching = renderToStaticMarkup(<EmptyState model={{ state: "switching", from: "a", to: "qwen3.8-flash-next", since: 0 }} />);
    expect(switching).toContain("Switching models");
    expect(switching).toContain("Loading qwen3.8-flash-next. The prompt opens once it serves.");
    const failed = renderToStaticMarkup(<EmptyState model={{ state: "error", message: "Switch failed: x", failed: true }} />);
    expect(failed).toContain("A model switch failed. The Server view says why.");
    expect(renderToStaticMarkup(<EmptyState model={{ state: "error", message: "down" }} />)).toContain("Start ignis-server");
  });

  it("points at the Image view while the image model is loaded, or unloaded while idle: the chat never loads it (spec image-model/05)", () => {
    for (const state of ["ready", "unloaded"] as const) {
      const html = renderToStaticMarkup(<EmptyState model={{ state, id: "ignis-image-2.1-turbo", contextLimit: null, kind: "image" }} />);
      expect(html).toContain("makes images, not replies. Make them in the Image view");
      expect(html).not.toContain("Send a prompt");
    }
    const unloaded = renderToStaticMarkup(<EmptyState model={{ state: "unloaded", id: "qwen3.8-27b", contextLimit: 786432, kind: "text" }} />);
    expect(unloaded).toContain("was unloaded while idle. Send a prompt: it loads");
  });
});
