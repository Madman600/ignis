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
});
