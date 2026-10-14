import type { ModelState } from "../api/model.ts";
import flame from "../brand/flame.webp";

const HEADLINE: Record<ModelState["state"], string> = {
  ready: "Light it up",
  reading: "Warming up",
  switching: "Switching models",
  unloaded: "Light it up",
  loading: "Warming up",
  error: "No model to talk to",
};

export function EmptyState({ model }: { model: ModelState }) {
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-7 py-10 text-center">
      <img src={flame} alt="" className="h-44 w-auto drop-shadow-[0_18px_40px_rgb(200_22_29/0.35)]" />
      <div className="flex flex-col gap-2">
        <p className="font-display text-4xl font-semibold tracking-tight text-ink">
          {HEADLINE[model.state]}
        </p>
        <p className="text-sm text-ash">
          {(model.state === "ready" || model.state === "unloaded") && model.kind === "image" && (
            <>
              <span className="font-display text-ink">{model.id}</span> makes images, not replies. Make them in the Image view; a chat model
              loads from the Server view.
            </>
          )}
          {model.state === "ready" && model.kind !== "image" && (
            <>
              Send a prompt to <span className="font-display text-ink">{model.id}</span>. Timings for each reply show up
              under it and in this session's log below.
            </>
          )}
          {model.state === "reading" && "Asking ignis which model it serves."}
          {model.state === "switching" && `Loading ${model.to}. The prompt opens once it serves.`}
          {model.state === "unloaded" && model.kind !== "image" && (
            <>
              <span className="font-display text-ink">{model.id}</span> was unloaded while idle. Send a prompt: it loads
              again first, and the first reply takes as long as the load.
            </>
          )}
          {model.state === "loading" && `Loading ${model.to}. The prompt opens once it serves.`}
          {model.state === "error" && (model.failed ? "A model switch failed. The Server view says why." : "Start ignis-server, then reload this page.")}
        </p>
      </div>
    </div>
  );
}
