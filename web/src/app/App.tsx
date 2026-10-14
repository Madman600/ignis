import { useCallback, useEffect, useState } from "react";
import { forgetKey, useAuth } from "../api/auth.ts";
import { canChat, loadUnderWay, type ModelState, useModel } from "../api/model.ts";
import { Composer } from "../conversation/Composer.tsx";
import { imageFromFile, type PromptImage } from "../conversation/images.ts";
import { DecideView, type Drawer as DecideDrawer } from "../decide/DecideView.tsx";
import { type OpenAgent, Transcript } from "../conversation/Transcript.tsx";
import { contextUsage } from "../metrics/context.ts";
import { Readout } from "../metrics/Readout.tsx";
import { MonitorView } from "../monitor/MonitorView.tsx";
import { useMonitor, useMonitorVisible } from "../monitor/useMonitor.ts";
import { SessionLog } from "../sessions/SessionLog.tsx";
import { SessionsPanel } from "../sessions/SessionsPanel.tsx";
import { ImageView } from "../image/ImageView.tsx";
import { useStudio } from "../image/useStudio.ts";
import { ServerView } from "../server/ServerView.tsx";
import { useServer, useTick } from "../server/useServer.ts";
import { DEFAULT_SETTINGS, type PlaygroundSettings } from "../settings/defaults.ts";
import { SettingsPanel } from "../settings/SettingsPanel.tsx";
import { AgentReader } from "../tools/agents/AgentReader.tsx";
import { MemoryManager } from "../tools/local/MemoryManager.tsx";
import { AGENT_SYSTEM_PROMPT, type AgentRun } from "../tools/agents/agents.ts";
import { ALL_TOOLS, type ToolsState } from "../tools/index.ts";
import { Header, type View } from "./Header.tsx";
import { KeyPage } from "./KeyPage.tsx";
import { TransportNotice } from "./TransportNotice.tsx";
import { STORED_FLAGS, useStoredFlag } from "./storedFlag.ts";
import { useConversation } from "./useConversation.ts";

// The Playground (GitHub #164): a streaming chat against ignis — over its
// Responses WebSocket, or /v1/chat/completions (GitHub #283) — with
// per-request figures measured in the browser.
// Sessions, settings and figures live in memory; a reload starts over, apart
// from the switches this browser stores (`storedFlag.ts`).
// This is the page's layout; the conversation loop is useConversation.
// The header switches between the chat, the Decide tab (GitHub #247), —
// when ignis serves metrics — the Monitor (GitHub #165), and the Server view
// (GitHub #316). Whichever is showing, the chat stays mounted underneath, so
// a streaming reply carries on.

type Drawer = "sessions" | "settings" | null;

export function App() {
  const auth = useAuth();
  const [model, refreshModel] = useModel();
  const monitorVisible = useMonitorVisible();
  const [view, setView] = useState<View>("chat");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [tools, setTools] = useState<ToolsState>(ALL_TOOLS);
  const [markdown, setMarkdown] = useState(true);
  // Off: the page runs one turn at a time, whichever session it is in. Stored,
  // so turning it on survives a reload rather than looking like it never took.
  const [parallel, setParallel] = useStoredFlag(STORED_FLAGS.parallel, false);
  const [input, setInput] = useState("");
  // The images the next prompt will carry. They belong to the box, not to a
  // session: once sent they live on the user message, and the box is empty again.
  const [images, setImages] = useState<PromptImage[]>([]);
  const [imageError, setImageError] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [reader, setReader] = useState<OpenAgent | null>(null);
  const closeReader = useCallback(() => setReader(null), []);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const closeMemory = useCallback(() => setMemoryOpen(false), []);
  const [drawer, setDrawer] = useState<Drawer>(null);
  // The bench has its own two drawers below `lg`; the header's buttons open
  // whichever view is showing.
  const [decideDrawer, setDecideDrawer] = useState<DecideDrawer>(null);
  const chat = useConversation({ model, refreshModel, settings, tools, parallel });
  const { active } = chat;
  const monitoring = view === "monitor" && monitorVisible;
  const deciding = view === "decide";
  const serving = view === "server";
  const imaging = view === "image";
  // Mounted from its first visit on, like the Decide tab: a draft of changes
  // and a reload being followed survive a look at the chat.
  const [serverOpened, setServerOpened] = useState(false);
  // The Image view the same: its prompt, its condition images and its gallery outlive a look at the chat.
  const [imageOpened, setImageOpened] = useState(false);
  // A switch or a load runs the header's flame and heat bar as a reply does (spec playground/05).
  const busy = chat.busy || loadUnderWay(model);

  const readerRun: AgentRun | undefined = reader
    ? active.messages.find((m) => m.id === reader.messageId)?.agents?.find((r) => r.callId === reader.callId)
    : undefined;

  // Metrics going away drops back to the chat *from the Monitor*, so their
  // return never flips the page on its own — and a reader in the Decide tab,
  // which needs no metrics, is left where they are.
  useEffect(() => {
    if (!monitorVisible) setView((current) => (current === "monitor" ? "chat" : current));
  }, [monitorVisible]);

  useEffect(() => {
    if (!drawer) return;
    const close = (e: globalThis.KeyboardEvent) => e.key === "Escape" && setDrawer(null);
    window.addEventListener("keydown", close);
    return () => window.removeEventListener("keydown", close);
  }, [drawer]);

  function selectSession(id: number) {
    chat.selectSession(id);
    setDrawer(null);
    setReader(null);
  }

  function newSession() {
    chat.newSession();
    setDrawer(null);
  }

  function send() {
    const text = input.trim();
    if ((!text && images.length === 0) || !chat.canRun) return;
    setInput("");
    setImages([]);
    setImageError(null);
    chat.send(text, images);
  }

  /** Picked or pasted files as prompt images; the ones that cannot be read are named instead. */
  async function addImages(files: File[]) {
    const results = await Promise.all(files.map(imageFromFile));
    const errors = results.flatMap((r) => (r.ok ? [] : [r.error]));
    setImageError(errors.length > 0 ? errors.join(" ") : null);
    const added = results.flatMap((r) => (r.ok ? [r.image] : []));
    if (added.length > 0) setImages((current) => [...current, ...added]);
  }

  const set = <K extends keyof PlaygroundSettings>(key: K, value: PlaygroundSettings[K]) =>
    setSettings((s) => ({ ...s, [key]: value }));

  // A 401 anywhere shows the key prompt in place of the page; App stays
  // mounted, so the sessions are still here once the key is in.
  if (auth.needsKey) return <KeyPage rejected={auth.rejected} />;

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <Header
        model={model}
        busy={busy}
        onOpen={(which) => (deciding ? setDecideDrawer(which) : setDrawer(which))}
        onForgetKey={auth.key ? forgetKey : undefined}
        onOpenMemory={() => setMemoryOpen(true)}
        view={monitoring ? "monitor" : deciding ? "decide" : serving ? "server" : imaging ? "image" : "chat"}
        onView={(next) => {
          if (next === "server") setServerOpened(true);
          if (next === "image") setImageOpened(true);
          setView(next);
        }}
        monitorAvailable={monitorVisible}
      />

      <TransportNotice />

      {monitoring && <Monitor />}

      {/* Mounted whichever view is showing, like the chat below it: the tab
          holds a whole typed request, and a glance at the Monitor is not a
          reason to lose it. Nothing here fetches on mount. */}
      <div className={deciding ? "contents" : "hidden"}>
        <DecideView ready={canChat(model)} drawer={decideDrawer} onDrawer={setDecideDrawer} />
      </div>

      {serverOpened && (
        <div className={serving ? "contents" : "hidden"}>
          <Server model={model} refreshModel={refreshModel} streaming={chat.busy} visible={serving} />
        </div>
      )}

      {imageOpened && (
        <div className={imaging ? "contents" : "hidden"}>
          <Image model={model} refreshModel={refreshModel} visible={imaging} />
        </div>
      )}

      <div className={monitoring || deciding || serving || imaging ? "hidden" : "contents"}>
        <div className="relative flex min-h-0 flex-1">
          {drawer && (
            <div className="fixed inset-0 z-30 bg-[#1c2026]/60 lg:hidden" onClick={() => setDrawer(null)} aria-hidden />
          )}

          <SessionsPanel
            open={drawer === "sessions"}
            list={chat.list}
            activeId={active.id}
            streaming={chat.streaming}
            onNew={newSession}
            onSelect={selectSession}
            onRemove={chat.deleteSession}
          />

          <main className="flex min-w-0 flex-1 flex-col">
            <Transcript
              session={active}
              model={model}
              markdown={markdown}
              canRerun={chat.canRun}
              following={chat.following}
              openAgent={reader}
              onOpenAgent={setReader}
              onRerun={chat.rerun}
              onSave={chat.saveEdit}
              onFork={chat.fork}
              onAnswer={chat.answer}
            />
            <Composer
              value={input}
              onChange={setInput}
              ready={canChat(model)}
              busy={chat.sendBlocked}
              streamingHere={chat.streamingHere}
              usage={contextUsage(active.log, settings.maxTokens, canChat(model) ? model.contextLimit : null)}
              onSend={send}
              onStop={chat.stop}
              canAttach={tools.enabled && tools.readFiles}
              attachments={active.attachments}
              attachError={chat.attachError}
              onAttach={(files) => void chat.attach(files)}
              onDetach={chat.detach}
              images={images}
              imageError={imageError}
              onAddImages={(files) => void addImages(files)}
              onRemoveImage={(index) => setImages((current) => current.filter((_, i) => i !== index))}
            />
          </main>

          <SettingsPanel
            open={drawer === "settings"}
            settings={settings}
            set={set}
            tools={tools}
            onToolsChange={setTools}
            markdown={markdown}
            onMarkdownChange={setMarkdown}
            parallel={parallel}
            onParallelChange={setParallel}
            attachments={active.attachments}
            onOpenMemory={() => setMemoryOpen(true)}
          />
        </div>

        <SessionLog rows={active.log} open={logOpen} onToggle={() => setLogOpen((o) => !o)} />
      </div>

      {memoryOpen && <MemoryManager onClose={closeMemory} />}

      {readerRun && (
        <>
          <div className="fixed inset-0 z-40 bg-[#1c2026]/40" onClick={closeReader} aria-hidden />
          <AgentReader
            key={readerRun.callId}
            run={readerRun}
            markdown={markdown}
            systemPrompt={readerRun.systemPrompt ?? AGENT_SYSTEM_PROMPT}
            figures={readerRun.figures && <Readout figures={readerRun.figures} />}
            onClose={closeReader}
          />
        </>
      )}
    </div>
  );
}

/** The Monitor subscribes on its own, so a scrape re-renders it and not the chat. */
function Monitor() {
  return <MonitorView state={useMonitor()} />;
}

/** The Server view's state lives here, and its clock ticks only while a switch runs. */
function Server({ model, refreshModel, streaming, visible }: { model: ModelState; refreshModel: () => void; streaming: boolean; visible: boolean }) {
  const [state, actions] = useServer(model, refreshModel, visible);
  const now = useTick(loadUnderWay(model));
  return <ServerView model={model} state={state} actions={actions} streaming={streaming} now={now} />;
}

/** The Image view's state lives here, so the form and the gallery stay while the chat shows. */
function Image({ model, refreshModel, visible }: { model: ModelState; refreshModel: () => void; visible: boolean }) {
  const [state, actions] = useStudio(model, refreshModel, visible);
  return <ImageView model={model} state={state} actions={actions} />;
}
