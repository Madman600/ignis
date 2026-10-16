import { type View } from "./Header.tsx";

// A view the URL can name, so a Playground tab is linkable (the header's view
// switch is plain `useState`, which resets to the chat on every reload).
//
// `?view=monitor` is the whole protocol: it is read once when the page mounts
// and rewritten on every switch, so the address bar always holds something that
// can be copied, bookmarked or opened in another tab.

/** The views a link may name. The Monitor only exists while metrics are served. */
const VIEWS: readonly View[] = ["chat", "decide", "image", "monitor", "server"];

/** The view `search` names, or `null` when it names none or an unknown one. */
export function viewFromSearch(search: string): View | null {
  const named = new URLSearchParams(search).get("view");
  if (!named) return null;
  return VIEWS.find((view) => view === named) ?? null;
}

/** The same URL as `href` with its `view` set to `view` (or cleared for the chat). */
export function withViewParam(href: string, view: View): string {
  const url = new URL(href);
  // The chat is the page's own default, so a link back to it carries nothing:
  // `?view=chat` and no parameter are the same page, and one URL per view keeps
  // a bookmark stable across a switch away and back.
  if (view === "chat") url.searchParams.delete("view");
  else url.searchParams.set("view", view);
  return `${url.pathname}${url.search}${url.hash}`;
}

/**
 * The view this page should open on: the URL's, or the chat.
 *
 * A `view=monitor` link on a server without metrics still returns `"monitor"`
 * here — `App` owns the fallback to the chat, because whether metrics answer is
 * a fact learned after the first render, not at mount.
 */
export function initialView(search: string): View {
  return viewFromSearch(search) ?? "chat";
}
