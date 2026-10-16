import { describe, expect, it } from "vitest";
import { initialView, viewFromSearch, withViewParam } from "./viewParam.ts";

// A linkable view (the header switch was useState only, so /ui/# had no way to
// say "open the Monitor"). These are the three rules that make a URL a link.

describe("viewFromSearch", () => {
  it("names the view the query gives it", () => {
    expect(viewFromSearch("?view=monitor")).toBe("monitor");
    expect(viewFromSearch("?view=server")).toBe("server");
    expect(viewFromSearch("?view=decide")).toBe("decide");
    expect(viewFromSearch("?view=image")).toBe("image");
    expect(viewFromSearch("?view=chat")).toBe("chat");
  });

  it("names nothing when the query gives no view, so the page keeps its default", () => {
    expect(viewFromSearch("")).toBeNull();
    expect(viewFromSearch("?view=")).toBeNull();
    expect(viewFromSearch("?model=27b")).toBeNull();
  });

  it("names nothing for a view ignis does not have, rather than guessing", () => {
    expect(viewFromSearch("?view=Monitor")).toBeNull();
    expect(viewFromSearch("?view=metrics")).toBeNull();
    expect(viewFromSearch("?view=troll")).toBeNull();
  });

  it("keeps the view when the query carries other keys too", () => {
    expect(viewFromSearch("?theme=dark&view=monitor&x=1")).toBe("monitor");
  });
});

describe("initialView", () => {
  it("is the chat whenever the URL does not say otherwise", () => {
    expect(initialView("")).toBe("chat");
    expect(initialView("?view=nope")).toBe("chat");
    expect(initialView("?view=chat")).toBe("chat");
  });

  it("is the Monitor on a monitor link even where metrics may not be served", () => {
    // App owns the fallback: whether metrics answer is learned after mount.
    expect(initialView("?view=monitor")).toBe("monitor");
  });
});

describe("withViewParam", () => {
  it("rewrites the view of a link in place", () => {
    expect(withViewParam("http://h/ui/?view=monitor", "server")).toBe("/ui/?view=server");
    expect(withViewParam("http://h/ui/", "monitor")).toBe("/ui/?view=monitor");
  });

  it("carries no query at all for the chat, so the bare page is the only chat URL", () => {
    expect(withViewParam("http://h/ui/?view=monitor", "chat")).toBe("/ui/");
    expect(withViewParam("http://h/ui/?view=server", "chat")).toBe("/ui/");
  });

  it("keeps other keys and a hash of the link it rewrites", () => {
    expect(withViewParam("http://h/ui/?theme=dark&view=chat#k", "monitor")).toBe("/ui/?theme=dark&view=monitor#k");
  });

  it("round-trips: the URL it writes is one the page opens on again", () => {
    for (const view of ["monitor", "server", "decide", "image"] as const) {
      expect(initialView(new URL(withViewParam("http://h/ui/", view), "http://h").search)).toBe(view);
    }
    expect(initialView(new URL(withViewParam("http://h/ui/?view=monitor", "chat"), "http://h").search)).toBe("chat");
  });
});
