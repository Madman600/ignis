import { describe, expect, it } from "vitest";
import { trapIndex } from "./focus.ts";

describe("trapIndex", () => {
  it("moves on with Tab and back with Shift+Tab", () => {
    expect(trapIndex(0, 3, false)).toBe(1);
    expect(trapIndex(2, 3, true)).toBe(1);
  });

  it("wraps at the ends: the last Tab goes to the first, the first Shift+Tab to the last", () => {
    expect(trapIndex(2, 3, false)).toBe(0);
    expect(trapIndex(0, 3, true)).toBe(2);
  });

  it("enters from outside at the first or the last, and has nowhere to go with nothing focusable", () => {
    expect(trapIndex(-1, 3, false)).toBe(0);
    expect(trapIndex(-1, 3, true)).toBe(2);
    expect(trapIndex(-1, 0, false)).toBe(-1);
  });
});
