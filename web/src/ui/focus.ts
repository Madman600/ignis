// Keyboard focus inside a dialog (WAI-ARIA dialog pattern): Tab and
// Shift+Tab wrap at the ends instead of leaving it.

/** The index Tab (or Shift+Tab, `back`) moves to among `count` focusable elements, from `current` (-1: none of them has focus). */
export function trapIndex(current: number, count: number, back: boolean): number {
  if (count <= 0) return -1;
  if (current < 0) return back ? count - 1 : 0;
  return back ? (current - 1 + count) % count : (current + 1) % count;
}

/** What a dialog's Tab can reach. */
export const FOCUSABLE = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
