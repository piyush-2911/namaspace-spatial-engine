// Keep keyboard and pointer holds separate: releasing one must not cancel the other.
export function createMovementInput(win, doc, viewport, buttons, enabled) {
  const keyboard = new Set(), pointers = new Map();
  const codes = new Set(['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight']);
  const clear = () => { keyboard.clear(); pointers.clear(); };
  win.addEventListener('keydown', e => {
    if (e.code === 'Escape') { clear(); return; }
    if (!enabled() || doc.activeElement !== viewport || e.ctrlKey || e.altKey || e.metaKey) return;
    if (codes.has(e.code)) { e.preventDefault(); keyboard.add(e.code); }
  });
  // Capture release events even when another control handles them.
  win.addEventListener('keyup', e => keyboard.delete(e.code), true);
  win.addEventListener('blur', clear);
  win.addEventListener('pagehide', clear);
  doc.addEventListener('visibilitychange', clear);
  doc.addEventListener('focusin', e => { if (e.target !== viewport) clear(); });
  for (const button of buttons) {
    button.addEventListener('pointerdown', e => {
      if (!enabled()) return;
      e.preventDefault();
      viewport.focus({preventScroll:true});
      button.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, button.dataset.move);
    });
    button.addEventListener('lostpointercapture', e => pointers.delete(e.pointerId));
  }
  for (const event of ['pointerup','pointercancel']) {
    win.addEventListener(event, e => pointers.delete(e.pointerId), true);
  }
  return {clear, has: code => keyboard.has(code) || [...pointers.values()].includes(code)};
}
