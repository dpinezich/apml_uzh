export default function () {
  // ── Close the SideEditor panel ─────────────────────────────────────────
  // showEditor is persisted via useLocalStorage('slidev-show-editor').
  // Setting it here before Vue reads the key ensures it stays closed.
  localStorage.setItem('slidev-show-editor', 'false')

  // ── Runtime style overrides ─────────────────────────────────────────────
  // Injected after the full CSS bundle so nothing can override these.
  // SideEditor panel — belt-and-suspenders in case localStorage races
  const style = document.createElement('style')
  style.textContent = `#page-root > [class*="shadow"] { display: none !important; }`
  document.head.appendChild(style)
}
