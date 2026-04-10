import { showEditor } from '@slidev/client/state/index'

export default function () {
  // ── Close the SideEditor panel if it was left open ─────────────────────
  // showEditor is a useLocalStorage ref; setting it here also clears the
  // persisted value so it stays closed across restarts.
  showEditor.value = false

  // ── Hide the NavControls arrow bar ─────────────────────────────────────
  // Injected at runtime so it wins over UnoCSS cascade regardless of order.
  const style = document.createElement('style')
  style.textContent = `
    /* Prev/Next arrow toolbar — hidden; keyboard navigation still works */
    nav.flex.flex-col { display: none !important; }
    /* SideEditor panel — belt-and-suspenders in case the ref reset races */
    #page-root > .shadow { display: none !important; }
  `
  document.head.appendChild(style)
}
