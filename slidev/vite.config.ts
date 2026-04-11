import { defineConfig } from 'vite'
import { existsSync, realpathSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))

const IMAGE_RE = /\.(png|jpe?g|gif|svg|webp|avif)$/i

function getChapterDir(): string {
  try {
    return dirname(realpathSync(resolve(__dirname, 'current.md')))
  } catch {
    return __dirname
  }
}

export default defineConfig({
  plugins: [
    {
      name: 'apml-resolve-chapter-assets',
      resolveId(id, importer) {
        // Slidev creates virtual modules named current.md__slidev_N.md
        // whose directory context is slidev/, not the real chapter dir.
        // Intercept image imports from those virtual modules and redirect
        // them to the actual chapter's 01-slides/ directory.
        if (!importer?.includes('current.md')) return
        const filename = id.startsWith('./') ? id.slice(2) : id
        if (!IMAGE_RE.test(filename)) return
        const candidate = resolve(getChapterDir(), filename)
        if (existsSync(candidate)) return candidate
      }
    }
  ]
})
