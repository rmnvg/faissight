/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // Relative asset URLs so the UI works behind path-prefixing proxies (Jupyter, Colab, HF Spaces).
  base: './',
  build: {
    outDir: '../src/faissight/static',
    emptyOutDir: true,
    // deck.gl (lazy map chunk) and recharts (overview) are each ~550 kB minified, ~170 kB gzip.
    chunkSizeWarningLimit: 700,
  },
  // Unit tests only; e2e/*.spec.ts belong to Playwright.
  test: { include: ['src/**/*.test.ts'] },
  server: {
    // `npm run dev` talks to a running `faissight serve` on the default port.
    proxy: { '/api': 'http://127.0.0.1:8765' },
  },
})
