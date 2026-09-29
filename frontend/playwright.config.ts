import { defineConfig, devices } from '@playwright/test'

// Smoke test against a real `faissight serve` on a small synthetic index.
// Needs the frontend built (npm run build) and the Python env (uv sync --extra faiss-cpu).
const PORT = 8799
const DATA = 'e2e/.data'

export default defineConfig({
  testDir: 'e2e',
  timeout: 30_000,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? 'github' : 'list',
  use: { baseURL: `http://127.0.0.1:${PORT}`, trace: 'retain-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command:
      `uv run --project .. python ../examples/make_synthetic.py --out ${DATA} --n 2000 --d 32 --nlist 16` +
      ` && FAISSIGHT_CACHE_DIR=${DATA}/cache uv run --project .. faissight serve ${DATA}/ivf_flat.index` +
      ` --vectors ${DATA}/vectors.npy --meta ${DATA}/chunks.jsonl --port ${PORT} --no-browser`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
})
