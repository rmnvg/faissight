import { expect, test } from '@playwright/test'

test('overview loads with index health', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
  await expect(page.getByText('IVF_FLAT').first()).toBeVisible()
  await expect(page.getByText('List size distribution')).toBeVisible()
})

test('query explorer: run a query and see results with miss explanations', async ({ page }) => {
  await page.goto('/#/query')
  await page.getByRole('radio', { name: 'Stored id' }).click()
  await page.getByPlaceholder('e.g. 42').fill('42')
  await page.getByRole('button', { name: 'Search', exact: true }).click()

  await expect(page.getByText('Recall@10')).toBeVisible()
  const results = page.locator('section', { hasText: 'Approximate results' }).locator('tbody tr')
  await expect(results).toHaveCount(10)
  const truth = page.locator('section', { hasText: 'Exact nearest neighbours' }).locator('tbody tr')
  await expect(truth).toHaveCount(10)
  // Each true neighbour carries an outcome badge.
  await expect(truth.first()).toContainText(/Found|Cell not probed|Quantization|Transform|Ranked out/)
  await expect(page.getByText('Probe order')).toBeVisible()
})

test('tuner: sweep and get a recommendation', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText('faiss.SearchParametersIVF(nprobe=')).toBeVisible()
})

test('hnsw graph: trace a search and see the layers and outcomes', async ({ page }) => {
  await page.goto('http://127.0.0.1:8798/#/hnsw?id=42&ef=16')
  await expect(page.getByText('Reconstructed trace.')).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText(/Level \d · step \d+ of \d+/)).toBeVisible()
  await expect(page.getByRole('button', { name: 'Next step' })).toBeVisible()
  const truth = page.locator('section', { hasText: 'Exact nearest neighbours' }).locator('li')
  await expect(truth).toHaveCount(10)
  await expect(truth.first()).toContainText(/Found|Not reached|ranked out/)
  await expect(page.getByText('Graph structure')).toBeVisible()
})
