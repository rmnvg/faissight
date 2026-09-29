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

test('query explorer: one request per search, and no exact work when compare is off', async ({ page }) => {
  const calls: string[] = []
  page.on('request', (r) => {
    if (r.url().includes('/api/search') || r.url().includes('/api/trace/')) calls.push(r.url())
  })
  await page.goto('/#/query')
  await page.getByRole('radio', { name: 'Stored id' }).click()
  await page.getByPlaceholder('e.g. 42').fill('7')
  let searched = page.waitForResponse((r) => r.url().endsWith('/api/search'))
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  let response = await searched
  expect(response.request().postDataJSON()).toMatchObject({ compare: true, trace: true })
  expect((await response.json()).ivf_trace).not.toBeNull()
  await expect(page.getByText('Recall@10')).toBeVisible()
  expect(calls).toHaveLength(1)

  await page.getByLabel('Compare with exact').uncheck()
  searched = page.waitForResponse((r) => r.url().endsWith('/api/search'))
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  response = await searched
  expect(response.request().postDataJSON()).toMatchObject({ compare: false, trace: false })
  const body = await response.json()
  expect(body.truth).toBeNull()
  expect(body.ivf_trace).toBeNull()
  await expect(page.getByText('Enable “compare with exact” to see the probe order')).toBeVisible()
  expect(calls).toHaveLength(2)
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

test('quantization: analysis loads with error, compression and distortion', async ({ page }) => {
  await page.goto('/#/quantization')
  await expect(page.getByText('Mean squared error')).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText('Distance fidelity (near pairs)')).toBeVisible()
  await expect(page.getByText('True vs approximate distance')).toBeVisible()
  const worst = page.locator('section', { hasText: 'Worst-reconstructed vectors' }).locator('tbody tr')
  await expect(worst).toHaveCount(50)
})

test('large int64 ids survive searches, result links, reloads and HNSW traces', async ({ page }) => {
  const id = '9007199254740993'
  await page.goto('http://127.0.0.1:8797/#/query')
  await page.getByRole('radio', { name: 'Stored id' }).click()
  await page.getByPlaceholder('e.g. 42').fill(id)
  const searched = page.waitForResponse((r) => r.url().endsWith('/api/search') && r.request().method() === 'POST')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  const response = await searched
  expect(response.request().postDataJSON().query.id).toBe(id)
  const body = await response.json()
  expect(body.results).toHaveLength(10)
  expect(body.results.every((r: { id: unknown }) => typeof r.id === 'string')).toBe(true)
  await expect(page.getByText('Recall@10')).toBeVisible()
  expect(page.url()).toContain(`id=${id}`)

  const next = body.results[0].id
  const rows = page.locator('section', { hasText: 'Approximate results' })
  const linked = page.waitForResponse((r) => r.url().endsWith('/api/search') && r.request().postDataJSON()?.query.id === next)
  await rows.getByRole('button', { name: next, exact: true }).click()
  await linked
  const reloaded = page.waitForResponse((r) => r.url().endsWith('/api/search'))
  await page.reload()
  expect((await reloaded).request().postDataJSON().query.id).toBe(next)

  const traced = page.waitForResponse((r) => r.url().endsWith('/api/trace/hnsw'))
  await page.goto(`http://127.0.0.1:8797/#/hnsw?id=${id}&ef=16`)
  const traceResponse = await traced
  expect(traceResponse.request().postDataJSON().query.id).toBe(id)
  const trace = await traceResponse.json()
  expect(trace.nodes.ids.every((v: unknown) => typeof v === 'string')).toBe(true)
  await expect(page.getByText('Reconstructed trace.')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Next step' })).toBeVisible()
})
