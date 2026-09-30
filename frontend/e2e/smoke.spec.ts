import { expect, test } from '@playwright/test'
import { readFile } from 'node:fs/promises'

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

test('a view that fails to load shows a recoverable error, not a blank page', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
  await page.route(/assets\/Tuner-[^/]+\.js$/, (route) => route.abort())
  await page.getByRole('button', { name: /Tuner/ }).click()
  await expect(page.getByText('This view could not be loaded from the faissight server')).toBeVisible()
  await expect(page.getByRole('button', { name: 'Reload page' })).toBeVisible()
  // The rest of the app still works.
  await page.getByRole('button', { name: /Overview/ }).click()
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
})

test('a dropped server connection surfaces as an error and the explorer recovers', async ({ page }) => {
  await page.goto('/#/query')
  await page.getByRole('radio', { name: 'Stored id' }).click()
  await page.getByPlaceholder('e.g. 42').fill('7')
  await page.route('**/api/search', (route) => route.abort('connectionreset'))
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('NETWORK_ERROR')
  await expect(page.getByRole('button', { name: 'Search', exact: true })).toBeEnabled()

  await page.unroute('**/api/search')
  await page.getByPlaceholder('e.g. 42').fill('8')
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await expect(page.getByText('Recall@10')).toBeVisible()
})

test('a newer search cancels the one still in flight', async ({ page }) => {
  await page.goto('/#/query')
  await page.getByRole('radio', { name: 'Stored id' }).click()
  // Hold the first search so the second overtakes it.
  let held = true
  await page.route('**/api/search', async (route) => {
    if (held) {
      held = false
      await new Promise((r) => setTimeout(r, 1500))
    }
    await route.continue().catch(() => {})  // the browser may have cancelled it meanwhile
  })
  // Record which searches the client itself aborted. In-page code is a string because
  // e2e/ is type-checked without DOM types.
  await page.addInitScript(`
    window.aborted = []
    const fetch = window.fetch
    window.fetch = (input, init) => {
      if (String(input).endsWith('api/search'))
        init?.signal?.addEventListener('abort', () => window.aborted.push(JSON.parse(init.body).query.id))
      return fetch(input, init)
    }
  `)
  await page.reload()
  await page.getByRole('radio', { name: 'Stored id' }).click()
  await page.getByPlaceholder('e.g. 42').fill('7')
  const first = page.waitForRequest((r) => r.url().endsWith('/api/search'))
  await page.getByRole('button', { name: 'Search', exact: true }).click()
  await first
  const second = page.waitForResponse(
    (r) => r.url().endsWith('/api/search') && r.request().postDataJSON().query.id === 8,
  )
  await page.evaluate(`location.hash = '#/query?id=8'`)
  const expected = (await (await second).json()).results[0]
  await expect(page.getByText('Recall@10')).toBeVisible()
  expect(await page.evaluate('window.aborted')).toEqual([7])
  const firstResult = page.locator('section', { hasText: 'Approximate results' }).locator('tbody tr').first()
  await expect(firstResult.getByRole('button', { name: String(expected.id), exact: true })).toBeVisible()
  await expect(page.getByRole('alert')).toHaveCount(0)
})

test('tuner: sweep and get a recommendation', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })
  await expect(page.getByText('faiss.SearchParametersIVF(nprobe=')).toBeVisible()
  // The e2e server samples stored vectors as queries: the checklist says so.
  const trust = page.locator('section', { hasText: 'How far to trust this' })
  await expect(trust.getByText('Queries are sampled stored vectors')).toBeVisible()
  await expect(trust.getByText('Recall is not relevance')).toBeVisible()
})

test('tuner: per-query recall and worst queries lead to the explorer', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText(/Per-query recall@10 at nprobe \d+/)).toBeVisible({ timeout: 20_000 })
  // The smallest value misses neighbours on this data; inspect it from the table.
  await page.locator('section', { hasText: 'All measurements' }).locator('tbody tr').first().click()
  await expect(page.getByText('Worst queries at nprobe 1')).toBeVisible()
  await page.getByRole('radio', { name: '95% lower bound ≥ target' }).click()
  await expect(page.getByText('smallest value confidently meeting the target')).toBeVisible()
  // The e2e server has no --queries, so worst queries are stored ids the explorer can open.
  await page.getByRole('button', { name: 'Explain' }).first().click()
  await expect(page).toHaveURL(/#\/query\?id=\d+&k=10&nprobe=1/)
  await expect(page.getByRole('heading', { name: 'Query explorer' })).toBeVisible()
  await expect(page.getByText('Recall@10', { exact: true })).toBeVisible()
})

test('tuner: suggested next steps explain a miss and run the follow-up sweep', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('textbox').first().fill('1')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })
  await page.getByRole('slider').fill('1')
  const steps = page.locator('section', { hasText: 'Suggested next steps' })
  // IVF-Flat codes are exact, so every miss at nprobe 1 is an unprobed list.
  await expect(steps.getByText(/Probe more lists: try nprobe up to \d+/)).toBeVisible()
  await expect(steps.getByText(/In probed lists at nprobe 1/)).toBeVisible()
  await expect(steps.getByText('Compression is limiting recall')).toHaveCount(0)
  await expect(page.getByRole('columnheader', { name: 'In probed lists' }).first()).toBeVisible()

  await steps.getByRole('button', { name: /^Sweep nprobe 1, / }).click()
  await expect(page).toHaveURL(/values=1%2C2/, { timeout: 10_000 })
  await expect(page.getByRole('textbox').first()).toHaveValue(/^1, 2/)
  await expect(page.locator('section', { hasText: 'All measurements' }).locator('tbody tr')).not.toHaveCount(1, {
    timeout: 20_000,
  })
})

test('tuner: a held-out evaluation query opens in the explorer by its row', async ({ page }) => {
  // The HNSW server has --queries: worst queries have no stored id, only a row.
  await page.goto('http://127.0.0.1:8798/#/tuner')
  await page.getByRole('textbox').first().fill('4, 16')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  const worst = page.locator('section', { hasText: /Worst queries at efSearch/ })
  await expect(worst.getByText(/^query #\d+$/).first()).toBeVisible({ timeout: 20_000 })
  await worst.getByRole('button', { name: 'Explain' }).first().click()
  await expect(page).toHaveURL(/#\/query\?row=\d+&k=10&ef=\d+/)
  await expect(page.getByRole('radio', { name: 'Evaluation query' })).toBeChecked()
  await expect(page.getByText('Recall@10', { exact: true })).toBeVisible()
  const truth = page.locator('section', { hasText: 'Exact nearest neighbours' }).locator('tbody tr')
  await expect(truth).toHaveCount(10)
})

test('tuner: a p95 latency budget the target can\'t fit in is explained', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })
  const budget = page.getByLabel('p95 latency budget (ms)')
  await budget.fill('0.000001')
  await expect(page.getByText('Not reached')).toBeVisible()
  await expect(page.getByText(/recall needs \d+, p95 .* ms: over budget/)).toBeVisible()
  const steps = page.locator('section', { hasText: 'Suggested next steps' })
  await expect(steps.getByText('No setting meets both the recall target and the latency budget')).toBeVisible()
  await expect(page.getByText('over budget', { exact: false }).first()).toBeVisible()
  await budget.fill('')
  await expect(page.getByText('Not reached')).toHaveCount(0)
  await expect(steps.getByText('No setting meets both')).toHaveCount(0)
})

test('tuner: save a run, then compare later sweeps with it', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('textbox').first().fill('1, 2')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })

  const downloading = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Save run' }).click()
  const saved = await downloading
  expect(saved.suggestedFilename()).toMatch(/^faissight-run-nprobe-\d{4}-\d{2}-\d{2}\.json$/)
  const run = JSON.parse(await readFile(await saved.path(), 'utf8'))
  expect(run.format).toBe('faissight.sweep-run')

  const card = page.locator('section', { hasText: 'Compare with a saved run' })
  const file = card.getByLabel('Saved run file')
  await file.setInputFiles({ name: 'same.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(run)) })
  await expect(card.getByText('No regressions against this run')).toBeVisible()
  await expect(card.locator('tbody tr')).toHaveCount(2)

  // A baseline that was much faster (a tiny fixed latency, not this run's real one --
  // recall on this small dataset is already near-ceiling, leaving no room to regress it):
  // this sweep regressed on latency at every setting. A near-zero floor makes that
  // detectable regardless of how fast real search is on this tiny synthetic index.
  await card.getByLabel('and at least (ms)').fill('0')
  const slower = JSON.parse(JSON.stringify(run))
  for (const p of slower.points) p.latency_p95_ms = 0.0001
  await file.setInputFiles({ name: 'faster-baseline.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(slower)) })
  await expect(card.getByText(/Regressed at 2 of 2 settings/)).toBeVisible()
  await expect(card.getByText('regressed: p95').first()).toBeVisible()

  // A malformed file must clear that regression verdict, not leave it looking current.
  await file.setInputFiles({ name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"format": "x"}') })
  await expect(card.getByText(/Not a faissight sweep run/)).toBeVisible()
  await expect(card.getByText(/Regressed at 2 of 2 settings/)).toHaveCount(0)
  await expect(card.locator('tbody tr')).toHaveCount(0)

  // Changing a threshold without re-running flags the shown result as stale.
  await file.setInputFiles({ name: 'same2.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(run)) })
  await expect(card.getByText('No regressions against this run')).toBeVisible()
  await expect(card.getByText(/thresholds below changed/)).toHaveCount(0)
  await card.getByLabel('p95 growth allowed (%)').fill('999')
  await expect(card.getByText(/thresholds below changed/)).toBeVisible()
  await card.getByRole('button', { name: 'Compare again' }).click()
  await expect(card.getByText(/thresholds below changed/)).toHaveCount(0)
})

test('tuner: a saved run round-trips large int64 ids exactly', async ({ page }) => {
  await page.goto('http://127.0.0.1:8797/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByText(/Recommended (nprobe|efSearch)/)).toBeVisible({ timeout: 20_000 })

  const downloading = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Save run' }).click()
  const saved = await downloading
  const text = await readFile(await saved.path(), 'utf8')
  const run = JSON.parse(text)
  const worstIds = run.points.flatMap((p: { worst_queries: { id: string | number | null }[] }) =>
    p.worst_queries.map((w) => w.id).filter((id: unknown) => id !== null),
  )
  expect(worstIds.length).toBeGreaterThan(0)
  for (const id of worstIds) {
    // The large ids in this fixture are all >= 2**53 + 1: JS can only hold those exactly
    // as strings, so a bare (roundable) number in the saved file would mean precision loss.
    expect(typeof id).toBe('string')
    expect(BigInt(id) >= 2n ** 53n + 1n).toBe(true)
  }
})

test('compare: side-by-side recall, latency, size and changed neighbours', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: /Compare/ }).click()
  await expect(page.getByRole('heading', { name: 'Compare' })).toBeVisible()
  await page.getByLabel('nprobe (other)').fill('1')
  await page.getByRole('button', { name: 'Compare', exact: true }).click()
  await expect(page.getByText('ivf_pq.index vs ivf_flat.index:')).toBeVisible({ timeout: 20_000 })
  await expect(page).toHaveURL(/#\/compare\?job=\w+&candidate=0/)
  const side = page.locator('section', { hasText: 'Side by side' })
  await expect(side.getByRole('cell', { name: 'IVF_PQ', exact: true })).toBeVisible()
  await expect(side.getByText('p95 latency')).toBeVisible()
  await expect(page.getByText(/Queries with different results/)).toBeVisible()
  const changes = page.locator('section', { hasText: 'Changed neighbours' }).locator('tbody tr')
  await expect(changes.first()).toBeVisible()
  // A reload keeps the comparison and the settings it ran with.
  await page.reload()
  await expect(page.getByText('ivf_pq.index vs ivf_flat.index:')).toBeVisible()
  await expect(page.getByLabel('nprobe (other)')).toHaveValue('1')
  await changes.first().getByRole('button', { name: 'Explain' }).click()
  await expect(page.getByRole('heading', { name: 'Query explorer' })).toBeVisible()
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

test('a failed analysis stays failed until "Try again" restarts it', async ({ page }) => {
  const urls: string[] = []
  await page.route('**/api/pq/error*', async (route) => {
    urls.push(route.request().url())
    if (urls.length === 1)
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error_code: 'PQ_FAILED', message: 'Quantization analysis failed: boom', hint: null }),
      })
    else await route.continue()
  })
  await page.goto('/#/quantization')
  await expect(page.getByText('Quantization analysis failed: boom').first()).toBeVisible()
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('Mean squared error')).toBeVisible({ timeout: 20_000 })
  expect(urls[1]).toContain('retry=true')
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

test('overview: a failed list-size request shows an error with a working retry', async ({ page }) => {
  await page.route('**/api/ivf/lists', (route) =>
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({ error_code: 'INTERNAL', message: 'boom', hint: 'Check the server log.' }),
    }),
  )
  await page.goto('/')
  await expect(page.getByText('Could not read the inverted lists.')).toBeVisible({ timeout: 10_000 })
  await expect(page.getByRole('main').getByText('boom Check the server log.')).toBeVisible()
  await page.unroute('**/api/ivf/lists')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByText('List size distribution')).toBeVisible()
  await expect(page.getByText('Could not read the inverted lists.')).toHaveCount(0)
})

test('narrow frames hide the sidebar and open it as an overlay', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 })
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Overview' })).toBeVisible()
  const nav = page.getByRole('navigation', { name: 'Views' })
  await expect(nav).toBeHidden()
  await page.getByRole('button', { name: '☰ Menu' }).click()
  await expect(nav).toBeVisible()
  await nav.getByRole('button', { name: /Tuner/ }).click()
  await expect(page.getByRole('heading', { name: 'Tuner' })).toBeVisible()
  await expect(nav).toBeHidden()
  // Widening the frame brings the sidebar back.
  await page.setViewportSize({ width: 1280, height: 800 })
  await expect(nav).toBeVisible()
})

test('tuner: cancel, then restart with identical settings, resumes the sweep', async ({ page }) => {
  await page.goto('/#/tuner')
  // Enough work (16 values x 2000 queries x 20 repeats) to cancel while it runs.
  await page.getByRole('textbox').first().fill(Array.from({ length: 16 }, (_, i) => i + 1).join(', '))
  await page.getByLabel('queries (sampled)').fill('2000')
  await page.getByLabel('Timing repeats').fill('20')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await page.getByRole('button', { name: 'Cancel sweep' }).click()
  await expect(page.getByText('Sweep cancelled.')).toBeVisible({ timeout: 10_000 })
  const url = page.url()

  // Same settings -> same job id and URL; the cached "cancelled" state must not stick.
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByRole('progressbar')).toBeVisible()
  await expect(page.getByText('Sweep cancelled.')).toHaveCount(0)
  expect(page.url()).toBe(url)
  await page.getByRole('button', { name: 'Cancel sweep' }).click()
  await expect(page.getByText('Sweep cancelled.')).toBeVisible({ timeout: 10_000 })
  // "Run again" restarts it the same way.
  await page.getByRole('button', { name: 'Run again' }).click()
  await expect(page.getByRole('progressbar')).toBeVisible()
  await page.getByRole('button', { name: 'Cancel sweep' }).click()
  await expect(page.getByText('Sweep cancelled.')).toBeVisible({ timeout: 10_000 })
})

test('tuner: a job the server no longer has can be run again from the link', async ({ page }) => {
  // e.g. a bookmarked result after the server restarted: the settings live in the URL.
  await page.goto('/#/tuner?job=000000000000&values=1,2&k=5&n=30&repeats=1&seed=4')
  await expect(page.getByText('This sweep is no longer on the server')).toBeVisible()
  await expect(page.getByRole('textbox').first()).toHaveValue('1, 2')
  await expect(page.getByLabel('Seed')).toHaveValue('4')
  await page.getByRole('button', { name: 'Run again' }).click()
  await expect(page.getByText('Recommended nprobe')).toBeVisible({ timeout: 20_000 })
  await expect(page).toHaveURL(/job=(?!000000000000)\w+&values=1%2C2&k=5&n=30&repeats=1&seed=4/)
  await expect(page.getByText('no longer on the server')).toHaveCount(0)
})

test('compare: an expired comparison offers "Run again" with its settings', async ({ page }) => {
  await page.goto('/#/compare?job=000000000000&candidate=0&k=10&n=40&repeats=1&seed=0&left=4&right=2')
  await expect(page.getByText('This comparison is no longer on the server')).toBeVisible()
  await expect(page.getByLabel('nprobe (main)')).toHaveValue('4')
  await expect(page.getByLabel('nprobe (other)')).toHaveValue('2')
  await page.getByRole('button', { name: 'Run again' }).click()
  await expect(page.getByText('ivf_pq.index vs ivf_flat.index:')).toBeVisible({ timeout: 20_000 })
  const side = page.locator('section', { hasText: 'Side by side' })
  await expect(side.getByRole('cell', { name: 'nprobe 4', exact: true })).toBeVisible()
  await expect(side.getByRole('cell', { name: 'nprobe 2', exact: true })).toBeVisible()
})

test('tuner: a server restart during a sweep shows "no longer on the server", not stale progress', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('textbox').first().fill(Array.from({ length: 16 }, (_, i) => i + 1).join(', '))
  await page.getByLabel('queries (sampled)').fill('2000')
  await page.getByLabel('Timing repeats').fill('20')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  await expect(page.getByRole('progressbar')).toBeVisible()
  const jobUrl = new URL(page.url().replace('#/', '')).searchParams.get('job')
  // The restarted server no longer knows the job.
  await page.route(`**/api/sweep/${jobUrl}`, (route) =>
    route.fulfill({
      status: 404,
      contentType: 'application/json',
      body: JSON.stringify({ error_code: 'NOT_FOUND', message: 'No sweep.', hint: null }),
    }),
  )
  await expect(page.getByText('This sweep is no longer on the server')).toBeVisible()
  await expect(page.getByRole('progressbar')).toHaveCount(0)
  await expect(page.getByRole('button', { name: 'Run sweep' })).toBeEnabled()
  // Clean up the real job.
  await page.unroute(`**/api/sweep/${jobUrl}`)
  await page.request.delete(`/api/sweep/${jobUrl}`)
})
