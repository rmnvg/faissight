import { expect, test, type Page } from '@playwright/test'

/** The history row's name input whose value matches, once the list shows it. */
async function runName(page: Page, pattern: RegExp) {
  const find = async () => {
    for (const input of await page.getByLabel('Run name').all()) {
      if (pattern.test(await input.inputValue())) return input
    }
    return null
  }
  await expect.poll(find, { timeout: 10_000 }).not.toBeNull()
  return (await find())!
}

// Non-WebGL workflows run on all three browser engines.
test('navigation menu traps focus, closes with Escape, and restores focus', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 })
  await page.goto('/')
  const open = page.getByRole('button', { name: '☰ Menu' })
  await open.focus()
  await page.keyboard.press('Enter')
  const dialog = page.getByRole('dialog', { name: 'Navigation menu' })
  await expect(dialog).toBeVisible()
  await expect(dialog.getByRole('button', { name: /Overview/ })).toBeFocused()
  await page.keyboard.press('Shift+Tab')
  await expect(dialog.getByRole('button', { name: 'Hide sidebar' })).toBeFocused()
  await page.keyboard.press('Tab')
  await expect(dialog.getByRole('button', { name: /Overview/ })).toBeFocused()
  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await expect(open).toBeFocused()
})

test('relevance evaluation loads judgements, reranks, exports, and archives the result', async ({ page }, testInfo) => {
  const base = 'http://127.0.0.1:8798'
  // Browsers share one server; distinct settings give each its own job and saved run.
  // Keyed on the browser, not the worker: with one worker every project gets index 0.
  const browser = ['chromium', 'firefox', 'webkit'].indexOf(testInfo.project.name)
  const candidates = 20 + Math.max(browser, 0) + 3 * testInfo.repeatEachIndex
  const info = await (await page.request.get(`${base}/api/info`)).json()
  await page.goto(`${base}/#/evaluation`)
  await expect(page.getByRole('heading', { name: 'Relevance', exact: true })).toBeVisible()
  const judgements = Array.from({ length: info.inputs.queries }, (_, row) =>
    JSON.stringify({ row, relevant: { '0': 1 } }),
  ).join('\n')
  await page.getByLabel('Judgements file').setInputFiles({
    name: 'labels.jsonl',
    mimeType: 'application/json',
    buffer: Buffer.from(judgements),
  })
  await expect(page.getByLabel('…or paste JSONL')).toHaveValue(judgements)
  await page.getByLabel('Rerank candidates (0 = off)').fill(String(candidates))
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click()
  await expect(page.getByRole('heading', { name: 'Labelled relevance @10' })).toBeVisible({ timeout: 20000 })
  await expect(page.getByText(/after reranking/).first()).toBeVisible()
  await expect(page.getByRole('button', { name: 'Explain query row 0' })).toBeVisible()
  const download = page.waitForEvent('download')
  await page.getByRole('button', { name: 'JSON', exact: true }).click()
  expect((await download).suggestedFilename()).toBe('faissight-relevance.json')

  await page.goto(`${base}/#/history`)
  const name = await runName(page, new RegExp(`^Relevance@10, rerank ${candidates} `))
  const renamed = `Browser evaluation ${testInfo.project.name} ${candidates}`
  await name.fill(renamed)
  await name.press('Enter')
  await page.reload()
  await runName(page, new RegExp(`^${renamed}$`))
})

test('tuner measurements can be inspected using the keyboard', async ({ page }) => {
  await page.goto('/#/tuner')
  await page.getByRole('button', { name: 'Run sweep' }).click()
  const row = page.getByRole('row', { name: 'Inspect nprobe 1', exact: true })
  await expect(row).toBeVisible({ timeout: 20000 })
  await row.focus()
  await page.keyboard.press('Enter')
  await expect(page.getByText('Worst queries at nprobe 1')).toBeVisible()
})

test('relevance explains missing inputs and invalid judgements', async ({ page }) => {
  // The default server has raw vectors but no --queries.
  await page.goto('/#/evaluation')
  await expect(page.getByText('Needs held-out queries and raw vectors')).toBeVisible()
  await expect(page.getByRole('button', { name: /Relevance/ })).toBeDisabled()
  await page.goto('http://127.0.0.1:8798/#/evaluation')
  await page.getByLabel('…or paste JSONL').fill('{}')
  await page.getByRole('button', { name: 'Evaluate', exact: true }).click()
  await expect(page.getByRole('alert')).toContainText('Invalid judgements')
})
