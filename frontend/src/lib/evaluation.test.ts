import { describe, expect, it } from 'vitest'
import { evaluationChecklist, halfWidth } from './evaluation'

const info = (queries: number | null, truth: 'raw' | 'reconstructed' | null = 'raw') => ({
  ground_truth_source: truth,
  inputs: {
    raw_vectors: truth === 'raw',
    metadata_rows: null,
    metadata_columns: [],
    metadata_coverage: null,
    embedder: null,
    embedder_status: null,
    queries,
  },
})
const byKey = (items: ReturnType<typeof evaluationChecklist>) => Object.fromEntries(items.map((i) => [i.key, i]))

describe('evaluationChecklist', () => {
  it('flags sampled queries and reconstructed ground truth before a run', () => {
    const c = byKey(evaluationChecklist(info(null, 'reconstructed')))
    expect([c.queries.ok, c.truth.ok, c.precision.ok, c.relevance.ok]).toEqual([false, false, null, null])
    expect(c.queries.detail).toContain('--queries')
  })
  it('uses the finished run’s query origin and precision', () => {
    const good = byKey(evaluationChecklist(info(null), { queryOrigin: 'given', nQueries: 500, halfWidth: 0.004 }))
    expect([good.queries.ok, good.truth.ok, good.precision.ok]).toEqual([true, true, true])
    const loose = byKey(evaluationChecklist(info(1000), { queryOrigin: 'sampled', nQueries: 200, halfWidth: 0.02 }))
    expect(loose.queries.ok).toBe(false)
    expect(loose.precision.ok).toBe(false)
    // Width shrinks with sqrt(n): halving it needs 4x the queries.
    expect(loose.precision.detail).toContain('about 800 queries')
  })
})

describe('halfWidth', () => {
  it('is half the interval, or null without one', () => {
    expect(halfWidth(0.9, 0.94)).toBeCloseTo(0.02)
    expect(halfWidth(null, 0.9)).toBeNull()
  })
})
