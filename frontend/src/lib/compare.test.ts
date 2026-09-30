import { describe, expect, it } from 'vitest'
import type { CompareMeasurement } from '../api/types'
import { fmtBytes, idList, recallIntervalsOverlap, relative, verdict } from './compare'

const m = (over: Partial<CompareMeasurement>): CompareMeasurement => ({
  name: 'a.index',
  kind: 'IVF_FLAT',
  params: { nprobe: 8 },
  serialized_bytes: 1_000_000,
  recall: 0.99,
  recall_ci_low: 0.98,
  recall_ci_high: 1,
  latency_mean_ms: 0.2,
  latency_p95_ms: 0.4,
  ...over,
})

describe('fmtBytes', () => {
  it('uses binary units', () => {
    expect(fmtBytes(512)).toBe('512 B')
    expect(fmtBytes(2048)).toBe('2.00 KiB')
    expect(fmtBytes(50 * 1024 * 1024)).toBe('50.0 MiB')
  })
})

describe('relative', () => {
  it('reads as smaller/larger factors with a tolerance band', () => {
    expect(relative(0.4, 0.2, ['faster', 'slower'])).toEqual({ text: '2.0× faster', better: true })
    expect(relative(0.2, 0.5, ['faster', 'slower'])).toEqual({ text: '2.5× slower', better: false })
    expect(relative(1, 1.03, ['smaller', 'larger']).text).toBe('about the same')
    expect(relative(0, 1, ['smaller', 'larger']).text).toBe('—')
  })
})

describe('verdict', () => {
  it('summarises recall, p95 latency and size', () => {
    const right = m({ name: 'b.index', recall: 0.71, latency_p95_ms: 0.1, serialized_bytes: 125_000 })
    expect(verdict(m({}), right, 10)).toBe('b.index vs a.index: −0.280 recall@10, 4.0× faster at p95, 8.0× smaller.')
    expect(verdict(m({}), m({ name: 'c.index' }), 10)).toBe(
      'c.index vs a.index: the same recall@10, about the same p95 latency, about the same size.',
    )
  })
})

describe('recallIntervalsOverlap', () => {
  it('flags gaps that may be sampling noise', () => {
    expect(recallIntervalsOverlap(m({}), m({ recall_ci_low: 0.95, recall_ci_high: 0.985 }))).toBe(true)
    expect(recallIntervalsOverlap(m({}), m({ recall_ci_low: 0.6, recall_ci_high: 0.7 }))).toBe(false)
    expect(recallIntervalsOverlap(m({}), m({ recall_ci_low: null }))).toBe(false)
  })
})

describe('idList', () => {
  it('truncates long lists', () => {
    expect(idList([])).toBe('—')
    expect(idList([1, 2, '9007199254740993'])).toBe('1, 2, 9007199254740993')
    expect(idList([1, 2, 3, 4, 5, 6, 7], 3)).toBe('1, 2, 3 +4 more')
  })
})
