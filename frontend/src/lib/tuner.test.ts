import { describe, expect, it } from 'vitest'
import type { SweepPoint } from '../api/types'
import { codeSnippet, parseValues, recommend, speedup, prefersLogAxis } from './tuner'

const pt = (value: number, recall: number, latency: number): SweepPoint => ({
  value,
  recall,
  latency_mean_ms: latency,
  latency_p95_ms: latency * 1.2,
})
const points = [pt(1, 0.65, 0.1), pt(2, 0.88, 0.15), pt(4, 0.99, 0.2), pt(8, 1, 0.4), pt(16, 1, 0.8)]

describe('recommend', () => {
  it('picks the smallest value meeting the target', () => {
    expect(recommend(points, 0.95)?.value).toBe(4)
    expect(recommend(points, 0.88)?.value).toBe(2)
    expect(recommend(points, 1)?.value).toBe(8)
  })
  it('ignores input order', () => {
    expect(recommend([...points].reverse(), 0.95)?.value).toBe(4)
  })
  it('returns null when no value reaches the target', () => {
    expect(recommend(points.slice(0, 2), 0.95)).toBeNull()
  })
})

describe('speedup', () => {
  it('compares with the largest value tried', () => {
    expect(speedup(points, points[2])).toBeCloseTo(4)
    expect(speedup(points, points[4])).toBeNull()
  })
})

describe('parseValues', () => {
  it('parses, dedupes and sorts', () => {
    expect(parseValues('8, 1 2,,4 2', null)).toEqual([1, 2, 4, 8])
  })
  it('rejects bad input', () => {
    expect(parseValues('', null)).toBeNull()
    expect(parseValues('1, x', null)).toBeNull()
    expect(parseValues('0', null)).toBeNull()
    expect(parseValues('1.5', null)).toBeNull()
    expect(parseValues('1, 200', 128)).toBeNull()
  })
})

describe('codeSnippet', () => {
  it('writes IVF and HNSW snippets', () => {
    expect(codeSnippet('nprobe', 24, false)).toContain('faiss.SearchParametersIVF(nprobe=24)')
    expect(codeSnippet('nprobe', 24, false)).toContain('set_index_parameter(index, "nprobe", 24)')
    expect(codeSnippet('efSearch', 64, false)).toContain('faiss.SearchParametersHNSW(efSearch=64)')
  })
  it('wraps params for refine indexes', () => {
    expect(codeSnippet('nprobe', 8, true)).toContain('IndexRefineSearchParameters')
  })
})

describe('prefersLogAxis', () => {
  it('uses log scale for wide positive ranges only', () => {
    expect(prefersLogAxis([1, 2, 4, 8, 16])).toBe(true)
    expect(prefersLogAxis([10, 12, 14])).toBe(false)
    expect(prefersLogAxis([1, 64])).toBe(false)
  })
})
