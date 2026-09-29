import { describe, expect, it } from 'vitest'
import type { HnswTrace } from '../api/types'
import { buildFrames, frameLabel, stateAt } from './hnswFrames'

const v = (node: number, accepted: boolean) => ({ node, distance: node, accepted })

// Level 1: 10 -> 11 (improves), 11 -> nothing better. Level 0 from 11: expand 11, then 12.
const trace = {
  levels: [
    { level: 1, entry: 10, steps: [
      { expanded: 10, expanded_distance: 10, visits: [v(11, true), v(13, false)] },
      { expanded: 11, expanded_distance: 11, visits: [v(10, false)] },
    ] },
    { level: 0, entry: 11, steps: [
      { expanded: 11, expanded_distance: 11, visits: [v(12, true), v(14, true), v(15, false)] },
      { expanded: 12, expanded_distance: 12, visits: [v(16, true)] },
    ] },
  ],
} as unknown as HnswTrace

describe('buildFrames', () => {
  it('orders frames top level first', () => {
    const frames = buildFrames(trace)
    expect(frames.map((f) => [f.level, f.expanded])).toEqual([[1, 10], [1, 11], [0, 11], [0, 12]])
    expect(frameLabel(frames[3])).toBe('Level 0 · step 2 of 2')
  })
})

describe('stateAt', () => {
  const frames = buildFrames(trace)
  it('tracks the greedy nearest on upper levels', () => {
    const s = stateAt(trace, frames, 0)
    expect([...s.visited].sort()).toEqual([10, 11, 13])
    expect([...s.frontier]).toEqual([11])
    expect(s.entries).toEqual([{ level: 1, node: 10 }])
  })
  it('resets per level and tracks the candidate queue on level 0', () => {
    const s = stateAt(trace, frames, 2)
    expect([...s.visited].sort()).toEqual([11, 12, 14, 15])
    expect([...s.frontier].sort()).toEqual([12, 14])
    expect(s.entries.map((e) => e.level)).toEqual([1, 0])
    const last = stateAt(trace, frames, 3)
    expect([...last.frontier].sort()).toEqual([14, 16])
    expect([...last.expanded].sort()).toEqual([11, 12])
    expect(last.isLast).toBe(true)
  })
  it('clamps out-of-range indices', () => {
    expect(stateAt(trace, frames, 99).frame.expanded).toBe(12)
    expect(stateAt(trace, frames, -5).frame.expanded).toBe(10)
  })
})
