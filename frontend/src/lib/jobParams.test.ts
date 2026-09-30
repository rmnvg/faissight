import { describe, expect, it } from 'vitest'
import { compareParams, runSettingsFromParams, sweepParams, sweepValuesText } from './jobParams'
import { toQueryString } from './route'

const roundTrip = (p: Record<string, string | number | null>) => new URLSearchParams(toQueryString(p))

describe('sweep URL settings', () => {
  it('round-trips the settings needed to run a sweep again', () => {
    const url = roundTrip(sweepParams('abc', { param: 'nprobe', values: [1, 4, 16], k: 5, n_queries: 50, repeats: 2, seed: 7 }))
    expect(url.get('job')).toBe('abc')
    expect(sweepValuesText(url)).toBe('1, 4, 16')
    expect(runSettingsFromParams(url)).toEqual({ k: 5, nQueries: 50, repeats: 2, seed: 7 })
  })
  it('falls back to defaults for missing or invalid values', () => {
    const url = new URLSearchParams('k=0&n=abc&seed=-1&values=1,x')
    expect(runSettingsFromParams(url)).toEqual({ k: 10, nQueries: 200, repeats: 3, seed: 0 })
    expect(sweepValuesText(url)).toBeNull()
    expect(sweepValuesText(new URLSearchParams())).toBeNull()
  })
})

describe('compare URL settings', () => {
  it('keeps each side’s search parameter', () => {
    const url = roundTrip(
      compareParams('j1', { candidate: 1, k: 10, n_queries: 100, repeats: 3, seed: 0, left_nprobe: 8, right_ef_search: 64 }),
    )
    expect([url.get('candidate'), url.get('left'), url.get('right'), url.get('seed')]).toEqual(['1', '8', '64', '0'])
    expect(runSettingsFromParams(url).nQueries).toBe(100)
  })
})
